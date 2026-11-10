"""訓練單一方法，輸出 OOF 與 test 機率到 outputs/oof/<name>.npz，供 arena.py 比較與集成。

用法：
  python run_method.py lgbm            # 單一方法
  python run_method.py lgbm xgb cat    # 依序多個
  python run_method.py --list          # 列出所有方法
  python run_method.py gru --device cuda --fast
"""
from __future__ import annotations
import argparse
import time

import numpy as np

from ttmatch import config
from ttmatch.data import build_datasets
from ttmatch.history import HIST_FEATURES


def _trees(kind, **kw):
    def f(D, a):
        from ttmatch.methods.trees import run
        return run(kind, D, a.folds, a.fast, **kw)
    f.uses_prog = bool(kw.get("use_prog", False))  # arena 依此判斷是否用了比分推移特徵
    return f


def _nn(arch, **kw):
    def f(D, a):
        from ttmatch.methods.nn import train_nn
        return train_nn(D, arch=arch, n_folds=a.folds, device=a.device, fast=a.fast,
                        epochs=a.epochs, n_seeds=a.seeds, **kw)
    f.uses_prog = False
    return f


# 名稱 → (說明, 訓練函式)。說明標註方法出處。
METHODS = {
    "lgbm": ("LightGBM + 選手 TE + test 轉移擴充（第 3 名 0000workjason）", _trees("lgbm")),
    "lgbm_note": ("LightGBM，不加選手 TE（對照組，檢驗第 3 名的選手 TE 是否有效）", _trees("lgbm", player_te=False)),
    "lgbm_te20": ("LightGBM action，選手 TE 改用較強平滑 alpha=20（本 repo 調整）",
                  _trees("lgbm", targets=("a",), te_alpha=20.0)),
    "lgbm_noplayer": ("LightGBM action/point，不用任何選手 ID 特徵（對照組）",
                      _trees("lgbm", targets=("a", "p"), player_te=False, drop=("next_player",))),
    # lgbm_note 證實選手 TE 在本資料無益，XGBoost / CatBoost 直接用同一組最佳特徵
    "xgb": ("XGBoost，特徵同 lgbm_note（lucas6028 多樹模型）", _trees("xgb", player_te=False)),
    "cat": ("CatBoost 原生類別特徵，特徵同 lgbm_note（lucas6028 多樹模型）", _trees("cat", player_te=False)),
    "lgbm_prog": ("LightGBM winner + 比分推移特徵（本 repo 新增）", _trees("lgbm", targets=("w",), use_prog=True)),
    "cat_prog": ("CatBoost winner + 比分推移特徵（本 repo 新增）", _trees("cat", targets=("w",), use_prog=True)),
    # --- 改善回合 1（本 repo 新增的實驗；全部用 lgbm_note 的特徵，不加選手 TE）---
    "lgbm_mirror": ("LightGBM action/point + 左右鏡像訓練列（第 22 名的增強套到樹模型）",
                    _trees("lgbm", targets=("a", "p"), player_te=False, mirror=True)),
    "lgbm_bag3": ("LightGBM action/point，3 個 seed 平均（seed bagging）",
                  _trees("lgbm", targets=("a", "p"), player_te=False, n_seeds=3)),
    "lgbm_reg": ("LightGBM action/point，較強正則化（葉數 31、min_data 80、feature_fraction 0.6、l2 5）",
                 _trees("lgbm", targets=("a", "p"), player_te=False,
                        params=dict(num_leaves=31, min_data_in_leaf=80, feature_fraction=0.6, lambda_l2=5.0))),
    "lgbm_big": ("LightGBM action/point，較大模型（葉數 127、min_data 20、lr 0.03、3000 輪上限）",
                 _trees("lgbm", targets=("a", "p"), player_te=False,
                        params=dict(num_leaves=127, min_data_in_leaf=20, learning_rate=0.03, rounds=3000))),
    "xgb_mirror": ("XGBoost action/point + 左右鏡像訓練列",
                   _trees("xgb", targets=("a", "p"), player_te=False, mirror=True)),
    "xgb_prog": ("XGBoost winner + 比分推移特徵", _trees("xgb", targets=("w",), use_prog=True)),
    "lgbm_prog_bag3": ("LightGBM winner + 比分推移，3 seed", _trees("lgbm", targets=("w",), use_prog=True, n_seeds=3)),
    "cat_prog_bag3": ("CatBoost winner + 比分推移，3 seed", _trees("cat", targets=("w",), use_prog=True, n_seeds=3)),
    "lgbm_pair": ("LightGBM action/point + 目前擊球者 ID（對戰組合資訊）",
                  _trees("lgbm", targets=("a", "p"), player_te=False, extra=("cur_player",))),
    "lgbm_hist": ("LightGBM action/point + 同場先前 rally 的選手傾向（history 特徵）",
                  _trees("lgbm", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES))),
    "lgbm_pair_hist": ("LightGBM action/point + 目前擊球者 ID + history 特徵",
                       _trees("lgbm", targets=("a", "p"), player_te=False,
                              extra=("cur_player",) + tuple(HIST_FEATURES))),
    "lgbm_hist_w": ("LightGBM winner + 比分推移 + history 特徵",
                    _trees("lgbm", targets=("w",), use_prog=True, extra=tuple(HIST_FEATURES))),
    "xgb_hist": ("XGBoost action/point + history 特徵",
                 _trees("xgb", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES))),
    "transformer_hist": ("因果 Transformer + history 特徵（每拍附下一拍擊球者的同場傾向）",
                         _nn("transformer", hist=True)),
    "gru_pemb_hist": ("GRU + 選手 ID 嵌入 + history 特徵", _nn("gru", player_emb=True, player_te=False, hist=True)),
    "gru": ("因果 GRU 多任務序列模型（第 3 名 0000workjason）", _nn("gru")),
    "transformer": ("因果 Transformer 多任務序列模型（第 3 名 0000workjason）", _nn("transformer")),
    "gru_t2v_film": ("GRU + Time2Vec + FiLM（第 5 名 SugarSquirrel v8）", _nn("t2v_film")),
    "gru_pemb": ("GRU + 選手 ID 嵌入（取代選手 TE，本 repo 新增）", _nn("gru", player_emb=True, player_te=False)),
    "transformer_pemb": ("因果 Transformer + 選手 ID 嵌入（本 repo 新增）",
                         _nn("transformer", player_emb=True, player_te=False)),
    "gru_mirror":("GRU + 左右鏡像資料增強（第 22 名 jabir95tsai）", _nn("gru", mirror=True)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("methods", nargs="*")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--fast", action="store_true", help="少量 boosting 輪數 / epoch，只用於冒煙測試")
    ap.add_argument("--folds", type=int, default=config.N_FOLDS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--suffix", default="", help="輸出檔名後綴，例如 _s3 → outputs/oof/<name>_s3.npz（不覆寫原檔）")
    a = ap.parse_args()
    if a.list or not a.methods:
        for k, (desc, _) in METHODS.items():
            print(f"{k:14s} {desc}")
        return

    config.ensure_dirs()
    D = build_datasets()
    for name in a.methods:
        desc, fn = METHODS[name]
        t0 = time.time()
        print(f">> {name}: {desc}", flush=True)
        out = fn(D, a)
        out = {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}
        out["rally_uid"] = D["ap_base"]["rally_uid"].to_numpy()
        out["prefix_len"] = D["ap_base"]["prefix_len"].to_numpy()
        out["test_rally_uid"] = D["test_pred"]["rally_uid"].to_numpy()
        out["uses_prog"] = np.array(fn.uses_prog)
        suffix = a.suffix + ("_fast" if a.fast else "")
        path = config.OOF_DIR / f"{name}{suffix}.npz"
        np.savez_compressed(path, **out)
        print(f"   saved {path.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
