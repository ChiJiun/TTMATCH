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
from ttmatch.features import PROG_FEATURES
from ttmatch.history import HIST_FEATURES, OHIST_FEATURES


def _trees(kind, **kw):
    def f(D, a):
        from ttmatch.methods.trees import run
        return run(kind, D, a.folds, a.fast, **kw)
    # arena 依此判斷是否用了比分推移特徵（winner 的 use_prog，或 action/point 的 extra 裡含 PROG_FEATURES）
    f.uses_prog = bool(kw.get("use_prog", False)) or bool(set(kw.get("extra", ())) & set(PROG_FEATURES))
    return f


def _nn(arch, **kw):
    def f(D, a):
        from ttmatch.methods.nn import train_nn
        return train_nn(D, arch=arch, n_folds=a.folds, device=a.device, fast=a.fast,
                        epochs=a.epochs, n_seeds=a.seeds, **kw)
    f.uses_prog = bool(kw.get("prog", False))
    return f


# 名稱 → (說明, 訓練函式)。說明標註方法出處。
METHODS = {
    "lgbm": ("LightGBM + 選手 TE + test 轉移擴充（第 3 名 0000workjason；保留原作法，其餘樹模型都不用 test 轉移擴充）",
             _trees("lgbm", use_aug=True)),
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
    "xgb_hist2": ("XGBoost action/point + 雙方 history 特徵（下一拍擊球者 + 目前擊球者）",
                  _trees("xgb", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES + OHIST_FEATURES))),
    "cat_hist": ("CatBoost action/point + history 特徵",
                 _trees("cat", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES))),
    "lgbm_hist2": ("LightGBM action/point + 雙方 history 特徵",
                   _trees("lgbm", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES + OHIST_FEATURES))),
    "transformer_hist": ("因果 Transformer + history 特徵（每拍附下一拍擊球者的同場傾向）",
                         _nn("transformer", hist=True)),
    "gru_pemb_hist": ("GRU + 選手 ID 嵌入 + history 特徵", _nn("gru", player_emb=True, player_te=False, hist=True)),
    "transformer_hist2": ("因果 Transformer + 雙方 history 特徵", _nn("transformer", hist="both")),
    # 比分推移揭露 rally 勝方 → 下一拍擊球者若是輸家，這拍有 39% 是失誤（落點 0）；若是勝方只有 0.7%
    "xgb_hist2_prog": ("XGBoost action/point + 雙方 history + 比分推移特徵",
                       _trees("xgb", targets=("a", "p"), player_te=False,
                              extra=tuple(HIST_FEATURES + OHIST_FEATURES + PROG_FEATURES))),
    "transformer_hist2_prog": ("因果 Transformer + 雙方 history + 比分推移特徵（三任務都用）",
                               _nn("transformer", hist="both", prog=True)),
    "lgbm_hist_prog": ("LightGBM action/point + history + 比分推移特徵",
                       _trees("lgbm", targets=("a", "p"), player_te=False,
                              extra=tuple(HIST_FEATURES + PROG_FEATURES))),
    "gru_hist": ("因果 GRU + history 特徵", _nn("gru", hist=True)),
    # --- 改善回合 2：序列模型的損失也依 test 截斷機制加權（樹模型預設就加權）---
    "gru_hist_al": ("因果 GRU + history 特徵，損失依截斷機制加權", _nn("gru", hist=True, align=True)),
    "transformer_hist2_al": ("因果 Transformer + 雙方 history，損失依截斷機制加權",
                             _nn("transformer", hist="both", align=True)),
    "transformer_hist2_prog_al": ("因果 Transformer + 雙方 history + 比分推移，損失依截斷機制加權",
                                  _nn("transformer", hist="both", prog=True, align=True)),
    "gru_pemb_hist_al": ("GRU + 選手 ID 嵌入 + history，損失依截斷機制加權",
                         _nn("gru", player_emb=True, player_te=False, hist=True, align=True)),
    "cat_hist_prog": ("CatBoost action/point + history + 比分推移特徵",
                      _trees("cat", targets=("a", "p"), player_te=False,
                             extra=tuple(HIST_FEATURES + PROG_FEATURES))),
    "lgbm_hist2_prog": ("LightGBM action/point + 雙方 history + 比分推移特徵",
                        _trees("lgbm", targets=("a", "p"), player_te=False,
                               extra=tuple(HIST_FEATURES + OHIST_FEATURES + PROG_FEATURES))),
    "xgb_hist_w": ("XGBoost winner + 比分推移 + history 特徵",
                   _trees("xgb", targets=("w",), use_prog=True, extra=tuple(HIST_FEATURES))),
    "cat_hist_w": ("CatBoost winner + 比分推移 + history 特徵",
                   _trees("cat", targets=("w",), use_prog=True, extra=tuple(HIST_FEATURES))),
    # 對照：樹模型加回 test 內部轉移擴充。CV 分數差不多，但 test 上幾乎不預測落點 0（見 trees.py 開頭）
    "lgbm_hist_aug": ("LightGBM action/point + history + test 轉移擴充（對照：test 上會低估終結拍）",
                      _trees("lgbm", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES), use_aug=True)),
    # 對照：action/point 不做截斷對齊（所有前綴等權，與改善回合 1 相同）。樹模型的 winner 加權後明顯進步，
    # 但 action 加權後有效樣本數只剩 56%，單模分數略降；兩種都留給 arena 用巢狀估計挑。
    "xgb_eq": ("XGBoost action/point，訓練列不加權（對照）",
               _trees("xgb", targets=("a", "p"), player_te=False, align=False)),
    "lgbm_note_eq": ("LightGBM action/point，訓練列不加權（對照）",
                     _trees("lgbm", targets=("a", "p"), player_te=False, align=False)),
    "lgbm_hist_eq": ("LightGBM action/point + history，訓練列不加權（對照）",
                     _trees("lgbm", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES), align=False)),
    "lgbm_hist2_eq": ("LightGBM action/point + 雙方 history，訓練列不加權（對照）",
                      _trees("lgbm", targets=("a", "p"), player_te=False,
                             extra=tuple(HIST_FEATURES + OHIST_FEATURES), align=False)),
    "xgb_hist_eq": ("XGBoost action/point + history，訓練列不加權（對照）",
                    _trees("xgb", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES), align=False)),
    "xgb_hist2_eq": ("XGBoost action/point + 雙方 history，訓練列不加權（對照）",
                     _trees("xgb", targets=("a", "p"), player_te=False,
                            extra=tuple(HIST_FEATURES + OHIST_FEATURES), align=False)),
    "cat_hist_eq": ("CatBoost action/point + history，訓練列不加權（對照）",
                    _trees("cat", targets=("a", "p"), player_te=False, extra=tuple(HIST_FEATURES), align=False)),
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
    ap.add_argument("--no-relocate", action="store_true",
                    help="不把 test 打散的 rally 放回原本的比賽（見 ttmatch/relocate.py）")
    a = ap.parse_args()
    if a.list or not a.methods:
        for k, (desc, _) in METHODS.items():
            print(f"{k:14s} {desc}")
        return

    config.ensure_dirs()
    D = build_datasets(relocate=not a.no_relocate)
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
