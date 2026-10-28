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


def _trees(kind, **kw):
    def f(D, a):
        from ttmatch.methods.trees import run
        return run(kind, D, a.folds, a.fast, **kw)
    return f


def _nn(arch, **kw):
    def f(D, a):
        from ttmatch.methods.nn import train_nn
        return train_nn(D, arch=arch, n_folds=a.folds, device=a.device, fast=a.fast,
                        epochs=a.epochs, n_seeds=a.seeds, **kw)
    return f


# 名稱 → (說明, 訓練函式)。說明標註方法出處。
METHODS = {
    "lgbm": ("LightGBM + 選手 TE + test 轉移擴充（第 3 名 0000workjason）", _trees("lgbm")),
    "lgbm_note": ("LightGBM，不加選手 TE（對照組，檢驗第 3 名的選手 TE 是否有效）", _trees("lgbm", player_te=False)),
    "lgbm_te20": ("LightGBM action，選手 TE 改用較強平滑 alpha=20（本 repo 調整）",
                  _trees("lgbm", targets=("a",), te_alpha=20.0)),
    "lgbm_noplayer": ("LightGBM action/point，不用任何選手 ID 特徵（對照組）",
                      _trees("lgbm", targets=("a", "p"), player_te=False, drop=("next_player",))),
    "xgb": ("XGBoost 同特徵（lucas6028 多樹模型）", _trees("xgb")),
    "cat": ("CatBoost 原生類別特徵（lucas6028 多樹模型）", _trees("cat")),
    "lgbm_prog": ("LightGBM winner + 比分推移特徵（本 repo 新增）", _trees("lgbm", targets=("w",), use_prog=True)),
    "cat_prog": ("CatBoost winner + 比分推移特徵（本 repo 新增）", _trees("cat", targets=("w",), use_prog=True)),
    "gru": ("因果 GRU 多任務序列模型（第 3 名 0000workjason）", _nn("gru")),
    "transformer": ("因果 Transformer 多任務序列模型（第 3 名 0000workjason）", _nn("transformer")),
    "gru_t2v_film": ("GRU + Time2Vec + FiLM（第 5 名 SugarSquirrel v8）", _nn("t2v_film")),
    "gru_mirror": ("GRU + 左右鏡像資料增強（第 22 名 jabir95tsai）", _nn("gru", mirror=True)),
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
        suffix = "_fast" if a.fast else ""
        path = config.OOF_DIR / f"{name}{suffix}.npz"
        np.savez_compressed(path, **out)
        print(f"   saved {path.name} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
