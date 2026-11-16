"""依 match 分組的交叉驗證 + 對齊 test 前綴長度的評估。

test 的 rally 被截在不同長度（多半很短），因此評估時每個驗證 rally 依 test 前綴長度分佈抽一個前綴。
改寫自 0000workjason/aicup2026-tabletennis-prediction solution/cv.py。
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config


def get_folds(groups, n_splits: int = config.N_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, dummy, groups=groups))


def sample_one_per_rally(df: pd.DataFrame, length_probs: dict, rng: np.random.Generator):
    lens = np.array(sorted(length_probs))
    probs = np.array([length_probs[int(l)] for l in lens], dtype=float)
    probs = probs / probs.sum()
    chosen = []
    plen = df["prefix_len"].to_numpy()
    for _, idx in df.groupby("rally_uid", sort=False).indices.items():
        sub = plen[idx]
        k = int(rng.choice(lens, p=probs))
        k = max(1, min(k, int(sub.max())))
        chosen.append(idx[np.argmin(np.abs(sub - k))])
    return np.array(chosen, dtype=int)


def length_matched_indices(df: pd.DataFrame, length_probs: dict, n_repeats: int = 5,
                           seed: int = config.SEED) -> list[np.ndarray]:
    return [sample_one_per_rally(df, length_probs, np.random.default_rng(seed + r))
            for r in range(n_repeats)]
