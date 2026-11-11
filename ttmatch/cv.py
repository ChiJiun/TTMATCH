"""依 match 分組的交叉驗證。

改寫自 0000workjason/aicup2026-tabletennis-prediction solution/cv.py。原作評估時「每個 rally 依 test 前綴
長度分佈抽一個長度，超過就截到 rally 的最後一個前綴」；本 repo 改用 ttmatch/truncation.py 的
cut_matched_indices（前綴長度分佈與 test 完全吻合）。
"""
from __future__ import annotations
import numpy as np
from sklearn.model_selection import GroupKFold

from . import config


def get_folds(groups, n_splits: int = config.N_FOLDS):
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(groups))
    return list(gkf.split(dummy, dummy, groups=groups))

