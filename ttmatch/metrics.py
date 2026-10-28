"""競賽指標：Macro-F1（action、point）、AUC（serverGetPoint）、總分。

改寫自 0000workjason/aicup2026-tabletennis-prediction solution/metrics.py；
macro_f1 改用 bincount 實作（決策調參會呼叫上千次），結果與 sklearn 預設行為一致：
類別集合取 y_true 與 y_pred 的聯集。
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score

from . import config


def macro_f1(y_true, y_pred) -> float:
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    n = int(max(y_true.max(), y_pred.max())) + 1
    tp = np.bincount(y_true[y_true == y_pred], minlength=n).astype(float)
    t_cnt = np.bincount(y_true, minlength=n).astype(float)
    p_cnt = np.bincount(y_pred, minlength=n).astype(float)
    present = (t_cnt + p_cnt) > 0
    denom = t_cnt + p_cnt
    f1 = np.where(denom > 0, 2 * tp / np.where(denom > 0, denom, 1), 0.0)
    return float(f1[present].mean())


def winner_auc(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def overall_score(s1: float, s2: float, s3: float) -> float:
    return config.W_ACTION * s1 + config.W_POINT * s2 + config.W_WINNER * s3
