"""Macro-F1 決策規則：pred = argmax_c [alpha * log p_c + w_c]。

alpha 為全域溫度，w_c 為各類別偏置（先以類別先驗冪次初始化，再做座標上升）。
在 OOF 上調參，可讓稀有類別被預測到，直接拉高 Macro-F1。
改寫自 0000workjason/aicup2026-tabletennis-prediction solution/decision.py
（第 5 名 SugarSquirrel 的 prior-correction β 是其中 alpha=1、只有先驗冪次的特例）。
"""
from __future__ import annotations
import numpy as np

from .metrics import macro_f1

ALPHA_GRID = (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.6)
_W_GRID = np.array([-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4])


def apply_decision(probs, classes, decision):
    alpha, logw = decision
    scores = alpha * np.log(np.clip(probs, 1e-12, None)) + logw[None, :]
    return np.asarray(classes)[scores.argmax(axis=1)]


def argmax_decision(n_classes: int):
    return (1.0, np.zeros(n_classes))


def tune_decision(probs, y_true, classes, n_coord_passes: int = 2, alpha_grid=ALPHA_GRID):
    classes = np.asarray(classes)
    y_true = np.asarray(y_true)
    C = len(classes)
    freq = np.clip(np.array([(y_true == c).mean() for c in classes]), 1e-6, None)
    log_freq = np.log(freq)

    best_f, alpha, logw = -1.0, 1.0, np.zeros(C)
    for a in alpha_grid:
        for beta in np.linspace(0.0, 1.2, 13):
            lw = -beta * log_freq
            f = macro_f1(y_true, apply_decision(probs, classes, (a, lw)))
            if f > best_f:
                best_f, alpha, logw = f, a, lw.copy()

    for _ in range(n_coord_passes):
        for ci in range(C):
            base = logw[ci]
            local_f, local_w = best_f, base
            for d in _W_GRID:
                logw[ci] = base + d
                f = macro_f1(y_true, apply_decision(probs, classes, (alpha, logw)))
                if f > local_f:
                    local_f, local_w = f, logw[ci]
            logw[ci] = local_w
            best_f = local_f

    for a in alpha_grid:
        f = macro_f1(y_true, apply_decision(probs, classes, (a, logw)))
        if f > best_f:
            best_f, alpha = f, a
    return (alpha, logw), best_f


# --- 依前綴深度分桶的決策：接發球、第三板、相持段的下一拍類別組成差很多 ---
DEPTH_BUCKETS = ((1, 1), (2, 2), (3, 999))


def _bucket_id(prefix_len):
    pl = np.asarray(prefix_len)
    bid = np.full(len(pl), len(DEPTH_BUCKETS) - 1, dtype=int)
    for i, (lo, hi) in enumerate(DEPTH_BUCKETS):
        bid[(pl >= lo) & (pl <= hi)] = i
    return bid


def tune_decision_by_depth(probs, y_true, classes, prefix_len, n_coord_passes: int = 2,
                           min_rows: int = 300):
    """每個深度桶各自調參；樣本太少的桶沿用全域決策。回傳 dict(bucket -> decision)。"""
    y_true = np.asarray(y_true)
    glob, _ = tune_decision(probs, y_true, classes, n_coord_passes)
    bid = _bucket_id(prefix_len)
    decs = {}
    for i in range(len(DEPTH_BUCKETS)):
        m = bid == i
        decs[i] = tune_decision(probs[m], y_true[m], classes, n_coord_passes)[0] if m.sum() >= min_rows else glob
    return decs


def apply_decision_by_depth(probs, classes, decs, prefix_len):
    bid = _bucket_id(prefix_len)
    pred = np.empty(len(probs), dtype=int)
    for i, dec in decs.items():
        m = bid == i
        if m.any():
            pred[m] = apply_decision(probs[m], classes, dec)
    return pred
