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
