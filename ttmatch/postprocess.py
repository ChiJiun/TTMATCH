"""規則修正：若預測類別在訓練資料的同一情境 (prev_action, last_action, last_point) 中從未出現，
且該情境樣本數足夠，改成該情境的眾數。

改寫自 jabir95tsai/pingpong_aicup_2026（私榜第 22 名）src/apply_rule_override.py；
改成直接吃 ap_base 格式的列，方便在 OOF 上評估。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

MIN_CONTEXT_SAMPLES = 30
CTX = ["prev_action", "last_action", "last_point"]


def fit_table(rows: pd.DataFrame, target_col: str) -> dict:
    table = {}
    for key, vals in rows.groupby(CTX)[target_col]:
        if len(vals) < MIN_CONTEXT_SAMPLES:
            continue
        vc = vals.value_counts()
        table[tuple(int(k) for k in key)] = (set(vc.index.astype(int)), int(vc.idxmax()))
    return table


def apply_override(rows: pd.DataFrame, pred: np.ndarray, table: dict) -> np.ndarray:
    pred = np.asarray(pred).copy()
    keys = map(tuple, rows[CTX].to_numpy().astype(int))
    for i, k in enumerate(keys):
        hit = table.get(k)
        if hit is not None and pred[i] not in hit[0]:
            pred[i] = hit[1]
    return pred
