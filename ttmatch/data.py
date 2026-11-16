"""讀取原始 CSV 並組出各模型共用的資料集。

樣本構成（沿用 0000workjason/aicup2026-tabletennis-prediction solution/data.py 的設計）：
- ap_base   = train 中有下一拍的每一列（rally 的每個前綴），CV 依 match 分組
- ap_aug    = test 內部已觀測的轉移（第 i 拍 → 第 i+1 拍），不洩漏任何答案
- test_pred = 每個 test rally 的最後一個已觀測拍（每個 rally_uid 一列）
- 另附比分推移特徵（PROG_FEATURES），train 端以 test 的 rally 抽樣間隔模擬
"""
from __future__ import annotations
import pandas as pd

from . import config
from .features import (add_features, test_gap_distribution, inconsistent_matches,
                       score_progression_train, score_progression_test)


def load_raw():
    d = config.data_dir()
    return (pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv"),
            pd.read_csv(d / "sample_submission.csv"))


def _strip_serve_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["has_next"]].reset_index(drop=True).copy()
    df["next_action"] = df["next_action"].astype(int)
    df["next_point"] = df["next_point"].astype(int)
    return df[~df["next_action"].isin(config.SERVE_IDS)].reset_index(drop=True)


def build_datasets(use_test_aug: bool = True) -> dict:
    tr, te, sample = load_raw()
    trf = add_features(tr, has_winner=True)
    tef = add_features(te, has_winner=False)

    gap_dist = test_gap_distribution(te)
    trf = trf.merge(score_progression_train(tr, gap_dist), on="rally_uid", how="left")
    tef = tef.merge(score_progression_test(te), on="rally_uid", how="left")

    ap_base = _strip_serve_targets(trf)
    ap_base["winner"] = ap_base["winner"].astype(int)
    ap_aug = _strip_serve_targets(tef) if use_test_aug else None
    test_pred = tef[~tef["has_next"]].reset_index(drop=True).copy()
    # 異常 match 的 rally 沒有比分推移資訊，arena 對這些列改用不含比分推移的 winner 集成
    test_pred["prog_unreliable"] = test_pred["match"].isin(inconsistent_matches(te)).astype(int)

    # 提交順序以 sample_submission 為準
    order = {u: i for i, u in enumerate(sample["rally_uid"])}
    test_pred = (test_pred.assign(_o=test_pred["rally_uid"].map(order))
                 .sort_values("_o").drop(columns="_o").reset_index(drop=True))

    return dict(ap_base=ap_base, ap_aug=ap_aug, test_pred=test_pred,
                trf=trf, tef=tef, sample=sample, gap_dist=gap_dist)


def test_length_probs(tef: pd.DataFrame) -> dict:
    """test rally 已觀測前綴長度的分佈（用來做長度對齊的評估）。"""
    lens = tef.groupby("rally_uid", sort=False)["strikeNumber"].max().astype(int)
    vc = lens.value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in vc.items()}
