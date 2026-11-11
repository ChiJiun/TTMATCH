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
                       score_progression_train, score_progression_test, mirror_raw)
from .history import match_history, attach_history, test_visible_rally_frac
from .relocate import relocate_scrambled, apply_relocation
from .truncation import fit_cut_distribution, cut_weights


def load_raw():
    d = config.data_dir()
    return (pd.read_csv(d / "train.csv"), pd.read_csv(d / "test.csv"),
            pd.read_csv(d / "sample_submission.csv"))


def _strip_serve_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["has_next"]].reset_index(drop=True).copy()
    df["next_action"] = df["next_action"].astype(int)
    df["next_point"] = df["next_point"].astype(int)
    return df[~df["next_action"].isin(config.SERVE_IDS)].reset_index(drop=True)


def build_datasets(use_test_aug: bool = True, relocate: bool = True) -> dict:
    tr, te, sample = load_raw()
    # test 的 match 340–354 是被打散的 rally：能定位的先放回原本的比賽，之後的比分推移、history 與
    # 各項 test 分佈統計都以放回後的編號計算（見 ttmatch/relocate.py）
    placed = relocate_scrambled(te) if relocate else None
    if placed is not None:
        te = apply_relocation(te, placed)
    trf = add_features(tr, has_winner=True)
    tef = add_features(te, has_winner=False)

    gap_dist = test_gap_distribution(te)
    trf = trf.merge(score_progression_train(tr, gap_dist), on="rally_uid", how="left")
    tef = tef.merge(score_progression_test(te), on="rally_uid", how="left")

    # 同場先前 rally 的選手傾向；train 依 test 的可見 rally 比例與前綴長度分佈模擬
    trf = attach_history(trf, match_history(tr, test_length_probs(tef), test_visible_rally_frac(te)))
    tef = attach_history(tef, match_history(te))

    ap_base = _strip_serve_targets(trf)
    ap_base["winner"] = ap_base["winner"].astype(int)
    # 截斷機制對齊（見 ttmatch/truncation.py）：rally_len / is_terminal / align_w 只用於樣本權重與評估抽樣，
    # 不是特徵（test 不知道 rally 的長度）
    rally_len = tr.groupby("rally_uid")["strikeNumber"].max()
    ap_base["rally_len"] = ap_base["rally_uid"].map(rally_len).astype(int)
    ap_base["is_terminal"] = (ap_base["rally_len"] == ap_base["prefix_len"] + 1).astype(int)
    cut_g = fit_cut_distribution(rally_len.to_numpy(), tef.groupby("rally_uid")["strikeNumber"].max().to_numpy())
    w = cut_weights(ap_base["prefix_len"], ap_base["rally_len"], cut_g)
    ap_base["align_w"] = w / w.mean()
    # 左右鏡像版（只給樹模型當額外訓練列；鏡像不改 actionId，所以列與 ap_base 一一對齊）
    ap_base_mirror = _strip_serve_targets(add_features(mirror_raw(tr), has_winner=True))
    assert (ap_base_mirror["rally_uid"].to_numpy() == ap_base["rally_uid"].to_numpy()).all()
    assert (ap_base_mirror["prefix_len"].to_numpy() == ap_base["prefix_len"].to_numpy()).all()
    ap_aug = _strip_serve_targets(tef) if use_test_aug else None
    if ap_aug is not None:
        # aug 列（test 內部轉移）一定不是終結拍：起始權重取同前綴長度的 base 非終結列平均權重
        nt = ap_base[ap_base["is_terminal"] == 0]
        k_cap = len(cut_g) - 1
        nt_mean = nt["align_w"].groupby(nt["prefix_len"].clip(upper=k_cap)).mean()
        ap_aug["align_w"] = (ap_aug["prefix_len"].clip(upper=k_cap).map(nt_mean)
                             .fillna(float(nt["align_w"].mean())).to_numpy())
    test_pred = tef[~tef["has_next"]].reset_index(drop=True).copy()
    # 異常 match 的 rally 沒有比分推移資訊，arena 對這些列改用不含比分推移的 winner 集成
    test_pred["prog_unreliable"] = test_pred["match"].isin(inconsistent_matches(te)).astype(int)

    # 提交順序以 sample_submission 為準
    order = {u: i for i, u in enumerate(sample["rally_uid"])}
    test_pred = (test_pred.assign(_o=test_pred["rally_uid"].map(order))
                 .sort_values("_o").drop(columns="_o").reset_index(drop=True))

    return dict(ap_base=ap_base, ap_base_mirror=ap_base_mirror, ap_aug=ap_aug, test_pred=test_pred,
                trf=trf, tef=tef, sample=sample, gap_dist=gap_dist, relocated=placed, cut_g=cut_g)


def test_length_probs(tef: pd.DataFrame) -> dict:
    """test rally 已觀測前綴長度的分佈（用來做長度對齊的評估）。"""
    lens = tef.groupby("rally_uid", sort=False)["strikeNumber"].max().astype(int)
    vc = lens.value_counts(normalize=True).sort_index()
    return {int(k): float(v) for k, v in vc.items()}
