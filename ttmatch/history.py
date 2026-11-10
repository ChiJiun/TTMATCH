"""同場比賽的先前資訊（本 repo 新增）：下一拍擊球者在這場比賽「較早的 rally」裡的擊球傾向。

test 每場比賽平均可見 23 個 rally，而且 train 與 test 的比賽不重疊，所以「這位選手這場比賽到目前為止
怎麼打」是選手 ID 之外合法可用的訊息（只用該 rally 之前的 rally，不碰任何要預測的那一拍）。

分佈對齊（與比分推移特徵的 train 模擬同一精神）：test 跟 train 有三個差異，train 端都要模擬——
1. test 每局只看得到約 34% 的 rally（rally_id 間隔 2–4）；
2. test 的 rally 只看得到前綴 k 拍（多半 1–3 拍），且第 k+1 拍一定存在（那是要預測的拍），所以
   看不到 rally 的最後一拍，而且長 rally 比短 rally 更容易成為可見 rally；
→ train 先以 34% 機率決定每個 rally 是否可見，再把 test 的前綴長度分佔「配對」到可見 rally 上：
   由長到短的 k 依序從「長度 ≥ k+1 且尚未指派」的 rally 中隨機挑出應有的數量；配不到的 rally 取
   k = 長度 − 1。這樣可見 rally 的前綴長度分佔與 test 幾乎相同，且永遠不含 rally 的最後一拍。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config
from .features import inconsistent_matches

HIST_PRIOR = 5.0          # 平滑用的虛擬樣本數（往全體分佈收縮）
_N_ACT, _N_PT = 15, 10    # 歷史統計只看 0–14 的非發球球種與 0–9 落點

HIST_FEATURES = (["hist_n", "hist_attack_rate", "hist_mean_strength"]
                 + [f"hist_act{c}" for c in range(_N_ACT)] + [f"hist_pt{c}" for c in range(_N_PT)])
HIST_KEYS = ["match", "numberGame", "rally_id", "next_player"]
# 同一組統計也可以套在「目前擊球者」（對手）身上：欄名加前綴 o
OHIST_FEATURES = ["o" + c for c in HIST_FEATURES]


def test_visible_rally_frac(test_df: pd.DataFrame) -> float:
    """test 正常 match 裡每局可見 rally 數 / 該局最大 rally_id，約 0.34。"""
    rt = test_df.groupby("rally_uid", sort=False).first().reset_index()
    rt = rt[~rt["match"].isin(inconsistent_matches(test_df))]
    g = rt.groupby(["match", "numberGame"])["rally_id"].agg(["count", "max"])
    return float(g["count"].sum() / g["max"].sum())


def _visible_mask(df: pd.DataFrame, len_probs: dict | None, keep_prob: float | None, seed: int) -> np.ndarray:
    """train 的模擬（見模組說明）。len_probs 與 keep_prob 都是 None 時（test）全部拍可見。"""
    if len_probs is None and keep_prob is None:
        return np.ones(len(df), dtype=bool)
    rng = np.random.default_rng(seed)
    uids = df["rally_uid"].to_numpy()
    uniq, inv = np.unique(uids, return_inverse=True)
    sn = df["strikeNumber"].to_numpy()
    rally_len = np.zeros(len(uniq), dtype=int)
    np.maximum.at(rally_len, inv, sn)
    kept = np.where(rng.random(len(uniq)) < (1.0 if keep_prob is None else keep_prob))[0]
    k = np.full(len(uniq), -1)                      # -1：不可見
    if len_probs is None:
        k[kept] = rally_len[kept] - 1
    else:
        lens = np.array(sorted(len_probs))[::-1]
        probs = np.array([len_probs[int(v)] for v in lens], dtype=float)
        counts = np.round(probs / probs.sum() * len(kept)).astype(int)
        assigned = np.zeros(len(uniq), dtype=bool)
        for kj, nj in zip(lens, counts):             # 由長到短配對，長 k 先拿長 rally
            pool = kept[(rally_len[kept] >= kj + 1) & ~assigned[kept]]
            pick = rng.choice(pool, size=min(int(nj), len(pool)), replace=False)
            k[pick] = kj
            assigned[pick] = True
        rest = kept[~assigned[kept]]
        k[rest] = rally_len[rest] - 1
    return (k[inv] >= 1) & (sn <= k[inv])


def match_history(df: pd.DataFrame, len_probs: dict | None = None, keep_prob: float | None = None,
                  seed: int = config.SEED) -> pd.DataFrame:
    """回傳以 HIST_KEYS 為鍵的表：該選手在同場比賽、(numberGame, rally_id) 之前的 rally 裡可見拍的統計量。

    df 是原始擊球表。test：len_probs=keep_prob=None，全部已觀測的拍都算。
    train：給 test 的前綴長度分佔與可見 rally 比例，模擬 test 的截斷與抽樣（見模組說明）。
    鍵集合 = df 中所有 (match, game, rally, 擊球者) ∪ (match, game, rally, 對手)，所以每一列都能對到。
    """
    vis = df[_visible_mask(df, len_probs, keep_prob, seed)]
    act = np.clip(vis["actionId"].to_numpy(dtype=int), 0, 18)
    pt = np.clip(vis["pointId"].to_numpy(dtype=int), 0, _N_PT - 1)
    agg = pd.DataFrame({
        "match": vis["match"].to_numpy(), "numberGame": vis["numberGame"].to_numpy(),
        "rally_id": vis["rally_id"].to_numpy(), "next_player": vis["gamePlayerId"].to_numpy(),
        "n": 1.0, "attack": np.isin(act, config.ATTACK_IDS).astype(float),
        "strength": vis["strengthId"].to_numpy(dtype=float),
    })
    for c in range(_N_ACT):
        agg[f"a{c}"] = (act == c).astype(float)
    for c in range(_N_PT):
        agg[f"p{c}"] = (pt == c).astype(float)
    agg = agg.groupby(HIST_KEYS, sort=False).sum()

    # 鍵集合：每個 rally 的兩位選手都要有一列（即使這個 rally 看不到他的拍）
    keys = pd.concat([
        df[["match", "numberGame", "rally_id", "gamePlayerId"]].rename(columns={"gamePlayerId": "next_player"}),
        df[["match", "numberGame", "rally_id", "gamePlayerOtherId"]].rename(columns={"gamePlayerOtherId": "next_player"}),
    ]).drop_duplicates()
    tab = keys.merge(agg, left_on=HIST_KEYS, right_index=True, how="left").fillna(0.0)
    tab = tab.sort_values(["match", "next_player", "numberGame", "rally_id"]).reset_index(drop=True)
    stat_cols = [c for c in tab.columns if c not in HIST_KEYS]
    cum = tab.groupby(["match", "next_player"], sort=False)[stat_cols].cumsum()
    before = cum - tab[stat_cols]          # 嚴格在這個 rally 之前

    n = before["n"].to_numpy()
    g_att = agg["attack"].sum() / max(agg["n"].sum(), 1.0)
    g_str = agg["strength"].sum() / max(agg["n"].sum(), 1.0)
    g_act = agg[[f"a{c}" for c in range(_N_ACT)]].sum().to_numpy() / max(agg["n"].sum(), 1.0)
    g_pt = agg[[f"p{c}" for c in range(_N_PT)]].sum().to_numpy() / max(agg["n"].sum(), 1.0)
    out = tab[HIST_KEYS].copy()
    out["hist_n"] = n
    out["hist_attack_rate"] = (before["attack"].to_numpy() + HIST_PRIOR * g_att) / (n + HIST_PRIOR)
    out["hist_mean_strength"] = (before["strength"].to_numpy() + HIST_PRIOR * g_str) / (n + HIST_PRIOR)
    for c in range(_N_ACT):
        out[f"hist_act{c}"] = (before[f"a{c}"].to_numpy() + HIST_PRIOR * g_act[c]) / (n + HIST_PRIOR)
    for c in range(_N_PT):
        out[f"hist_pt{c}"] = (before[f"p{c}"].to_numpy() + HIST_PRIOR * g_pt[c]) / (n + HIST_PRIOR)
    return out


def attach_history(feat_df: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    """把 match_history 的表合併到特徵表兩次：依 next_player（hist_*）與依 cur_player（ohist_*，對手的傾向）。"""
    out = feat_df.merge(hist, on=HIST_KEYS, how="left")
    oh = hist.rename(columns={"next_player": "cur_player", **{c: "o" + c for c in HIST_FEATURES}})
    out = out.merge(oh, on=["match", "numberGame", "rally_id", "cur_player"], how="left")
    assert len(out) == len(feat_df)
    assert not out["hist_n"].isna().any() and not out["ohist_n"].isna().any(), "有特徵列對不到歷史統計"
    return out
