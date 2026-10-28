"""下一拍預測的特徵工程。

「預測點」= 觀測完第 1..t 拍後預測第 t+1 拍。每一列（第 t 拍）只帶入 1..t 拍可得的資訊；
同一 rally 有下一拍的列成為訓練樣本，test rally 的最後一列是預測點。

prefix 特徵與 target encoding 改寫自 0000workjason/aicup2026-tabletennis-prediction
（AI CUP 2026 私榜第 3 名）solution/features.py；比分推移特徵（score progression）為本 repo 新增。
"""
from __future__ import annotations
from collections import defaultdict

import numpy as np
import pandas as pd

from . import config

MISSING_CAT = 50  # 「還沒有這一拍」的哨兵類別（真實 id 為 0..18）

CATEGORICAL = [
    "sex", "numberGame", "next_strikeId",
    "last_action", "last_point", "last_hand", "last_strength", "last_spin",
    "last_strikeId", "last_position",
    "prev_action", "prev_point", "prev_hand", "prev_spin", "prev_strength",
    "prev2_action", "prev2_point",
    "serve_action", "serve_spin", "serve_strength", "serve_point", "serve_hand",
    "next_player",
]
NUMERIC = [
    "prefix_len", "striker_is_server", "next_is_server",
    "is_predict_receive", "is_predict_third",
    "server_score", "receiver_score", "score_diff", "total_points",
    "cum_attack", "cum_control", "cum_defensive",
    "cum_out", "cum_net", "cum_long",
    "cum_atk_srv", "cum_atk_rcv", "attack_init_diff", "last_is_attack",
    "mean_strength", "attack_ratio",
    "score_max", "srv_game_point", "rcv_game_point", "is_deuce",
    "prev_is_attack", "recent_attack2",
]
FEATURES = CATEGORICAL + NUMERIC

# 比分推移特徵：同一局下一個「可見」rally 的比分，揭露本 rally 誰得分
PROG_FEATURES = ["prog_has_next", "prog_gap", "prog_srv_gain", "prog_rcv_gain", "prog_frac"]


def add_features(df: pd.DataFrame, has_winner: bool = True) -> pd.DataFrame:
    df = df.sort_values(["rally_uid", "strikeNumber"]).reset_index(drop=True).copy()
    g = df.groupby("rally_uid", sort=False)
    t = df["strikeNumber"].astype(int)

    # 第 t 拍為奇數 → 發球方擊球
    df["prefix_len"] = t
    df["striker_is_server"] = (t % 2 == 1).astype(int)
    df["next_is_server"] = (t % 2 == 0).astype(int)
    df["next_strikeId"] = np.where(t == 1, 2, 4).astype(int)
    df["is_predict_receive"] = (t == 1).astype(int)
    df["is_predict_third"] = (t == 2).astype(int)
    df["next_player"] = df["gamePlayerOtherId"].astype(int)

    df["last_action"] = df["actionId"].astype(int)
    df["last_point"] = df["pointId"].astype(int)
    df["last_hand"] = df["handId"].astype(int)
    df["last_strength"] = df["strengthId"].astype(int)
    df["last_spin"] = df["spinId"].astype(int)
    df["last_strikeId"] = df["strikeId"].astype(int)
    df["last_position"] = df["positionId"].astype(int)

    for col, src in [("prev_action", "actionId"), ("prev_point", "pointId"),
                     ("prev_hand", "handId"), ("prev_spin", "spinId"),
                     ("prev_strength", "strengthId")]:
        df[col] = g[src].shift(1).fillna(MISSING_CAT).astype(int)
    df["prev2_action"] = g["actionId"].shift(2).fillna(MISSING_CAT).astype(int)
    df["prev2_point"] = g["pointId"].shift(2).fillna(MISSING_CAT).astype(int)

    for col, src in [("serve_action", "actionId"), ("serve_spin", "spinId"),
                     ("serve_strength", "strengthId"), ("serve_point", "pointId"),
                     ("serve_hand", "handId")]:
        df[col] = g[src].transform("first").astype(int)

    # scoreSelf/Other 是擊球者視角，轉成發球方視角
    ss = df["scoreSelf"].astype(int)
    so = df["scoreOther"].astype(int)
    is_odd = (t % 2 == 1)
    df["server_score"] = np.where(is_odd, ss, so)
    df["receiver_score"] = np.where(is_odd, so, ss)
    df["score_diff"] = df["server_score"] - df["receiver_score"]
    df["total_points"] = df["server_score"] + df["receiver_score"]

    act = df["actionId"].astype(int)
    pt = df["pointId"].astype(int)
    ind = pd.DataFrame({
        "attack": act.isin(config.ATTACK_IDS).astype(int),
        "control": act.isin(config.CONTROL_IDS).astype(int),
        "defensive": act.isin(config.DEFENSIVE_IDS).astype(int),
        "out": (pt == 0).astype(int),
        "net": pt.between(1, 3).astype(int),
        "long": pt.between(7, 9).astype(int),
        "strength": df["strengthId"].astype(int),
        "atk_srv": (act.isin(config.ATTACK_IDS) & is_odd).astype(int),
        "atk_rcv": (act.isin(config.ATTACK_IDS) & ~is_odd).astype(int),
    }, index=df.index)
    ind["rally_uid"] = df["rally_uid"].values
    cum = ind.groupby("rally_uid", sort=False).cumsum()
    for c in ["attack", "control", "defensive", "out", "net", "long", "atk_srv", "atk_rcv"]:
        df["cum_" + c] = cum[c].astype(int)
    df["mean_strength"] = cum["strength"] / df["prefix_len"]
    df["attack_ratio"] = df["cum_attack"] / df["prefix_len"]
    df["attack_init_diff"] = df["cum_atk_srv"] - df["cum_atk_rcv"]
    df["last_is_attack"] = act.isin(config.ATTACK_IDS).astype(int)

    df["score_max"] = np.maximum(df["server_score"], df["receiver_score"])
    df["srv_game_point"] = ((df["server_score"] >= 10) & (df["score_diff"] >= 1)).astype(int)
    df["rcv_game_point"] = ((df["receiver_score"] >= 10) & (df["score_diff"] <= -1)).astype(int)
    df["is_deuce"] = ((df["server_score"] >= 10) & (df["receiver_score"] >= 10)).astype(int)
    df["prev_is_attack"] = df["prev_action"].isin(config.ATTACK_IDS).astype(int)
    df["recent_attack2"] = df["last_is_attack"] + df["prev_is_attack"]

    df["next_action"] = g["actionId"].shift(-1)
    df["next_point"] = g["pointId"].shift(-1)
    df["has_next"] = df["next_action"].notna()
    if has_winner and "serverGetPoint" in df.columns:
        df["winner"] = df["serverGetPoint"].astype(int)
    else:
        df["winner"] = np.nan
    return df


def get_X(df: pd.DataFrame, extra: list[str] | None = None) -> pd.DataFrame:
    return df[FEATURES + (extra or [])].copy()


# --- 比分推移特徵 -----------------------------------------------------------

def _rally_table(df: pd.DataFrame) -> pd.DataFrame:
    """每個 rally 一列：發球者與發球方視角比分（取第 1 拍）。"""
    first = df.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).first()
    return pd.DataFrame({
        "rally_uid": first.index.to_numpy(),
        "match": first["match"].to_numpy(),
        "numberGame": first["numberGame"].to_numpy(),
        "rally_id": first["rally_id"].to_numpy(),
        "server": first["gamePlayerId"].to_numpy(),
        "pair": [frozenset(p) for p in zip(first["gamePlayerId"], first["gamePlayerOtherId"])],
        "srv_score": first["scoreSelf"].to_numpy(),
        "rcv_score": first["scoreOther"].to_numpy(),
        # 正常資料中 rally_id 就是局內第幾分，等於開局比分總和 + 1；test 的 match 340–354 不符合
        # （每個「match」混了 20 多位選手、rally_id 與比分對不上，像是被打散重編號的 rally）
        "consistent": (first["rally_id"] - 1 == first["scoreSelf"] + first["scoreOther"]).to_numpy(),
    })


def _gain(cur: pd.DataFrame, nxt: pd.DataFrame):
    """從 cur rally 開始到 nxt rally 開始之間，cur 的發球方 / 接發方各得幾分。"""
    same = nxt["server"].to_numpy() == cur["server"].to_numpy()
    nxt_srv = np.where(same, nxt["srv_score"], nxt["rcv_score"])
    nxt_rcv = np.where(same, nxt["rcv_score"], nxt["srv_score"])
    return nxt_srv - cur["srv_score"].to_numpy(), nxt_rcv - cur["rcv_score"].to_numpy()


def _prog_frame(cur: pd.DataFrame, nxt: pd.DataFrame, has: np.ndarray, gap: np.ndarray):
    a, b = _gain(cur, nxt)
    # 一致性檢查：同一對選手、兩個 rally 的 rally_id 都與比分一致、得分非負且總和等於間隔；
    # 不通過就當作沒有下一個 rally，避免異常資料產生訓練時從未見過的值
    has = (has & (cur["pair"].to_numpy() == nxt["pair"].to_numpy())
           & cur["consistent"].to_numpy() & nxt["consistent"].to_numpy().astype(bool)
           & (a >= 0) & (b >= 0) & (a + b == np.nan_to_num(gap, nan=-1)))
    a = np.where(has, a, np.nan).astype(float)
    b = np.where(has, b, np.nan).astype(float)
    tot = a + b
    frac = np.where(has & (tot > 0), a / np.where(tot > 0, tot, 1), np.nan)
    return pd.DataFrame({
        "rally_uid": cur["rally_uid"].to_numpy(),
        "prog_has_next": has.astype(int),
        "prog_gap": np.where(has, gap, np.nan),
        "prog_srv_gain": a, "prog_rcv_gain": b, "prog_frac": frac,
    })


def inconsistent_matches(df: pd.DataFrame) -> set:
    """rally_id 與比分大多對不上的 match（test 的 340–354），其 rally 拿不到比分推移資訊。"""
    rt = _rally_table(df)
    rate = rt.groupby("match")["consistent"].mean()
    return set(rate[rate < 0.5].index.tolist())


def test_gap_distribution(test_df: pd.DataFrame) -> dict:
    """test 內同一局相鄰可見 rally 的 rally_id 間隔分佈；最後一個可見 rally 記為 gap=0（沒有下一個）。
    只統計 rally_id 與比分一致的正常 match。"""
    rt = _rally_table(test_df)
    rt = rt[~rt["match"].isin(inconsistent_matches(test_df))].sort_values(["match", "numberGame", "rally_id"])
    d = rt.groupby(["match", "numberGame"])["rally_id"].diff(-1).mul(-1)
    d = d.fillna(0).astype(int)
    vc = d.value_counts(normalize=True)
    return {int(k): float(v) for k, v in vc.items()}


def score_progression_test(test_df: pd.DataFrame) -> pd.DataFrame:
    """test：直接用同一局下一個可見 rally 的比分。"""
    rt = _rally_table(test_df).sort_values(["match", "numberGame", "rally_id"]).reset_index(drop=True)
    nxt = rt.groupby(["match", "numberGame"]).shift(-1)
    has = nxt["rally_id"].notna().to_numpy()
    gap = (nxt["rally_id"] - rt["rally_id"]).to_numpy()
    nxt = nxt.fillna({"server": -1, "srv_score": 0, "rcv_score": 0, "consistent": False})
    nxt["pair"] = [p if isinstance(p, frozenset) else frozenset() for p in nxt["pair"]]
    return _prog_frame(rt, nxt, has, gap)


def score_progression_train(train_df: pd.DataFrame, gap_dist: dict, seed: int = config.SEED) -> pd.DataFrame:
    """train：每個 rally 依 test 的間隔分佈（只取 gap>0）抽一個 gap，模擬「下一個可見 rally」。
    該局沒有 rally_id+gap 的 rally → 視為沒有下一個；局末自然產生的比例與 test 的 gap=0 比例相近，
    所以不再額外抽 gap=0，以免重複計算。"""
    rt = _rally_table(train_df).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    gaps = np.array(sorted(g for g in gap_dist if g > 0))
    probs = np.array([gap_dist[g] for g in gaps])
    draw = rng.choice(gaps, size=len(rt), p=probs / probs.sum())
    # train 有些局中間缺 rally：r+g 不存在時取該局 rally_id >= r+g 的第一個 rally，
    # 只有局末之後才算沒有下一個（與 test 只在局末出現 gap=0 一致）
    games = {}
    for i, (m, n, r) in enumerate(zip(rt["match"], rt["numberGame"], rt["rally_id"])):
        games.setdefault((m, n), []).append((r, i))
    games = {k: (np.array([r for r, _ in sorted(v)]), np.array([i for _, i in sorted(v)]))
             for k, v in games.items()}
    idx = np.full(len(rt), -1)
    for i, (m, n, r, g) in enumerate(zip(rt["match"], rt["numberGame"], rt["rally_id"], draw)):
        ids, rows = games[(m, n)]
        j = np.searchsorted(ids, r + g)
        if j < len(ids):
            idx[i] = rows[j]
    has = idx >= 0
    nxt = rt.iloc[np.where(has, idx, 0)].reset_index(drop=True)
    gap = np.where(has, nxt["rally_id"].to_numpy() - rt["rally_id"].to_numpy(), np.nan)
    return _prog_frame(rt, nxt, has, gap)


# --- OOF target encoding（必須只用該 fold 的訓練列 fit）----------------------

def _key_tuples(df: pd.DataFrame, key_cols):
    return list(map(tuple, df[key_cols].to_numpy()))


def fit_target_encoding(df, target_col, classes, key_cols, alpha=20.0):
    C = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    y = df[target_col].astype(int).to_numpy()
    valid = np.isin(y, classes)
    yi = np.array([c2i[v] for v in y[valid]], dtype=int)
    prior = np.bincount(yi, minlength=C).astype(float)
    prior = prior / max(prior.sum(), 1.0)
    acc = defaultdict(lambda: np.zeros(C))
    for k, c in zip(_key_tuples(df.loc[valid], key_cols), yi):
        acc[k][c] += 1.0
    enc = {k: (v + alpha * prior) / (v.sum() + alpha) for k, v in acc.items()}
    return enc, prior


def oof_target_encoding(df, target_col, classes, key_cols, alpha, groups, n_inner=5):
    """訓練列自己的 TE 必須用巢狀 OOF 計算（依 match 分組的內層 fold），否則每列的 TE 含有自己的答案，
    模型會過度信任這個特徵。原第 3 名作法直接用整個訓練集 fit 再套回訓練列，在本資料上反而讓 action 退步。"""
    from sklearn.model_selection import GroupKFold
    C = len(classes)
    out = np.empty((len(df), C), dtype=np.float32)
    groups = np.asarray(groups)
    n_inner = min(n_inner, len(np.unique(groups)))
    for tr, va in GroupKFold(n_splits=n_inner).split(df, groups=groups):
        enc, prior = fit_target_encoding(df.iloc[tr], target_col, classes, key_cols, alpha)
        out[va] = apply_target_encoding(df.iloc[va], enc, prior, key_cols, classes, "x").to_numpy()
    return out


def apply_target_encoding(df, enc, prior, key_cols, classes, prefix):
    C = len(classes)
    out = np.empty((len(df), C), dtype=np.float32)
    for i, k in enumerate(_key_tuples(df, key_cols)):
        out[i] = enc.get(k, prior)
    return pd.DataFrame(out, columns=[f"{prefix}_p{j}" for j in range(C)], index=df.index)
