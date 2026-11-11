import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ttmatch.features import (score_progression_test, score_progression_train,
                              inconsistent_matches)
from ttmatch.features import test_gap_distribution as gap_distribution
from ttmatch.metrics import macro_f1
from ttmatch.methods.nn import POINT_MIRROR, POSITION_MIRROR


def test_macro_f1_matches_sklearn():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 15, 3000)
    p = rng.integers(0, 12, 3000)
    p[:50] = 17  # 只出現在預測裡的類別
    assert abs(macro_f1(y, p) - f1_score(y, p, average="macro")) < 1e-12


def _rally(uid, rid, server, other, s_srv, s_rcv):
    return dict(rally_uid=uid, match=1, numberGame=1, rally_id=rid, strikeNumber=1,
                gamePlayerId=server, gamePlayerOtherId=other, scoreSelf=s_srv, scoreOther=s_rcv)


def test_score_progression_reads_next_visible_rally():
    df = pd.DataFrame([
        _rally(10, 1, 7, 8, 0, 0),   # 發球 7，0:0
        _rally(11, 4, 8, 7, 1, 2),   # 3 分後換 8 發球：8 拿 1 分、7 拿 2 分
        _rally(12, 5, 8, 7, 2, 2),   # 局內最後一個可見 rally
    ])
    out = score_progression_test(df).set_index("rally_uid")
    assert out.loc[10, "prog_gap"] == 3
    assert out.loc[10, "prog_srv_gain"] == 2 and out.loc[10, "prog_rcv_gain"] == 1
    assert out.loc[11, "prog_srv_gain"] == 1 and out.loc[11, "prog_rcv_gain"] == 0
    assert out.loc[12, "prog_has_next"] == 0 and np.isnan(out.loc[12, "prog_frac"])
    assert gap_distribution(df) == {3: 1 / 3, 1: 1 / 3, 0: 1 / 3}


def test_score_progression_rejects_inconsistent_rallies():
    df = pd.DataFrame([
        _rally(20, 1, 7, 8, 0, 0),
        _rally(21, 4, 30, 31, 2, 1),   # 同一局卻換了一對選手（test 的異常 match）
        _rally(22, 6, 30, 31, 9, 3),   # rally_id 與比分不一致
        _rally(23, 9, 30, 31, 9, 4),
    ])
    out = score_progression_test(df).set_index("rally_uid")
    assert (out["prog_has_next"] == 0).all()
    assert out["prog_frac"].isna().all()


def test_score_progression_train_simulation():
    # 一局 6 分（rally_id 1..6，中間缺 3），發球輪替；比分與 rally_id 一致
    rows = []
    s_a, s_b = 0, 0
    winners = [1, 1, 2, 2, 1, 2]  # 每分由哪位選手拿下
    for rid, w in enumerate(winners, start=1):
        server = 1 if ((rid - 1) // 2) % 2 == 0 else 2
        other = 2 if server == 1 else 1
        ss, so = (s_a, s_b) if server == 1 else (s_b, s_a)
        if rid != 3:
            rows.append(_rally(100 + rid, rid, server, other, ss, so))
        s_a, s_b = s_a + (w == 1), s_b + (w == 2)
    df = pd.DataFrame(rows)
    out = score_progression_train(df, {2: 1.0}, seed=0).set_index("rally_uid")
    # rally 1 → 抽 gap=2 → rally 3 不存在 → 取 rally_id>=3 的第一個（rally 4），實際間隔 3
    assert out.loc[101, "prog_has_next"] == 1 and out.loc[101, "prog_gap"] == 3
    assert out.loc[101, "prog_srv_gain"] + out.loc[101, "prog_rcv_gain"] == 3
    # rally 1 發球者是選手 1，期間選手 1 拿 2 分（第 1、2 分）、選手 2 拿 1 分
    assert out.loc[101, "prog_srv_gain"] == 2 and out.loc[101, "prog_rcv_gain"] == 1
    # 局末之後沒有 rally → 沒有下一個
    assert out.loc[105, "prog_has_next"] == 0 and out.loc[106, "prog_has_next"] == 0


def test_inconsistent_matches():
    df = pd.DataFrame([
        _rally(1, 1, 7, 8, 0, 0), _rally(2, 3, 7, 8, 1, 1),
        dict(_rally(3, 1, 30, 31, 5, 2), match=99), dict(_rally(4, 2, 32, 33, 0, 7), match=99),
    ])
    assert inconsistent_matches(df) == {99}


def test_mirror_is_involution():
    assert (POINT_MIRROR[POINT_MIRROR] == np.arange(10)).all()
    assert (POSITION_MIRROR[POSITION_MIRROR] == np.arange(4)).all()


def test_winner_extra_dedups_and_keeps_order():
    from ttmatch.features import PROG_FEATURES
    from ttmatch.methods.trees import winner_extra
    assert winner_extra(False) == []
    assert winner_extra(True) == list(PROG_FEATURES)
    # extra 與 PROG_FEATURES 重疊時只保留一份，順序為 PROG_FEATURES 在前
    out = winner_extra(True, ["hist_n", PROG_FEATURES[0], "hist_n"])
    assert out == list(PROG_FEATURES) + ["hist_n"]
    assert len(out) == len(set(out))


def test_uses_prog_flag_and_fallback():
    from arena import uses_prog
    assert uses_prog("cat_prog", {}) and not uses_prog("cat", {})
    # 新版 npz 的旗標優先於名稱
    assert uses_prog("winner_v2", {"uses_prog": np.array(True)})
    assert not uses_prog("odd_prog", {"uses_prog": np.array(False)})


def test_mixed_auc_interpolates_between_blends():
    from arena import mixed_auc
    from sklearn.metrics import roc_auc_score

    class FakeEval:
        def __init__(self, y):
            self.y = y

        def auc(self, p):
            return roc_auc_score(self.y, p)

    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(50), 20)
    y = rng.integers(0, 2, groups.size)
    good = y + rng.normal(0, 0.3, y.size)    # 幾乎完美
    plain = rng.normal(0, 1, y.size)          # 隨機
    ev = FakeEval(y)
    lo, hi = ev.auc(plain), ev.auc(good)
    m0, _ = mixed_auc(ev, groups, good, plain, frac=0.0)
    m1, _ = mixed_auc(ev, groups, good, plain, frac=1.0)
    m, s = mixed_auc(ev, groups, good, plain, frac=0.2)
    assert abs(m1 - lo) < 1e-12
    assert abs(m0 - hi) < 1e-12
    assert lo < m < hi and s > 0


def _fake_D(n_matches=8, rallies_per_match=30, seed=0):
    rng = np.random.default_rng(seed)
    rows, uid = [], 0
    for m in range(n_matches):
        for _ in range(rallies_per_match):
            for t in range(1, int(rng.integers(1, 5)) + 1):
                rows.append(dict(rally_uid=uid, match=m, prefix_len=t, next_action=int(rng.integers(0, 15)),
                                 next_point=int(rng.integers(0, 10)), winner=int(rng.integers(0, 2))))
            uid += 1
    tef = pd.DataFrame(dict(rally_uid=np.repeat(np.arange(50), 2), strikeNumber=np.tile([1, 2], 50)))
    return dict(ap_base=pd.DataFrame(rows), tef=tef)


def test_evaluator_halves_and_restricted():
    from arena import Evaluator
    D = _fake_D()
    ev = Evaluator(D, repeats=2)
    rng = np.random.default_rng(1)
    p = rng.dirichlet(np.ones(15), len(D["ap_base"]))
    full = ev.f1(p, "a")
    assert abs(full - np.mean([ev.f1(p, "a", halves=(0,)), ev.f1(p, "a", halves=(1,))])) < 1e-12
    # auc(halves=(h,)) 只用第 h 半的列
    w = rng.random(len(D["ap_base"]))
    s = ev.sel[0]
    from sklearn.metrics import roc_auc_score
    manual = np.mean([roc_auc_score(ev.y["w"][s[ev.half[s] == 0]], w[s[ev.half[s] == 0]]) for s in ev.sel])
    assert abs(ev.auc(w, halves=(0,)) - manual) < 1e-12
    # restricted：抽樣列全在 mask 內，且兩半都有列、沿 match 切
    mask = ev.half != 1
    sub = ev.restricted(mask)
    assert all(mask[s].all() for s in sub.sel)
    rows = np.where(mask)[0]
    assert set(sub.half[rows]) == {0, 1}
    m = D["ap_base"]["match"].to_numpy()
    assert not (set(m[rows][sub.half[rows] == 0]) & set(m[rows][sub.half[rows] == 1]))


def test_nested_estimate_returns_scores():
    from arena import Evaluator, nested_estimate
    D = _fake_D()
    ev = Evaluator(D, repeats=2)
    rng = np.random.default_rng(2)
    n = len(D["ap_base"])
    oof = {"m1": rng.dirichlet(np.ones(10), n), "m2": rng.dirichlet(np.ones(10), n)}
    b, s = nested_estimate(ev, list(oof), oof, "p")
    assert 0.0 <= b <= 1.0 and 0.0 <= s <= 1.0
    w = {"m1": rng.random(n), "m2": rng.random(n)}
    b, s = nested_estimate(ev, list(w), w, "w")
    assert 0.0 <= b <= 1.0 and 0.0 <= s <= 1.0


def test_match_history_uses_only_earlier_rallies():
    from ttmatch.history import match_history, HIST_PRIOR
    rows = []
    # 一場比賽、一局、三個 rally：選手 1 在 rally 1 打 2 拍攻擊（action 1）、rally 2 打 1 拍控制（action 10）
    for rid, strokes in {1: [(1, 1, 1), (2, 2, 8), (3, 1, 1)], 2: [(1, 1, 10), (2, 2, 8)], 3: [(1, 2, 8)]}.items():
        for sn, player, action in strokes:
            rows.append(dict(rally_uid=rid, match=1, numberGame=1, rally_id=rid, strikeNumber=sn,
                             gamePlayerId=player, gamePlayerOtherId=3 - player, actionId=action,
                             pointId=5, strengthId=2))
    df = pd.DataFrame(rows)
    h = match_history(df).set_index(["match", "numberGame", "rally_id", "next_player"])
    assert h.loc[(1, 1, 1, 1), "hist_n"] == 0 and h.loc[(1, 1, 1, 2), "hist_n"] == 0
    assert h.loc[(1, 1, 2, 1), "hist_n"] == 2 and h.loc[(1, 1, 3, 1), "hist_n"] == 3
    # rally 3 之前，選手 1 的 3 拍有 2 拍攻擊；全體攻擊率 2/6
    g_att = 2 / 6
    assert abs(h.loc[(1, 1, 3, 1), "hist_attack_rate"] - (2 + HIST_PRIOR * g_att) / (3 + HIST_PRIOR)) < 1e-12
    # 選手 2 在 rally 3 本身的拍不能算進 rally 3 的歷史
    assert h.loc[(1, 1, 3, 2), "hist_n"] == 2
    # 截斷模擬：只看第 1 拍 → 選手 1 在 rally 3 之前只剩 2 拍（rally 1、2 的第 1 拍）
    h1 = match_history(df, {1: 1.0}).set_index(["match", "numberGame", "rally_id", "next_player"])
    assert h1.loc[(1, 1, 3, 1), "hist_n"] == 2 and h1.loc[(1, 1, 3, 2), "hist_n"] == 0
    # 全部 k=2：只有 rally 1（3 拍）配得到 k=2；rally 2（2 拍）配不到 → k = 長度 − 1 = 1
    h2 = match_history(df, {2: 1.0}).set_index(["match", "numberGame", "rally_id", "next_player"])
    assert h2.loc[(1, 1, 3, 1), "hist_n"] == 2 and h2.loc[(1, 1, 3, 2), "hist_n"] == 1
    # 全部 k=10：沒有 rally 配得到 → 都退回 k = 長度 − 1，永遠看不到最後一拍
    h10 = match_history(df, {10: 1.0}).set_index(["match", "numberGame", "rally_id", "next_player"])
    assert h10.loc[(1, 1, 3, 1), "hist_n"] == 2 and h10.loc[(1, 1, 3, 2), "hist_n"] == 1
    assert (h10["hist_pt0"] < 0.5).all()  # 沒有任何 pointId 0 的最後一拍被算進去（此例全是 pointId 5）
    # keep_prob=0 → 沒有任何 rally 可見
    h3 = match_history(df, None, 0.0).set_index(["match", "numberGame", "rally_id", "next_player"])
    assert (h3["hist_n"] == 0).all()


def test_mixed_score_two_dimensional():
    from arena import mixed_score
    rng = np.random.default_rng(3)
    groups = np.repeat(np.arange(20), 10)
    main, plain = np.zeros((200, 3)), np.ones((200, 3))
    seen = []
    m, s = mixed_score(lambda p: seen.append(p) or float(p[:, 0].mean()), groups, main, plain, 0.25, repeats=3)
    assert abs(m - 0.25) < 1e-12 and s == 0.0          # 每次剛好 5 / 20 場 match 換成 plain
    assert all(((p == 0) | (p == 1)).all() and (p.min(1) == p.max(1)).all() for p in seen)  # 整列一起換
