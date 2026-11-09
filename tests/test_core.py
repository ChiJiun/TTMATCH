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
