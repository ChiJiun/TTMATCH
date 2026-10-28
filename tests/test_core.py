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
