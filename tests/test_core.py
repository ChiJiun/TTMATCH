import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from ttmatch.features import score_progression_test
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


def test_mirror_is_involution():
    assert (POINT_MIRROR[POINT_MIRROR] == np.arange(10)).all()
    assert (POSITION_MIRROR[POSITION_MIRROR] == np.arange(4)).all()
