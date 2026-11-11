"""檢查各方法在 test 上是否低估終結拍：比較 CV 抽樣與 test 上的平均 P(落點 0)（README 的對照表）。

終結拍的落點一定是 0，所以平均 P(落點 0) 就是模型認為「要預測的那一拍是終結拍」的比例。CV 的評估樣本已對齊
test 的截斷機制，兩邊應該接近。樹模型若把 test 內部轉移（一定不是終結拍，而且和要預測的列來自同一個
rally）加進訓練，test 上的數字會掉到接近 0，而 CV 完全看不出來（見 ttmatch/methods/trees.py 開頭）。

用法：python tools/terminal_bias.py lgbm_hist_aug lgbm_hist transformer_hist2     （不給名稱 = outputs/oof 全部）
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Evaluator                    # noqa: E402
from ttmatch import config                     # noqa: E402
from ttmatch.data import build_datasets        # noqa: E402


def main():
    names = sys.argv[1:] or sorted(f.stem for f in config.OOF_DIR.glob("*.npz"))
    D = build_datasets()
    base, tp = D["ap_base"], D["test_pred"]
    sel = np.concatenate(Evaluator(D).sel)
    k_cv = np.minimum(base["prefix_len"].to_numpy()[sel], 3)
    k_te = np.minimum(tp["prefix_len"].to_numpy(), 3)
    unseen = ~tp["next_player"].isin(set(base["next_player"])).to_numpy()
    rows = []
    for n in names:
        z = np.load(config.OOF_DIR / f"{n}.npz")
        if "oof_p" not in z:
            continue
        cv, te = z["oof_p"][sel, 0], z["te_p"][:, 0]
        r = {"方法": n, "CV": cv.mean(), "test": te.mean()}
        for k, lab in ((1, "k=1"), (2, "k=2"), (3, "k≥3")):
            r[f"CV {lab}"], r[f"test {lab}"] = cv[k_cv == k].mean(), te[k_te == k].mean()
        r["test（選手沒在 train 出現過）"] = te[unseen].mean()
        rows.append(r)
    term = base["is_terminal"].to_numpy()[sel]
    print(f"CV 抽樣的終結拍比例：{term.mean():.3f}；test 中下一拍擊球者沒在 train 出現過的列：{unseen.mean():.1%}")
    print()
    print(pd.DataFrame(rows).set_index("方法").round(3).to_markdown())


if __name__ == "__main__":
    main()
