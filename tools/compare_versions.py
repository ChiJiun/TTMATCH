"""兩個版本的集成差多少、差異是否超過抽樣雜訊：依 match 做成對 bootstrap。

兩個版本都用 arena 報告裡的配方重建集成的 OOF 機率，決策規則照正式流程跨半調參（每一列的預測都來自
「另一半比賽」調出來的規則），評估樣本是同一組截斷機制對齊的抽樣。接著把 match 有放回地重抽，
每次算兩個版本的 action / point / winner / overall 之差，回報平均與 95% 區間。

注意：集成的成員與權重是各版本在全部 OOF 上挑的（in-sample），兩邊都有同樣性質的選擇偏差；
這裡量的是「換一批比賽，差距還在不在」，不是巢狀估計。

用法：python tools/compare_versions.py outputs/r1 outputs      （後者 − 前者）
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Evaluator, TASKS                               # noqa: E402
from ttmatch import config                                       # noqa: E402
from ttmatch.data import build_datasets                          # noqa: E402
from ttmatch.decision import tune_decision, apply_decision       # noqa: E402
from ttmatch.metrics import macro_f1, winner_auc, overall_score  # noqa: E402
from tools.sensitivity import load_blends                        # noqa: E402


def honest_predictions(ev: Evaluator, blends: dict):
    """回傳 (評估列索引, {a, p: 預測類別；w: 機率})，每一列的決策規則都在另一半比賽上調。"""
    rows = np.concatenate(ev.sel)
    pred = {"w": blends["w"][rows]}
    for t in ("a", "p"):
        classes, y = TASKS[t][1], ev.y[t]
        out = np.empty(len(rows), dtype=int)
        for h in (0, 1):
            tune, _ = ev._split(h)
            dec, _ = tune_decision(blends[t][tune], y[tune], classes)
            m = ev.half[rows] == h
            out[m] = apply_decision(blends[t][rows[m]], classes, dec)
        pred[t] = out
    return rows, pred


def scores(ev: Evaluator, rows: np.ndarray, pred: dict, take: np.ndarray) -> np.ndarray:
    r = rows[take]
    a = macro_f1(ev.y["a"][r], pred["a"][take])
    p = macro_f1(ev.y["p"][r], pred["p"][take])
    w = winner_auc(ev.y["w"][r], pred["w"][take])
    return np.array([a, p, w, overall_score(a, p, w)])


def main(n_boot: int = 300, seed: int = config.SEED):
    d_old, d_new = Path(sys.argv[1]), Path(sys.argv[2])
    ev = Evaluator(build_datasets())
    rows, p_old = honest_predictions(ev, load_blends(d_old))
    _, p_new = honest_predictions(ev, load_blends(d_new))
    match = ev.base["match"].to_numpy()[rows]
    by_match = {m: np.where(match == m)[0] for m in np.unique(match)}
    keys = np.array(list(by_match))
    full = np.arange(len(rows))
    s_old, s_new = scores(ev, rows, p_old, full), scores(ev, rows, p_new, full)
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        take = np.concatenate([by_match[m] for m in rng.choice(keys, size=len(keys))])
        diffs.append(scores(ev, rows, p_new, take) - scores(ev, rows, p_old, take))
    diffs = np.array(diffs)
    lo, hi = np.percentile(diffs, [2.5, 97.5], axis=0)
    print(f"| | {d_old.name} | {d_new.name} | 差 | 95% 區間（依 match 重抽 {n_boot} 次） | 差 > 0 的比例 |")
    print("|---|---:|---:|---:|---|---:|")
    for i, lab in enumerate(("action", "point", "winner", "overall")):
        print(f"| {lab} | {s_old[i]:.4f} | {s_new[i]:.4f} | {s_new[i] - s_old[i]:+.4f} "
              f"| [{lo[i]:+.4f}, {hi[i]:+.4f}] | {(diffs[:, i] > 0).mean():.1%} |")


if __name__ == "__main__":
    main()
