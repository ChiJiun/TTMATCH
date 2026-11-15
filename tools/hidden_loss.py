"""估計「集成在 test 上預測落點 0 / 球種 0 的比例與 CV 不同」造成的損失（README「各版本比較」的隱藏損失）。

樹模型誤用 test 內部轉移時，集成在 test 上幾乎不預測落點 0，而 CV 分數看不出來。這支工具在 CV 上重現這個
情況：把類別 0 的機率乘上一個倍率，直到 CV 上「預測為 0」的比例等於 test 上的比例，再算 Macro-F1
（決策規則照正式流程跨半調參）。前提是 test 的終結拍比例如 F0 的估計（ttmatch/truncation.py），而且只模擬了
類別 0 的預測比例，沒有模擬其他類別的變化，所以是粗估。

用法：python tools/hidden_loss.py [輸出資料夾 ...]     （預設 outputs/；每個資料夾要有 arena_report.md 與 oof/）
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Evaluator, TASKS                               # noqa: E402
from ttmatch import config                                       # noqa: E402
from ttmatch.data import build_datasets                          # noqa: E402
from ttmatch.decision import tune_decision, apply_decision       # noqa: E402
from ttmatch.metrics import macro_f1                             # noqa: E402


def load_oof_and_test(out_dir: Path, task: str):
    """最終集成的 (OOF 機率, test 機率)：有 blend.npz 就直接讀，否則依報告的配方重建主集成。"""
    if (out_dir / "blend.npz").exists():
        z = np.load(out_dir / "blend.npz")
        return z[f"oof_{task}"].astype(float), z[f"te_{task}"].astype(float)
    text = (out_dir / "arena_report.md").read_text(encoding="utf-8")
    recipe = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))[task]
    parts = [(w, np.load(out_dir / "oof" / f"{n}.npz")) for n, w in recipe]
    return (sum(w * z[f"oof_{task}"].astype(float) for w, z in parts),
            sum(w * z[f"te_{task}"].astype(float) for w, z in parts))


def squeeze_class0(p: np.ndarray, classes, dec, target: float) -> np.ndarray:
    """把類別 0 的機率乘上一個倍率（二分搜尋），讓「預測為 0」的比例等於 target。"""
    lo, hi = 0.0, 64.0
    for _ in range(40):
        mid = (lo + hi) / 2
        q = p.copy()
        q[:, 0] *= mid
        if (np.asarray(apply_decision(q, classes, dec)) == 0).mean() > target:
            hi = mid
        else:
            lo = mid
    q = p.copy()
    q[:, 0] *= lo
    return q


def hidden_loss(ev: Evaluator, oof: np.ndarray, te: np.ndarray, task: str) -> dict:
    classes, y = TASKS[task][1], ev.y[task]
    sel = np.concatenate(ev.sel)
    dec, _ = tune_decision(oof[sel], y[sel], classes)
    share_cv = float((np.asarray(apply_decision(oof[sel], classes, dec)) == 0).mean())
    share_te = float((np.asarray(apply_decision(te, classes, dec)) == 0).mean())
    plain, squeezed = [], []
    for h in (0, 1):
        tune, evals = ev._split(h)
        rows = np.concatenate(evals)
        dec_h, _ = tune_decision(oof[tune], y[tune], classes)
        plain.append(macro_f1(y[rows], apply_decision(oof[rows], classes, dec_h)))
        squeezed.append(macro_f1(y[rows], apply_decision(squeeze_class0(oof[rows], classes, dec_h, share_te),
                                                         classes, dec_h)))
    return dict(share_cv=share_cv, share_te=share_te, f1=float(np.mean(plain)), f1_squeezed=float(np.mean(squeezed)))


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [config.OUT_DIR]
    ev = Evaluator(build_datasets())
    print("| 集成 | 任務 | 預測為 0 的比例：CV | test | Macro-F1 | 壓到 test 的比例後 | 差 | 換算 overall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for d in dirs:
        for task, lab, weight in (("a", "action", config.W_ACTION), ("p", "point", config.W_POINT)):
            r = hidden_loss(ev, *load_oof_and_test(d, task), task)
            diff = r["f1_squeezed"] - r["f1"]
            print(f"| {d.name} | {lab} | {r['share_cv']:.3f} | {r['share_te']:.3f} | {r['f1']:.4f} "
                  f"| {r['f1_squeezed']:.4f} | {diff:+.4f} | {weight * diff:+.4f} |", flush=True)


if __name__ == "__main__":
    main()
