"""`_fs` 方法（終結拍分解 + test 內部轉移 + 驗證 match 的模擬內部轉移）的 OOF 與 test 是不是同一種行為。

兩張表：

1. 各方法的條件分佈平均最大機率，CV 抽樣 vs test。落點先拿掉類別 0 再重新歸一（只看「不是終結拍時」的條件
   分佈，與終結拍比例無關）；球種直接取最大機率。只加 test 內部轉移的 `_fa`，test 會比 CV 尖銳；`_fs` 應該接近。
2. 「少了自己 rally 前幾拍的轉移」差多少。`_fs` 的 OOF 只有一部分列與 test 完全同條件（可見 rally、前綴長度
   等於模擬可見前綴長度 k；訓練資料裡有它自己的第 1..k−1 列），其餘的列少了自己 rally 的轉移。這裡在
   k ≥ 2 的同條件列上，比較同一列的兩個預測：模型 g（訓練過它自己的第 1..k−1 列）與另一個模型
   （看不到這個 rally，只有同場其他 rally 的轉移）。兩者接近，代表其餘的列也接近 test 的行為。
   只看非終結列（非終結條件模型只在這些列上有標籤）。

用法：python tools/sim_aug_check.py lgbm_hist_prog_fa lgbm_hist_prog_fs lgbm_hist_prog_fs1   （不給名稱 = 全部）
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Evaluator, TASKS, confidence as cond_max   # noqa: E402
from ttmatch import config                     # noqa: E402
from ttmatch.data import build_datasets        # noqa: E402


def row_kinds(base: pd.DataFrame) -> np.ndarray:
    """每一列與 test 的關係：0 = 完全同條件（前綴長度 1，或可見 rally 且前綴 == 模擬可見前綴 k）；
    1 = 少了一部分自己 rally 的轉移（可見 rally、前綴 > k ≥ 2）；2 = 少了全部（其餘前綴 ≥ 2 的列）。"""
    k = base[sorted(c for c in base.columns if c.startswith("vis_k"))].to_numpy().max(1)
    p = base["prefix_len"].to_numpy()
    return np.where((p == 1) | (p == k), 0, np.where((k >= 2) & (p > k), 1, 2))


def own_rally_effect(z: dict, base: pd.DataFrame, task: str) -> dict | None:
    """在 k ≥ 2 的同條件非終結列上，比較「訓練過自己 rally 前幾拍」與「沒有」的兩個模型。"""
    if f"simnt_{task}" not in z:
        return None
    col, classes = TASKS[task]
    vis = base[sorted(c for c in base.columns if c.startswith("vis_k"))].to_numpy()
    k, g = vis.max(1), vis.argmax(1)
    y = base[col].to_numpy().astype(int)
    rows = np.where((k >= 2) & (base["prefix_len"].to_numpy() == k) & (base["is_terminal"].to_numpy() == 0)
                    & np.isin(y, classes))[0]
    assert (z["sim_use"][rows] == g[rows]).all()
    yi = np.searchsorted(classes, y[rows])
    nt = z[f"simnt_{task}"].astype(float)
    G = nt.shape[1]
    out = {"n": len(rows)}
    for lab, gg in (("with", g[rows]), ("without", (g[rows] + 1) % G)):
        p = nt[rows, gg]
        if task == "p":                          # 非終結條件模型的類別 0 幾乎是 0；與第 1 張表一致，拿掉再歸一
            p, yy, keep = p[:, 1:], yi - 1, yi >= 1
        else:
            yy, keep = yi, np.ones(len(rows), dtype=bool)
        p = np.clip(p / np.clip(p.sum(1, keepdims=True), 1e-12, None), 1e-9, 1.0)
        out[f"{lab}_logloss"] = float(-np.log(p[keep, yy[keep]]).mean())
        out[f"{lab}_acc"] = float((p.argmax(1) == yy)[keep].mean())
        out[f"{lab}_maxp"] = float(p.max(1).mean())
    return out


def main():
    names = sys.argv[1:] or sorted(f.stem for f in config.OOF_DIR.glob("*.npz"))
    D = build_datasets()
    base, tp = D["ap_base"], D["test_pred"]
    sel = np.concatenate(Evaluator(D).sel)
    reliable = tp["prog_unreliable"].to_numpy() == 0
    kind = row_kinds(base)[sel]
    print("CV 評估列與 test 的關係：完全同條件 {:.1%}、少一部分自己 rally 的轉移 {:.1%}、少全部 {:.1%}".format(
        *[(kind == i).mean() for i in range(3)]))
    print()
    rows, own = [], []
    for n in names:
        z = dict(np.load(config.OOF_DIR / f"{n}.npz"))
        if "oof_a" not in z and "oof_p" not in z:
            continue
        r = {"方法": n}
        for t, lab in (("a", "action"), ("p", "point（非終結）")):
            if f"oof_{t}" not in z:
                continue
            cv, te = cond_max(z[f"oof_{t}"][sel], t == "p"), cond_max(z[f"te_{t}"][reliable], t == "p")
            r[f"{lab}：CV"], r[f"{lab}：test"], r[f"{lab}：差"] = cv, te, te - cv
            if f"simnt_{t}" in z:
                same = cond_max(z[f"oof_{t}"][sel[kind == 0]], t == "p")
                r[f"{lab}：CV 同條件列"] = same
            e = own_rally_effect(z, base, t)
            if e is not None:
                own.append({"方法": n, "任務": lab, "列數": e["n"],
                            "logloss：有自己的轉移": e["with_logloss"], "沒有": e["without_logloss"],
                            "正確率：有": e["with_acc"], "正確率：沒有": e["without_acc"],
                            "平均最大機率：有": e["with_maxp"], "平均最大機率：沒有": e["without_maxp"]})
        rows.append(r)
    print(pd.DataFrame(rows).set_index("方法").round(3).to_markdown())
    if own:
        print()
        print("k ≥ 2 的同條件非終結列：同一列，「訓練過自己 rally 的第 1..k−1 列」的模型 vs 沒有的模型")
        print()
        print(pd.DataFrame(own).set_index(["方法", "任務"]).round(4).to_markdown())


if __name__ == "__main__":
    main()
