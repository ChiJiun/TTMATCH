"""`_fs` 方法（終結拍分解 + test 內部轉移 + 驗證 match 的模擬內部轉移）的 OOF 與 test 是不是同一種行為，
以及它比 `_fa` 高的分數是從哪裡來的。

四張表：

1. 各方法的條件分佈平均最大機率，CV 抽樣 vs test。落點先拿掉類別 0 再重新歸一（只看「不是終結拍時」的條件
   分佈，與終結拍比例無關）；球種直接取最大機率。只加 test 內部轉移的 `_fa`，test 會比 CV 尖銳；`_fs` 應該接近。
2. 「少了自己 rally 前幾拍的轉移」差多少。`_fs` 的 OOF 只有一部分列與 test 完全同條件（可見 rally、前綴長度
   等於模擬可見前綴長度 k；訓練資料裡有它自己的第 1..k−1 列），其餘的列少了自己 rally 的轉移。這裡在
   k ≥ 2 的同條件列上，比較同一列的兩個預測：模型 g（訓練過它自己的第 1..k−1 列）與另一個模型
   （看不到這個 rally，只有同場其他 rally 的轉移）。兩者接近，代表其餘的列也接近 test 的行為。
   只看非終結列（非終結條件模型只在這些列上有標籤）。
3. 兩個模型是否對稱。模型 0 的模擬轉移與 history 特徵來自同一組可見 rally（與 test 相同）；模型 1 的轉移來自
   另一組，與 history 特徵互補，看到的同場資訊可能比 test 多。這裡在「兩組都看不到的 rally」（兩個模型都沒
   訓練過它的任何一列）的非終結列上比較兩者；模型 1 明顯比較好的話，用模型 1 的 OOF 列就偏樂觀。
4. 分數差在哪些列（要給至少兩個方法才印）。評估列依「下一拍擊球者在訓練 fold 出現的列數」以及「與 test 的
   關係」分組，列出每個方法完整機率的 logloss 與正確率。`_fs` 比 `_fa` 多出來的部分如果是同場其他 rally 的
   轉移帶來的，它應該集中在訓練 fold 少見的選手，而且在「訓練資料裡完全沒有自己 rally」的列上一樣大。

用法：python tools/sim_aug_check.py lgbm_hist lgbm_hist_fa lgbm_hist_fs lgbm_hist_fs1   （不給名稱 = 全部）
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arena import Evaluator, TASKS, confidence as cond_max   # noqa: E402
from ttmatch import config                     # noqa: E402
from ttmatch.cv import get_folds               # noqa: E402
from ttmatch.data import build_datasets        # noqa: E402

LABELS = (("a", "action"), ("p", "point（非終結）"))
PLAYER_BINS = ((-1, 0, "0"), (0, 300, "1–300"), (300, 1500, "301–1500"), (1500, np.inf, "> 1500"))


def _vis(base: pd.DataFrame) -> np.ndarray:
    return base[sorted(c for c in base.columns if c.startswith("vis_k"))].to_numpy()


def row_kinds(base: pd.DataFrame) -> np.ndarray:
    """每一列與 test 的關係：0 = 完全同條件（前綴長度 1，或可見 rally 且前綴 == 模擬可見前綴 k）；
    1 = 少了一部分自己 rally 的轉移（可見 rally、前綴 > k ≥ 2）；2 = 少了全部（其餘前綴 ≥ 2 的列）。"""
    k = _vis(base).max(1)
    p = base["prefix_len"].to_numpy()
    return np.where((p == 1) | (p == k), 0, np.where((k >= 2) & (p > k), 1, 2))


def loss_acc(p: np.ndarray, y: np.ndarray, classes, rows: np.ndarray) -> tuple[float, float, int]:
    """rows 中標籤屬於 classes 的列：歸一後的 logloss、正確率、列數。"""
    classes = np.asarray(classes)
    rows = rows[np.isin(y[rows], classes)]
    yi = np.searchsorted(classes, y[rows])
    q = np.asarray(p[rows], dtype=float)
    q = np.clip(q / np.clip(q.sum(1, keepdims=True), 1e-12, None), 1e-9, 1.0)
    return float(-np.log(q[np.arange(len(rows)), yi]).mean()), float((q.argmax(1) == yi).mean()), len(rows)


def _nonterminal(z: dict, task: str, g) -> tuple[np.ndarray, list]:
    """第 g 個非終結條件模型的輸出與對應的類別（g 可以是每一列各自的模型編號）。
    落點的類別 0 幾乎是 0，與第 1 張表一致，拿掉。"""
    classes = list(TASKS[task][1])
    nt = z[f"simnt_{task}"]
    p = nt[np.arange(len(nt)), g] if isinstance(g, np.ndarray) else nt[:, g]
    return (p[:, 1:], classes[1:]) if task == "p" else (p, classes)


def own_rally_effect(z: dict, base: pd.DataFrame, task: str) -> dict | None:
    """在 k ≥ 2 的同條件非終結列上，比較「訓練過自己 rally 前幾拍」與「沒有」的兩個模型。"""
    if f"simnt_{task}" not in z:
        return None
    vis = _vis(base)
    k, g = vis.max(1), vis.argmax(1)
    y = base[TASKS[task][0]].to_numpy().astype(int)
    rows = np.where((k >= 2) & (base["prefix_len"].to_numpy() == k) & (base["is_terminal"].to_numpy() == 0))[0]
    assert (z["sim_use"][rows] == g[rows]).all()
    out = {}
    for lab, gg in (("with", g), ("without", (g + 1) % vis.shape[1])):
        p, classes = _nonterminal(z, task, gg)
        out[f"{lab}_logloss"], out[f"{lab}_acc"], out["n"] = loss_acc(p, y, classes, rows)
        out[f"{lab}_maxp"] = cond_max(p[rows] / np.clip(p[rows].sum(1, keepdims=True), 1e-12, None))
    return out


def model_symmetry(z: dict, base: pd.DataFrame, task: str) -> dict | None:
    """兩組都看不到的 rally 的非終結列（每個模型都沒訓練過它）：各個模型的 logloss 與正確率。"""
    if f"simnt_{task}" not in z:
        return None
    vis = _vis(base)
    y = base[TASKS[task][0]].to_numpy().astype(int)
    rows = np.where((vis.max(1) == 0) & (base["is_terminal"].to_numpy() == 0))[0]
    out = {}
    for g in range(vis.shape[1]):
        p, classes = _nonterminal(z, task, g)
        out[f"logloss：模型 {g}"], out[f"正確率：模型 {g}"], out["列數"] = loss_acc(p, y, classes, rows)
    return out


def train_fold_player_rows(base: pd.DataFrame) -> np.ndarray:
    """每一列的下一拍擊球者，在它所屬驗證 fold 的訓練資料裡有幾列（0 = 訓練時沒看過這位選手）。"""
    player = base["next_player"].to_numpy()
    out = np.zeros(len(base))
    for tr_idx, va_idx in get_folds(base["match"].to_numpy(), config.N_FOLDS):
        counts = pd.Series(player[tr_idx]).value_counts()
        out[va_idx] = pd.Series(player[va_idx]).map(counts).fillna(0).to_numpy()
    return out


def main():
    names = sys.argv[1:] or sorted(f.stem for f in config.OOF_DIR.glob("*.npz") if not f.stem.endswith("_fast"))
    D = build_datasets()
    base, tp = D["ap_base"], D["test_pred"]
    sel = np.concatenate(Evaluator(D).sel)
    reliable = tp["prog_unreliable"].to_numpy() == 0
    kinds = row_kinds(base)
    kind = kinds[sel]
    print("CV 評估列與 test 的關係：完全同條件 {:.1%}、少一部分自己 rally 的轉移 {:.1%}、少全部 {:.1%}".format(
        *[(kind == i).mean() for i in range(3)]))
    print()
    zs = {}
    for n in names:
        z = dict(np.load(config.OOF_DIR / f"{n}.npz"))
        if "oof_a" in z or "oof_p" in z:
            zs[n] = z
    rows, own, sym = [], [], []
    for n, z in zs.items():
        r = {"方法": n}
        for t, lab in LABELS:
            if f"oof_{t}" not in z:
                continue
            cv, te = cond_max(z[f"oof_{t}"][sel], t == "p"), cond_max(z[f"te_{t}"][reliable], t == "p")
            r[f"{lab}：CV"], r[f"{lab}：test"], r[f"{lab}：差"] = cv, te, te - cv
            e = own_rally_effect(z, base, t)
            if e is not None:
                own.append({"方法": n, "任務": lab, "列數": e["n"],
                            "logloss：有自己的轉移": e["with_logloss"], "沒有": e["without_logloss"],
                            "正確率：有": e["with_acc"], "正確率：沒有": e["without_acc"],
                            "平均最大機率：有": e["with_maxp"], "平均最大機率：沒有": e["without_maxp"]})
            s = model_symmetry(z, base, t)
            if s is not None:
                sym.append({"方法": n, "任務": lab, **s})
        rows.append(r)
    print(pd.DataFrame(rows).set_index("方法").round(3).to_markdown())
    if own:
        print()
        print("k ≥ 2 的同條件非終結列：同一列，「訓練過自己 rally 的第 1..k−1 列」的模型 vs 沒有的模型")
        print()
        print(pd.DataFrame(own).set_index(["方法", "任務"]).round(4).to_markdown())
    if sym:
        print()
        print("兩組都看不到的 rally 的非終結列：兩個模型都沒訓練過它，表現應該相同")
        print()
        print(pd.DataFrame(sym).set_index(["方法", "任務"]).round(4).to_markdown())
    if len(zs) < 2:
        return
    seen = train_fold_player_rows(base)
    all_train = base["next_player"].value_counts()
    seen_te = tp["next_player"].map(all_train).fillna(0).to_numpy()
    plen = base["prefix_len"].to_numpy()
    groups = [(f"擊球者在訓練 fold 的列數 {lab}", (seen > lo) & (seen <= hi)) for lo, hi, lab in PLAYER_BINS]
    groups += [("同條件，前綴長度 1", (kinds == 0) & (plen == 1)), ("同條件，前綴長度 ≥ 2", (kinds == 0) & (plen >= 2)),
               ("少一部分自己 rally 的轉移", kinds == 1), ("少全部", kinds == 2), ("全部", np.ones(len(base), dtype=bool))]
    print()
    print("下一拍擊球者在訓練資料裡的列數（CV：所屬 fold 的訓練資料；test：整個 train）各佔多少：")
    print("  " + "、".join(f"{lab}：CV {((seen[sel] > lo) & (seen[sel] <= hi)).mean():.1%} / "
                         f"test {((seen_te > lo) & (seen_te <= hi)).mean():.1%}" for lo, hi, lab in PLAYER_BINS))
    for t, lab in (("a", "action"), ("p", "point")):
        have = {n: z for n, z in zs.items() if f"oof_{t}" in z}
        y = base[TASKS[t][0]].to_numpy().astype(int)
        out = []
        for glab, mask in groups:
            rows_g = sel[mask[sel]]
            r = {"評估列": glab, "列數": len(rows_g)}
            for n, z in have.items():
                ll, acc, _ = loss_acc(z[f"oof_{t}"], y, TASKS[t][1], rows_g)
                r[f"{n}：logloss"], r[f"{n}：正確率"] = ll, acc
            out.append(r)
        print()
        print(f"{lab}：完整機率在各組評估列上的 logloss 與正確率")
        print()
        print(pd.DataFrame(out).set_index("評估列").round(4).to_markdown())


if __name__ == "__main__":
    main()
