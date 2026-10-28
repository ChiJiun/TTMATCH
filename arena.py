"""方法競技場：在同一套 CV 下比較 outputs/oof/*.npz 的所有方法，淘汰弱者，集成強者，產生提交檔。

評估方式（所有方法一致）：
1. OOF 來自依 match 分組的 5-fold（test 的 79 場比賽與 train 完全不重疊，所以必須按 match 切）。
2. 每個 rally 依 test 前綴長度分佈抽一個前綴（重複 5 次），讓 CV 的長度組成跟 test 一樣。
3. Macro-F1 的決策規則（溫度 + 類別偏置）用「另一半比賽」調參、在這一半評估，兩半互換取平均，
   避免在同一批資料上調參又評分而高估。
4. 集成：每個任務各自做貪婪前向選擇（優勝劣汰）——從最佳單一方法開始，只有能讓誠實分數上升的
   方法才會被加入，其餘淘汰。

用法：python arena.py [--fast] [--no-override]
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from ttmatch import config
from ttmatch.cv import length_matched_indices
from ttmatch.data import build_datasets, test_length_probs
from ttmatch.decision import (tune_decision, apply_decision,
                              tune_decision_by_depth, apply_decision_by_depth)
from ttmatch.metrics import macro_f1, winner_auc, overall_score
from ttmatch.postprocess import fit_table, apply_override

TASKS = {"a": ("next_action", config.ACTION_CLASSES), "p": ("next_point", config.POINT_CLASSES)}
BLEND_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
MIN_GAIN = 5e-4


class Evaluator:
    def __init__(self, D, repeats=5):
        self.base = D["ap_base"]
        self.sel = length_matched_indices(self.base, test_length_probs(D["tef"]), repeats)
        half = np.zeros(len(self.base), dtype=int)
        gkf = GroupKFold(n_splits=2)
        for h, (_, idx) in enumerate(gkf.split(self.base, groups=self.base["match"])):
            half[idx] = h
        self.half = half
        self.y = {k: self.base[col].to_numpy() for k, (col, _) in TASKS.items()}
        self.y["w"] = self.base["winner"].to_numpy()

    def _split(self, h):
        """回傳 (調參用索引, 每次重複的評估索引)：調參用另一半，評估用第 h 半。"""
        tune = np.concatenate([s[self.half[s] != h] for s in self.sel])
        evals = [s[self.half[s] == h] for s in self.sel]
        return tune, evals

    def f1(self, probs, task, quick=False, override_tables=None, depth=False):
        """誠實的 Macro-F1：決策規則在另一半調參。depth=True 時依前綴深度分桶調參。"""
        classes = TASKS[task][1]
        y = self.y[task]
        plen = self.base["prefix_len"].to_numpy()
        passes = 0 if quick else 2
        scores = []
        for h in (0, 1):
            tune, evals = self._split(h)
            if depth:
                decs = tune_decision_by_depth(probs[tune], y[tune], classes, plen[tune], passes)
            else:
                dec, _ = tune_decision(probs[tune], y[tune], classes, n_coord_passes=passes)
            for ev in evals:
                if depth:
                    pred = apply_decision_by_depth(probs[ev], classes, decs, plen[ev])
                else:
                    pred = apply_decision(probs[ev], classes, dec)
                if override_tables is not None:
                    pred = apply_override(self.base.iloc[ev], pred, override_tables[h])
                scores.append(macro_f1(y[ev], pred))
        return float(np.mean(scores))

    def f1_argmax(self, probs, task):
        classes = np.asarray(TASKS[task][1])
        y = self.y[task]
        return float(np.mean([macro_f1(y[s], classes[probs[s].argmax(1)]) for s in self.sel]))

    def auc(self, score):
        return float(np.nanmean([winner_auc(self.y["w"][s], score[s]) for s in self.sel]))

    def override_tables(self, task):
        col = TASKS[task][0]
        return [fit_table(self.base[self.half != h], col) for h in (0, 1)]


def load_methods(oof_dir: Path, base: pd.DataFrame, test_uids: np.ndarray, fast: bool):
    methods = {}
    for f in sorted(oof_dir.glob("*.npz")):
        is_fast = f.stem.endswith("_fast")
        if is_fast != fast:
            continue
        z = dict(np.load(f))
        assert np.array_equal(z["rally_uid"], base["rally_uid"].to_numpy()), f"{f.name} OOF 列順序不符"
        assert np.array_equal(z["prefix_len"], base["prefix_len"].to_numpy()), f"{f.name} OOF 列順序不符"
        assert np.array_equal(z["test_rally_uid"], test_uids), f"{f.name} test 列順序不符"
        methods[f.stem.removesuffix("_fast")] = z
    return methods


def greedy_blend(names, oof, test, score_fn, log):
    """貪婪前向選擇：每輪嘗試把一個方法以權重 w 混入目前集成，取最大增益；增益不足即停止。"""
    single = {n: score_fn(oof[n]) for n in names}
    best_name = max(single, key=single.get)
    cur_oof, cur_test = oof[best_name].copy(), test[best_name].copy()
    cur_score = single[best_name]
    recipe = [(best_name, 1.0)]
    log.append(f"  起點 {best_name}: {cur_score:.4f}")
    remaining = [n for n in names if n != best_name]
    while remaining:
        cand = None
        for n in remaining:
            for w in BLEND_GRID:
                s = score_fn((1 - w) * cur_oof + w * oof[n])
                if cand is None or s > cand[0]:
                    cand = (s, n, w)
        if cand is None or cand[0] < cur_score + MIN_GAIN:
            break
        s, n, w = cand
        cur_oof = (1 - w) * cur_oof + w * oof[n]
        cur_test = (1 - w) * cur_test + w * test[n]
        recipe = [(m, c * (1 - w)) for m, c in recipe] + [(n, w)]
        log.append(f"  + {n} (w={w:.1f}) → {s:.4f}")
        cur_score = s
        remaining.remove(n)
    if remaining:
        log.append(f"  淘汰：{', '.join(remaining)}")
    return cur_oof, cur_test, recipe, single


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="改讀 *_fast.npz（冒煙測試用）")
    ap.add_argument("--no-override", action="store_true")
    args = ap.parse_args()

    D = build_datasets()
    base, test_df = D["ap_base"], D["test_pred"]
    methods = load_methods(config.OOF_DIR, base, test_df["rally_uid"].to_numpy(), args.fast)
    if not methods:
        raise SystemExit("outputs/oof/ 沒有任何 npz，請先執行 run_method.py")
    ev = Evaluator(D)
    log, rows = [], []

    # --- 1. 單一方法排行榜（同一套誠實評估）---
    for name, z in methods.items():
        r = {"method": name}
        if "oof_a" in z:
            r["action_argmax"] = ev.f1_argmax(z["oof_a"], "a")
            r["action_tuned"] = ev.f1(z["oof_a"], "a")
        if "oof_p" in z:
            r["point_argmax"] = ev.f1_argmax(z["oof_p"], "p")
            r["point_tuned"] = ev.f1(z["oof_p"], "p")
        if "oof_w" in z:
            r["winner_auc"] = ev.auc(z["oof_w"])
        if {"action_tuned", "point_tuned", "winner_auc"} <= r.keys():
            r["overall"] = overall_score(r["action_tuned"], r["point_tuned"], r["winner_auc"])
        rows.append(r)
        print(f"   評估 {name} 完成", flush=True)
    board = pd.DataFrame(rows).set_index("method").sort_values("overall", ascending=False, na_position="last")

    # --- 2. 各任務貪婪集成 ---
    final = {}
    for task, key, label in (("a", "oof_a", "action"), ("p", "oof_p", "point")):
        names = [n for n, z in methods.items() if key in z]
        log.append(f"[{label}] 貪婪集成（快速調參版分數）")
        oof = {n: methods[n][key] for n in names}
        test = {n: methods[n]["te" + key[3:]] for n in names}
        b_oof, b_test, recipe, _ = greedy_blend(
            names, oof, test, lambda p, t=task: ev.f1(p, t, quick=True), log)
        score = ev.f1(b_oof, task)
        s_depth = ev.f1(b_oof, task, depth=True)
        log.append(f"  決策規則：全域 {score:.4f} / 分深度 {s_depth:.4f}")
        use_depth = s_depth > score + MIN_GAIN
        score = max(score, s_depth) if use_depth else score
        use_override = False
        if not args.no_override:
            tables = ev.override_tables(task)
            s_ov = ev.f1(b_oof, task, override_tables=tables, depth=use_depth)
            log.append(f"  規則修正：{score:.4f} → {s_ov:.4f}")
            if s_ov > score + MIN_GAIN:
                score, use_override = s_ov, True
        final[task] = dict(oof=b_oof, test=b_test, recipe=recipe, score=score,
                           override=use_override, depth=use_depth)

    names = [n for n, z in methods.items() if "oof_w" in z]
    log.append("[winner] 貪婪集成（AUC）")
    b_oof, b_test, recipe, _ = greedy_blend(
        names, {n: methods[n]["oof_w"] for n in names}, {n: methods[n]["te_w"] for n in names},
        ev.auc, log)
    final["w"] = dict(oof=b_oof, test=b_test, recipe=recipe, score=ev.auc(b_oof))

    overall = overall_score(final["a"]["score"], final["p"]["score"], final["w"]["score"])

    # --- 3. 產生提交檔：決策規則用全部長度對齊樣本調參 ---
    all_sel = np.concatenate(ev.sel)
    sub = pd.DataFrame({"rally_uid": test_df["rally_uid"].to_numpy()})
    for task, col in (("a", "actionId"), ("p", "pointId")):
        classes = TASKS[task][1]
        y = ev.y[task]
        if final[task]["depth"]:
            plen = base["prefix_len"].to_numpy()
            decs = tune_decision_by_depth(final[task]["oof"][all_sel], y[all_sel], classes, plen[all_sel])
            pred = apply_decision_by_depth(final[task]["test"], classes, decs,
                                           test_df["prefix_len"].to_numpy())
        else:
            dec, _ = tune_decision(final[task]["oof"][all_sel], y[all_sel], classes)
            pred = apply_decision(final[task]["test"], classes, dec)
        if final[task]["override"]:
            pred = apply_override(test_df, pred, fit_table(base, TASKS[task][0]))
        sub[col] = pred.astype(int)
    sub["serverGetPoint"] = np.clip(final["w"]["test"], 0, 1)
    assert sub["rally_uid"].tolist() == D["sample"]["rally_uid"].tolist()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    sub_path = config.OUT_DIR / ("submission_fast.csv" if args.fast else "submission.csv")
    sub.to_csv(sub_path, index=False)

    # --- 4. 報告 ---
    fmt = board.round(4)
    lines = ["# Arena 報告", "", "## 單一方法（同一套 CV、決策規則跨半調參）", "",
             fmt.to_markdown(), "", "## 集成過程", "", "```", *log, "```", "",
             "## 最終集成（誠實 CV）", "",
             *[f"- {lab} Macro-F1：{final[t]['score']:.4f}"
               f"（分深度決策：{'是' if final[t]['depth'] else '否'}；規則修正：{'是' if final[t]['override'] else '否'}）"
               for t, lab in (("a", "action"), ("p", "point"))],
             f"- winner AUC：{final['w']['score']:.4f}",
             f"- **overall：{overall:.4f}**", "",
             "配方：", "```json",
             json.dumps({t: [(n, round(w, 3)) for n, w in final[t]["recipe"]] for t in ("a", "p", "w")},
                        ensure_ascii=False, indent=1), "```", "",
             f"提交檔：`outputs/{sub_path.name}`"]
    report = config.OUT_DIR / ("arena_report_fast.md" if args.fast else "arena_report.md")
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
