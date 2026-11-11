"""方法競技場：在同一套 CV 下比較 outputs/oof/*.npz 的所有方法，淘汰弱者，集成強者，產生提交檔。

評估方式（所有方法一致）：
1. OOF 來自依 match 分組的 5-fold（test 的 79 場比賽與 train 完全不重疊，所以必須按 match 切）。
2. 每個 rally 依 test 前綴長度分佈抽一個前綴（重複 5 次），讓 CV 的長度組成跟 test 一樣。
3. Macro-F1 的決策規則（溫度 + 類別偏置）用「另一半比賽」調參、在這一半評估，兩半互換取平均，
   避免在同一批資料上調參又評分而高估。
4. 集成：每個任務各自做貪婪前向選擇（優勝劣汰）——從最佳單一方法開始，只有能讓誠實分數上升的
   方法才會被加入，其餘淘汰。

用法：
  python arena.py                                         # 全部方法 → outputs/submission.csv
  python arena.py --exclude lgbm_prog cat_prog --tag noprog
                                                          # 排除指定方法 → outputs/submission_noprog.csv
  python arena.py --fast                                  # 讀 *_fast.npz（冒煙測試）
  python arena.py --no-override                           # 不嘗試規則修正
  python arena.py --no-nested                             # 跳過巢狀估計
  python arena.py --preset final                          # 只用 README 結果所用的方法集（FINAL_METHODS）
"""
from __future__ import annotations
import argparse
import copy
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

# README「結果」所用的方法集（改善回合 1 的結論）。依巢狀估計挑選：再加入 action/point 也用比分推移的
# lgbm_hist_prog / xgb_hist2_prog / transformer_hist2_prog，巢狀估計反而從 0.4177 降到 0.4157，因此不納入。
FINAL_METHODS = [
    "cat", "cat_hist", "cat_prog", "cat_prog_bag3", "gru", "gru_hist", "gru_mirror", "gru_pemb", "gru_pemb_hist",
    "gru_t2v_film", "lgbm_hist", "lgbm_hist2", "lgbm_note", "lgbm_prog", "transformer", "transformer_hist",
    "transformer_hist2", "transformer_pemb", "xgb", "xgb_hist", "xgb_hist2", "xgb_prog",
]
# 不含比分推移特徵的版本（課堂規則若不允許比分推移時使用）
NOPROG_METHODS = [m for m in FINAL_METHODS if not m.endswith("_prog") and m != "cat_prog_bag3"]
PRESETS = {"final": FINAL_METHODS, "noprog": NOPROG_METHODS}
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

    def restricted(self, mask: np.ndarray) -> "Evaluator":
        """只看 mask 為 True 的列的評估器；兩半在這些列的 match 上重新切（巢狀估計用）。"""
        sub = copy.copy(self)
        sub.sel = [s[mask[s]] for s in self.sel]
        rows = np.where(mask)[0]
        half = np.zeros(len(self.base), dtype=int)
        for h, (_, idx) in enumerate(GroupKFold(n_splits=2).split(
                rows, groups=self.base["match"].to_numpy()[rows])):
            half[rows[idx]] = h
        sub.half = half
        return sub

    def f1(self, probs, task, quick=False, override_tables=None, depth=False, halves=(0, 1)):
        """誠實的 Macro-F1：決策規則在另一半調參。depth=True 時依前綴深度分桶調參。
        halves 可只評其中一半（調參仍用另一半）。"""
        classes = TASKS[task][1]
        y = self.y[task]
        plen = self.base["prefix_len"].to_numpy()
        passes = 0 if quick else 2
        scores = []
        for h in halves:
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

    def auc(self, score, halves=(0, 1)):
        return float(np.nanmean([winner_auc(self.y["w"][s[np.isin(self.half[s], halves)]],
                                            score[s[np.isin(self.half[s], halves)]]) for s in self.sel]))

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


def uses_prog(name: str, z: dict) -> bool:
    """方法是否使用比分推移特徵。新版 npz 由 run_method.py 存 uses_prog 旗標；舊檔依名稱後綴判斷。"""
    if "uses_prog" in z:
        return bool(np.asarray(z["uses_prog"]).item())
    return name.endswith("_prog")


def mixed_score(score_fn, groups: np.ndarray, main_oof: np.ndarray, plain_oof: np.ndarray,
                frac: float, repeats: int = 20, seed: int = config.SEED) -> tuple[float, float]:
    """模擬 test 的情況：隨機挑 frac 比例的 match 改用 plain 集成，其餘用主集成，回傳整體分數的平均與標準差。
    main_oof / plain_oof 可以是一維（winner 機率）或二維（類別機率）。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    k = int(round(frac * len(uniq)))
    if k == 0:
        return float(score_fn(main_oof)), 0.0
    scores = []
    for _ in range(repeats):
        pick = np.isin(groups, rng.choice(uniq, size=k, replace=False))
        if main_oof.ndim == 2:
            pick = pick[:, None]
        scores.append(score_fn(np.where(pick, plain_oof, main_oof)))
    return float(np.mean(scores)), float(np.std(scores))


def mixed_auc(ev, groups, main_oof, plain_oof, frac, repeats: int = 20, seed: int = config.SEED):
    return mixed_score(ev.auc, groups, main_oof, plain_oof, frac, repeats, seed)


def nested_estimate(ev: Evaluator, names, oof, task):
    """巢狀估計：集成成員、權重與決策規則全部只用「另一半比賽」挑選 / 調參，再在這一半評分，兩半互換。

    回傳 (集成的巢狀分數, 另一半挑出的最佳單一方法在這一半的分數)。
    主流程的集成分數是用全部 OOF 挑成員，所以略為樂觀；這個數字沒有那層選擇偏差。

    涵蓋範圍：只重做集成成員、權重與決策規則的挑選。主流程的其他挑選——分深度決策、規則修正、
    異常 match 的 plain 集成混合——不在這裡重做；目前前兩者都沒有被採用，plain 混合另由 mixed_score
    估計。此外，若用這個數字在多個方法集（版本）之間挑選，被選中的版本仍會有一點選擇偏差。
    """
    blend_scores, single_scores = [], []
    for h in (0, 1):
        sel_mask = ev.half != h
        ev_s = ev.restricted(sel_mask)
        fn = ev_s.auc if task == "w" else (lambda p, t=task: ev_s.f1(p, t, quick=True))
        b_oof, _, _, single = greedy_blend(names, oof, oof, fn, log=[])
        best_single = max(single, key=single.get)
        if task == "w":
            blend_scores.append(ev.auc(b_oof, halves=(h,)))
            single_scores.append(ev.auc(oof[best_single], halves=(h,)))
        else:
            blend_scores.append(ev.f1(b_oof, task, halves=(h,)))
            single_scores.append(ev.f1(oof[best_single], task, halves=(h,)))
    return float(np.mean(blend_scores)), float(np.mean(single_scores))


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


def _display_path(p: Path) -> str:
    """repo 內的路徑顯示相對路徑，否則顯示絕對路徑（設了 TTMATCH_OUT_DIR 時）。"""
    try:
        return p.resolve().relative_to(Path(__file__).resolve().parent).as_posix()
    except ValueError:
        return str(p.resolve())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true", help="改讀 *_fast.npz（冒煙測試用）")
    ap.add_argument("--no-override", action="store_true")
    ap.add_argument("--exclude", nargs="*", default=[], help="不納入比較的方法名稱，例如 lgbm_prog cat_prog")
    ap.add_argument("--tag", default="", help="輸出檔名後綴，例如 noprog → submission_noprog.csv")
    ap.add_argument("--no-nested", action="store_true", help="跳過巢狀估計（省時間）")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="只用預先定義的方法集（final / noprog）")
    args = ap.parse_args()

    D = build_datasets()
    base, test_df = D["ap_base"], D["test_pred"]
    methods = load_methods(config.OOF_DIR, base, test_df["rally_uid"].to_numpy(), args.fast)
    if not methods:
        raise SystemExit("outputs/oof/ 沒有任何 npz，請先執行 run_method.py")
    unknown = sorted(set(args.exclude) - methods.keys())
    if unknown:
        print(f"警告：--exclude 中的 {', '.join(unknown)} 不在 outputs/oof/ 裡，已忽略", flush=True)
    methods = {n: z for n, z in methods.items() if n not in set(args.exclude)}
    if args.preset:
        missing = sorted(set(PRESETS[args.preset]) - methods.keys())
        if missing:
            raise SystemExit(f"--preset {args.preset} 需要的方法還沒訓練：{', '.join(missing)}（請先執行 run_method.py）")
        methods = {n: methods[n] for n in PRESETS[args.preset]}
    if not methods:
        raise SystemExit("--exclude 排除了所有方法，沒有剩下可比較的方法")
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
    board = pd.DataFrame(rows).set_index("method")
    if "overall" in board.columns:
        board = board.sort_values("overall", ascending=False, na_position="last")
    for key in ("oof_a", "oof_p", "oof_w"):
        if not any(key in z for z in methods.values()):
            raise SystemExit(f"沒有任何方法提供 {key}，無法產生提交檔")

    # --- 2. 各任務貪婪集成 ---
    final = {}
    unreliable = test_df["prog_unreliable"].to_numpy().astype(bool)
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
        # 有些 action/point 方法也用比分推移特徵；test 異常 match 的列改用不含它的集成（同 winner）
        plain = [n for n in names if not uses_prog(n, methods[n])]
        if unreliable.any() and plain and len(plain) < len(names):
            log.append(f"[{label}] 不含比分推移的集成（用於 test 異常 match 的 {int(unreliable.sum())} 列）")
            p_oof, p_test, p_recipe, _ = greedy_blend(
                plain, {n: oof[n] for n in plain}, {n: test[n] for n in plain},
                lambda p, t=task: ev.f1(p, t, quick=True), log)
            final[task].update(oof_plain=p_oof, test_plain=p_test, recipe_plain=p_recipe,
                               score_plain=ev.f1(p_oof, task), frac_plain=float(unreliable.mean()))
            final[task]["score_mixed"] = mixed_score(
                lambda p, t=task: ev.f1(p, t, quick=True), base["match"].to_numpy(), b_oof, p_oof,
                float(unreliable.mean()), repeats=10)
            # mixed 用快速調參估計；換算成與 score 同尺度：score − (快速版主集成 − 快速版混合)
            q_main = ev.f1(b_oof, task, quick=True)
            final[task]["score_mixed"] = (score - (q_main - final[task]["score_mixed"][0]),
                                          final[task]["score_mixed"][1])

    names = [n for n, z in methods.items() if "oof_w" in z]
    log.append("[winner] 貪婪集成（AUC）")
    b_oof, b_test, recipe, _ = greedy_blend(
        names, {n: methods[n]["oof_w"] for n in names}, {n: methods[n]["te_w"] for n in names},
        ev.auc, log)
    final["w"] = dict(oof=b_oof, test=b_test, recipe=recipe, score=ev.auc(b_oof))
    # test 異常 match 的 rally 拿不到比分推移資訊；train 中「沒有下一個 rally」幾乎都在局末，
    # 直接用含比分推移的模型會把它們當成局末。這些列改用不含比分推移特徵的方法另做一組集成。
    plain = [n for n in names if not uses_prog(n, methods[n])]
    if unreliable.any() and plain and len(plain) < len(names):
        log.append(f"[winner] 不含比分推移的集成（用於 test 異常 match 的 {int(unreliable.sum())} 列）")
        p_oof, p_test, p_recipe, _ = greedy_blend(
            plain, {n: methods[n]["oof_w"] for n in plain}, {n: methods[n]["te_w"] for n in plain},
            ev.auc, log)
        final["w"]["test"] = np.where(unreliable, p_test, final["w"]["test"])
        final["w"]["recipe_plain"] = p_recipe
        final["w"]["score_plain"] = ev.auc(p_oof)
        # CV 的 0.846 假設每一列都有比分推移；test 約 14% 列只能用 plain 集成，而 AUC 是整份一起算
        frac = float(unreliable.mean())
        final["w"]["frac_plain"] = frac
        final["w"]["score_mixed"] = mixed_auc(ev, base["match"].to_numpy(), b_oof, p_oof, frac)

    overall = overall_score(final["a"]["score"], final["p"]["score"], final["w"]["score"])

    # --- 2b. 巢狀估計：集成與決策都在另一半選，在這一半評 ---
    nested = {}
    if not args.no_nested:
        for task, key in (("a", "oof_a"), ("p", "oof_p"), ("w", "oof_w")):
            names = [n for n, z in methods.items() if key in z]
            nested[task] = nested_estimate(ev, names, {n: methods[n][key] for n in names}, task)
            print(f"   巢狀估計 {task} 完成", flush=True)
        nested["overall"] = overall_score(nested["a"][0], nested["p"][0], nested["w"][0])
    has_mixed = any("score_mixed" in final[t] for t in ("a", "p", "w"))
    overall_mixed = (overall_score(*[final[t].get("score_mixed", (final[t]["score"],))[0] for t in ("a", "p", "w")])
                     if has_mixed else None)

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
        if "test_plain" in final[task]:
            dec_p, _ = tune_decision(final[task]["oof_plain"][all_sel], y[all_sel], classes)
            pred_plain = apply_decision(final[task]["test_plain"], classes, dec_p)
            pred = np.where(unreliable, pred_plain, pred)
        if final[task]["override"]:
            pred = apply_override(test_df, pred, fit_table(base, TASKS[task][0]))
        sub[col] = pred.astype(int)
    sub["serverGetPoint"] = np.clip(final["w"]["test"], 0, 1)
    assert sub["rally_uid"].tolist() == D["sample"]["rally_uid"].tolist()
    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = ("_fast" if args.fast else "") + (f"_{args.tag}" if args.tag else "")
    sub_path = config.OUT_DIR / f"submission{suffix}.csv"
    sub.to_csv(sub_path, index=False)

    # --- 4. 報告 ---
    fmt = board.round(4)
    lines = ["# Arena 報告", "",
             *([f"排除的方法：{', '.join(args.exclude)}", ""] if args.exclude else []),
             "## 單一方法（同一套 CV、決策規則跨半調參）", "",
             fmt.to_markdown(), "", "## 集成過程", "", "```", *log, "```", "",
             "## 最終集成", "",
             "以下分數的決策規則仍是跨半調參，但集成成員、權重、分深度決策與規則修正都是依同一套分數挑選的，"
             "因此含選擇偏差、會略為樂觀；單一方法排行榜沒有這層選擇。", "",
             *[f"- {lab} Macro-F1：{final[t]['score']:.4f}"
               f"（分深度決策：{'是' if final[t]['depth'] else '否'}；規則修正：{'是' if final[t]['override'] else '否'}"
               + (f"；test 異常 match 改用不含比分推移的集成，其 CV {final[t]['score_plain']:.4f}"
                  if "score_plain" in final[t] else "") + "）"
               for t, lab in (("a", "action"), ("p", "point"))],
             f"- winner AUC：{final['w']['score']:.4f}"
             + (f"（test 異常 match 改用不含比分推移的集成，其 CV AUC {final['w']['score_plain']:.4f}）"
                if "score_plain" in final["w"] else ""),
             f"- **overall：{overall:.4f}**",
             *([f"- 巢狀估計（成員、權重、決策都只用另一半挑；無選擇偏差）：action {nested['a'][0]:.4f}、"
                f"point {nested['p'][0]:.4f}、winner {nested['w'][0]:.4f}、**overall {nested['overall']:.4f}**；"
                f"同樣方式只挑單一最佳方法：action {nested['a'][1]:.4f}、point {nested['p'][1]:.4f}、"
                f"winner {nested['w'][1]:.4f}"] if nested else []),
             *([f"- 預期 test 表現：test 有 {float(unreliable.mean()):.1%} 的列改用不含比分推移的集成，"
                f"在 CV 上隨機挑同比例的 match 模擬："
                + "、".join(f"{lab} 約 {final[t]['score_mixed'][0]:.4f}（±{final[t]['score_mixed'][1]:.4f}）"
                           for t, lab in (("a", "action"), ("p", "point"), ("w", "winner"))
                           if "score_mixed" in final[t])
                + f"，overall 約 **{overall_mixed:.4f}**"]
               if overall_mixed is not None else []), "",
             "配方：", "```json",
             json.dumps({**{t: [(n, round(w, 3)) for n, w in final[t]["recipe"]] for t in ("a", "p", "w")},
                         **{f"{t}_plain": [(n, round(w, 3)) for n, w in final[t]["recipe_plain"]]
                            for t in ("a", "p", "w") if "recipe_plain" in final[t]}},
                        ensure_ascii=False, indent=1), "```", "",
             f"提交檔：`{_display_path(sub_path)}`"]
    report = config.OUT_DIR / f"arena_report{suffix}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
