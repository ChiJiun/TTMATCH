"""方法競技場：在同一套 CV 下比較 outputs/oof/*.npz 的所有方法，淘汰弱者，集成強者，產生提交檔。

評估方式（所有方法一致）：
1. OOF 來自依 match 分組的 5-fold（test 的 79 場比賽與 train 完全不重疊，所以必須按 match 切）。
2. 每個 rally 依 test 的截斷機制抽一個前綴（重複 5 次；見 ttmatch/truncation.py），讓 CV 的前綴長度
   分佈與「要預測的是不是 rally 最後一拍」的比例都跟 test 一樣。
3. Macro-F1 的決策規則（溫度 + 類別偏置）用「另一半比賽」調參、在這一半評估，兩半互換取平均，
   避免在同一批資料上調參又評分而高估。
4. 集成：每個任務各自做貪婪前向選擇（優勝劣汰）——從最佳單一方法開始，只有能讓誠實分數上升的
   方法才會被加入，其餘淘汰。
5. 終結拍更新：rally 最後一拍的擊球者一定輸掉這一分，所以 winner 集成的勝負機率可以用來修正
   action / point 集成裡「下一拍是最後一拍」的機率（terminal_update）；有進步才採用，巢狀估計也重做這一步。
6. 報告最後列出集成在 CV 抽樣與在 test 上「預測為各類別」的比例（prediction_shift）。兩邊差很多代表
   test 端的預測與 CV 不一致，這種問題 CV 分數看不出來（樹模型誤用 test 內部轉移就是這樣發現的）。

用法：
  python arena.py                                         # 全部方法 → outputs/submission.csv
  python arena.py --preset noprog --tag noprog            # 不用比分推移特徵 → outputs/submission_noprog.csv
  python arena.py --exclude lgbm_prog cat_prog --tag x    # 排除指定方法（可與 --preset 併用）
  python arena.py --fast                                  # 讀 *_fast.npz（冒煙測試）
  python arena.py --no-override                           # 不嘗試規則修正
  python arena.py --no-nested                             # 跳過巢狀估計
  python arena.py --no-terminal-update                    # 不做終結拍更新（對照）
  python arena.py --preset final                          # 改善回合 3 的方法集（FINAL_METHODS，含 _fa）
  python arena.py --preset nofa --tag nofa                # 同上但不含 _fa（README 最後採用）→ outputs/submission_nofa.csv
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
from ttmatch.data import build_datasets
from ttmatch.truncation import cut_matched_indices
from ttmatch.decision import (tune_decision, apply_decision,
                              tune_decision_by_depth, apply_decision_by_depth)
from ttmatch.metrics import macro_f1, winner_auc, overall_score
from ttmatch.postprocess import fit_table, apply_override

TASKS = {"a": ("next_action", config.ACTION_CLASSES), "p": ("next_point", config.POINT_CLASSES)}

# 改善回合 1 的方法集（PR #4 的 README 結果）。回合 2 的比較基準：這組 npz 在新的評估抽樣下重新計分。
R1_METHODS = [
    "cat", "cat_hist", "cat_prog", "cat_prog_bag3", "gru", "gru_hist", "gru_mirror", "gru_pemb", "gru_pemb_hist",
    "gru_t2v_film", "lgbm_hist", "lgbm_hist2", "lgbm_note", "lgbm_prog", "transformer", "transformer_hist",
    "transformer_hist2", "transformer_pemb", "xgb", "xgb_hist", "xgb_hist2", "xgb_prog",
]
# 改善回合 3 的方法集（PR #7 的 README 結果）。直接分類的樹模型都不用 test 內部轉移擴充；`_fa` 是終結拍分解的
# 樹模型，只在「非終結拍」的條件模型裡用 test 內部轉移（見 ttmatch/methods/trees.py）。
# winner 一律依截斷機制加權；action/point 兩種都放（預設加權，`_eq` 不加權），序列模型含 history 的加權版
# （`_al`）與不加權版都放，交給貪婪選擇與巢狀估計挑。CatBoost 的 action/point 太慢（單一方法約 1–1.5 小時）
# 而且前兩輪都沒被選進集成，這一輪沒有重訓，不在候選裡。
FINAL_METHODS = [
    # winner（比分推移）
    "cat_prog", "cat_prog_bag3", "lgbm_prog", "xgb_prog", "cat_hist_w", "lgbm_hist_w", "xgb_hist_w",
    # 三任務的樹模型（不含比分推移；winner 也供放不回去的打散 rally 使用）
    "xgb", "lgbm_note",
    # action / point：加權版
    "lgbm_hist", "lgbm_hist2", "xgb_hist2", "lgbm_hist_prog", "xgb_hist2_prog",
    # action / point：不加權的對照
    "xgb_eq", "lgbm_note_eq", "lgbm_hist_eq", "lgbm_hist2_eq", "xgb_hist2_eq",
    # action / point：終結拍分解 + test 內部轉移（改善回合 3）
    "xgb_fa", "lgbm_note_fa", "lgbm_hist_fa", "lgbm_hist2_fa", "xgb_hist2_fa", "lgbm_hist_prog_fa", "xgb_hist2_prog_fa",
    # 序列模型
    "gru_hist_al", "transformer_hist2_al", "transformer_hist2_prog_al", "gru_pemb_hist_al", "transformer_hist2",
    "gru", "transformer", "gru_pemb", "transformer_pemb", "gru_t2v_film", "gru_mirror",
]
# `_fs`：`_fa` 再加上驗證 match 的模擬內部轉移，讓 OOF 與 test 預測是同一種行為（train_factored 的 sim_aug）。
# `_fs` 的轉移列權重 ×5（與 `_fa` 相同），`_fs1` 是 ×1。
FS_METHODS = [
    "lgbm_hist_fs", "lgbm_hist_prog_fs", "xgb_hist2_fs", "xgb_hist2_prog_fs",
    "lgbm_hist_fs1", "lgbm_hist_prog_fs1", "xgb_hist2_fs1", "xgb_hist2_prog_fs1",
]
# 不含比分推移特徵的版本（課堂規則若不允許比分推移時使用）
PROG_METHODS = {"cat_prog", "cat_prog_bag3", "lgbm_prog", "xgb_prog", "cat_hist_w", "lgbm_hist_w", "xgb_hist_w",
                "lgbm_hist_prog", "xgb_hist2_prog", "transformer_hist2_prog_al",
                "lgbm_hist_prog_fa", "xgb_hist2_prog_fa",
                "lgbm_hist_prog_fs", "xgb_hist2_prog_fs", "lgbm_hist_prog_fs1", "xgb_hist2_prog_fs1"}
NOPROG_METHODS = [m for m in FINAL_METHODS if m not in PROG_METHODS]
# 拿掉終結拍分解樹模型（`_fa`）的版本，README 最後採用的是這一個：`_fa` 在 test 上比 CV 更有把握（見 CONF_WARN），
# 這部分 CV 量不到；而拿掉之後巢狀估計沒有變差。
FA_METHODS = {m for m in FINAL_METHODS if m.endswith("_fa")}
NOFA_METHODS = [m for m in FINAL_METHODS if m not in FA_METHODS]
# `_fs` 的版本（截止後分析）：`nofa` 再加 8 個 `_fs`，也就是回合 3 的方法集把 7 個 `_fa` 換成 `_fs`；
# `final_fs` 是 `final` 再加 `_fs`（`_fa` 也留著，確認貪婪選擇是不是還會選它們）；`noprog_fs` 拿掉比分推移的方法。
FS_PRESET = NOFA_METHODS + FS_METHODS
PRESETS = {"final": FINAL_METHODS, "nofa": NOFA_METHODS, "noprog": NOPROG_METHODS, "r1": R1_METHODS,
           "fs": FS_PRESET, "final_fs": FINAL_METHODS + FS_METHODS,
           "noprog_fs": [m for m in FS_PRESET if m not in PROG_METHODS]}
BLEND_GRID = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
MIN_GAIN = 5e-4
# 對照用的方法：樹模型直接併入 test 內部轉移，test 上會低估終結拍（見 ttmatch/methods/trees.py）。
# CV 分數看不出這個偏差，貪婪選擇可能把它們選進集成，所以沒指定 --preset 時預設排除。
CONTROL_METHODS = {"lgbm", "lgbm_hist_aug"}
TU_LAMBDAS = (0.5, 0.75, 1.0)   # 終結拍更新採用 winner 資訊的程度（見 terminal_update）
TU_KCAP = 8                     # 終結拍更新的先驗依前綴長度分桶，k ≥ 8 併成一桶


class Evaluator:
    def __init__(self, D, repeats=5):
        self.base = D["ap_base"]
        self.sel = cut_matched_indices(self.base, D["cut_g"], repeats)
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


def striker_loses(prefix_len, winner) -> np.ndarray:
    """要預測的那一拍（第 k+1 拍）的擊球者是否輸掉這一分。發球方打奇數拍；winner = 1 表示發球方得分。"""
    server_strikes = (np.asarray(prefix_len) + 1) % 2 == 1
    return np.where(server_strikes, np.asarray(winner) == 0, np.asarray(winner) == 1)


def fit_terminal_stats(base: pd.DataFrame, mask=None) -> dict:
    """終結拍更新要用的先驗，依截斷機制加權（mask 限定只用哪些列估計）。

    q[k]：前綴長度 k、下一拍不是終結拍時，下一拍的擊球者最後仍輸掉這一分的比例。
    pri_t / pri_n：終結拍 / 非終結拍的球種分佈。"""
    b = base if mask is None else base[mask]
    w = b["align_w"].to_numpy(dtype=float)
    term = b["is_terminal"].to_numpy().astype(bool)
    k = np.minimum(b["prefix_len"].to_numpy(), TU_KCAP)
    lose = striker_loses(b["prefix_len"].to_numpy(), b["winner"].to_numpy())
    q = np.full(TU_KCAP + 1, float(np.average(lose[~term], weights=w[~term])))
    for i in range(1, TU_KCAP + 1):
        m = (k == i) & ~term
        if w[m].sum() > 0:
            q[i] = float(np.average(lose[m], weights=w[m]))
    ya, C = b["next_action"].to_numpy().astype(int), len(TASKS["a"][1])
    pri_t = np.bincount(ya[term], weights=w[term], minlength=C) + 1e-9
    pri_n = np.bincount(ya[~term], weights=w[~term], minlength=C) + 1e-9
    return dict(q=q, pri_t=pri_t / pri_t.sum(), pri_n=pri_n / pri_n.sum())


def terminal_update(p, a, pw, prefix_len, stats: dict, lam: float):
    """用 winner 機率更新「下一拍是終結拍」的機率，再重組落點與球種機率。回傳 (action 機率, point 機率)。

    終結拍的擊球者一定輸掉這一分，所以 P(終結 | 擊球者輸) = p0 / P(擊球者輸)，其中 p0 是落點模型給的
    P(落點 0)，模型隱含的 P(擊球者輸) = p0 + (1 − p0)·q[k]。winner 集成（含比分推移）另外給了一個更準的
    P(擊球者輸)，兩者的比值就是修正倍率：
        P(終結) ← p0 · (winner 集成的 P(擊球者輸) / 模型隱含的 P(擊球者輸)) ^ lam
    lam = 1 是完全採用 winner 集成的資訊，lam < 1 是打折（兩邊的資訊有重疊時）。
    落點：類別 0 換成更新後的 P(終結)，其餘類別依原比例分配剩下的機率。
    球種：依終結 / 非終結拍的球種先驗比調整，P(終結) 沒變的列不動。"""
    k = np.asarray(prefix_len)
    pw = np.clip(np.asarray(pw, dtype=float), 1e-4, 1 - 1e-4)
    lose = np.where((k + 1) % 2 == 1, 1 - pw, pw)
    p0 = np.clip(p[:, 0].astype(float), 1e-6, 1 - 1e-6)
    prior = p0 + (1 - p0) * stats["q"][np.minimum(k, TU_KCAP)]
    # prior ≥ p0、lose ≤ 1，所以 λ ≤ 1 時更新後的機率不超過 p0^(1−λ) ≤ 1；clip 只是保險，不會真的截到
    post = np.clip(p0 * (lose / prior) ** lam, 0.0, 1.0)
    rest = p[:, 1:] / np.clip(p[:, 1:].sum(1, keepdims=True), 1e-12, None)
    p_new = np.concatenate([post[:, None], (1 - post)[:, None] * rest], axis=1)
    t, n = stats["pri_t"][None, :], stats["pri_n"][None, :]
    a_new = a * (post[:, None] * t + (1 - post)[:, None] * n) / (p0[:, None] * t + (1 - p0)[:, None] * n)
    return a_new / a_new.sum(1, keepdims=True), p_new


def select_terminal_update(ev: Evaluator, oof: dict, test: dict, k_oof, k_test, stats: dict, log: list) -> dict:
    """在「原集成」與各種終結拍更新版之間做貪婪選擇。回傳 {任務: (oof, test, 配方)}；沒有進步的任務不回傳。

    oof / test 的鍵：a、p、w（主集成），以及有的話 a_plain、p_plain（不含比分推移的集成）。
    更新版有兩種底：不含比分推移的集成（勝負資訊只從 winner 進來，不重複計算）與主集成。"""
    cand_oof = {t: {"blend": oof[t]} for t in ("a", "p")}
    cand_test = {t: {"blend": test[t]} for t in ("a", "p")}
    bases = [("main", "a", "p")]
    if "a_plain" in oof or "p_plain" in oof:
        bases.insert(0, ("plain", "a_plain" if "a_plain" in oof else "a", "p_plain" if "p_plain" in oof else "p"))
    for tag, ka, kp in bases:
        for lam in TU_LAMBDAS:
            name = f"tu_{tag}_{lam:g}"
            a_o, p_o = terminal_update(oof[kp], oof[ka], oof["w"], k_oof, stats, lam)
            a_t, p_t = terminal_update(test[kp], test[ka], test["w"], k_test, stats, lam)
            cand_oof["a"][name], cand_oof["p"][name] = a_o, p_o
            cand_test["a"][name], cand_test["p"][name] = a_t, p_t
    out = {}
    for t, label in (("a", "action"), ("p", "point")):
        fn = lambda p, t=t: ev.f1(p, t, quick=True)
        sub_log = []
        b_oof, b_test, recipe, single = greedy_blend(list(cand_oof[t]), cand_oof[t], cand_test[t], fn, sub_log)
        gain = fn(b_oof) - single["blend"]
        log.append(f"[{label}] 終結拍更新（候選：原集成 blend、tu_<底>_<lam>；快速調參版分數）")
        log.extend(sub_log)
        log.append(f"  原集成 {single['blend']:.4f} → {single['blend'] + gain:.4f}"
                   + ("（採用）" if gain >= MIN_GAIN else "（進步不足，不採用）"))
        if gain >= MIN_GAIN:
            out[t] = (b_oof, b_test, recipe)
    return out


def nested_estimate(ev: Evaluator, methods: dict, base: pd.DataFrame, terminal: bool = True) -> dict:
    """巢狀估計：集成成員、權重、終結拍更新與決策規則全部只用「另一半比賽」挑選 / 調參，再在這一半評分，
    兩半互換。回傳 {任務: (集成的巢狀分數, 另一半挑出的最佳單一方法在這一半的分數, 不做終結拍更新的巢狀分數)}。

    主流程的集成分數是用全部 OOF 挑成員，所以略為樂觀；這個數字沒有那層選擇偏差。

    涵蓋範圍：集成成員與權重、終結拍更新（先驗、要不要用、用哪一種）、決策規則。主流程的其他挑選——
    分深度決策、規則修正、異常 match 的 plain 集成混合——不在這裡重做；目前前兩者都沒有被採用，plain 混合
    另由 mixed_score 估計。此外，若用這個數字在多個方法集（版本）之間挑選，被選中的版本仍會有一點選擇偏差。
    """
    k = base["prefix_len"].to_numpy()
    res = {t: ([], [], []) for t in ("a", "p", "w")}
    for h in (0, 1):
        mask = ev.half != h
        ev_s = ev.restricted(mask)
        bl = {}
        for t in ("a", "p", "w"):
            names = [n for n, z in methods.items() if f"oof_{t}" in z]
            oof = {n: methods[n][f"oof_{t}"] for n in names}
            fn = ev_s.auc if t == "w" else (lambda p, t=t: ev_s.f1(p, t, quick=True))
            bl[t], _, _, single = greedy_blend(names, oof, oof, fn, log=[])
            score = (lambda p: ev.auc(p, halves=(h,))) if t == "w" else (lambda p, t=t: ev.f1(p, t, halves=(h,)))
            res[t][1].append(score(oof[max(single, key=single.get)]))
            res[t][2].append(score(bl[t]))
            plain = [n for n in names if not uses_prog(n, methods[n])]
            if t != "w" and plain and len(plain) < len(names):
                bl[f"{t}_plain"], _, _, _ = greedy_blend(plain, {n: oof[n] for n in plain},
                                                         {n: oof[n] for n in plain}, fn, log=[])
        upd = (select_terminal_update(ev_s, bl, bl, k, k, fit_terminal_stats(base, mask), log=[])
               if terminal else {})
        res["w"][0].append(res["w"][2][-1])
        for t in ("a", "p"):
            res[t][0].append(ev.f1(upd[t][0], t, halves=(h,)) if t in upd else res[t][2][-1])
    return {t: tuple(float(np.mean(v)) for v in res[t]) for t in res}


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


def prediction_shift(pred_cv, pred_test, classes):
    """「預測為各類別」的比例在 CV 抽樣與 test 上各是多少，以及兩者的總變異距離（0 = 相同，1 = 完全不同）。

    CV 的評估樣本已對齊 test 的截斷機制，所以兩邊的預測分佈應該接近；差很多代表 test 端的預測與 CV 不一致
    （例如訓練時混入了只有 test 才有的列），這種問題 CV 分數看不出來。"""
    pred_cv, pred_test = np.asarray(pred_cv), np.asarray(pred_test)
    cv = np.array([(pred_cv == c).mean() for c in classes])
    te = np.array([(pred_test == c).mean() for c in classes])
    return cv, te, 0.5 * float(np.abs(cv - te).sum())


# 預測分佈的總變異距離超過這個值就在報告裡警告。test 的選手組成與 train 不同，正常情況約 0.07–0.10；
# 樹模型誤用 test 內部轉移時 point 是 0.32–0.40。
SHIFT_WARN = 0.15
# 終結拍另外單獨檢查：「預測為落點 0」的比例在 CV 與 test 的差超過這個值就警告。正常約 0.01–0.03；
# 樹模型誤用 test 內部轉移時差 0.3 以上，但只混入一部分這種方法時總變異距離可能還在 SHIFT_WARN 以下。
P0_WARN = 0.05
# 「模型有多大把握」也單獨檢查（見 confidence）：test 減 CV 的差超過這個值就在報告裡標出來。
# 不用 test 內部轉移的樹模型與序列模型，落點約 ±0.01、球種約 −0.02～−0.005；
# 用了 test 內部轉移的樹模型（_fa）落點是 +0.02～+0.10：test 的每一列都有同場的內部轉移可學，CV 的驗證 fold 沒有。
# 這個值是看過上面的數字才定的，離不用 test 內部轉移的樹模型（−0.022）不遠，只當提醒用。
CONF_WARN = 0.03


def confidence(p, drop0: bool = False) -> float:
    """機率分佈的平均最大機率（模型有多大把握）。drop0=True 時先拿掉類別 0 再重新歸一，
    也就是只看「不是終結拍時」的條件分佈；這樣量到的差異與終結拍比例無關。

    prediction_shift 比的是決策之後的類別比例，看不到這種差異：一個方法可以在 test 上比 CV 更有把握
    （它在 test 上學到了 CV 驗證 fold 沒有的資訊），而預測的類別比例幾乎不變。"""
    p = np.asarray(p, dtype=float)
    if drop0:
        p = p[:, 1:] / np.clip(p[:, 1:].sum(1, keepdims=True), 1e-12, None)
    return float(p.max(1).mean())


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
    ap.add_argument("--preset", choices=sorted(PRESETS), help="只用預先定義的方法集（" + " / ".join(sorted(PRESETS)) + "）")
    ap.add_argument("--no-relocate", action="store_true",
                    help="不把 test 打散的 rally 放回原比賽（對照用；OOF 也必須是在同樣設定下訓練的）")
    ap.add_argument("--no-terminal-update", action="store_true",
                    help="不用 winner 集成更新 action / point 的終結拍機率（對照用）")
    ap.add_argument("--include-controls", action="store_true",
                    help=f"沒指定 --preset 時也納入對照用的方法（{', '.join(sorted(CONTROL_METHODS))}）")
    args = ap.parse_args()

    D = build_datasets(relocate=not args.no_relocate)
    base, test_df = D["ap_base"], D["test_pred"]
    methods = load_methods(config.OOF_DIR, base, test_df["rally_uid"].to_numpy(), args.fast)
    if not methods:
        raise SystemExit("outputs/oof/ 沒有任何 npz，請先執行 run_method.py")
    unknown = sorted(set(args.exclude) - methods.keys())
    if unknown:
        print(f"警告：--exclude 中的 {', '.join(unknown)} 不在 outputs/oof/ 裡，已忽略", flush=True)
    if args.preset:
        missing = sorted(set(PRESETS[args.preset]) - methods.keys())
        if missing:
            raise SystemExit(f"--preset {args.preset} 需要的方法還沒訓練：{', '.join(missing)}（請先執行 run_method.py）")
        methods = {n: methods[n] for n in PRESETS[args.preset]}
    methods = {n: z for n, z in methods.items() if n not in set(args.exclude)}   # 可與 --preset 併用
    controls = sorted(CONTROL_METHODS & methods.keys())
    if controls and not args.preset and not args.include_controls:
        print(f"略過對照用的方法（test 上會低估終結拍，CV 看不出來）：{', '.join(controls)}；"
              "要納入請加 --include-controls", flush=True)
        methods = {n: z for n, z in methods.items() if n not in CONTROL_METHODS}
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
        final[task] = dict(oof=b_oof, test=b_test, recipe=recipe)
        # 有些 action/point 方法也用比分推移特徵；不含它的集成有兩個用途：
        # test 異常 match 的列改用它（同 winner），以及當作終結拍更新的底
        plain = [n for n in names if not uses_prog(n, methods[n])]
        if plain and len(plain) < len(names):
            log.append(f"[{label}] 不含比分推移的集成（用於 test 異常 match 的 {int(unreliable.sum())} 列，"
                       "以及終結拍更新的底）")
            p_oof, p_test, p_recipe, _ = greedy_blend(
                plain, {n: oof[n] for n in plain}, {n: test[n] for n in plain},
                lambda p, t=task: ev.f1(p, t, quick=True), log)
            final[task].update(oof_plain=p_oof, test_plain=p_test, recipe_plain=p_recipe,
                               score_plain=ev.f1(p_oof, task), frac_plain=float(unreliable.mean()))

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
        # CV 的分數假設每一列都有比分推移；test 有一部分列只能用 plain 集成，而 AUC 是整份一起算
        frac = float(unreliable.mean())
        final["w"]["frac_plain"] = frac
        final["w"]["score_mixed"] = mixed_auc(ev, base["match"].to_numpy(), b_oof, p_oof, frac)

    # --- 2a. 終結拍更新：用 winner 集成的勝負機率修正 action / point 集成 ---
    if not args.no_terminal_update:
        oofs = {t: final[t]["oof"] for t in ("a", "p", "w")}
        tests = {t: final[t]["test"] for t in ("a", "p", "w")}
        for t in ("a", "p"):
            if "oof_plain" in final[t]:
                oofs[f"{t}_plain"], tests[f"{t}_plain"] = final[t]["oof_plain"], final[t]["test_plain"]
        upd = select_terminal_update(ev, oofs, tests, base["prefix_len"].to_numpy(),
                                     test_df["prefix_len"].to_numpy(), fit_terminal_stats(base), log)
        for t, (o, te, rec) in upd.items():
            final[t].update(oof=o, test=te, recipe_tu=rec)

    # --- 2b. action / point 的決策規則、規則修正與 test 異常列的折減 ---
    for task, label in (("a", "action"), ("p", "point")):
        b_oof = final[task]["oof"]
        score = ev.f1(b_oof, task)
        s_depth = ev.f1(b_oof, task, depth=True)
        log.append(f"[{label}] 決策規則：全域 {score:.4f} / 分深度 {s_depth:.4f}")
        use_depth = s_depth > score + MIN_GAIN
        score = max(score, s_depth) if use_depth else score
        use_override = False
        if not args.no_override:
            tables = ev.override_tables(task)
            s_ov = ev.f1(b_oof, task, override_tables=tables, depth=use_depth)
            log.append(f"  規則修正：{score:.4f} → {s_ov:.4f}")
            if s_ov > score + MIN_GAIN:
                score, use_override = s_ov, True
        final[task].update(score=score, override=use_override, depth=use_depth)
        if unreliable.any() and "oof_plain" in final[task]:
            mixed = mixed_score(lambda p, t=task: ev.f1(p, t, quick=True), base["match"].to_numpy(), b_oof,
                                final[task]["oof_plain"], float(unreliable.mean()), repeats=10)
            # mixed 用快速調參估計；換算成與 score 同尺度：score − (快速版主集成 − 快速版混合)
            q_main = ev.f1(b_oof, task, quick=True)
            final[task]["score_mixed"] = (score - (q_main - mixed[0]), mixed[1])

    overall = overall_score(final["a"]["score"], final["p"]["score"], final["w"]["score"])

    # --- 2c. 巢狀估計：集成、終結拍更新與決策都在另一半選，在這一半評 ---
    nested = {}
    if not args.no_nested:
        nested = nested_estimate(ev, methods, base, terminal=not args.no_terminal_update)
        nested["overall"] = overall_score(nested["a"][0], nested["p"][0], nested["w"][0])
        nested["overall_no_tu"] = overall_score(nested["a"][2], nested["p"][2], nested["w"][2])
        print("   巢狀估計完成", flush=True)
    has_mixed = any("score_mixed" in final[t] for t in ("a", "p", "w"))
    overall_mixed = (overall_score(*[final[t].get("score_mixed", (final[t]["score"],))[0] for t in ("a", "p", "w")])
                     if has_mixed else None)
    # 巢狀估計再扣掉「異常列改用 plain 集成」的折減（折減量取自 in-sample 的 主集成 − 混合）
    nested_mixed = (overall_score(*[nested[t][0] - (final[t]["score"] - final[t].get("score_mixed", (final[t]["score"],))[0])
                                    for t in ("a", "p", "w")]) if (nested and has_mixed) else None)

    # --- 3. 產生提交檔：決策規則用全部長度對齊樣本調參 ---
    all_sel = np.concatenate(ev.sel)
    sub = pd.DataFrame({"rally_uid": test_df["rally_uid"].to_numpy()})
    shift = {}
    for task, col in (("a", "actionId"), ("p", "pointId")):
        classes = TASKS[task][1]
        y = ev.y[task]
        if final[task]["depth"]:
            plen = base["prefix_len"].to_numpy()
            decs = tune_decision_by_depth(final[task]["oof"][all_sel], y[all_sel], classes, plen[all_sel])
            pred = apply_decision_by_depth(final[task]["test"], classes, decs,
                                           test_df["prefix_len"].to_numpy())
            pred_cv = apply_decision_by_depth(final[task]["oof"][all_sel], classes, decs, plen[all_sel])
        else:
            dec, _ = tune_decision(final[task]["oof"][all_sel], y[all_sel], classes)
            pred = apply_decision(final[task]["test"], classes, dec)
            pred_cv = apply_decision(final[task]["oof"][all_sel], classes, dec)
        shift[task] = prediction_shift(pred_cv, pred[~unreliable] if unreliable.any() else pred, classes)
        if unreliable.any() and "test_plain" in final[task]:
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
    # 最終集成的機率（tools/ 的分析工具直接讀這個檔，不必照配方重建）
    np.savez_compressed(config.OUT_DIR / f"blend{suffix}.npz",
                        **{f"oof_{t}": final[t]["oof"].astype(np.float32) for t in ("a", "p", "w")},
                        **{f"te_{t}": np.asarray(final[t]["test"], dtype=np.float32) for t in ("a", "p", "w")})
    p0_cv, p0_te = float(shift["p"][0][0]), float(shift["p"][1][0])
    # 最終集成與它的成員在 CV 與 test 上「有多大把握」（落點只看非終結拍的條件分佈；test 只算有比分推移的列）
    reliable = ~unreliable
    members = list(dict.fromkeys(n for t in ("a", "p") for k in ("recipe", "recipe_plain")
                                 for n, _ in final[t].get(k, [])))
    conf_rows = [("最終集成", *[(confidence(final[t]["oof"][all_sel], t == "p"),
                               confidence(final[t]["test"][reliable], t == "p")) for t in ("a", "p")])]
    for n in members:
        conf_rows.append((f"`{n}`", *[(confidence(methods[n][f"oof_{t}"][all_sel], t == "p"),
                                       confidence(methods[n][f"te_{t}"][reliable], t == "p"))
                                      if f"oof_{t}" in methods[n] else None for t in ("a", "p")]))
    conf_flagged = [name for name, *cells in conf_rows[1:]
                    if any(c is not None and abs(c[1] - c[0]) > CONF_WARN for c in cells)]

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
                f"winner {nested['w'][1]:.4f}；不做終結拍更新：action {nested['a'][2]:.4f}、"
                f"point {nested['p'][2]:.4f}、overall {nested['overall_no_tu']:.4f}"] if nested else []),
             *([f"- 預期 test 表現：test 有 {float(unreliable.mean()):.1%} 的列改用不含比分推移的集成，"
                f"在 CV 上隨機挑同比例的 match 模擬："
                + "、".join(f"{lab} 約 {final[t]['score_mixed'][0]:.4f}（±{final[t]['score_mixed'][1]:.4f}）"
                           for t, lab in (("a", "action"), ("p", "point"), ("w", "winner"))
                           if "score_mixed" in final[t])
                + f"，overall 約 **{overall_mixed:.4f}**"
                + (f"；同樣的折減套在巢狀估計上：overall 約 **{nested_mixed:.4f}**" if nested_mixed is not None else "")]
               if overall_mixed is not None else []), "",
             "## CV 與 test 的預測分佈", "",
             "同一組決策規則下「預測為各類別」的比例。CV 的評估樣本已對齊 test 的截斷機制，兩邊應該接近；"
             "差很多代表 test 端的預測與 CV 不一致，CV 分數不可信。", "",
             "| 任務 | | " + " | ".join(str(c) for c in range(max(len(TASKS[t][1]) for t in shift))) + " | 總變異距離 |",
             "|---|---|" + "---:|" * (max(len(TASKS[t][1]) for t in shift) + 1),
             *[f"| {lab} | {where} | " + " | ".join(
                 [f"{v:.3f}" for v in vals] + [""] * (max(len(TASKS[x][1]) for x in shift) - len(vals)))
               + f" | {shift[t][2]:.3f} |"
               for t, lab in (("a", "action"), ("p", "point"))
               for where, vals in (("CV", shift[t][0]), ("test", shift[t][1]))],
             "",
             f"- winner 平均機率：CV {float(final['w']['oof'][all_sel].mean()):.3f}、"
             f"test {float(np.mean(final['w']['test'])):.3f}",
             f"- 預測為落點 0（終結拍）的比例：CV {p0_cv:.3f}、test {p0_te:.3f}",
             *[f"- ⚠️ {lab} 的預測分佈在 CV 與 test 差很多（總變異距離 {shift[t][2]:.3f} > {SHIFT_WARN}）"
               for t, lab in (("a", "action"), ("p", "point")) if shift[t][2] > SHIFT_WARN],
             *([f"- ⚠️ 預測為落點 0 的比例在 CV 與 test 差了 {abs(p0_cv - p0_te):.3f}（> {P0_WARN}）："
                "test 端可能低估或高估終結拍"] if abs(p0_cv - p0_te) > P0_WARN else []),
             "",
             "機率分佈的平均最大機率（模型有多大把握；落點只看「不是終結拍時」的條件分佈）。"
             "上面的類別比例看不到這種差異。", "",
             "| | action：CV | test | 差 | point（非終結）：CV | test | 差 |",
             "|---|---:|---:|---:|---:|---:|---:|",
             *["| " + name + " | " + " | ".join(
                 f"{c[0]:.3f} | {c[1]:.3f} | {c[1] - c[0]:+.3f}" if c is not None else "— | — | —" for c in cells) + " |"
               for name, *cells in conf_rows],
             "",
             *([f"- ⚠️ {', '.join(conf_flagged)} 在 test 上比 CV 更有把握（差 > {CONF_WARN}）：它在 test 上用到了 "
                "CV 驗證 fold 沒有的資訊（例如同場的 test 內部轉移），CV 分數量不到這部分在 test 上的好處或壞處"]
               if conf_flagged else []),
             "",
             "配方：", "```json",
             json.dumps({**{t: [(n, round(w, 3)) for n, w in final[t]["recipe"]] for t in ("a", "p", "w")},
                         **{f"{t}_final": [(n, round(w, 3)) for n, w in final[t]["recipe_tu"]]
                            for t in ("a", "p") if "recipe_tu" in final[t]},
                         **{f"{t}_plain": [(n, round(w, 3)) for n, w in final[t]["recipe_plain"]]
                            for t in ("a", "p", "w") if "recipe_plain" in final[t]}},
                        ensure_ascii=False, indent=1), "```", "",
             f"提交檔：`{_display_path(sub_path)}`"]
    report = config.OUT_DIR / f"arena_report{suffix}.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
