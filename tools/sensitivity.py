"""敏感度分析：如果 test 的截斷機制不是 F0，最終集成的分數會差多少？

ttmatch/truncation.py 的 F0 機制只有 k = 1 的終結率被 test 的比分資料直接支持；k ≥ 2 是假設。
這支工具讀 arena 存下來的最終集成 OOF 機率（blend.npz；舊的輸出資料夾則依報告裡的配方重建），
決策規則照正式流程在 F0 抽樣上調參，
再換成不同假設下的評估樣本計分：

  F0   ：正式評估用的機制
  M1   ：k = 1 照 F0；k ≥ 2 的 rally 長度分佈照 train 的所有前綴等權（終結率約 0.2–0.29）
  等權 ：所有 k 都照 train 的所有前綴等權（上一輪訓練時的隱含假設；k = 1 的終結率只有 0.125）

三種假設的前綴長度分佈都等於 test 的。

另外一個與截斷無關的情境：
  F0＋比分推移組成照 test：train 端的比分推移是模擬的（每個 rally 獨立抽間隔），「比分能直接確定勝負」的
        rally 比 test 多（test 把一部分這種 rally 之後的 rally 打散了，見 ttmatch/relocate.py）。這個情境把
        評估列依（有沒有下一個可見 rally、間隔、能否確定勝負）重新加權成 test 的組成。

用法：python tools/sensitivity.py [輸出資料夾 ...]     （預設 outputs/；每個資料夾要有 arena_report.md 與 oof/）
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
from ttmatch.metrics import macro_f1, winner_auc, overall_score  # noqa: E402


def load_blends(out_dir: Path) -> dict:
    """最終集成的 action / point / winner OOF 機率。

    arena.py 會把最終集成（含終結拍更新）存成 blend.npz，有就直接讀；較舊的輸出資料夾沒有這個檔，
    改依報告的配方（主集成）重建。"""
    if (out_dir / "blend.npz").exists():
        z = np.load(out_dir / "blend.npz")
        return {t: z[f"oof_{t}"].astype(float) for t in ("a", "p", "w")}
    text = (out_dir / "arena_report.md").read_text(encoding="utf-8")
    recipe = json.loads(re.search(r"```json\n(.*?)\n```", text, re.S).group(1))
    out = {}
    for t in ("a", "p", "w"):
        out[t] = sum(w * np.load(out_dir / "oof" / f"{n}.npz")[f"oof_{t}"].astype(float) for n, w in recipe[t])
    return out


def prog_cell(df) -> np.ndarray:
    """比分推移的組成：0 = 沒有下一個可見 rally；否則 10 × min(間隔, 5) + (比分能否直接確定勝負)。"""
    has = df["prog_has_next"].to_numpy() == 1
    gap = np.nan_to_num(df["prog_gap"].to_numpy(dtype=float), nan=0.0)
    det = (df["prog_srv_gain"].to_numpy() == 0) | (df["prog_rcv_gain"].to_numpy() == 0)
    return np.where(has, 10 * np.minimum(gap, 5).astype(int) + det.astype(int), 0)


def scenario_weights(base, g, test_pred=None) -> dict:
    """各假設下每個訓練列的抽樣權重（總和為 1，前綴長度邊際都等於 F0 的）。"""
    k = np.clip(base["prefix_len"].to_numpy(), 1, len(g) - 1)
    f0 = base["align_w"].to_numpy(dtype=float)
    f0 = f0 / f0.sum()
    mass = np.bincount(k, weights=f0, minlength=len(g))        # 各 k 的質量（= test 的前綴長度分佈）
    equal = mass[k] / np.bincount(k, minlength=len(g))[k]
    out = {"F0": f0, "M1": np.where(k == 1, f0, equal), "等權": equal}
    if test_pred is not None:
        # 只看有比分推移資訊的 test 列（放不回去的打散 rally 另由 arena 的 mixed 估計處理）
        cell_tr, cell_te = prog_cell(base), prog_cell(test_pred[test_pred["prog_unreliable"] == 0])
        share_tr = np.bincount(cell_tr, weights=f0, minlength=60)
        share_te = np.bincount(cell_te, minlength=60) / len(cell_te)
        ratio = np.divide(share_te, share_tr, out=np.zeros(60), where=share_tr > 0)
        w = f0 * ratio[cell_tr]
        out["F0＋比分推移組成照 test"] = w / w.sum()
    return out


def evaluate(ev: Evaluator, blends: dict, w: np.ndarray, repeats: int = 5, seed: int = config.SEED) -> dict:
    """決策規則在另一半的 F0 樣本上調參；評估列依 w 在這一半內有放回抽樣。"""
    base, res = ev.base, {}
    n_draw = base["rally_uid"].nunique() // 2
    draws = []
    for r in range(repeats):
        rng = np.random.default_rng(seed + r)
        draws.append([rng.choice(np.where(ev.half == h)[0], size=n_draw,
                                 p=w[ev.half == h] / w[ev.half == h].sum()) for h in (0, 1)])
    for t in ("a", "p"):
        classes, y, scores = TASKS[t][1], ev.y[t], []
        for h in (0, 1):
            tune, _ = ev._split(h)
            dec, _ = tune_decision(blends[t][tune], y[tune], classes)
            scores += [macro_f1(y[d[h]], apply_decision(blends[t][d[h]], classes, dec)) for d in draws]
        res[t] = float(np.mean(scores))
    res["w"] = float(np.mean([winner_auc(ev.y["w"][np.concatenate(d)], blends["w"][np.concatenate(d)])
                              for d in draws]))
    res["overall"] = overall_score(res["a"], res["p"], res["w"])
    res["terminal"] = float(np.average(base["is_terminal"], weights=w))
    if "prog_has_next" in base:
        res["det"] = float(np.average(prog_cell(base) % 10 == 1, weights=w))
    return res


def main():
    dirs = [Path(a) for a in sys.argv[1:]] or [config.OUT_DIR]
    D = build_datasets()
    ev = Evaluator(D)
    W = scenario_weights(D["ap_base"], D["cut_g"], D["test_pred"])
    tp = D["test_pred"]
    print(f"test 中比分能直接確定勝負的列：{float((prog_cell(tp[tp['prog_unreliable'] == 0]) % 10 == 1).mean()):.3f}"
          "（只算有比分推移資訊的列）")
    print()
    print("| 集成 | 假設 | 終結率 | 能確定勝負的列 | action | point | winner | overall |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for d in dirs:
        blends = load_blends(d)
        for name, w in W.items():
            r = evaluate(ev, blends, w)
            print(f"| {d.name} | {name} | {r['terminal']:.3f} | {r.get('det', float('nan')):.3f} "
                  f"| {r['a']:.4f} | {r['p']:.4f} | {r['w']:.4f} "
                  f"| {r['overall']:.4f} |", flush=True)


if __name__ == "__main__":
    main()
