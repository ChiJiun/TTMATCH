"""快速評估：用 arena 同一套誠實評估，只對指定的 outputs/oof/<name>.npz 算分，不做集成。

用法：
  python eval_oof.py lgbm_note lgbm_mirror lgbm_bag3   # 比較幾個方法
  python eval_oof.py --all                            # 全部 npz
實驗時用它比較「改動前 / 改動後」，比跑整個 arena.py 快得多。
"""
from __future__ import annotations
import argparse

import pandas as pd

from arena import Evaluator, load_methods
from ttmatch import config
from ttmatch.data import build_datasets
from ttmatch.metrics import overall_score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--fast", action="store_true")
    args = ap.parse_args()

    D = build_datasets()
    methods = load_methods(config.OOF_DIR, D["ap_base"], D["test_pred"]["rally_uid"].to_numpy(), args.fast)
    names = list(methods) if args.all else args.names
    missing = [n for n in names if n not in methods]
    if missing:
        raise SystemExit(f"找不到：{', '.join(missing)}（有的：{', '.join(methods)}）")
    ev = Evaluator(D)
    rows = []
    for n in names:
        z, r = methods[n], {"method": n}
        if "oof_a" in z:
            r["action_argmax"], r["action_tuned"] = ev.f1_argmax(z["oof_a"], "a"), ev.f1(z["oof_a"], "a")
        if "oof_p" in z:
            r["point_argmax"], r["point_tuned"] = ev.f1_argmax(z["oof_p"], "p"), ev.f1(z["oof_p"], "p")
        if "oof_w" in z:
            r["winner_auc"] = ev.auc(z["oof_w"])
        if {"action_tuned", "point_tuned", "winner_auc"} <= r.keys():
            r["overall"] = overall_score(r["action_tuned"], r["point_tuned"], r["winner_auc"])
        rows.append(r)
        print(f"   {n} 完成", flush=True)
    print(pd.DataFrame(rows).set_index("method").round(4).to_markdown())


if __name__ == "__main__":
    main()
