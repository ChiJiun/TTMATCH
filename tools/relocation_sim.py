"""在 train 上模擬 test 的「公開 / 打散」樣式，量 ttmatch/relocate.py 放回的數量與正確率。

test 沒有打散 rally 的真實位置可以對答案，所以用 train（每局的 rally 都完整）重現 test 的樣式：
1. 每局依 test 的間隔分佈挑出「被選進 test 的 rally」（test 相鄰 rally 的 rally_id 差只有 2、3、4）。
2. 其中符合 test 觀察到的打散條件的 rally——前一個被選的 rally 是公開的而且在前 2 分，下一個被選的
   rally 在後 2 分且這 2 分雙方各得 1 分（或它是該局最後一個被選的）——以 test 估出來的比例打散：
   match / numberGame / rally_id 換成假的，只留選手、性別與比分。
3. 用 relocate_heads 放回，與真實位置比對。

每次重複抽與 test 相同數量的 match。條件 7 是依 test 的觀察設計的，而這個模擬也照同樣的觀察產生打散
rally，所以它量的是「樣式如同觀察到的那樣時，候選唯一是否就代表位置正確」，不能證明樣式本身。

用法：python tools/relocation_sim.py [重複次數，預設 20]
"""
from __future__ import annotations
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttmatch import config                                                             # noqa: E402
from ttmatch.data import load_raw                                                      # noqa: E402
from ttmatch.relocate import (rally_heads, scrambled_matches, relocate_heads, relocate_scrambled,  # noqa: E402
                              apply_relocation, REQUIRED_DPREV, REQUIRED_DNEXT)

FAKE_MATCH = 10 ** 6      # 假 match 編號的起點


def _score(r) -> dict:
    return {int(r.server): int(r.srv_score), int(r.receiver): int(r.rcv_score)}


def _eligible(prev, cur, nxt) -> bool:
    """cur 是否符合 test 觀察到的打散條件（prev / nxt 是同一局前後被選的 rally，沒有則為 None）。"""
    if prev is None or cur.rally_id - prev.rally_id != REQUIRED_DPREV:
        return False
    if nxt is None:
        return True
    if nxt.rally_id - cur.rally_id != REQUIRED_DNEXT:
        return False
    s, o = _score(cur), _score(nxt)
    return all(o[p] - s[p] == 1 for p in s)


def test_pattern(te: pd.DataFrame) -> dict:
    """從 test（打散的 rally 放回後）估計：每局第一個 rally_id、相鄰間隔、符合條件的 rally 被打散的比例。"""
    placed = relocate_scrambled(te)
    heads = rally_heads(apply_relocation(te, placed))
    bad = scrambled_matches(rally_heads(te))
    left = int((heads["match"].isin(bad)).sum())
    heads = heads[~heads["match"].isin(bad)].sort_values(["match", "numberGame", "rally_id"])
    is_scr = heads["rally_uid"].isin(set(placed["rally_uid"])).to_numpy()
    first, steps, n_elig_visible = Counter(), Counter(), 0
    rows = list(heads.itertuples())
    for i, r in enumerate(rows):
        same = lambda j: 0 <= j < len(rows) and (rows[j].match, rows[j].numberGame) == (r.match, r.numberGame)
        prev, nxt = (rows[i - 1] if same(i - 1) else None), (rows[i + 1] if same(i + 1) else None)
        if prev is None:
            first[int(r.rally_id)] += 1
        else:
            steps[int(r.rally_id - prev.rally_id)] += 1
        if not is_scr[i] and prev is not None and not is_scr[i - 1] and _eligible(prev, r, nxt):
            n_elig_visible += 1
    n_scr = len(placed) + left               # 放不回去的也算打散（它們多半同樣符合條件）
    return dict(first=first, steps=steps, rate=n_scr / (n_scr + n_elig_visible), n_matches=heads["match"].nunique())


def simulate(train_heads: pd.DataFrame, pat: dict, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """回傳 (模擬的 test rally 表, {打散 rally 的 rally_uid: 真實 (match, game, rally_id)})。"""
    f_ids, f_p = np.array(list(pat["first"])), np.array(list(pat["first"].values()), dtype=float)
    s_ids, s_p = np.array(list(pat["steps"])), np.array(list(pat["steps"].values()), dtype=float)
    matches = rng.choice(train_heads["match"].unique(), size=pat["n_matches"], replace=False)
    sub = train_heads[train_heads["match"].isin(matches)]
    keep, truth = [], {}
    for _, d in sub.groupby(["match", "numberGame"], sort=True):
        d = d.sort_values("rally_id")
        by_id = {int(r.rally_id): r for r in d.itertuples()}
        last, rid, chosen = max(by_id), int(rng.choice(f_ids, p=f_p / f_p.sum())), []
        while rid <= last:
            if rid in by_id:
                chosen.append(by_id[rid])
            rid += int(rng.choice(s_ids, p=s_p / s_p.sum()))
        prev_visible = None
        for i, r in enumerate(chosen):
            prev = chosen[i - 1] if i else None
            nxt = chosen[i + 1] if i + 1 < len(chosen) else None
            scr = (prev is not None and prev is prev_visible and _eligible(prev, r, nxt)
                   and rng.random() < pat["rate"])
            if scr:
                truth[int(r.rally_uid)] = (r.match, r.numberGame, int(r.rally_id))
            else:
                prev_visible = r
            keep.append((r, scr))
    out = pd.DataFrame([r._asdict() for r, _ in keep]).drop(columns="Index")
    scr = np.array([s for _, s in keep])
    # 打散：假的 match（每個約 17 個 rally）、局數與 rally_id
    n = int(scr.sum())
    order = rng.permutation(n)
    out.loc[scr, "match"] = FAKE_MATCH + order // 17
    out.loc[scr, "numberGame"] = 1
    out.loc[scr, "rally_id"] = 100 + order % 17
    return out, truth


def main(repeats: int = 20, seed: int = config.SEED):
    tr, te, _ = load_raw()
    pat = test_pattern(te)
    heads = rally_heads(tr)
    heads = heads[heads["rally_id"] == heads["true_rid"]]      # 編號與比分一致的 rally（train 幾乎全部）
    print(f"test 的樣式：間隔 {dict(sorted(pat['steps'].items()))}；符合條件的 rally 被打散的比例 {pat['rate']:.2f}")
    rng = np.random.default_rng(seed)
    res = {"條件 1–6": [], "條件 1–7": []}
    for _ in range(repeats):
        sim, truth = simulate(heads, pat, rng)
        for name, use_next in (("條件 1–6", False), ("條件 1–7", True)):
            placed = relocate_heads(sim, use_next=use_next)
            ok = sum(truth[int(u)] == (m, g, int(rid)) for u, m, g, rid in placed.itertuples(index=False))
            res[name].append((len(truth), len(placed), ok))
    print("\n| 放回條件 | 打散的 rally（平均） | 放回的比例 | 放回者位置正確的比例 | 放錯的 rally（平均每次） |")
    print("|---|---:|---:|---:|---:|")
    for name, r in res.items():
        n, p, ok = np.array(r).sum(0)
        print(f"| {name} | {n / repeats:.0f} | {p / n:.1%} | {ok / p:.1%} | {(p - ok) / repeats:.1f} |")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 20)
