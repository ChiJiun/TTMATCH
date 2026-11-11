"""檢驗 test 截斷機制的證據（重現 ttmatch/truncation.py 與 README 引用的數字）。

只用 test 已公開的欄位：前綴長度，以及同一局下一個可見 rally 的比分（比分推移）。不用任何答案。

原理：train 中「發球方贏 ⇔ rally 長度 L 是偶數」（最後一拍的擊球者輸）成立 99.9%。所以某個截斷機制
給出的 P(L | k) 直接決定 P(發球方贏 | k)；而 test 的比分推移資料（間隔 g 個 rally 內發球方共得幾分）
可以算出每個機制的對數概似。間隔內其他 rally 的得分分佈取自 train。

用法：python tools/truncation_evidence.py
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ttmatch.data import build_datasets, load_raw          # noqa: E402
from ttmatch.features import _rally_table                    # noqa: E402
from ttmatch.truncation import KMAX, _length_dist            # noqa: E402


def other_gain_dist(train_df, max_gap=11):
    """train：從 rally r 到 r+g，扣掉 r 自己之後，r 的發球方在其他 g−1 個 rally 得幾分的分佈。"""
    rt = _rally_table(train_df)
    rt["w"] = rt["rally_uid"].map(train_df.groupby("rally_uid")["serverGetPoint"].first())
    rt = rt.sort_values(["match", "numberGame", "rally_id"]).reset_index(drop=True)
    key = {k: i for i, k in enumerate(zip(rt.match, rt.numberGame, rt.rally_id))}
    srv, ss, rs, w, ok = (rt[c].to_numpy() for c in ("server", "srv_score", "rcv_score", "w", "consistent"))
    out = {}
    for g in range(2, max_gap + 1):
        cnt = np.zeros(g)
        for i, (m, n, r) in enumerate(zip(rt.match, rt.numberGame, rt.rally_id)):
            j = key.get((m, n, r + g))
            if j is None or not ok[i] or not ok[j] or (ss[j] + rs[j]) - (ss[i] + rs[i]) != g:
                continue
            o = int((ss[j] if srv[j] == srv[i] else rs[j]) - ss[i] - w[i])
            if 0 <= o < g:
                cnt[o] += 1
        out[g] = cnt / cnt.sum()
    return out


def main():
    D = build_datasets()
    tr, _, _ = load_raw()
    tp, g = D["test_pred"], D["cut_g"]
    pi = _length_dist(tr.groupby("rally_uid")["strikeNumber"].max().to_numpy(), KMAX)
    pk = np.bincount(np.clip(tp["prefix_len"].to_numpy(), 1, KMAX), minlength=KMAX + 1) / len(tp)
    other = other_gain_dist(tr)
    t = tp[(tp["prog_has_next"] == 1) & (tp["prog_gap"] <= 11)]
    T_k = np.clip(t["prefix_len"].to_numpy(dtype=int), 1, KMAX)
    T_g, T_a = t["prog_gap"].to_numpy(dtype=int), t["prog_srv_gain"].to_numpy(dtype=int)
    Ls = np.arange(KMAX + 2)

    def ll_rows(p_srv, sel=slice(None)):
        s = 0.0
        for k, gap, a in zip(T_k[sel], T_g[sel], T_a[sel]):
            o, p = other[gap], np.clip(p_srv[k], 1e-3, 1 - 1e-3)
            s += np.log(max(p * (o[a - 1] if 1 <= a <= gap else 0.0) + (1 - p) * (o[a] if a < gap else 0.0), 1e-12))
        return s

    def posterior(m):                        # m[k, L] = m(k | L) → P(L | k)
        J = m * pi[None, :]
        return J / np.clip(J.sum(1, keepdims=True), 1e-300, None), J.sum(1)

    mech = {}
    m = np.zeros((KMAX + 1, KMAX + 2))       # 所有前綴等權：各 k 內 L 依 train 中 L ≥ k+1 的分佈
    for L in range(2, KMAX + 2):
        m[1:L, L] = 1.0
    mech["所有前綴等權（未對齊的訓練分佈）"] = m
    m = np.zeros_like(m)                     # 要預測的那一拍永遠不是最後一拍（k ≤ L−2 等權）
    for L in range(3, KMAX + 2):
        m[1:L - 1, L] = 1.0
    mech["從不截在終結拍（test 內部轉移隱含的假設）"] = m
    m = np.zeros_like(m)                     # 原評估抽樣：K ~ test 前綴分佈，超過就截到 L−1
    for L in range(2, KMAX + 2):
        m[1:L - 1, L] = pk[1:L - 1]
        m[L - 1, L] = pk[L - 1:].sum()
    mech["原評估抽樣（K 超過就截到 L−1）"] = m
    m = np.zeros_like(m)                     # F0：k ~ g，限制在 1..L−1
    for L in range(2, KMAX + 2):
        m[1:L, L] = g[1:L] / g[1:L].sum()
    mech["F0（本 repo 採用）"] = m

    print(f"有比分推移資訊的 test rally：{len(t)} 個（共 {len(tp)} 個）\n")
    print(f"{'機制':32s} {'對數概似':>10s} {'整體終結率':>8s}  k=1..6 的終結率               k 邊際與 test 的最大差")
    for name, m in mech.items():
        PL, marg = posterior(m)
        p_srv = (PL * (Ls % 2 == 0)).sum(1)
        term = [PL[k, k + 1] for k in range(1, 7)]
        T = float(sum(marg[k] * PL[k, k + 1] for k in range(1, KMAX)) / marg[1:].sum())
        print(f"{name:32s} {ll_rows(p_srv):10.2f} {T:8.3f}  " + " ".join(f"{v:.3f}" for v in term)
              + f"   {np.abs(marg[1:] / marg[1:].sum() - pk[1:]).max():.4f}")

    print("\n各前綴長度：只讓終結率 t 自由（非終結的長度分佈照 train），以比分資料做最大概似估計")
    PLC, _ = posterior(mech["所有前綴等權（未對齊的訓練分佈）"])
    PLF, _ = posterior(mech["F0（本 repo 採用）"])
    print(" k    n  等權   F0   估計值 [95% 區間]")
    for k in range(1, 8):
        sel = T_k == k
        nt = PLC[k].copy()
        nt[k + 1] = 0
        q = (nt / nt.sum() * (Ls % 2 == 0)).sum()
        term_srv = 1.0 if (k + 1) % 2 == 0 else 0.0
        ts = np.linspace(0, 1, 201)
        lls = []
        for tv in ts:
            p = np.full(KMAX + 1, tv * term_srv + (1 - tv) * q)
            lls.append(ll_rows(p, sel))
        lls = np.array(lls)
        ok = ts[lls >= lls.max() - 1.92]
        print(f"{k:2d} {int(sel.sum()):4d}  {PLC[k, k + 1]:.3f} {PLF[k, k + 1]:.3f}  {ts[lls.argmax()]:.3f} [{ok.min():.2f}, {ok.max():.2f}]")


if __name__ == "__main__":
    main()
