"""test 的截斷機制（本 repo 新增）：估計「長度 L 的 rally 被截在第 k 拍」的機率，讓訓練與評估對齊 test。

資料上的事實：rally 的最後一拍 pointId 一定是 0，而且最後一拍的擊球者輸掉這一分（train 中 99.9%）。
所以「要預測的那一拍是不是 rally 的最後一拍」（以下稱終結拍）同時決定了落點 0、球種分佈與勝負。
train 的每個 rally 提供所有前綴（k = 1..L−1），其中終結拍只佔 21%；test 每個 rally 只有一個前綴，
終結拍的比例取決於主辦單位怎麼截。

機制假設（F0）：每個 rally 從一個固定的分佈 g 抽截斷位置 k，抽到 k > L−1 就重抽，也就是
    m(k | L) = g(k) / Σ_{j ≤ L−1} g(j)，k = 1..L−1。
再假設 test 的 rally 長度分佈與 train 相同，g 就能由 test 的前綴長度分佈唯一解出（k 的邊際分佈完全吻合）。

佐證（不用任何 test 答案，只用 test 已公開的比分；`python tools/truncation_evidence.py` 可重現）：
比分推移揭露了部分 rally 的勝負，而勝負等同 L 的奇偶。以這份資料的對數概似比較：「所有前綴等權」
−1788.3、原評估抽樣 −1786.6、F0 −1778.9（F0 比等權高 9.3，約 1 萬倍的概似比）；
「要預測的那一拍從來不是最後一拍」只有 −1817.4。
差異幾乎全部來自 k = 1：長度 2 的 rally 一定只能截在 k = 1，所以 test 中 k = 1 的終結率是
P(L=2) / P_test(k=1) ≈ 0.45（比分資料單獨估出來是 0.47，95% 區間 0.32–0.61），而所有前綴等權時只有 0.125。
k ≥ 2 的終結率，比分資料分辨不出 0.2–0.45 之間的值（k ≥ 4 的點估計甚至偏低），F0 給 0.35–0.50；
這一段是機制假設推出來的，不是資料直接證實的。
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from . import config

KMAX = 30    # 前綴長度上限；更長的併入最後一格
KTAIL = 9    # k ≥ KTAIL 的 test 樣本太少（每格不到 15 個），g 在這一段只調一個共同倍率


def _length_dist(rally_len: np.ndarray, kmax: int) -> np.ndarray:
    """pi[L]，L = 2..kmax+1（更長的併入 kmax+1）。長度 1 的 rally 沒有下一拍，不會成為預測樣本。"""
    L = np.clip(np.asarray(rally_len, dtype=int), 0, kmax + 1)
    pi = np.bincount(L[L >= 2], minlength=kmax + 2).astype(float)
    return pi / pi.sum()


def _marginal(g: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """給定 g 與長度分佈，k 的邊際分佈 Σ_L pi_L g(k)/S_L。"""
    S = np.concatenate([[0.0, 0.0], np.cumsum(g[1:])])          # S[L] = Σ_{j ≤ L−1} g(j)
    c = np.divide(pi, S, out=np.zeros_like(pi), where=S > 0)
    tail = np.concatenate([np.cumsum(c[::-1])[::-1][1:], [0.0]])  # tail[k] = Σ_{L > k} c[L]
    return g * tail[:len(g)]


def fit_cut_distribution(train_rally_len, test_prefix_len, kmax: int = KMAX, ktail: int = KTAIL,
                         iters: int = 3000) -> np.ndarray:
    """解 g(k)（k = 1..kmax，g[0] = 0），使 F0 機制下 k 的邊際分佈等於 test 的前綴長度分佈。"""
    pi = _length_dist(train_rally_len, kmax)
    pk = np.bincount(np.clip(np.asarray(test_prefix_len, dtype=int), 1, kmax), minlength=kmax + 1).astype(float)
    pk /= pk.sum()
    g = np.ones(kmax + 1)
    g[0] = 0.0
    for _ in range(iters):
        marg = _marginal(g, pi)
        r = np.ones(kmax + 1)
        r[1:ktail] = pk[1:ktail] / np.clip(marg[1:ktail], 1e-300, None)
        r[ktail:] = pk[ktail:].sum() / max(marg[ktail:].sum(), 1e-300)
        g = g * r
        g /= g.sum()
    return g


def cut_weights(prefix_len, rally_len, g: np.ndarray) -> np.ndarray:
    """每個 (前綴 k, rally 長度 L) 訓練列的權重 m(k|L) = g(k)/S_L：加權後的列分佈就是 F0 下的 test 分佈。"""
    kmax = len(g) - 1
    k = np.clip(np.asarray(prefix_len, dtype=int), 1, kmax)
    L = np.clip(np.asarray(rally_len, dtype=int), 2, kmax + 1)
    S = np.concatenate([[0.0, 0.0], np.cumsum(g[1:])])
    return g[k] / S[L]


def cut_matched_indices(df: pd.DataFrame, g: np.ndarray, n_repeats: int = 5,
                        seed: int = config.SEED) -> list[np.ndarray]:
    """評估用抽樣：每個 rally 依 g(k) 抽一個前綴（只在它實際有的前綴裡抽），重複 n_repeats 次。

    這就是 F0 機制本身，所以抽出來的樣本在前綴長度分佈與終結拍比例上都對齊 test。"""
    kmax = len(g) - 1
    k = np.clip(df["prefix_len"].to_numpy(dtype=int), 1, kmax)
    groups = list(df.groupby("rally_uid", sort=False).indices.values())
    out = []
    for r in range(n_repeats):
        rng = np.random.default_rng(seed + r)
        u = rng.random(len(groups))
        chosen = np.empty(len(groups), dtype=int)
        for i, idx in enumerate(groups):
            cdf = np.cumsum(g[k[idx]])
            chosen[i] = idx[min(int(np.searchsorted(cdf, u[i] * cdf[-1], side="right")), len(idx) - 1)]
        out.append(chosen)
    return out


def blend_aug_weights(w_base, k_base, terminal_base, w_aug, k_aug):
    """把 test 內部轉移（aug 列，全部是非終結拍）併入訓練時，維持各前綴長度的終結拍比例。

    aug 列只會增加「非終結」的質量；這裡把每個 k 的非終結池（base 非終結列 + aug 列）整體縮回 base 非終結列
    原本的總質量，終結列不動。回傳 (base 權重, aug 權重)。"""
    wb, wa = np.asarray(w_base, dtype=float).copy(), np.asarray(w_aug, dtype=float).copy()
    k_base, k_aug, terminal_base = np.asarray(k_base), np.asarray(k_aug), np.asarray(terminal_base, dtype=bool)
    for kv in np.unique(k_aug):
        mb, ma = (k_base == kv) & ~terminal_base, k_aug == kv
        tot = wb[mb].sum()
        if tot <= 0:
            wa[ma] = 0.0       # 這個前綴長度在訓練 fold 沒有非終結列可對照，aug 列不參與
            continue
        s = tot / (tot + wa[ma].sum())
        wb[mb] *= s
        wa[ma] *= s
    return wb, wa
