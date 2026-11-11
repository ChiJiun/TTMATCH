"""把 test「打散的 rally」放回原本的比賽（本 repo 新增）。

test 的 match 340–354 不是真的比賽：每個「match」混了 20 多位選手，rally_id 與比分對不上。比對後發現，
這些 rally 的選手組合全部出現在其他正常的 test match 裡，選手 ID、性別與開局比分都是真的，只有
match / numberGame / rally_id 三個編號被打亂。也就是說，它們是正常 test match 裡沒有公開的 rally。

正常資料中 rally_id − 1 = 開局比分總和，所以打散的 rally 的真實 rally_id 可以直接由比分算出；剩下要找的是
它屬於哪一場、哪一局。候選的局必須同時滿足：
1. 同一對選手、同一性別；
2. 該局沒有公開過這個 rally_id；
3. 比分夾在該局前一個與後一個已知 rally 之間（雙方得分都不能倒退）；
4. 發球者符合該局的發球輪替（每 2 分換發，10 平後每分換發）；
5. 這個比分下該局還沒結束。
只用上面 5 個條件就能唯一定位的 rally 有很強的位置規律：幾乎都在「前一個公開 rally 的後 2 分」
（126 個裡有 124 個）。因此再加第 6 個條件：
6. rally_id 必須等於該局前一個公開 rally 的 rally_id + 2。
6 個條件都滿足的候選只剩一個才放回去；放回去的 rally 成為新的已知點（收緊條件 2、3），再反覆處理其餘的
rally。定位不出來的維持原狀（之後仍當作沒有比分推移資訊）。在 train 上模擬同樣的公開 / 打散樣式，
放回去的 rally 約 99% 位置正確。

用途：放回去之後，(1) 這些 rally 自己有了比分推移與同場 history 特徵；(2) 它們前面的公開 rally 多了一個
更近的比分檢查點，比分推移的間隔從 4 縮短成 2。只用到 test 已公開的欄位（選手、比分），不涉及任何答案。
"""
from __future__ import annotations
from collections import Counter

import numpy as np
import pandas as pd

REQUIRED_DPREV = 2   # 打散的 rally 與前一個公開 rally 的 rally_id 差（觀察到的規律，當作必要條件）


def _serve_turn(total: int) -> int:
    """比分總和為 total 時是第幾個發球輪次（10 平前每 2 分一輪，之後每分一輪）。"""
    return total // 2 if total < 20 else 10 + (total - 20)


def rally_heads(df: pd.DataFrame) -> pd.DataFrame:
    """每個 rally 一列（取第 1 拍）：編號、發球者、接發者、雙方開局比分。"""
    first = df.sort_values(["rally_uid", "strikeNumber"]).groupby("rally_uid", sort=False).first()
    out = first[["sex", "match", "numberGame", "rally_id", "gamePlayerId", "gamePlayerOtherId",
                 "scoreSelf", "scoreOther"]].reset_index()
    out = out.rename(columns={"gamePlayerId": "server", "gamePlayerOtherId": "receiver",
                              "scoreSelf": "srv_score", "scoreOther": "rcv_score"})
    out["true_rid"] = out["srv_score"] + out["rcv_score"] + 1
    return out


def scrambled_matches(heads: pd.DataFrame) -> set:
    """rally_id 與比分大多對不上的 match。"""
    rate = (heads["rally_id"] == heads["true_rid"]).groupby(heads["match"]).mean()
    return set(rate[rate < 0.5].index.tolist())


class _Game:
    """一局已知的檢查點：rally_id → (發球者, {選手: 得分})。"""

    def __init__(self, match, game, sex, players):
        self.match, self.game, self.sex, self.players = match, game, sex, players
        self.points: dict[int, tuple[int, dict]] = {}
        self.visible: set[int] = set()

    def add(self, rid, server, score, visible=False):
        self.points[int(rid)] = (int(server), score)
        if visible:
            self.visible.add(int(rid))

    def expected_server(self, total):
        votes = Counter()
        a, b = self.players
        for rid, (srv, _) in self.points.items():
            same = (_serve_turn(rid - 1) - _serve_turn(total)) % 2 == 0
            votes[srv if same else (b if srv == a else a)] += 1
        return votes.most_common(1)[0][0] if votes else None

    def fit(self, rid, server, score):
        """這個 rally 能不能放進這一局；可以則回傳 (與前一個公開 rally 的差, 與下一個已知 rally 的差)。"""
        if rid in self.points or not self.points:
            return None
        ids = sorted(self.points)
        prev = [i for i in ids if i < rid]
        nxt = [i for i in ids if i > rid]
        for nb, sign in ((prev[-1:], 1), (nxt[:1], -1)):
            for i in nb:
                other = self.points[i][1]
                if any(sign * (score[p] - other[p]) < 0 for p in score):
                    return None
        hi, lo = max(score.values()), min(score.values())
        if hi >= 11 and hi - lo >= 2:
            return None
        if self.expected_server(rid - 1) != server:
            return None
        vis_prev = [i for i in sorted(self.visible) if i < rid]
        return (rid - vis_prev[-1] if vis_prev else None, nxt[0] - rid if nxt else None)


def relocate_scrambled(test_df: pd.DataFrame, max_rounds: int = 20) -> pd.DataFrame:
    """回傳能唯一定位的打散 rally：rally_uid, match, numberGame, rally_id（真實位置）。"""
    heads = rally_heads(test_df)
    bad = scrambled_matches(heads)
    normal, scr = heads[~heads["match"].isin(bad)], heads[heads["match"].isin(bad)]
    games: dict[tuple, list[_Game]] = {}
    for (m, g), d in normal.groupby(["match", "numberGame"], sort=True):
        d = d[d["rally_id"] == d["true_rid"]]          # 只用編號與比分一致的 rally 當檢查點
        if d.empty:
            continue
        players = tuple(sorted((int(d["server"].iloc[0]), int(d["receiver"].iloc[0]))))
        gm = _Game(m, g, int(d["sex"].iloc[0]), players)
        for r in d.itertuples():
            if tuple(sorted((int(r.server), int(r.receiver)))) == players:
                gm.add(r.true_rid, r.server, {int(r.server): int(r.srv_score), int(r.receiver): int(r.rcv_score)},
                       visible=True)
        games.setdefault((players, gm.sex), []).append(gm)

    todo = {int(r.rally_uid): r for r in scr.itertuples()}
    placed = {}
    for _ in range(max_rounds):
        picks = {}
        for uid, r in todo.items():
            pair = tuple(sorted((int(r.server), int(r.receiver))))
            score = {int(r.server): int(r.srv_score), int(r.receiver): int(r.rcv_score)}
            cands = []
            for gm in games.get((pair, int(r.sex)), []):
                d = gm.fit(int(r.true_rid), int(r.server), score)
                if d is not None and d[0] == REQUIRED_DPREV:
                    cands.append(gm)
            if len(cands) == 1:
                picks[uid] = cands[0]
        # 兩個 rally 搶同一個位置時都不放
        slots = Counter((id(gm), int(todo[uid].true_rid)) for uid, gm in picks.items())
        new = {uid: gm for uid, gm in picks.items() if slots[(id(gm), int(todo[uid].true_rid))] == 1}
        if not new:
            break
        for uid, gm in new.items():
            r = todo.pop(uid)
            gm.add(r.true_rid, r.server, {int(r.server): int(r.srv_score), int(r.receiver): int(r.rcv_score)})
            placed[uid] = (gm.match, gm.game, int(r.true_rid))
    return pd.DataFrame([(u, m, g, rid) for u, (m, g, rid) in sorted(placed.items())],
                        columns=["rally_uid", "match", "numberGame", "rally_id"])


def apply_relocation(test_df: pd.DataFrame, placed: pd.DataFrame) -> pd.DataFrame:
    """把定位到的打散 rally 的 match / numberGame / rally_id 換成真實位置；其他列不動。"""
    out = test_df.copy()
    if placed.empty:
        return out
    idx = placed.set_index("rally_uid")
    hit = out["rally_uid"].isin(idx.index).to_numpy()
    for col in ("match", "numberGame", "rally_id"):
        out.loc[hit, col] = out.loc[hit, "rally_uid"].map(idx[col]).to_numpy()
    return out
