"""多任務序列模型：一次讀入整個 rally，在每個位置 t 預測第 t+1 拍的 action / point 與 rally 勝負。

- GRU / 因果 Transformer、因果 next-step 訓練、OOF 對齊方式改寫自
  0000workjason/aicup2026-tabletennis-prediction（私榜第 3 名）solution/model_nn.py
- t2v_film：Time2Vec 數值嵌入 + FiLM 條件化，改寫自 SugarSquirrel/aicup2026-table-tennis
  （私榜第 5 名）src/gen_submission_v8_t2v_film.py；原作只在最後一拍做 FiLM，這裡改成每個位置都做，
  以符合因果 next-step 訓練
- mirror：左右鏡像資料增強（落點 1↔3、4↔6、7↔9，站位 1↔3），來自 jabir95tsai/pingpong_aicup_2026（私榜第 22 名）
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config
from ..cv import get_folds
from ..features import fit_target_encoding

CAT_FEATS = ["actionId", "pointId", "handId", "strengthId", "spinId",
             "strikeId", "positionId", "sex"]
SCALAR_FEATS = ["score_diff", "total_points", "striker_is_server",
                "last_is_attack", "attack_ratio", "cum_attack"]
SCALAR_DIV = np.array([11.0, 22.0, 1.0, 1.0, 1.0, 10.0], dtype=np.float32)
POS_CAP = 40

POINT_MIRROR = np.array([0, 3, 2, 1, 6, 5, 4, 9, 8, 7])
POSITION_MIRROR = np.array([0, 3, 2, 1])


def _build_records(df: pd.DataFrame):
    df = df.sort_values(["rally_uid", "strikeNumber"])
    recs = []
    for uid, sub in df.groupby("rally_uid", sort=False):
        w = sub["winner"].to_numpy()
        recs.append(dict(
            uid=int(uid), match=sub["match"].iloc[0],
            cats=sub[CAT_FEATS].to_numpy(dtype=np.int64),
            scal=(sub[SCALAR_FEATS].to_numpy(dtype=np.float32) / SCALAR_DIV),
            pos=np.clip(sub["strikeNumber"].to_numpy(dtype=np.int64), 0, POS_CAP),
            action=sub["actionId"].to_numpy(dtype=np.int64),
            point=sub["pointId"].to_numpy(dtype=np.int64),
            pl=sub["next_player"].to_numpy(dtype=np.int64),
            win=-1 if pd.isna(w[0]) else int(w[0]), T=len(sub),
        ))
    return recs


def _mirror(rec):
    r = dict(rec)
    cats = rec["cats"].copy()
    pi, qi = CAT_FEATS.index("pointId"), CAT_FEATS.index("positionId")
    cats[:, pi] = POINT_MIRROR[cats[:, pi]]
    cats[:, qi] = POSITION_MIRROR[np.clip(cats[:, qi], 0, 3)]
    r["cats"] = cats
    r["point"] = POINT_MIRROR[rec["point"]]
    return r


def train_nn(D: dict, arch="gru", n_folds=config.N_FOLDS, device="cpu", epochs=16,
             fast=False, hidden=160, layers=2, dropout=0.3, lr=2e-3, weight_decay=1e-5,
             batch_size=128, n_seeds=2, seed=config.SEED, player_te=True, mirror=False):
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

    torch.set_num_threads(max(1, torch.get_num_threads()))
    if fast:
        epochs = min(epochs, 2)

    trf, tef = D["trf"], D["tef"]
    ap_base, test_df, ap_aug = D["ap_base"], D["test_pred"], D.get("ap_aug")
    vocabs = [max(int(trf[c].max()), int(tef[c].max())) + 1 for c in CAT_FEATS]
    emb_dims = [16 if c in ("actionId", "pointId") else 8 for c in CAT_FEATS]
    n_scal_base = len(SCALAR_FEATS)
    n_scal = n_scal_base + (config.N_ACTION if player_te else 0)

    train_recs = _build_records(trf)
    test_recs = _build_records(tef)
    train_match = np.array([r["match"] for r in train_recs])

    def _cls_w(y, n, temper=0.5):
        cnt = np.clip(np.bincount(np.clip(y, 0, n - 1), minlength=n).astype(float), 1.0, None)
        w = (cnt.sum() / cnt) ** temper
        return torch.tensor(w / w.mean(), dtype=torch.float32, device=device)

    wa = _cls_w(ap_base["next_action"].to_numpy(), config.N_ACTION)
    wp = _cls_w(ap_base["next_point"].to_numpy(), config.N_POINT)

    ap_idx = {(u, t): i for i, (u, t) in
              enumerate(zip(ap_base["rally_uid"].to_numpy(), ap_base["prefix_len"].to_numpy()))}
    te_idx = {u: i for i, u in enumerate(test_df["rally_uid"].to_numpy())}
    te_klen = dict(zip(test_df["rally_uid"].to_numpy(), test_df["prefix_len"].to_numpy().astype(int)))

    def _pad(batch):
        B, T = len(batch), max(r["T"] for r in batch)
        cats = np.zeros((B, T, len(CAT_FEATS)), np.int64)
        scal = np.zeros((B, T, n_scal), np.float32)
        pos = np.zeros((B, T), np.int64)
        for b, r in enumerate(batch):
            t = r["T"]
            cats[b, :t] = r["cats"]
            scal[b, :t, :n_scal_base] = r["scal"]
            if player_te:
                scal[b, :t, n_scal_base:] = r["te"]
            pos[b, :t] = r["pos"]
        return cats, scal, pos

    def collate(batch):
        cats, scal, pos = _pad(batch)
        B, T = pos.shape
        ta = np.full((B, T), -100, np.int64)
        tp = np.full((B, T), -100, np.int64)
        tw = np.zeros((B, T), np.float32)
        wmask = np.zeros((B, T), np.float32)
        for b, r in enumerate(batch):
            t = r["T"]
            if t > 1:
                nxt_a = r["action"][1:t]
                va = nxt_a <= 14
                ta[b, np.arange(t - 1)[va]] = nxt_a[va]
                tp[b, :t - 1] = r["point"][1:t]
            if r["win"] >= 0:
                tw[b, :t] = r["win"]
                wmask[b, :t] = 1.0
        to = lambda a: torch.from_numpy(a).to(device)
        return to(cats), to(scal), to(pos), to(ta), to(tp), to(tw), to(wmask)

    class Embedder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = nn.ModuleList([nn.Embedding(v, d) for v, d in zip(vocabs, emb_dims)])
            self.pos_emb = nn.Embedding(POS_CAP + 1, 16)
            self.out_dim = sum(emb_dims) + 16 + n_scal

        def forward(self, cats, scal, pos):
            xs = [emb(cats[..., i]) for i, emb in enumerate(self.embs)]
            return torch.cat(xs + [self.pos_emb(pos), scal], dim=-1)

    class Heads(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.drop = nn.Dropout(dropout)
            self.a, self.p, self.w = nn.Linear(d, config.N_ACTION), nn.Linear(d, config.N_POINT), nn.Linear(d, 1)

        def forward(self, h):
            h = self.drop(h)
            return self.a(h), self.p(h), self.w(h).squeeze(-1)

    class StrokeGRU(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = Embedder()
            self.gru = nn.GRU(self.emb.out_dim, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.heads = Heads(hidden)

        def forward(self, cats, scal, pos):
            h, _ = self.gru(self.emb(cats, scal, pos))
            return self.heads(h)

    class StrokeTransformer(nn.Module):
        """因果 Transformer：位置 t 只能看第 1..t 拍。"""
        def __init__(self):
            super().__init__()
            self.emb = Embedder()
            self.proj = nn.Linear(self.emb.out_dim, hidden)
            layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=4, dim_feedforward=2 * hidden,
                                               dropout=dropout, batch_first=True, activation="gelu")
            self.enc = nn.TransformerEncoder(layer, num_layers=layers)
            self.heads = Heads(hidden)

        def forward(self, cats, scal, pos):
            x = self.proj(self.emb(cats, scal, pos))
            T = x.size(1)
            causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            h = self.enc(x, mask=causal, src_key_padding_mask=(pos == 0))
            return self.heads(h)

    class Time2Vec(nn.Module):
        def __init__(self, in_dim, k=7):
            super().__init__()
            self.w_lin = nn.Parameter(torch.randn(in_dim) * 0.1)
            self.b_lin = nn.Parameter(torch.zeros(in_dim))
            self.W = nn.Parameter(torch.randn(in_dim, k) * 0.5)
            self.B = nn.Parameter(torch.randn(in_dim, k) * 0.5)

        def forward(self, x):
            lin = (x * self.w_lin + self.b_lin).unsqueeze(-1)
            per = torch.sin(x.unsqueeze(-1) * self.W + self.B)
            return torch.cat([lin, per], -1).flatten(-2)

    class StrokeT2VFiLM(nn.Module):
        """GRU 主體；數值特徵另經 Time2Vec，GRU 輸出以 [sex 嵌入, 至今 t2v 平均] 做 FiLM 條件化。"""
        def __init__(self):
            super().__init__()
            self.emb = Embedder()
            self.t2v = Time2Vec(n_scal_base, k=7)
            t2v_dim = n_scal_base * 8
            self.gru = nn.GRU(self.emb.out_dim + t2v_dim, hidden, num_layers=layers, batch_first=True,
                              dropout=dropout if layers > 1 else 0.0)
            self.film = nn.Linear(emb_dims[-1] + t2v_dim, 2 * hidden)
            self.heads = Heads(hidden)

        def forward(self, cats, scal, pos):
            t2v = self.t2v(scal[..., :n_scal_base])
            h, _ = self.gru(torch.cat([self.emb(cats, scal, pos), t2v], -1))
            steps = torch.arange(1, t2v.size(1) + 1, device=t2v.device).view(1, -1, 1)
            ctx = torch.cat([self.emb.embs[-1](cats[..., -1]), t2v.cumsum(1) / steps], -1)
            g, b = self.film(ctx).chunk(2, -1)
            return self.heads(h * (1.0 + torch.tanh(g)) + b)

    Model = {"gru": StrokeGRU, "transformer": StrokeTransformer, "t2v_film": StrokeT2VFiLM}[arch]

    ce_a = nn.CrossEntropyLoss(weight=wa, ignore_index=-100)
    ce_p = nn.CrossEntropyLoss(weight=wp, ignore_index=-100)
    bce = nn.BCEWithLogitsLoss(reduction="none")

    oof_a = np.zeros((len(ap_base), config.N_ACTION), np.float32)
    oof_p = np.zeros((len(ap_base), config.N_POINT), np.float32)
    oof_w = np.zeros(len(ap_base), np.float32)
    te_a = np.zeros((len(test_df), config.N_ACTION), np.float32)
    te_p = np.zeros((len(test_df), config.N_POINT), np.float32)
    te_w = np.zeros(len(test_df), np.float32)

    @torch.no_grad()
    def predict_fill(model, recs, is_oof, scale):
        model.eval()
        for b in DataLoader(recs, batch_size=256, shuffle=False, collate_fn=lambda x: x):
            cats, scal, pos = _pad(b)
            la, lp, lw = model(torch.from_numpy(cats).to(device), torch.from_numpy(scal).to(device),
                               torch.from_numpy(pos).to(device))
            pa = torch.softmax(la, -1).cpu().numpy()
            pp = torch.softmax(lp, -1).cpu().numpy()
            pw = torch.sigmoid(lw).cpu().numpy()
            for j, r in enumerate(b):
                u, t = r["uid"], r["T"]
                if is_oof:
                    for tt in range(1, t):
                        ri = ap_idx.get((u, tt))
                        if ri is not None:
                            oof_a[ri] += pa[j, tt - 1] * scale
                            oof_p[ri] += pp[j, tt - 1] * scale
                            oof_w[ri] += pw[j, tt - 1] * scale
                else:
                    ri = te_idx.get(u)
                    if ri is not None:
                        k = te_klen[u]
                        te_a[ri] += pa[j, k - 1] * scale
                        te_p[ri] += pp[j, k - 1] * scale
                        te_w[ri] += pw[j, k - 1] * scale

    for fold, (tr_i, va_i) in enumerate(get_folds(train_match, n_folds)):
        if player_te:
            # P(action | next_player) 只用本 fold 訓練場次（+ test 內部轉移）fit
            src = ap_base[ap_base["match"].isin(set(train_match[tr_i].tolist()))]
            if ap_aug is not None:
                src = pd.concat([src, ap_aug], ignore_index=True)
            enc, prior = fit_target_encoding(src, "next_action", config.ACTION_CLASSES, ["next_player"])
            prior = prior.astype(np.float32)
            for r in train_recs + test_recs:
                r["te"] = np.array([enc.get((int(p),), prior) for p in r["pl"]], dtype=np.float32)

        tr_recs = [train_recs[i] for i in tr_i] + test_recs
        if mirror:
            tr_recs = tr_recs + [_mirror(r) for r in tr_recs]
        va_recs = [train_recs[i] for i in va_i]

        for s in range(n_seeds):
            torch.manual_seed(seed + 100 * fold + s)
            np.random.seed(seed + 100 * fold + s)
            model = Model().to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            loader = DataLoader(tr_recs, batch_size=batch_size, shuffle=True, collate_fn=collate)
            for _ in range(epochs):
                model.train()
                for cats, scal, pos, ta, tp, tw, wmask in loader:
                    opt.zero_grad()
                    la, lp, lw = model(cats, scal, pos)
                    loss = (config.W_ACTION * ce_a(la.reshape(-1, config.N_ACTION), ta.reshape(-1))
                            + config.W_POINT * ce_p(lp.reshape(-1, config.N_POINT), tp.reshape(-1))
                            + config.W_WINNER * (bce(lw, tw) * wmask).sum() / wmask.sum().clamp(min=1.0))
                    loss.backward()
                    opt.step()
                sched.step()
            predict_fill(model, va_recs, True, 1.0 / n_seeds)
            predict_fill(model, test_recs, False, 1.0 / (n_folds * n_seeds))
        print(f"   [nn:{arch}{'+mirror' if mirror else ''}] fold {fold + 1}/{n_folds} done", flush=True)

    return dict(oof_a=oof_a, oof_p=oof_p, oof_w=oof_w, te_a=te_a, te_p=te_p, te_w=te_w)
