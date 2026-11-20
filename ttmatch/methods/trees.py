"""梯度提升樹模型：LightGBM / XGBoost / CatBoost，產生依 match 分組的 OOF 與 test 機率。

- LightGBM 流程（類別權重、test 內部轉移擴充、選手 target encoding）改寫自
  0000workjason/aicup2026-tabletennis-prediction（私榜第 3 名）solution/model_lgbm.py
- XGBoost / CatBoost 對照組參考 lucas6028/AI_CUP_2026_Table_Tennis_Prediction（前標）的多樹模型比較
- winner 模型可加入本 repo 新增的比分推移特徵（PROG_FEATURES）

test 內部轉移擴充（use_aug）預設關閉。這些列來自 test rally 自己較短的前綴，而且一定不是 rally 的最後一拍；
樹模型會記住「這個 rally / 這位選手的列都不是終結拍」，於是對同一個 rally 真正要預測的那一列幾乎不預測
落點 0：LightGBM + history 在 test 上的平均 P(落點 0) 只有 0.07，OOF 是 0.41；拿掉這些列後 test 是 0.40。
CV 看不到這個偏差（驗證 fold 的 rally 沒有自己的前綴在訓練集裡）。序列模型沒有這個現象，仍使用這些列。
"""
from __future__ import annotations
import os

import numpy as np
import pandas as pd

from .. import config
from ..cv import get_folds
from ..features import (CATEGORICAL, CATEGORICAL_OPTIONAL, PROG_FEATURES, get_X,
                        fit_target_encoding, apply_target_encoding, oof_target_encoding)
from ..truncation import blend_aug_weights

ACTION_TE_KEYS = [["next_player"], ["next_player", "last_action"], ["next_player", "next_strikeId"]]
# 同時跑多個訓練行程時用來限制每個行程的執行緒數（CatBoost 預設會佔滿所有核心）；0 = 套件預設
N_THREADS = int(os.environ.get("TTMATCH_THREADS", "0"))


def _sample_weights(y_idx, n_class, temper=0.5):
    counts = np.clip(np.bincount(y_idx, minlength=n_class).astype(float), 1.0, None)
    inv = (counts.sum() / counts) ** temper
    inv = inv / inv.mean()
    return inv[y_idx]


def _with_te(X, src_df, encoders, classes, tag):
    if not encoders:
        return X.reset_index(drop=True)
    extra = [apply_target_encoding(src_df, enc, prior, keys, classes, f"{tag}{i}").reset_index(drop=True)
             for i, (keys, enc, prior) in enumerate(encoders)]
    return pd.concat([X.reset_index(drop=True)] + extra, axis=1)


# --- 各套件的 fit / predict 包裝 ---------------------------------------------

def _fit_predict(kind, n_class, X_tr, y_tr, w_tr, X_va, y_va, X_preds, fast,
                 n_seeds=1, params_override=None):
    """n_seeds>1 時以不同隨機種子各訓練一次並平均（seed bagging）；params_override 覆寫套件預設參數。"""
    if n_seeds > 1:
        outs = [_fit_predict(kind, n_class, X_tr, y_tr, w_tr, X_va, y_va, X_preds, fast,
                             1, {**(params_override or {}), "_seed": config.SEED + 1000 * s})
                for s in range(n_seeds)]
        return [np.mean([o[i] for o in outs], axis=0) for i in range(len(X_preds))]
    po = dict(params_override or {})
    seed = int(po.pop("_seed", config.SEED))
    rounds = 120 if fast else int(po.pop("rounds", 1500))

    if kind == "lgbm":
        import lightgbm as lgb
        params = dict(learning_rate=0.05, num_leaves=63 if n_class > 2 else 47,
                      min_data_in_leaf=40 if n_class > 2 else 60,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, num_threads=N_THREADS, verbosity=-1, seed=seed)
        params.update(po)
        if n_class > 2:
            params.update(objective="multiclass", num_class=n_class, metric="multi_logloss")
        else:
            params.update(objective="binary", metric="auc")
        cats = [c for c in CATEGORICAL + CATEGORICAL_OPTIONAL if c in X_tr.columns]
        dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cats, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return [m.predict(X, num_iteration=m.best_iteration) for X in X_preds]

    if kind == "xgb":
        import xgboost as xgb
        params = dict(tree_method="hist", learning_rate=0.06, max_depth=7, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                      nthread=N_THREADS, seed=seed)
        params.update(po)
        if n_class > 2:
            params.update(objective="multi:softprob", num_class=n_class, eval_metric="mlogloss")
        else:
            params.update(objective="binary:logistic", eval_metric="auc")
        dtr = xgb.DMatrix(X_tr, label=y_tr, weight=w_tr)
        dva = xgb.DMatrix(X_va, label=y_va)
        m = xgb.train(params, dtr, num_boost_round=rounds, evals=[(dva, "va")],
                      early_stopping_rounds=50, verbose_eval=False)
        rng = (0, m.best_iteration + 1)
        return [m.predict(xgb.DMatrix(X), iteration_range=rng) for X in X_preds]

    if kind == "cat":
        from catboost import CatBoostClassifier
        cats = [c for c in CATEGORICAL + CATEGORICAL_OPTIONAL if c in X_tr.columns]
        # 多類別 + 大量類別特徵時 CTR 計算很慢：限制 CTR 組合、小詞彙類別改 one-hot
        cat_params = dict(
            iterations=150 if fast else 700, learning_rate=0.12, depth=6,
            max_ctr_complexity=1, one_hot_max_size=20,
            loss_function="MultiClass" if n_class > 2 else "Logloss",
            eval_metric="MultiClass" if n_class > 2 else "AUC",
            random_seed=seed, thread_count=N_THREADS or -1, verbose=False,
            od_type="Iter", od_wait=60, allow_writing_files=False)
        cat_params.update(po)
        m = CatBoostClassifier(**cat_params)
        m.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=cats, eval_set=(X_va, y_va))
        out = [m.predict_proba(X) for X in X_preds]
        return out if n_class > 2 else [p[:, 1] for p in out]

    raise ValueError(kind)


def train_multiclass(kind, D, target_col, classes, n_folds=config.N_FOLDS, fast=False,
                     te_keys=None, te_alpha=1.0, aug_weight_mult=5.0, use_aug=False, drop=(),
                     mirror=False, mirror_weight=1.0, n_seeds=1, params=None, extra=(), align=True):
    """回傳 (oof, test_pred)，欄位順序 = classes。

    align：訓練列依 test 的截斷機制加權（ap_base["align_w"]，見 ttmatch/truncation.py），讓訓練分佈的
    前綴長度與終結拍比例對齊 test；aug 列併入時維持各前綴長度的終結拍比例。
    use_aug：把 test 內部轉移（D["ap_aug"]）併入訓練。預設關閉，原因見本檔開頭；只有重現第 3 名原作法的
    `lgbm` 與對照用的 `lgbm_hist_aug` 會打開。

    drop：要拿掉的特徵欄。extra：額外加入的特徵欄（例如 cur_player、HIST_FEATURES）。
    mirror：把訓練 fold 的左右鏡像列也加進訓練（標籤為 next_point 時一併鏡像），驗證 / test 永遠用原始列；
    鏡像表沒有 extra 欄位，不能與 extra 同時使用。n_seeds：seed bagging。params：覆寫套件參數。
    """
    assert not (mirror and extra), "mirror 與 extra 不能同時使用"
    _X = lambda df: get_X(df, list(extra)).drop(columns=list(drop))
    base_df, test_df = D["ap_base"], D["test_pred"]
    aug_df = D["ap_aug"] if use_aug else None
    mir_df = D["ap_base_mirror"] if mirror else None
    C = len(classes)
    c2i = {c: i for i, c in enumerate(classes)}
    yb = base_df[target_col].astype(int).to_numpy()
    ya = aug_df[target_col].astype(int).to_numpy() if aug_df is not None else None
    oof = np.zeros((len(base_df), C))
    test_pred = np.zeros((len(test_df), C))

    for tr_idx, va_idx in get_folds(base_df["match"].to_numpy(), n_folds):
        rows = tr_idx[np.isin(yb[tr_idx], classes)]
        src_tr = base_df.iloc[rows]
        y_tr = np.array([c2i[v] for v in yb[rows]])
        n_base = len(y_tr)
        n_mirror = 0
        if mir_df is not None:
            ym = mir_df[target_col].astype(int).to_numpy()[rows]  # 鏡像表與 base 一一對齊
            src_tr = pd.concat([src_tr, mir_df.iloc[rows]], ignore_index=True)
            y_tr = np.concatenate([y_tr, [c2i[v] for v in ym]])
            n_mirror = len(rows)
        if aug_df is not None:
            ma = np.isin(ya, classes)
            src_tr = pd.concat([src_tr, aug_df.iloc[np.where(ma)[0]]], ignore_index=True)
            y_tr = np.concatenate([y_tr, [c2i[v] for v in ya[ma]]])
        encoders = [(k, *fit_target_encoding(src_tr, target_col, classes, k, te_alpha))
                    for k in (te_keys or [])]
        va_df = base_df.iloc[va_idx]
        # 訓練列的 TE 用巢狀 OOF；驗證列與 test 用整個訓練 fold fit 的編碼
        X_tr = pd.concat([_X(src_tr).reset_index(drop=True)] + [
            pd.DataFrame(oof_target_encoding(src_tr, target_col, classes, k, te_alpha, src_tr["match"]),
                         columns=[f"te{i}_p{j}" for j in range(C)])
            for i, k in enumerate(te_keys or [])], axis=1)
        X_va = _with_te(_X(va_df), va_df, encoders, classes, "te")
        X_te = _with_te(_X(test_df), test_df, encoders, classes, "te")
        w_tr = _sample_weights(y_tr, C)
        w_tr[n_base:n_base + n_mirror] *= mirror_weight
        w_tr[n_base + n_mirror:] *= aug_weight_mult
        if align:
            b = base_df.iloc[rows]
            wb = b["align_w"].to_numpy()
            if aug_df is not None:
                a = aug_df.iloc[np.where(ma)[0]]
                wb, wa = blend_aug_weights(wb, b["prefix_len"].to_numpy(), b["is_terminal"].to_numpy(),
                                           aug_weight_mult * a["align_w"].to_numpy(), a["prefix_len"].to_numpy())
                wa = wa / aug_weight_mult      # aug_weight_mult 已乘在 w_tr 上，這裡只留對齊的部分
            else:
                wa = np.zeros(0)
            w_al = np.concatenate([wb, np.tile(wb, 1 if n_mirror else 0), wa])
            w_tr = w_tr * w_al / np.average(w_al, weights=w_tr)
        y_va = np.array([c2i.get(v, 0) for v in yb[va_idx]])
        p_va, p_te = _fit_predict(kind, C, X_tr, y_tr, w_tr, X_va, y_va, [X_va, X_te], fast,
                                  n_seeds=n_seeds, params_override=params)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
    return oof, test_pred


def sim_transition_plan(prefix_len, vis_k, va_idx):
    """驗證 fold 的「模擬內部轉移」要放進哪個模型的訓練、每個驗證列又該用哪個模型預測。

    prefix_len：每一列的前綴長度。vis_k：(列數, G)，這一列的 rally 在第 g 組模擬裡的可見前綴長度
    （0 = 這一組看不到它；各組的可見 rally 互不重疊，見 history.simulated_visible_sets）。va_idx：驗證 fold 的列。

    回傳 (sim_rows, use)：
    - sim_rows[g]：第 g 個模型多拿來訓練的驗證列 = 第 g 組可見 rally 中 prefix_len < 可見前綴長度的列。
      這些列要預測的是第 2..k 拍，全部在模擬的可見前綴裡，所以一定不是終結拍。
    - use[i]：驗證列 va_idx[i] 用哪個模型的預測。規則只有一條：該模型的訓練資料裡，同一個 rally 的列都必須
      比這一列短（訓練列的目標是第 prefix_len + 1 拍，所以這等於「這一列要預測的那一拍與更後面的拍都不在
      訓練資料裡」）。
        · rally 在第 g 組可見、prefix_len ≥ 可見前綴長度 k：用模型 g（訓練裡有它自己的第 1..k−1 列）。
          prefix_len == k 的列與 test 完全同條件；prefix_len > k 的列少了自己的第 k..prefix_len−1 列。
        · rally 在第 g 組可見、prefix_len < k：這一列本身就在模型 g 的訓練資料裡，改用模型 (g+1) mod G
          （它看不到這個 rally，訓練裡沒有它的任何一列）。
        · rally 在每一組都不可見：用模型 0（訓練裡沒有它的任何一列）。
    """
    prefix_len, vis_k, va_idx = np.asarray(prefix_len), np.asarray(vis_k), np.asarray(va_idx)
    G = vis_k.shape[1]
    assert G >= 2, "至少要兩組互不重疊的模擬，否則可見 rally 較短的前綴沒有模型可用"
    assert ((vis_k >= 1).sum(1) <= 1).all(), "各組模擬的可見 rally 不能重疊"
    k_va, p_va = vis_k[va_idx], prefix_len[va_idx]
    sim_rows = [va_idx[(k_va[:, g] >= 1) & (p_va < k_va[:, g])] for g in range(G)]
    use = np.zeros(len(va_idx), dtype=int)
    for g in range(G):
        vis = k_va[:, g] >= 1
        use[vis] = np.where(p_va[vis] >= k_va[vis, g], g, (g + 1) % G)
    return sim_rows, use


def train_factored(kind, D, targets=("a", "p"), n_folds=config.N_FOLDS, fast=False, use_aug=True,
                   aug_weight_mult=5.0, n_seeds=1, params=None, extra=(), drop=(), align=True, sim_aug=False):
    """把「下一拍是不是終結拍」拆出來單獨建模，讓 test 內部轉移可以放心拿來訓練。回傳 {oof_a, te_a, oof_p, te_p}。

    P(類別 | x) = P(終結 | x) · P(類別 | 終結, x) + (1 − P(終結 | x)) · P(類別 | 非終結, x)
    - P(終結 | x)：二元模型，只用 train（test 沒有終結拍的標籤，也不能假設它的比例）。
    - P(類別 | 非終結, x)：多類別模型，用 train 的非終結列 + test 內部轉移。test 內部轉移全部是非終結拍，
      放進這個條件模型不會改變任何終結拍的比例，所以沒有本檔開頭說的偏差，又能用到 test 選手的資料。
    - P(類別 | 終結, x)：落點一定是 0；球種另訓練一個只用 train 終結列的多類別模型。

    sim_aug（方法後綴 `_fs`）：讓 OOF 與 test 預測是同一種行為。只加 test 內部轉移時，test 的每一列都有
    「同一場其他 rally、同一個 rally 前幾拍」的轉移在訓練資料裡，驗證 fold 的列卻沒有，OOF 量不到這份資訊的
    效果（test 的條件分佈比 OOF 尖銳，見 README）。sim_aug 把驗證 match 模擬成 test 的可見樣式
    （D["ap_base"] 的 vis_k*：每組的可見 rally 比例與前綴長度分佈都照 test；vis_k0 與 history 特徵是同一組
    模擬），把可見 rally 的內部轉移（prefix_len < 可見前綴長度的列）也加進非終結條件模型的訓練，權重規則
    與 test 內部轉移相同。

    驗證 rally 的每個前綴都要有 OOF（arena 的評估抽樣 cut_matched_indices 是每個 rally 依 g(k) 另外抽一個
    前綴，與模擬的可見前綴無關），而一個模型不能預測自己訓練過的列。所以每個 fold 訓練 G 個非終結條件模型
    （G = 模擬組數，各組的可見 rally 互不重疊），每個驗證列挑一個「訓練資料裡沒有它要預測的那一拍與更後面
    的拍」的模型（規則見 sim_transition_plan，tests/test_core.py 有對應的測試）。與 test 的差別：
    - 同場其他 rally 的轉移：每個模型都有一組（密度同 test）。
    - 同一個 rally 自己前幾拍的轉移：test 的列一定有第 1..k−1 拍；OOF 只有「可見 rally、prefix_len 等於模擬
      可見前綴長度」的列完全相同，其餘的列少了一部分或全部（前綴長度 1 的列本來就沒有，不受影響）。
      所以 OOF 在這一點上仍比 test 保守。
    - early stopping 用驗證 fold 中不在該模型訓練資料裡的非終結列。
    - test 預測是 fold × G 個模型的平均；每個模型的訓練資料比 test 內部轉移多了一組驗證 match 的模擬轉移。

    sim_aug 時另外回傳診斷用的陣列（tools/sim_aug_check.py 讀它們，量「少了自己 rally 的列」差多少）：
    sim_use（每一列用第幾個模型）與 simnt_<任務>（(列數, G, 類別數)，G 個非終結條件模型對每個驗證列的輸出；
    其中包含模型訓練過的列，那些值不能拿來評分）。
    """
    assert not (sim_aug and not use_aug), "sim_aug 是讓 OOF 重現 test 內部轉移的效果，必須同時 use_aug"
    _X = lambda df: get_X(df, list(extra)).drop(columns=list(drop))
    base_df, test_df = D["ap_base"], D["test_pred"]
    aug_df = D["ap_aug"] if use_aug else None
    term = base_df["is_terminal"].to_numpy().astype(int)
    plen = base_df["prefix_len"].to_numpy()
    w_al = base_df["align_w"].to_numpy() if align else np.ones(len(base_df))
    w_aug = (aug_df["align_w"].to_numpy() if align else np.ones(len(aug_df))) if aug_df is not None else None
    if sim_aug:
        vis_k = base_df[sorted(c for c in base_df.columns if c.startswith("vis_k"))].to_numpy()
        w_sim = base_df["sim_aug_w"].to_numpy() if align else np.ones(len(base_df))
    X_base, X_test = _X(base_df).reset_index(drop=True), _X(test_df).reset_index(drop=True)
    X_aug = _X(aug_df).reset_index(drop=True) if aug_df is not None else None
    specs = {"a": ("next_action", config.ACTION_CLASSES), "p": ("next_point", config.POINT_CLASSES)}
    out = {}
    for t in targets:
        out[f"oof_{t}"] = np.zeros((len(base_df), len(specs[t][1])))
        out[f"te_{t}"] = np.zeros((len(test_df), len(specs[t][1])))
        if sim_aug:
            out[f"simnt_{t}"] = np.zeros((len(base_df), vis_k.shape[1], len(specs[t][1])), dtype=np.float32)
    if sim_aug:
        out["sim_use"] = np.zeros(len(base_df), dtype=int)

    def _multiclass(rows_tr, rows_va, y, classes, with_aug, ya, X_va, seeds_kw, rows_sim=None):
        """在 rows_tr（可加 aug 列與 rows_sim）上訓練多類別模型，回傳對 X_va 與 test 的機率。
        rows_va 只用於 early stopping。"""
        c2i = {c: i for i, c in enumerate(classes)}
        C = len(classes)
        rows_tr = rows_tr[np.isin(y[rows_tr], classes)]
        X_tr, y_tr, w_tr = X_base.iloc[rows_tr], np.array([c2i[v] for v in y[rows_tr]]), w_al[rows_tr].copy()
        if with_aug:
            ma = np.where(np.isin(ya, classes))[0]
            X_tr = pd.concat([X_tr, X_aug.iloc[ma]], ignore_index=True)
            y_tr = np.concatenate([y_tr, [c2i[v] for v in ya[ma]]])
            w_tr = np.concatenate([w_tr, aug_weight_mult * w_aug[ma]])
        if rows_sim is not None and len(rows_sim):
            rows_sim = rows_sim[np.isin(y[rows_sim], classes)]
            X_tr = pd.concat([X_tr, X_base.iloc[rows_sim]], ignore_index=True)
            y_tr = np.concatenate([y_tr, [c2i[v] for v in y[rows_sim]]])
            w_tr = np.concatenate([w_tr, aug_weight_mult * w_sim[rows_sim]])
        w_tr = w_tr * _sample_weights(y_tr, C)
        w_tr = w_tr / w_tr.mean()
        rows_va = rows_va[np.isin(y[rows_va], classes)]
        y_va = np.array([c2i[v] for v in y[rows_va]])
        return _fit_predict(kind, C, X_tr, y_tr, w_tr, X_base.iloc[rows_va], y_va, [X_va, X_test], fast, **seeds_kw)

    seeds_kw = dict(n_seeds=n_seeds, params_override=params)
    for tr_idx, va_idx in get_folds(base_df["match"].to_numpy(), n_folds):
        X_va = X_base.iloc[va_idx]
        pt_va, pt_te = _fit_predict(kind, 2, X_base.iloc[tr_idx], term[tr_idx], w_al[tr_idx] / w_al[tr_idx].mean(),
                                    X_va, term[va_idx], [X_va, X_test], fast, **seeds_kw)
        nt_tr, nt_va = tr_idx[term[tr_idx] == 0], va_idx[term[va_idx] == 0]
        t_tr, t_va = tr_idx[term[tr_idx] == 1], va_idx[term[va_idx] == 1]
        if sim_aug:
            sim_rows, use = sim_transition_plan(plen, vis_k, va_idx)
            assert all(term[r].sum() == 0 for r in sim_rows), "模擬內部轉移不應該含終結拍"
            out["sim_use"][va_idx] = use
        for t in targets:
            col, classes = specs[t]
            y = base_df[col].astype(int).to_numpy()
            ya = aug_df[col].astype(int).to_numpy() if aug_df is not None else None
            if sim_aug:
                n_va, n_te = np.zeros((len(va_idx), len(classes))), np.zeros((len(test_df), len(classes)))
                for g, rows_sim in enumerate(sim_rows):
                    es = np.setdiff1d(nt_va, rows_sim)       # early stopping 不用已經在訓練資料裡的列
                    p_va, p_te = _multiclass(nt_tr, es, y, classes, True, ya, X_va, seeds_kw, rows_sim)
                    n_va[use == g] = p_va[use == g]
                    out[f"simnt_{t}"][va_idx, g] = p_va
                    n_te += p_te / len(sim_rows)
            else:
                n_va, n_te = _multiclass(nt_tr, nt_va, y, classes, aug_df is not None, ya, X_va, seeds_kw)
            if t == "p":      # 終結拍的落點一定是 0
                e_va, e_te = np.zeros_like(n_va), np.zeros_like(n_te)
                e_va[:, classes.index(0)] = 1.0
                e_te[:, classes.index(0)] = 1.0
            else:
                e_va, e_te = _multiclass(t_tr, t_va, y, classes, False, None, X_va, seeds_kw)
            out[f"oof_{t}"][va_idx] = pt_va[:, None] * e_va + (1 - pt_va)[:, None] * n_va
            out[f"te_{t}"] += (pt_te[:, None] * e_te + (1 - pt_te)[:, None] * n_te) / n_folds
    return out


def winner_extra(use_prog, extra=()):
    """winner 模型的額外特徵欄：use_prog 時加上 PROG_FEATURES，再接 extra；去重複並保留順序。

    目前的方法不會重疊（lgbm_hist_prog 等把比分推移放在 extra 的方法只訓練 action/point），這是防呆：
    若之後有方法同時 use_prog=True 又在 extra 放比分推移欄，不去重複會產生重複欄名，部分套件會報錯。
    """
    return list(dict.fromkeys((list(PROG_FEATURES) if use_prog else []) + list(extra)))


def train_winner(kind, D, n_folds=config.N_FOLDS, fast=False, use_prog=False, n_seeds=1, params=None,
                 extra=(), align=True):
    base_df, test_df = D["ap_base"], D["test_pred"]
    extra = winner_extra(use_prog, extra)
    yb = base_df["winner"].astype(int).to_numpy()
    oof = np.zeros(len(base_df))
    test_pred = np.zeros(len(test_df))
    X_all, X_te = get_X(base_df, extra), get_X(test_df, extra)
    for tr_idx, va_idx in get_folds(base_df["match"].to_numpy(), n_folds):
        w_tr = base_df["align_w"].to_numpy()[tr_idx] if align else None
        p_va, p_te = _fit_predict(kind, 2, X_all.iloc[tr_idx], yb[tr_idx], w_tr,
                                  X_all.iloc[va_idx], yb[va_idx], [X_all.iloc[va_idx], X_te], fast,
                                  n_seeds=n_seeds, params_override=params)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
    return oof, test_pred


def run(kind, D, n_folds=config.N_FOLDS, fast=False, player_te=True, targets=("a", "p", "w"),
        use_prog=False, te_alpha=1.0, drop=(), mirror=False, mirror_weight=1.0, n_seeds=1,
        params=None, extra=(), align=True, use_aug=False, factored=False, sim_aug=False, aug_weight_mult=5.0):
    """factored：action / point 改用終結拍分解（train_factored）；此時 use_aug 指條件模型要不要加 test 內部轉移，
    sim_aug 指交叉驗證時要不要再加驗證 match 的模擬內部轉移，aug_weight_mult 是這兩種轉移列的權重倍率
    （只對 factored 有作用）。"""
    out = {}
    if factored:
        assert not (player_te or mirror), "factored 不支援選手 TE 與鏡像"
        ap = tuple(t for t in targets if t in ("a", "p"))
        out.update(train_factored(kind, D, ap, n_folds, fast, use_aug=use_aug, aug_weight_mult=aug_weight_mult,
                                  n_seeds=n_seeds, params=params, extra=extra, drop=drop, align=align,
                                  sim_aug=sim_aug))
        targets = tuple(t for t in targets if t == "w")
    common = dict(drop=drop, mirror=mirror, mirror_weight=mirror_weight, n_seeds=n_seeds, params=params,
                  extra=extra, align=align, use_aug=use_aug)
    if "a" in targets:
        out["oof_a"], out["te_a"] = train_multiclass(
            kind, D, "next_action", config.ACTION_CLASSES, n_folds, fast,
            te_keys=ACTION_TE_KEYS if player_te else None, te_alpha=te_alpha, **common)
    if "p" in targets:
        # 第 3 名的 A/B：選手 TE 對 point 反而 -0.004，因此 point 不加
        out["oof_p"], out["te_p"] = train_multiclass(
            kind, D, "next_point", config.POINT_CLASSES, n_folds, fast, **common)
    if "w" in targets:
        out["oof_w"], out["te_w"] = train_winner(kind, D, n_folds, fast, use_prog,
                                                 n_seeds=n_seeds, params=params, extra=extra, align=align)
    return out
