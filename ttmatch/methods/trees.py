"""梯度提升樹模型：LightGBM / XGBoost / CatBoost，產生依 match 分組的 OOF 與 test 機率。

- LightGBM 流程（類別權重、test 內部轉移擴充、選手 target encoding）改寫自
  0000workjason/aicup2026-tabletennis-prediction（私榜第 3 名）solution/model_lgbm.py
- XGBoost / CatBoost 對照組參考 lucas6028/AI_CUP_2026_Table_Tennis_Prediction（前標）的多樹模型比較
- winner 模型可加入本 repo 新增的比分推移特徵（PROG_FEATURES）
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .. import config
from ..cv import get_folds
from ..features import (CATEGORICAL, PROG_FEATURES, get_X,
                        fit_target_encoding, apply_target_encoding, oof_target_encoding)

ACTION_TE_KEYS = [["next_player"], ["next_player", "last_action"], ["next_player", "next_strikeId"]]


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

def _fit_predict(kind, n_class, X_tr, y_tr, w_tr, X_va, y_va, X_preds, fast):
    rounds = 120 if fast else 1500
    if kind == "lgbm":
        import lightgbm as lgb
        params = dict(learning_rate=0.05, num_leaves=63 if n_class > 2 else 47,
                      min_data_in_leaf=40 if n_class > 2 else 60,
                      feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                      lambda_l2=1.0, num_threads=0, verbosity=-1, seed=config.SEED)
        if n_class > 2:
            params.update(objective="multiclass", num_class=n_class, metric="multi_logloss")
        else:
            params.update(objective="binary", metric="auc")
        cats = [c for c in CATEGORICAL if c in X_tr.columns]
        dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=cats, free_raw_data=False)
        dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cats, reference=dtr)
        m = lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        return [m.predict(X, num_iteration=m.best_iteration) for X in X_preds]

    if kind == "xgb":
        import xgboost as xgb
        params = dict(tree_method="hist", learning_rate=0.06, max_depth=7, subsample=0.8,
                      colsample_bytree=0.8, min_child_weight=3, reg_lambda=1.0,
                      nthread=0, seed=config.SEED)
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
        cats = [c for c in CATEGORICAL if c in X_tr.columns]
        # 多類別 + 大量類別特徵時 CTR 計算很慢：限制 CTR 組合、小詞彙類別改 one-hot
        m = CatBoostClassifier(
            iterations=150 if fast else 700, learning_rate=0.12, depth=6,
            max_ctr_complexity=1, one_hot_max_size=20,
            loss_function="MultiClass" if n_class > 2 else "Logloss",
            eval_metric="MultiClass" if n_class > 2 else "AUC",
            random_seed=config.SEED, thread_count=-1, verbose=False,
            od_type="Iter", od_wait=60, allow_writing_files=False)
        m.fit(X_tr, y_tr, sample_weight=w_tr, cat_features=cats, eval_set=(X_va, y_va))
        out = [m.predict_proba(X) for X in X_preds]
        return out if n_class > 2 else [p[:, 1] for p in out]

    raise ValueError(kind)


def train_multiclass(kind, D, target_col, classes, n_folds=config.N_FOLDS, fast=False,
                     te_keys=None, te_alpha=1.0, aug_weight_mult=5.0, use_aug=True, drop=()):
    """回傳 (oof, test_pred)，欄位順序 = classes。drop：要拿掉的特徵欄。"""
    _X = lambda df: get_X(df).drop(columns=list(drop))
    base_df, test_df = D["ap_base"], D["test_pred"]
    aug_df = D["ap_aug"] if use_aug else None
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
        w_tr[n_base:] *= aug_weight_mult
        y_va = np.array([c2i.get(v, 0) for v in yb[va_idx]])
        p_va, p_te = _fit_predict(kind, C, X_tr, y_tr, w_tr, X_va, y_va, [X_va, X_te], fast)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
    return oof, test_pred


def train_winner(kind, D, n_folds=config.N_FOLDS, fast=False, use_prog=False):
    base_df, test_df = D["ap_base"], D["test_pred"]
    extra = PROG_FEATURES if use_prog else None
    yb = base_df["winner"].astype(int).to_numpy()
    oof = np.zeros(len(base_df))
    test_pred = np.zeros(len(test_df))
    X_all, X_te = get_X(base_df, extra), get_X(test_df, extra)
    for tr_idx, va_idx in get_folds(base_df["match"].to_numpy(), n_folds):
        p_va, p_te = _fit_predict(kind, 2, X_all.iloc[tr_idx], yb[tr_idx], None,
                                  X_all.iloc[va_idx], yb[va_idx], [X_all.iloc[va_idx], X_te], fast)
        oof[va_idx] = p_va
        test_pred += p_te / n_folds
    return oof, test_pred


def run(kind, D, n_folds=config.N_FOLDS, fast=False, player_te=True, targets=("a", "p", "w"),
        use_prog=False, te_alpha=1.0, drop=()):
    out = {}
    if "a" in targets:
        out["oof_a"], out["te_a"] = train_multiclass(
            kind, D, "next_action", config.ACTION_CLASSES, n_folds, fast,
            te_keys=ACTION_TE_KEYS if player_te else None, te_alpha=te_alpha, drop=drop)
    if "p" in targets:
        # 第 3 名的 A/B：選手 TE 對 point 反而 -0.004，因此 point 不加
        out["oof_p"], out["te_p"] = train_multiclass(
            kind, D, "next_point", config.POINT_CLASSES, n_folds, fast, drop=drop)
    if "w" in targets:
        out["oof_w"], out["te_w"] = train_winner(kind, D, n_folds, fast, use_prog)
    return out
