"""Residual Boosting: Ridge + XGBoost on Ridge residuals.

Ridge handles linear extrapolation (national shifts, demographic trends).
HistGradientBoosting captures nonlinear residual patterns (geo clusters,
interaction effects) that Ridge misses — without extrapolation pressure
since residuals are near-zero centered.

Architecture:
  1. Ridge (best per-block config from preregistered.py) → deviation predictions
  2. LOO OOF Ridge residuals → train HistGradientBoosting on residuals
  3. Final = Ridge_pred + XGB_residual_pred

LOO evaluation uses nested LOO: outer fold holds out an election date,
inner LOO computes OOF residuals for the remaining dates → trains XGB →
predicts held-out fold. No validation tuning.

Usage:
    python3 -u -m src.residual_boost
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

from src.cross_type_dev import (
    load_cross_type_data, add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS, ABBR, ALPHAS, VAL_DATE, VAL_TYPE, TARGET_COLS,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.beat_it import build_extended_data

ALPHA_GRID = np.logspace(-2, 6, 20)
PREV_RAW = {"Gauche": 0.7414, "Centre+Droite": 0.5947,
            "Extreme_Droite": 0.8092, "Abstention": 0.7328}

# Best Ridge configs per block (from preregistered.py LOO selection)
# Each: (name, data_key, feat_key, cfg)
BEST_RIDGE = {
    "Gauche":          ("Legi-PCA5-devlag",  "legi_v1_2", "legi", {"pca_k": 5}),
    "Centre+Droite":   ("Legi-PCA7-devlag",  "legi_v1_2", "legi", {"pca_k": 7}),
    "Extreme_Droite":  ("CT-PCA5-devlag",    "ct_v1_2",   "ct",   {"pca_k": 5}),
    "Abstention":      ("CT-PCA10-devlag",   "ct_v1_2",   "ct",   {"pca_k": 10}),
}

# Single fixed config — no LOO selection, eliminates selection noise
XGB_FIXED = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 100,
             "min_samples_leaf": 500, "l2_regularization": 1.0}


def split_tv(df):
    val_mask = (
        np.isclose(df["date_float"], VAL_DATE, atol=1e-3)
        & (df["election_type"] == VAL_TYPE)
    )
    return df[~val_mask], df[val_mask]


def _apply_pca(X, pca, n_demo):
    if pca is None:
        return X
    return np.hstack([pca.transform(X[:, :n_demo]), X[:, n_demo:]])


def _build_fold_info(train, national_means):
    """Build fold masks and national means for LOO over election dates."""
    train_types = train["election_type"].values
    train_dates = train["date_float"].values
    train_td = (
        train[["election_type", "date_float"]]
        .drop_duplicates().sort_values("date_float").values.tolist()
    )
    fold_masks, fold_nats = [], []
    for etype, ddate in train_td:
        mask = (np.isclose(train_dates, ddate, atol=1e-3)
                & (train_types == etype))
        fold_masks.append(mask)
        nm_row = national_means[
            (national_means["election_type"] == etype)
            & np.isclose(national_means["date_float"], ddate, atol=1e-3)
        ]
        fold_nats.append(
            {tc: float(nm_row[tc].iloc[0]) for tc in TARGET_COLS}
            if len(nm_row) > 0 else {tc: 0.0 for tc in TARGET_COLS})
    return fold_masks, fold_nats


def run_residual_boost(tc, train, val, feat_cols, demo_cols, national_est,
                       national_means, cfg):
    """Ridge + XGB residual stacking for one block.

    Returns dict with ridge-only and combined results.
    """
    n_demo = len(demo_cols)
    pca_k = cfg.get("pca_k")

    # Scale features
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v_raw = scaler.transform(val[feat_cols].values.astype(np.float64))

    # PCA (fit on full training for val prediction)
    if pca_k:
        pca_full = PCA(n_components=pca_k).fit(X_tr_raw[:, :n_demo])
    else:
        pca_full = None

    X_tr = _apply_pca(X_tr_raw, pca_full, n_demo)
    X_v = _apply_pca(X_v_raw, pca_full, n_demo)

    dev_y = train[f"dev_{tc}"].values.astype(np.float64)
    y_true_train = train[tc].values.astype(np.float64)
    y_true_val = val[tc].values.astype(np.float64)
    nat_est = national_est.get(tc, 0.0)

    fold_masks, fold_nats = _build_fold_info(train, national_means)
    n_folds = len(fold_masks)

    # ── Step 1: Full Ridge → val prediction ──
    ridge_full = RidgeCV(alphas=ALPHA_GRID)
    ridge_full.fit(X_tr, dev_y)
    ridge_val_pred = ridge_full.predict(X_v) + nat_est
    ridge_val_r2 = r2_score(y_true_val, ridge_val_pred)

    # ── Step 2: LOO OOF Ridge residuals (for XGB training) ──
    oof_ridge_dev = np.full(len(train), np.nan)
    for f_idx, held_mask in enumerate(fold_masks):
        not_held = ~held_mask
        if pca_k:
            pca_fold = PCA(n_components=pca_k).fit(X_tr_raw[not_held, :n_demo])
        else:
            pca_fold = None
        X_ft = _apply_pca(X_tr_raw[not_held], pca_fold, n_demo)
        X_fh = _apply_pca(X_tr_raw[held_mask], pca_fold, n_demo)
        ridge = Ridge(alpha=ridge_full.alpha_, solver="cholesky")
        ridge.fit(X_ft, dev_y[not_held])
        oof_ridge_dev[held_mask] = ridge.predict(X_fh)

    # OOF residuals in deviation space: what Ridge missed
    oof_residuals = dev_y - oof_ridge_dev

    # Ridge-only LOO R²
    oof_ridge_pred = np.array([
        oof_ridge_dev[m] + fold_nats[i][tc]
        for i, m in enumerate(fold_masks)
        for _ in [None]  # unpack trick
    ], dtype=object)
    # Flatten properly
    oof_ridge_abs = np.full(len(train), np.nan)
    for i, m in enumerate(fold_masks):
        oof_ridge_abs[m] = oof_ridge_dev[m] + fold_nats[i][tc]
    ridge_oof_r2 = r2_score(y_true_train, oof_ridge_abs)

    # ── Step 3: XGB on OOF residuals — single fixed config, no selection ──
    X_tr_xgb = X_tr_raw  # trees don't need PCA
    X_v_xgb = X_v_raw

    # LOO evaluation of the combined model (Ridge OOF + XGB on OOF residuals)
    oof_combined = np.full(len(train), np.nan)
    for f_idx, held_mask in enumerate(fold_masks):
        not_held = ~held_mask
        # XGB trained on OOF residuals from other folds, predicts this fold
        train_res_ok = ~held_mask & ~np.isnan(oof_residuals)
        if train_res_ok.sum() < 100:
            continue
        xgb = HistGradientBoostingRegressor(
            early_stopping=False, random_state=42, **XGB_FIXED)
        xgb.fit(X_tr_xgb[train_res_ok], oof_residuals[train_res_ok])
        xgb_resid_pred = xgb.predict(X_tr_xgb[held_mask])
        oof_combined[held_mask] = (
            oof_ridge_dev[held_mask] + xgb_resid_pred + fold_nats[f_idx][tc])

    ok = ~np.isnan(oof_combined)
    best_xgb_oof_r2 = r2_score(y_true_train[ok], oof_combined[ok]) if ok.sum() > 100 else ridge_oof_r2

    # ── Step 4: Final val prediction ──
    ok_res = ~np.isnan(oof_residuals)
    xgb_final = HistGradientBoostingRegressor(
        early_stopping=False, random_state=42, **XGB_FIXED)
    xgb_final.fit(X_tr_xgb[ok_res], oof_residuals[ok_res])
    xgb_val_resid = xgb_final.predict(X_v_xgb)
    combined_val_pred = ridge_val_pred + xgb_val_resid
    combined_val_r2 = r2_score(y_true_val, combined_val_pred)

    return {
        "ridge_oof_r2": ridge_oof_r2,
        "ridge_val_r2": ridge_val_r2,
        "combined_oof_r2": best_xgb_oof_r2,
        "combined_val_r2": combined_val_r2,
        "xgb_params": XGB_FIXED,
        "ridge_alpha": ridge_full.alpha_,
        "n_folds": n_folds,
    }


def main():
    data_dir = Path("data")
    t0 = time.time()

    # ── Load data ──
    df, demo_indicators, national_means, poll_feats = \
        load_cross_type_data(data_dir)
    type_cols = add_election_type_onehot(df)

    df_ext, ext_indicators, ext_nm, ext_pf = build_extended_data(data_dir)
    ext_type_cols = add_election_type_onehot(df_ext)

    # ── National estimates ──
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    est["Abstention"] = abs_pred

    ext_est = dict(est)

    # ── Feature groups ──
    geo_time = ["latitude", "longitude", "date_float"]
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    # ── Datasets ──
    df_v1 = df.dropna(subset=demo_indicators)
    df_v1_2lag = df_v1.dropna(
        subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)

    df_legi = df[df["election_type"] == VAL_TYPE].copy()
    df_legi_v1 = df_legi.dropna(subset=demo_indicators)
    df_legi_v1_2 = df_legi_v1.dropna(
        subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)

    ext_raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    ext_raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    ext_dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    ext_dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    ext_v1 = df_ext.dropna(subset=ext_indicators)
    ext_v1_2 = ext_v1.dropna(
        subset=ext_raw_lag1 + ext_raw_lag2 + ext_dev_lag1 + ext_dev_lag2)

    nd_ct = geo_time + dev_lag1 + dev_lag2 + type_cols
    nd_legi = geo_time + dev_lag1 + dev_lag2
    ext_nd = geo_time + ext_dev_lag1 + ext_dev_lag2 + ext_type_cols

    # Map data/feat keys to actual objects
    datasets = {
        "ct_v1_2": df_v1_2lag,
        "legi_v1_2": df_legi_v1_2,
        "ext_v1_2": ext_v1_2,
    }
    feat_maps = {
        "ct":   (demo_indicators, nd_ct, demo_indicators + nd_ct, national_means),
        "legi": (demo_indicators, nd_legi, demo_indicators + nd_legi, national_means),
        "ext":  (ext_indicators, ext_nd, ext_indicators + ext_nd, ext_nm),
    }
    est_maps = {
        "ct": est,
        "legi": est,
        "ext": ext_est,
    }

    # ── Run residual boosting per block ──
    print("=" * 70)
    print("RESIDUAL BOOSTING: Ridge + HistGradientBoosting on OOF residuals")
    print("Fixed XGB config (no hyperparameter selection, no val tuning)")
    print("=" * 70)

    results = {}
    for tc in TARGET_COLS:
        ridge_name, data_key, feat_key, cfg = BEST_RIDGE[tc]
        demo_cols, nd_cols, all_cols, nm = feat_maps[feat_key]
        data = datasets[data_key]
        nat_est = est_maps[feat_key]

        cfg["n_demo"] = len(demo_cols)

        train, val = split_tv(data)
        ok_tr = train[all_cols].notna().all(axis=1)
        ok_v = val[all_cols].notna().all(axis=1)
        train_clean = train[ok_tr].copy()
        val_clean = val[ok_v].copy()

        print(f"\n{'─'*60}")
        print(f"  {ABBR[tc]} ({tc}) — base Ridge: {ridge_name}")
        print(f"  train={len(train_clean):,} val={len(val_clean):,} "
              f"feat={len(all_cols)} "
              f"dates={sorted(train_clean['date_float'].unique())}")
        print(f"  Fixed XGB config: {XGB_FIXED}")

        t1 = time.time()
        res = run_residual_boost(
            tc, train_clean, val_clean, all_cols, demo_cols,
            nat_est, nm, cfg)
        elapsed = time.time() - t1

        results[tc] = res
        delta = res["combined_val_r2"] - res["ridge_val_r2"]
        print(f"  Ridge-only:  OOF={res['ridge_oof_r2']:.4f}  "
              f"Val={res['ridge_val_r2']:.4f}")
        print(f"  Ridge+XGB:   OOF={res['combined_oof_r2']:.4f}  "
              f"Val={res['combined_val_r2']:.4f}  "
              f"(Δ={delta:+.4f})")
        print(f"  XGB params: {res['xgb_params']}")
        print(f"  ({elapsed:.0f}s)")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("SUMMARY: Ridge vs Ridge+XGB residual boost")
    print(f"{'='*70}")
    print(f"\n{'Block':20s} {'Ridge Val':>10s} {'+XGB Val':>10s} "
          f"{'Δ':>7s} {'prev':>7s} {'vs prev':>8s}")
    print("-" * 65)

    for tc in TARGET_COLS:
        r = results[tc]
        delta = r["combined_val_r2"] - r["ridge_val_r2"]
        vs_prev = r["combined_val_r2"] - PREV_RAW[tc]
        mark = "BEAT" if vs_prev > 0.0005 else (
            "~tie" if abs(vs_prev) <= 0.0005 else "miss")
        print(f"  {tc:20s} {r['ridge_val_r2']:9.4f} "
              f"{r['combined_val_r2']:9.4f}  {delta:+.4f} "
              f"{PREV_RAW[tc]:6.4f}  {vs_prev:+.4f} [{mark}]")

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
