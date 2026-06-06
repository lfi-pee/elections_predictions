"""Residual Boosting: Ridge + Approximate GP on Ridge residuals.

Same architecture as residual_boost.py but replaces HistGradientBoosting
with a Nystroem kernel approximation + BayesianRidge, which:
  1. Models smooth nonlinear residual patterns (like GP with RBF kernel)
  2. Provides per-BV uncertainty estimates (posterior predictive std)
  3. Conformal calibration of prediction intervals from LOO residuals

The Nystroem transform approximates a Gaussian process with RBF kernel.
BayesianRidge performs evidence maximization (automatic regularization)
and returns posterior predictive intervals.

Key change vs residual_boost.py: also computes calibrated per-BV
prediction intervals from LOO Ridge residuals — the main deliverable
for a political party that needs to know uncertainty, not just point
predictions.

Usage:
    python3 -u -m src.gp_residual_boost
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge, BayesianRidge
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import Nystroem
from sklearn.metrics import r2_score

from src.cross_type_dev import (
    load_cross_type_data,
    add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS,
    ABBR,
    ALPHAS,
    VAL_DATE,
    VAL_TYPE,
    TARGET_COLS,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.beat_it import build_extended_data
from src.cross_type_ridge import TARGET_BLOCKS as _TB

ALPHA_GRID = np.logspace(-2, 6, 20)
PREV_RAW = {
    "Gauche": 0.7414,
    "Centre+Droite": 0.5947,
    "Extreme_Droite": 0.8092,
    "Abstention": 0.4102,
}

BEST_RIDGE = {
    "Gauche": ("Legi-PCA5-devlag", "legi_v1_2", "legi", {"pca_k": 5}),
    "Centre+Droite": ("Legi-PCA7-devlag", "legi_v1_2", "legi", {"pca_k": 7}),
    "Extreme_Droite": ("CT-PCA5-devlag", "ct_v1_2", "ct", {"pca_k": 5}),
    "Abstention": ("CT-PCA10-devlag", "ct_v1_2", "ct", {"pca_k": 10}),
}

# Lighter GP: 100 components (less overfitting, 5x faster than 500)
GP_CONFIG = {
    "n_components": 100,
    "kernel": "rbf",
    "gamma": None,
    "random_state": 42,
}


def split_tv(df):
    val_mask = np.isclose(df["date_float"], VAL_DATE, atol=1e-3) & (
        df["election_type"] == VAL_TYPE
    )
    return df[~val_mask], df[val_mask]


def _apply_pca(X, pca, n_demo):
    if pca is None:
        return X
    return np.hstack([pca.transform(X[:, :n_demo]), X[:, n_demo:]])


def _build_fold_info(train, national_means):
    train_types = train["election_type"].values
    train_dates = train["date_float"].values
    train_td = (
        train[["election_type", "date_float"]]
        .drop_duplicates()
        .sort_values("date_float")
        .values.tolist()
    )
    fold_masks, fold_nats = [], []
    for etype, ddate in train_td:
        mask = np.isclose(train_dates, ddate, atol=1e-3) & (train_types == etype)
        fold_masks.append(mask)
        nm_row = national_means[
            (national_means["election_type"] == etype)
            & np.isclose(national_means["date_float"], ddate, atol=1e-3)
        ]
        fold_nats.append(
            {tc: float(nm_row[tc].iloc[0]) for tc in TARGET_COLS}
            if len(nm_row) > 0
            else {tc: 0.0 for tc in TARGET_COLS}
        )
    return fold_masks, fold_nats


def _fit_gp(X_train, y_train, gp_cfg):
    n_comp = min(gp_cfg["n_components"], X_train.shape[0] - 1, X_train.shape[1] * 3)
    nystroem = Nystroem(
        kernel=gp_cfg["kernel"],
        gamma=gp_cfg["gamma"],
        n_components=n_comp,
        random_state=gp_cfg["random_state"],
    )
    Z = nystroem.fit_transform(X_train)
    bridge = BayesianRidge(max_iter=300, tol=1e-5)
    bridge.fit(Z, y_train)
    return nystroem, bridge


def _predict_gp(nystroem, bridge, X_test):
    Z = nystroem.transform(X_test)
    return bridge.predict(Z, return_std=True)


def run_gp_residual_boost(
    tc, train, val, feat_cols, demo_cols, national_est, national_means, cfg, gp_cfg=None
):
    """Ridge + GP residual stacking for one block.

    Returns dict with:
      - Ridge-only and combined R² metrics
      - Per-BV predictions + calibrated uncertainty
      - Conformal prediction intervals from LOO
    """
    if gp_cfg is None:
        gp_cfg = GP_CONFIG

    n_demo = len(demo_cols)
    pca_k = cfg.get("pca_k")

    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v_raw = scaler.transform(val[feat_cols].values.astype(np.float64))

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

    # ── Step 1: Full Ridge → val prediction ──
    ridge_full = RidgeCV(alphas=ALPHA_GRID)
    ridge_full.fit(X_tr, dev_y)
    ridge_val_pred = ridge_full.predict(X_v) + nat_est
    ridge_val_r2 = r2_score(y_true_val, ridge_val_pred)

    # ── Step 2: LOO OOF Ridge residuals ──
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

    oof_residuals = dev_y - oof_ridge_dev

    # Ridge-only OOF predictions (absolute)
    oof_ridge_abs = np.full(len(train), np.nan)
    for i, m in enumerate(fold_masks):
        oof_ridge_abs[m] = oof_ridge_dev[m] + fold_nats[i][tc]
    ridge_oof_r2 = r2_score(y_true_train, oof_ridge_abs)

    # ── Conformal intervals from Ridge LOO ──
    # These are the raw prediction errors of the Ridge on held-out elections
    oof_ridge_errors = y_true_train - oof_ridge_abs
    abs_errors = np.abs(oof_ridge_errors[~np.isnan(oof_ridge_errors)])
    # Quantiles for different confidence levels
    conformal_q = {
        50: float(np.percentile(abs_errors, 50)),
        80: float(np.percentile(abs_errors, 80)),
        90: float(np.percentile(abs_errors, 90)),
        95: float(np.percentile(abs_errors, 95)),
    }

    # ── Step 3: GP on OOF residuals ──
    X_tr_gp = X_tr_raw
    X_v_gp = X_v_raw

    oof_combined = np.full(len(train), np.nan)
    oof_gp_std = np.full(len(train), np.nan)

    for f_idx, held_mask in enumerate(fold_masks):
        train_ok = ~held_mask & ~np.isnan(oof_residuals)
        if train_ok.sum() < 100:
            continue
        nystroem, bridge = _fit_gp(X_tr_gp[train_ok], oof_residuals[train_ok], gp_cfg)
        gp_pred, gp_std = _predict_gp(nystroem, bridge, X_tr_gp[held_mask])
        oof_combined[held_mask] = (
            oof_ridge_dev[held_mask] + gp_pred + fold_nats[f_idx][tc]
        )
        oof_gp_std[held_mask] = gp_std

    ok = ~np.isnan(oof_combined)
    gp_oof_r2 = (
        r2_score(y_true_train[ok], oof_combined[ok]) if ok.sum() > 100 else ridge_oof_r2
    )

    # Conformal calibration of GP uncertainty
    if ok.sum() > 100:
        oof_actual_err = y_true_train[ok] - oof_combined[ok]
        z_scores = oof_actual_err / np.clip(oof_gp_std[ok], 1e-6, None)
        cal_scale = float(np.percentile(np.abs(z_scores), 90) / 1.645)
    else:
        cal_scale = 1.0

    # ── Step 4: Final val prediction with uncertainty ──
    ok_res = ~np.isnan(oof_residuals)
    nystroem_final, bridge_final = _fit_gp(
        X_tr_gp[ok_res], oof_residuals[ok_res], gp_cfg
    )
    gp_val_pred, gp_val_std = _predict_gp(nystroem_final, bridge_final, X_v_gp)

    combined_val_pred = ridge_val_pred + gp_val_pred
    combined_val_r2 = r2_score(y_true_val, combined_val_pred)
    calibrated_std = gp_val_std * cal_scale

    return {
        "ridge_oof_r2": ridge_oof_r2,
        "ridge_val_r2": ridge_val_r2,
        "gp_oof_r2": gp_oof_r2,
        "gp_val_r2": combined_val_r2,
        "ridge_alpha": ridge_full.alpha_,
        "n_folds": len(fold_masks),
        "cal_scale": cal_scale,
        "conformal_q": conformal_q,
        # Predictions
        "ridge_val_pred": ridge_val_pred,
        "val_pred": combined_val_pred,
        "val_std": calibrated_std,
        "val_true": y_true_val,
    }


def main():
    data_dir = Path("data")
    t0 = time.time()

    # ── Load data ──
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)
    type_cols = add_election_type_onehot(df)

    df_ext, ext_indicators, ext_nm, ext_pf = build_extended_data(data_dir)
    ext_type_cols = add_election_type_onehot(df_ext)

    # ── National estimates (raw poll avg, matching current pipeline) ──
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    est["Abstention"] = abs_pred

    print("\n" + "=" * 70)
    print("STAGE 1: National estimates (raw poll avg + gap model)")
    print("=" * 70)
    for b in TARGET_COLS:
        print(f"    {b:20s}: {est[b]:.1f}%")

    ext_est = dict(est)

    # ── Feature groups ──
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    # ── Datasets ──
    df_v1 = df.dropna(subset=demo_indicators)
    df_v1_2lag = df_v1.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)

    df_legi = df[df["election_type"] == VAL_TYPE].copy()
    df_legi_v1 = df_legi.dropna(subset=demo_indicators)
    df_legi_v1_2 = df_legi_v1.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)

    ext_raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    ext_raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    ext_dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    ext_dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    ext_v1 = df_ext.dropna(subset=ext_indicators)
    ext_v1_2 = ext_v1.dropna(
        subset=ext_raw_lag1 + ext_raw_lag2 + ext_dev_lag1 + ext_dev_lag2
    )

    nd_ct = dev_lag1 + dev_lag2 + type_cols
    nd_legi = dev_lag1 + dev_lag2
    ext_nd = ext_dev_lag1 + ext_dev_lag2 + ext_type_cols

    datasets = {
        "ct_v1_2": df_v1_2lag,
        "legi_v1_2": df_legi_v1_2,
        "ext_v1_2": ext_v1_2,
    }
    feat_maps = {
        "ct": (demo_indicators, nd_ct, demo_indicators + nd_ct, national_means),
        "legi": (demo_indicators, nd_legi, demo_indicators + nd_legi, national_means),
        "ext": (ext_indicators, ext_nd, ext_indicators + ext_nd, ext_nm),
    }
    est_maps = {"ct": est, "legi": est, "ext": ext_est}

    # ── Run per block ──
    print("\n" + "=" * 70)
    print("STAGE 2: Ridge + GP residual boost (Nystroem + BayesianRidge)")
    print(
        f"  GP: {GP_CONFIG['n_components']} Nystroem components, "
        f"RBF kernel, BayesianRidge"
    )
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

        print(f"\n{'─' * 60}")
        print(f"  {ABBR[tc]} ({tc}) — Ridge: {ridge_name}")
        print(
            f"  train={len(train_clean):,} val={len(val_clean):,} "
            f"feat={len(all_cols)} "
            f"folds={len(train_clean['date_float'].unique())}"
        )

        t1 = time.time()
        res = run_gp_residual_boost(
            tc, train_clean, val_clean, all_cols, demo_cols, nat_est, nm, cfg
        )
        elapsed = time.time() - t1

        results[tc] = res
        delta_gp = res["gp_val_r2"] - res["ridge_val_r2"]
        print(
            f"  Ridge-only:  OOF={res['ridge_oof_r2']:.4f}  "
            f"Val={res['ridge_val_r2']:.4f}"
        )
        print(
            f"  Ridge+GP:    OOF={res['gp_oof_r2']:.4f}  "
            f"Val={res['gp_val_r2']:.4f}  "
            f"(delta={delta_gp:+.4f})"
        )
        print(f"  Conformal intervals (Ridge LOO):")
        for pct, q in sorted(res["conformal_q"].items()):
            print(f"    {pct}%: ±{q:.1f}pp")
        med_std = float(np.median(res["val_std"]))
        print(
            f"  GP calibrated uncertainty: median ±{med_std:.2f}pp  "
            f"(cal_scale={res['cal_scale']:.2f})"
        )
        print(f"  ({elapsed:.0f}s)")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    # R² comparison
    print(
        f"\n{'Block':20s} {'Ridge':>8s} {'+GP':>8s} {'delta':>7s} "
        f"{'prev':>7s} {'vs prev':>8s}"
    )
    print("-" * 62)
    for tc in TARGET_COLS:
        r = results[tc]
        dg = r["gp_val_r2"] - r["ridge_val_r2"]
        vs = max(r["gp_val_r2"], r["ridge_val_r2"]) - PREV_RAW[tc]
        best = max(r["gp_val_r2"], r["ridge_val_r2"])
        tag = "+GP" if r["gp_val_r2"] > r["ridge_val_r2"] else "Ridge"
        mark = "BEAT" if vs > 0.0005 else ("~tie" if abs(vs) <= 0.0005 else "")
        print(
            f"  {tc:20s} {r['ridge_val_r2']:7.4f} {r['gp_val_r2']:7.4f} "
            f"{dg:+.4f} {PREV_RAW[tc]:6.4f} {vs:+.4f} [{tag}] {mark}"
        )

    # Prediction intervals
    print(f"\n  PREDICTION INTERVALS (from Ridge LOO conformal):")
    print(f"  {'Block':20s} {'50%':>7s} {'80%':>7s} {'90%':>7s} {'95%':>7s}")
    print(f"  {'-' * 48}")
    for tc in TARGET_COLS:
        q = results[tc]["conformal_q"]
        print(f"  {tc:20s} ±{q[50]:5.1f} ±{q[80]:5.1f} ±{q[90]:5.1f} ±{q[95]:5.1f}")

    # Coverage check on val
    print(f"\n  COVERAGE CHECK (actual coverage on 2024 val):")
    print(f"  {'Block':20s} {'50%':>7s} {'80%':>7s} {'90%':>7s} {'95%':>7s}")
    print(f"  {'-' * 48}")
    for tc in TARGET_COLS:
        r = results[tc]
        y = r["val_true"]
        pred = r["ridge_val_pred"]  # use Ridge for conformal (GP may hurt)
        q = r["conformal_q"]
        covs = []
        for pct in [50, 80, 90, 95]:
            lo = pred - q[pct]
            hi = pred + q[pct]
            cov = float(np.mean((y >= lo) & (y <= hi))) * 100
            covs.append(f"{cov:5.1f}%")
        print(f"  {tc:20s} {'  '.join(covs)}")

    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
