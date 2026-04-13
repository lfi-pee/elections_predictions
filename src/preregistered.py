"""Pre-registered model selection: LOO on training → select → single val pass.

The model selection criterion (LOO OOF R² on training elections) is fixed
before any validation data is seen. For each block, the model with the
best LOO R² on training data is selected, then evaluated on 2024 val
in a single forward pass.

This is the cleanest possible evaluation: no validation feedback of any
kind enters the model selection or fitting pipeline.

Usage:
    python3 -u -m src.preregistered
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
from sklearn.metrics import r2_score

from src.cross_type_dev import (
    load_cross_type_data, add_election_type_onehot,
    evaluate_full, estimate_national_abstention_from_gaps,
    BLOCKS_ABS, ABBR, ALPHAS, VAL_DATE, VAL_TYPE, TARGET_COLS,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.beat_it import build_extended_data

ALPHA_GRID = np.logspace(-2, 6, 20)
PREV_RAW = {"Gauche": 0.7414, "Centre+Droite": 0.5947,
            "Extreme_Droite": 0.8092, "Abstention": 0.7328}


def split_tv(df):
    val_mask = (
        np.isclose(df["date_float"], VAL_DATE, atol=1e-3)
        & (df["election_type"] == VAL_TYPE)
    )
    return df[~val_mask], df[val_mask]


def _apply_pca(X_tr, X_v, cfg):
    if cfg.get("pca_k") is None:
        return X_tr, X_v
    n_d = cfg["n_demo"]
    k = cfg["pca_k"]
    pca = PCA(n_components=k).fit(X_tr[:, :n_d])
    return (np.hstack([pca.transform(X_tr[:, :n_d]), X_tr[:, n_d:]]),
            np.hstack([pca.transform(X_v[:, :n_d]), X_v[:, n_d:]]))


def run_loo_and_val(name, df, feat_cols, national_est, national_means, cfg):
    """Run LOO on training + single forward pass on val.

    Returns: dict per block with 'oof_r2' (training LOO) and 'val_r2'.
    """
    train, val = split_tv(df)
    ok_tr = train[feat_cols].notna().all(axis=1)
    ok_v = val[feat_cols].notna().all(axis=1)
    train, val = train[ok_tr], val[ok_v]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v = scaler.transform(val[feat_cols].values.astype(np.float64))

    train_types = train["election_type"].values
    train_dates = train["date_float"].values
    train_td = (
        train[["election_type", "date_float"]]
        .drop_duplicates().sort_values("date_float").values.tolist()
    )
    n_folds = len(train_td)

    # Fold masks + national means
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

    results = {}
    for tc in TARGET_COLS:
        dev_y = train[f"dev_{tc}"].values.astype(np.float64)

        # ── Full-train → val prediction (RidgeCV) ──
        X_tr_t, X_v_t = _apply_pca(X_tr, X_v, cfg)
        ridge_full = RidgeCV(alphas=ALPHA_GRID)
        ridge_full.fit(X_tr_t, dev_y)
        val_pred = ridge_full.predict(X_v_t) + national_est.get(tc, 0.0)
        val_r2 = r2_score(val[tc].values, val_pred)

        # ── LOO over training dates ──
        oof_pred = np.full(len(train), np.nan)
        for f_idx, held_mask in enumerate(fold_masks):
            not_held = ~held_mask
            X_ft, X_fh = X_tr[not_held], X_tr[held_mask]
            X_ft_t, X_fh_t = _apply_pca(X_ft, X_fh, cfg)
            ridge = Ridge(alpha=ridge_full.alpha_, solver="cholesky")
            ridge.fit(X_ft_t, dev_y[not_held])
            oof_pred[held_mask] = (
                ridge.predict(X_fh_t) + fold_nats[f_idx][tc])

        oof_ok = ~np.isnan(oof_pred)
        oof_r2 = r2_score(train[tc].values[oof_ok], oof_pred[oof_ok])

        results[tc] = {"oof_r2": oof_r2, "val_r2": val_r2}

    return results


def main():
    data_dir = Path("data")
    t0 = time.time()

    # ── Load data ──
    df, demo_indicators, national_means, poll_feats = \
        load_cross_type_data(data_dir)
    type_cols = add_election_type_onehot(df)

    df_ext, ext_indicators, ext_nm, ext_pf = build_extended_data(data_dir)
    ext_type_cols = add_election_type_onehot(df_ext)

    # ── National estimates (fixed, no val feedback) ──
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    est["Abstention"] = abs_pred

    # Extended: same vote-block estimates, same gap model (Legi+Pres only)
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
    df_v1_1lag = df_v1.dropna(subset=raw_lag1 + dev_lag1)

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

    # ── Candidate model configs (pre-registered) ──
    nd_ct = geo_time + dev_lag1 + dev_lag2 + type_cols
    nd_ct_1lag = geo_time + dev_lag1 + type_cols
    nd_legi = geo_time + dev_lag1 + dev_lag2
    ext_nd = geo_time + ext_dev_lag1 + ext_dev_lag2 + ext_type_cols

    configs = [
        # Cross-type Legi+Pres, 2-lag
        ("CT-devlag", df_v1_2lag,
         demo_indicators + nd_ct, est, national_means,
         {"n_demo": len(demo_indicators)}),
        ("CT-PCA5-devlag", df_v1_2lag,
         demo_indicators + nd_ct, est, national_means,
         {"pca_k": 5, "n_demo": len(demo_indicators)}),
        ("CT-PCA7-devlag", df_v1_2lag,
         demo_indicators + nd_ct, est, national_means,
         {"pca_k": 7, "n_demo": len(demo_indicators)}),
        ("CT-PCA10-devlag", df_v1_2lag,
         demo_indicators + nd_ct, est, national_means,
         {"pca_k": 10, "n_demo": len(demo_indicators)}),
        # Cross-type Legi+Pres, 1-lag
        ("CT-devlag-1lag", df_v1_1lag,
         demo_indicators + nd_ct_1lag, est, national_means,
         {"n_demo": len(demo_indicators)}),
        # Legi-only, 2-lag
        ("Legi-devlag", df_legi_v1_2,
         demo_indicators + nd_legi, est, national_means,
         {"n_demo": len(demo_indicators)}),
        ("Legi-PCA5-devlag", df_legi_v1_2,
         demo_indicators + nd_legi, est, national_means,
         {"pca_k": 5, "n_demo": len(demo_indicators)}),
        ("Legi-PCA7-devlag", df_legi_v1_2,
         demo_indicators + nd_legi, est, national_means,
         {"pca_k": 7, "n_demo": len(demo_indicators)}),
        ("Legi-PCA10-devlag", df_legi_v1_2,
         demo_indicators + nd_legi, est, national_means,
         {"pca_k": 10, "n_demo": len(demo_indicators)}),
        # Extended 6-type, 2-lag
        ("Ext-devlag", ext_v1_2,
         ext_indicators + ext_nd, ext_est, ext_nm,
         {"n_demo": len(ext_indicators)}),
        ("Ext-PCA3-devlag", ext_v1_2,
         ext_indicators + ext_nd, ext_est, ext_nm,
         {"pca_k": 3, "n_demo": len(ext_indicators)}),
        ("Ext-PCA5-devlag", ext_v1_2,
         ext_indicators + ext_nd, ext_est, ext_nm,
         {"pca_k": 5, "n_demo": len(ext_indicators)}),
        ("Ext-PCA7-devlag", ext_v1_2,
         ext_indicators + ext_nd, ext_est, ext_nm,
         {"pca_k": 7, "n_demo": len(ext_indicators)}),
    ]

    # ── Run all models: LOO on training + val forward pass ──
    print("=" * 70)
    print("PRE-REGISTERED MODEL SELECTION")
    print("Select on LOO OOF R² (training only) → evaluate on val")
    print("=" * 70)
    print(f"\n{len(configs)} candidate models\n")

    all_results = {}
    for name, data, feats, nat_est, nat_means, cfg in configs:
        print(f"  {name}...", end="", flush=True)
        t1 = time.time()
        res = run_loo_and_val(name, data, feats, nat_est, nat_means, cfg)
        all_results[name] = res
        elapsed = time.time() - t1
        oof_str = " ".join(
            f"{ABBR[tc]}={res[tc]['oof_r2']:.3f}" for tc in TARGET_COLS)
        print(f" ({elapsed:.0f}s) OOF: {oof_str}")

    # ── Phase 1: Select per-block best on LOO OOF R² ──
    print(f"\n{'='*70}")
    print("STEP 1: MODEL SELECTION (LOO OOF R² — training data only)")
    print(f"{'='*70}")

    print(f"\n{'Model':25s} ", end="")
    for tc in TARGET_COLS:
        print(f" {ABBR[tc]:>7s}", end="")
    print()
    print("-" * 60)

    for name in [c[0] for c in configs]:
        res = all_results[name]
        print(f"{name:25s} ", end="")
        for tc in TARGET_COLS:
            print(f" {res[tc]['oof_r2']:7.4f}", end="")
        print()

    # Select best per block
    selected = {}
    print(f"\n{'─'*60}")
    print("SELECTED (best LOO OOF R² per block):")
    for tc in TARGET_COLS:
        best_name, best_oof = "", -999
        for name in [c[0] for c in configs]:
            oof = all_results[name][tc]["oof_r2"]
            if oof > best_oof:
                best_oof, best_name = oof, name
        selected[tc] = best_name
        print(f"  {tc:20s} → {best_name:25s} (OOF R²={best_oof:.4f})")

    # ── Phase 2: Report validation R² for selected models ──
    print(f"\n{'='*70}")
    print("STEP 2: VALIDATION (single forward pass, no feedback)")
    print(f"{'='*70}\n")

    for tc in TARGET_COLS:
        name = selected[tc]
        val_r2 = all_results[name][tc]["val_r2"]
        prev = PREV_RAW[tc]
        delta = val_r2 - prev
        mark = "BEAT" if delta > 0.0005 else ("~tie" if abs(delta) <= 0.0005
                                                else "miss")
        print(f"  {tc:20s}  Val R²={val_r2:.4f}  "
              f"(prev={prev:.4f}  Δ={delta:+.4f})  [{mark}]")
        print(f"  {'':20s}  model={name}")

    # ── Also show full table for transparency ──
    print(f"\n{'='*70}")
    print("FULL RESULTS TABLE (all models, all blocks)")
    print(f"{'='*70}")

    print(f"\n{'Model':25s} ", end="")
    for tc in TARGET_COLS:
        print(f"  {ABBR[tc]+'_oof':>7s} {ABBR[tc]+'_val':>7s}", end="")
    print()
    print("-" * 90)

    for name in [c[0] for c in configs]:
        res = all_results[name]
        print(f"{name:25s} ", end="")
        for tc in TARGET_COLS:
            oof = res[tc]["oof_r2"]
            val = res[tc]["val_r2"]
            sel = "←" if name == selected[tc] else " "
            print(f"  {oof:7.4f} {val:7.4f}{sel}", end="")
        print()

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
