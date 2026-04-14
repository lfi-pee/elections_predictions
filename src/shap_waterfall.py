"""SHAP waterfall plots for the best pre-registered Ridge+XGB models.

Trains the LOO-selected best model for each block (from preregistered.py),
adds XGB residual boosting on LOO residuals (same as preregistered.py Step 2),
then generates SHAP waterfall plots for individual BV predictions.

Ridge SHAP: since Ridge is linear and PCA is a linear transform, we multiply
through the PCA loadings to attribute SHAP values back to the original 52
demographic indicators — no opaque "DemoPCA_k" components.

XGB SHAP: computed via shap.TreeExplainer on the HistGradientBoosting model
trained on LOO Ridge residuals. Both decompositions share the same original
feature space and are summed to produce the combined waterfall.

Math (Ridge):  effective_coef[i] = sum_j(ridge_coef[j] * pca_components[j, i])
               shap_ridge[i] = effective_coef[i] * x_standardized[i]
Math (combined): shap[i] = shap_ridge[i] + shap_xgb[i]
                 base = ridge_base + xgb_base

Usage:
    python3 -m src.shap_waterfall
    python3 -m src.shap_waterfall --bv-idx 42
    python3 -m src.shap_waterfall --location "01001_0001"
    python3 -m src.shap_waterfall --max-display 20
    python3 -m src.shap_waterfall --output-dir plots/shap
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor

import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cross_type_dev import (
    load_cross_type_data, add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS, ABBR, VAL_DATE, VAL_TYPE, TARGET_COLS,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.beat_it import build_extended_data

ALPHA_GRID = np.logspace(-2, 6, 20)

# Fixed XGB config for residual boosting (matches preregistered.py)
XGB_FIXED = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 100,
             "min_samples_leaf": 500, "l2_regularization": 1.0}

# Best models per block (LOO-selected in preregistered.py)
BEST_MODELS = {
    "Gauche":         "Legi-PCA5-devlag",
    "Centre+Droite":  "Legi-PCA7-devlag",
    "Extreme_Droite": "CT-PCA5-devlag",
    "Abstention":     "CT-PCA10-devlag",
}


def split_tv(df):
    val_mask = (
        np.isclose(df["date_float"], VAL_DATE, atol=1e-3)
        & (df["election_type"] == VAL_TYPE)
    )
    return df[~val_mask], df[val_mask]


def _clean_name(col):
    """Shorten column names for display."""
    return (col
            .replace("dev_", "")
            .replace("type_", "")
            .replace("Pct_", "")
            .replace("Taux_", "")
            .replace("_lag", " lag"))


def train_and_explain(block, df, df_ext, demo_indicators, ext_indicators,
                      national_means, ext_nm, est, ext_est):
    """Train the best Ridge+XGB model for a block.

    Returns combined SHAP values (Ridge + XGB residual boost) decomposed
    back to original features, plus all objects needed for waterfall plots.
    """
    model_name = BEST_MODELS[block]

    geo_time = ["latitude", "longitude", "date_float"]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]

    if model_name.startswith("Ext-"):
        type_cols = add_election_type_onehot(df_ext)
        non_demo = geo_time + dev_lag1 + dev_lag2 + type_cols
        feat_cols = ext_indicators + non_demo
        data = df_ext.dropna(subset=ext_indicators)
        data = data.dropna(
            subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)
        nat_est, indicators = ext_est, ext_indicators
    elif model_name.startswith("Legi-"):
        non_demo = geo_time + dev_lag1 + dev_lag2
        feat_cols = demo_indicators + non_demo
        data = df[df["election_type"] == VAL_TYPE].copy()
        data = data.dropna(subset=demo_indicators)
        data = data.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)
        nat_est, indicators = est, demo_indicators
    else:
        type_cols = add_election_type_onehot(df)
        non_demo = geo_time + dev_lag1 + dev_lag2 + type_cols
        feat_cols = demo_indicators + non_demo
        data = df.dropna(subset=demo_indicators)
        data = data.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)
        nat_est, indicators = est, demo_indicators

    pca_k = None
    for part in model_name.split("-"):
        if part.startswith("PCA"):
            pca_k = int(part[3:])

    n_demo = len(indicators)
    non_demo_cols = [c for c in feat_cols if c not in indicators]

    # Split train/val
    train, val = split_tv(data)
    ok_tr = train[feat_cols].notna().all(axis=1)
    ok_v = val[feat_cols].notna().all(axis=1)
    train, val = train[ok_tr], val[ok_v]

    # Standardize
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(
        train[feat_cols].values.astype(np.float64))
    X_v_scaled = scaler.transform(
        val[feat_cols].values.astype(np.float64))

    # PCA on demographics (if needed) and train Ridge
    if pca_k is not None:
        pca_obj = PCA(n_components=pca_k).fit(X_tr_scaled[:, :n_demo])
        X_tr_pca = np.hstack([
            pca_obj.transform(X_tr_scaled[:, :n_demo]),
            X_tr_scaled[:, n_demo:]])
        X_v_pca = np.hstack([
            pca_obj.transform(X_v_scaled[:, :n_demo]),
            X_v_scaled[:, n_demo:]])
    else:
        pca_obj = None
        X_tr_pca, X_v_pca = X_tr_scaled, X_v_scaled

    dev_y = train[f"dev_{block}"].values.astype(np.float64)
    ridge = RidgeCV(alphas=ALPHA_GRID)
    ridge.fit(X_tr_pca, dev_y)

    # ── Compute effective coefficients in original (standardized) space ──
    # Ridge coefs are in PCA space: [pca_0..pca_k, non_demo_0..non_demo_m]
    # We want coefs in original space: [demo_0..demo_n, non_demo_0..non_demo_m]
    if pca_k is not None:
        coef_pca = ridge.coef_[:pca_k]          # (k,)
        coef_rest = ridge.coef_[pca_k:]          # (m,)
        # effective_demo_coef[i] = sum_j(coef_pca[j] * components[j, i])
        coef_demo_eff = pca_obj.components_.T @ coef_pca  # (n_demo,)
        effective_coef = np.concatenate([coef_demo_eff, coef_rest])
    else:
        effective_coef = ridge.coef_

    # ── Compute SHAP values in original space ──
    # For linear model: shap[i] = coef[i] * (x[i] - E_train[x[i]])
    # After StandardScaler, E_train[x] = 0 for training data.
    # For val data: shap[i] = coef[i] * x_scaled[i]
    # But we should use the exact training mean (= 0 after scaling).
    # More precisely: shap[i] = coef[i] * (x_v_scaled[i] - mean_train_scaled[i])
    # mean_train_scaled = 0 by construction of StandardScaler.
    train_mean_scaled = X_tr_scaled.mean(axis=0)  # ≈ 0, but use exact value
    base_value = float(ridge.intercept_)
    if pca_k is not None:
        # Adjust base value for PCA centering:
        # In PCA space, the mean of z = pca.transform(X) is pca.transform(mean(X))
        # The Ridge intercept absorbs the PCA-space mean, but when we decompose
        # back, we need: base = intercept + coef_pca @ pca.transform(mean_train_demo)
        # + coef_rest @ mean_train_rest
        # Since mean_train_scaled ≈ 0, pca.transform(0) = -pca.mean_
        # But more precisely, let's compute directly:
        mean_pca_tr = X_tr_pca.mean(axis=0)
        base_value = float(ridge.intercept_ + ridge.coef_ @ mean_pca_tr)

    # SHAP values for each val sample
    shap_vals_v = (X_v_scaled - train_mean_scaled) * effective_coef  # (n_val, n_feat)

    # Sanity: compare sum of SHAP + base with Ridge prediction
    ridge_pred_v = ridge.predict(X_v_pca)
    recon = shap_vals_v.sum(axis=1) + base_value
    max_err = np.max(np.abs(ridge_pred_v - recon))
    print(f"  Ridge SHAP decomposition max error: {max_err:.2e}")

    # ── XGB residual boost on LOO residuals ──
    train_dates_arr = train["date_float"].values
    train_types_arr = train["election_type"].values
    train_td = (
        train[["election_type", "date_float"]]
        .drop_duplicates().sort_values("date_float").values.tolist()
    )
    fold_masks = []
    for etype, ddate in train_td:
        mask = (np.isclose(train_dates_arr, ddate, atol=1e-3)
                & (train_types_arr == etype))
        fold_masks.append(mask)

    oof_ridge_dev = np.full(len(train), np.nan)
    for f_idx, held_mask in enumerate(fold_masks):
        not_held = ~held_mask
        if pca_k is not None:
            pca_fold = PCA(n_components=pca_k).fit(
                X_tr_scaled[not_held, :n_demo])
            X_ft = np.hstack([
                pca_fold.transform(X_tr_scaled[not_held, :n_demo]),
                X_tr_scaled[not_held, n_demo:]])
            X_fh = np.hstack([
                pca_fold.transform(X_tr_scaled[held_mask, :n_demo]),
                X_tr_scaled[held_mask, n_demo:]])
        else:
            X_ft = X_tr_scaled[not_held]
            X_fh = X_tr_scaled[held_mask]
        ridge_fold = Ridge(alpha=ridge.alpha_, solver="cholesky")
        ridge_fold.fit(X_ft, dev_y[not_held])
        oof_ridge_dev[held_mask] = ridge_fold.predict(X_fh)

    oof_residuals = dev_y - oof_ridge_dev
    ok_res = ~np.isnan(oof_residuals)
    xgb = HistGradientBoostingRegressor(
        early_stopping=False, random_state=42, **XGB_FIXED)
    xgb.fit(X_tr_scaled[ok_res], oof_residuals[ok_res])

    xgb_explainer = shap.TreeExplainer(xgb)
    xgb_shap_expl = xgb_explainer(X_v_scaled)
    xgb_shap_v = xgb_shap_expl.values
    xgb_base = float(np.atleast_1d(xgb_shap_expl.base_values)[0])

    # Combined SHAP = Ridge + XGB
    shap_vals_v = shap_vals_v + xgb_shap_v
    base_value = base_value + xgb_base

    combined_pred = ridge_pred_v + xgb.predict(X_v_scaled)
    recon = shap_vals_v.sum(axis=1) + base_value
    max_err = np.max(np.abs(combined_pred - recon))
    print(f"  Combined SHAP decomposition max error: {max_err:.2e}")

    # All original feature names (readable)
    all_names = [_clean_name(c) for c in list(indicators) + non_demo_cols]

    # Original (unscaled) feature values for display
    val_raw = val[feat_cols].values.astype(np.float64)

    return {
        "shap_values": shap_vals_v,
        "base_value": base_value,
        "feature_names": all_names,
        "val_raw": val_raw,
        "val": val,
        "ridge": ridge,
        "xgb": xgb,
        "X_v_pca": X_v_pca,
        "national_est": nat_est.get(block, 0.0),
        "model_name": model_name,
        "n_train": len(X_tr_scaled),
        "n_feat": len(all_names),
        "alpha": ridge.alpha_,
    }


def make_waterfall(info, block, bv_idx, max_display=15, output=None):
    """Generate SHAP waterfall plot for one BV and one block."""
    sv = info["shap_values"][bv_idx]          # (n_feat,)
    base = info["base_value"]
    names = info["feature_names"]
    raw_vals = info["val_raw"][bv_idx]         # original feature values
    nat_est = info["national_est"]
    val = info["val"]

    bv_row = val.iloc[bv_idx]
    location = bv_row.get("location", "?")
    actual = bv_row[block]
    dev_pred = float(sv.sum() + base)
    final_pred = dev_pred + nat_est

    explanation = shap.Explanation(
        values=sv,
        base_values=base,
        data=raw_vals,
        feature_names=names,
    )

    title = (f"{block} — BV {location}\n"
             f"Dev pred: {dev_pred:+.1f}pp | "
             f"National est: {nat_est:.1f}% | "
             f"Final: {final_pred:.1f}% | "
             f"Actual: {actual:.1f}%")

    fig = plt.figure(figsize=(10, max(6, max_display * 0.4)))
    shap.plots.waterfall(explanation, max_display=max_display, show=False)
    plt.gca().set_title(title, fontsize=11, pad=12)
    plt.tight_layout()

    plt.savefig(output, dpi=150, bbox_inches="tight")
    print(f"  Saved: {output}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="SHAP waterfall plots for best Ridge models")
    parser.add_argument("--bv-idx", type=int, default=None,
                        help="Index in the validation set (default: random)")
    parser.add_argument("--location", type=str, default=None,
                        help="BV location code (e.g. '01001_0001')")
    parser.add_argument("--block", type=str, default=None,
                        help="Single block to plot (default: all 4)")
    parser.add_argument("--max-display", type=int, default=15,
                        help="Max features to show (default: 15)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots")
    args = parser.parse_args()

    data_dir = Path("data")

    print("Loading data...")
    df, demo_indicators, national_means, poll_feats = \
        load_cross_type_data(data_dir)
    _ = add_election_type_onehot(df)

    df_ext, ext_indicators, ext_nm, ext_pf = build_extended_data(data_dir)
    _ = add_election_type_onehot(df_ext)

    # National estimates
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    est["Abstention"] = abs_pred
    ext_est = dict(est)

    blocks = [args.block] if args.block else list(TARGET_COLS)

    out_dir = Path(args.output_dir or "plots/shap")
    out_dir.mkdir(parents=True, exist_ok=True)

    bv_idx_resolved = None
    for block in blocks:
        print(f"\n{'='*60}")
        print(f"  {block} — model: {BEST_MODELS[block]}")
        print(f"{'='*60}")

        info = train_and_explain(
            block, df, df_ext, demo_indicators, ext_indicators,
            national_means, ext_nm, est, ext_est)

        val = info["val"]
        print(f"  Train: {info['n_train']:,}  Val: {len(val):,}  "
              f"Features: {info['n_feat']}  alpha={info['alpha']:.1f}")

        # Resolve BV index
        if args.location:
            matches = val.index[val["location"] == args.location]
            if len(matches) == 0:
                print(f"  WARNING: location '{args.location}' not found in "
                      f"{block} val set. Using random BV.")
                bv_idx_resolved = np.random.randint(len(val))
            else:
                bv_idx_resolved = val.index.get_loc(matches[0])
        elif args.bv_idx is not None:
            bv_idx_resolved = min(args.bv_idx, len(val) - 1)
        elif bv_idx_resolved is None:
            bv_idx_resolved = np.random.randint(len(val))

        loc = val.iloc[bv_idx_resolved].get("location", "unknown")
        out_path = str(out_dir / f"shap_{ABBR[block]}_{loc}.png")

        make_waterfall(info, block, bv_idx_resolved,
                       max_display=args.max_display, output=out_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
