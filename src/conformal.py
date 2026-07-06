"""Conformal Prediction Intervals for BV-level election predictions.

Uses LOO residuals from the Ridge deviation model as conformal scores.
Provides distribution-free prediction intervals with finite-sample
coverage guarantees.

Two calibration modes:
  - Oracle: uses actual national means in LOO → captures deviation error only
  - Realistic: uses poll-estimated national means → captures total error

Three interval types:
  - Standard: single quantile, same width for all BVs
  - Adaptive: width varies by predicted deviation magnitude (5 bins)
  - Both provide guaranteed marginal coverage >= 1 - alpha

The primary deliverable for a political party: per-BV predictions with
confidence intervals, saved as predictions_with_intervals.csv.

Usage:
    python3 -u -m src.conformal
"""

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import time
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, Ridge, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
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
from src.cross_type_ridge import TARGET_BLOCKS, build_slate_presence
from src.beat_it import build_extended_data

DOM3 = {"971", "972", "973", "974", "975", "976", "977", "978"}
POLY3 = {"986", "987", "988"}


def territory_class(loc):
    p2 = loc[:2]
    if p2 in ("2A", "2B"):
        return "corsica"
    if p2 == "ZZ":
        return "abroad"
    p3 = loc[:3]
    if p3 in DOM3:
        return "DOM"
    if p3 in POLY3:
        return "polynesia"
    return "mainland"


ALPHA_GRID = np.logspace(-2, 6, 20)

# Best Ridge configs per block. ED/Ab switched CT→legi-only per the task-correct
# held-out-legislative OOF (preconisations.md §9, select_ct_vs_legi.py): on
# identical rows, legi-only ≥ CT on every block; the prior CT pick for ED/Ab was
# an artifact of the production all-fold OOF being inflated by easy held-out
# presidential folds. Selection follows OOF (the 2024 val prefers CT for ED/Ab;
# per the pre-registered rule a single test year does not override the LOO).
BEST_RIDGE = {
    "Gauche": ("Legi-PCA5-devlag", "legi_v1_2", "legi", {"pca_k": 5}),
    "Centre+Droite": ("Legi-PCA7-devlag", "legi_v1_2", "legi", {"pca_k": 7}),
    "Extreme_Droite": ("Legi-PCA5-devlag", "legi_v1_2", "legi", {"pca_k": 5}),
    "Abstention": ("Legi-PCA5-devlag", "legi_v1_2", "legi", {"pca_k": 5}),
}

INTERVAL_ALPHAS = [0.20, 0.10, 0.05]  # 80%, 90%, 95% intervals


# ── Core conformal functions ────────────────────────────────────────


def conformal_quantile(abs_residuals, alpha):
    """Compute conformal quantile with finite-sample correction."""
    n = len(abs_residuals)
    q_level = min(1.0, (1 - alpha) * (1 + 1 / n))
    return float(np.quantile(abs_residuals, q_level))


def evaluate_coverage(y_true, lower, upper):
    """Compute empirical coverage and interval width metrics."""
    covered = (y_true >= lower) & (y_true <= upper)
    width = upper - lower
    return {
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(width)),
        "median_width": float(np.median(width)),
        "n_covered": int(np.sum(covered)),
        "n_total": len(y_true),
    }


def per_territory_intervals(
    cal_residuals, cal_territories, val_territories, alpha, min_n=50
):
    """Conformal intervals stratified by territory class.

    Each (mainland / DOM / polynesia / corsica / abroad) gets its own
    quantile of |residual|. Cells with fewer than min_n calibration
    points fall back to the global quantile.

    Returns half-widths (one per val BV).
    """
    abs_res = np.abs(cal_residuals)
    n_cal = len(cal_residuals)
    q_level = min(1.0, (1 - alpha) * (1 + 1 / n_cal))
    global_q = float(np.quantile(abs_res, q_level))

    per_terr_q = {}
    for cls in ("mainland", "DOM", "polynesia", "corsica", "abroad"):
        mask = cal_territories == cls
        if mask.sum() >= min_n:
            n_c = int(mask.sum())
            ql = min(1.0, (1 - alpha) * (1 + 1 / n_c))
            per_terr_q[cls] = float(np.quantile(abs_res[mask], ql))
        else:
            per_terr_q[cls] = global_q

    half_widths = np.array([per_terr_q.get(t, global_q) for t in val_territories])
    return half_widths, per_terr_q


def adaptive_intervals(cal_residuals, cal_dev_preds, val_dev_preds, alpha, n_bins=5):
    """Adaptive conformal intervals: width varies by deviation magnitude.

    BVs with extreme predicted deviations may have larger residuals
    (harder to predict precisely). Binning by |deviation prediction|
    gives tighter intervals for moderate BVs and wider for extremes.
    """
    abs_res = np.abs(cal_residuals)
    abs_dev = np.abs(cal_dev_preds)

    bin_edges = np.quantile(abs_dev, np.linspace(0, 1, n_bins + 1))
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    n_cal = len(cal_residuals)
    q_level = min(1.0, (1 - alpha) * (1 + 1 / n_cal))

    # Compute per-bin quantile
    bin_quantiles = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (abs_dev >= bin_edges[i]) & (abs_dev < bin_edges[i + 1])
        if mask.sum() >= 50:
            bin_quantiles[i] = np.quantile(abs_res[mask], q_level)
        else:
            # Fall back to global quantile
            bin_quantiles[i] = np.quantile(abs_res, q_level)

    # Assign each val BV to a bin
    val_abs_dev = np.abs(val_dev_preds)
    half_widths = np.zeros(len(val_dev_preds))
    for i in range(n_bins):
        mask = (val_abs_dev >= bin_edges[i]) & (val_abs_dev < bin_edges[i + 1])
        half_widths[mask] = bin_quantiles[i]

    return half_widths


# ── LOO + conformal pipeline ───────────────────────────────────────


def split_tv(df):
    val_mask = np.isclose(df["date_float"], VAL_DATE, atol=1e-3) & (
        df["election_type"] == VAL_TYPE
    )
    return df[~val_mask], df[val_mask]


def _apply_pca(X, pca, n_demo):
    if pca is None:
        return X
    return np.hstack([pca.transform(X[:, :n_demo]), X[:, n_demo:]])


def run_conformal_for_block(
    tc,
    train,
    val,
    feat_cols,
    demo_cols,
    national_est_val,
    national_means,
    cfg,
    loo_national_ests=None,
    present_train=None,
    present_val=None,
):
    """Run Ridge LOO, collect calibration residuals, compute intervals.

    Args:
        loo_national_ests: dict {(etype, date_round): {block: est}}.
            If provided (realistic mode), uses these for calibration.
            If None (oracle mode), uses actual national means.
        present_train / present_val: optional bool arrays (aligned to the train /
            val rows) marking whether this block fields a candidate. Where absent,
            the predicted share is forced to 0 — the block is off the ballot so its
            actual share is exactly 0. LOO-selected slate mask (mask_renorm_eval.py).

    Returns dict with calibration and interval data.
    """
    n_demo = len(demo_cols)
    pca_k = cfg.get("pca_k")

    # Scale
    scaler = StandardScaler()
    X_tr_raw = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v_raw = scaler.transform(val[feat_cols].values.astype(np.float64))

    # PCA
    if pca_k:
        pca_full = PCA(n_components=pca_k).fit(X_tr_raw[:, :n_demo])
    else:
        pca_full = None

    X_tr = _apply_pca(X_tr_raw, pca_full, n_demo)
    X_v = _apply_pca(X_v_raw, pca_full, n_demo)

    dev_y = train[f"dev_{tc}"].values.astype(np.float64)
    y_true_train = train[tc].values.astype(np.float64)
    y_true_val = val[tc].values.astype(np.float64)
    nat_est_val = national_est_val.get(tc, 0.0)

    # Build fold info
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
            {tc2: float(nm_row[tc2].iloc[0]) for tc2 in TARGET_COLS}
            if len(nm_row) > 0
            else {tc2: 0.0 for tc2 in TARGET_COLS}
        )

    # ── Full Ridge → val prediction ──
    ridge_full = RidgeCV(alphas=ALPHA_GRID)
    ridge_full.fit(X_tr, dev_y)
    val_dev_pred = ridge_full.predict(X_v)
    val_pred = val_dev_pred + nat_est_val
    if present_val is not None:
        val_pred = np.where(present_val, val_pred, 0.0)

    # ── LOO: collect calibration residuals ──
    cal_residuals = []
    cal_dev_preds = []
    cal_territories = []
    train_locations = train["location"].values

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
        dev_pred = ridge.predict(X_fh)

        # National level for this fold
        etype, ddate = train_td[f_idx]
        key = (etype, round(ddate, 3))
        if loo_national_ests and key in loo_national_ests:
            nat_level = loo_national_ests[key].get(tc, fold_nats[f_idx][tc])
        else:
            nat_level = fold_nats[f_idx][tc]

        abs_pred = dev_pred + nat_level
        if present_train is not None:
            abs_pred = np.where(present_train[held_mask], abs_pred, 0.0)
        actual = y_true_train[held_mask]
        residuals = actual - abs_pred

        cal_residuals.append(residuals)
        cal_dev_preds.append(dev_pred)
        cal_territories.append(
            np.array([territory_class(loc) for loc in train_locations[held_mask]])
        )

    cal_residuals = np.concatenate(cal_residuals)
    cal_dev_preds = np.concatenate(cal_dev_preds)
    cal_territories = np.concatenate(cal_territories)
    abs_cal_res = np.abs(cal_residuals)
    val_territories = np.array([territory_class(loc) for loc in val["location"].values])

    # ── Compute intervals for each alpha ──
    intervals = {}
    for alpha in INTERVAL_ALPHAS:
        pct = int(100 * (1 - alpha))

        # Standard
        q = conformal_quantile(abs_cal_res, alpha)
        std_lower = val_pred - q
        std_upper = val_pred + q
        std_cov = evaluate_coverage(y_true_val, std_lower, std_upper)

        # Adaptive
        adap_hw = adaptive_intervals(cal_residuals, cal_dev_preds, val_dev_pred, alpha)
        adap_lower = val_pred - adap_hw
        adap_upper = val_pred + adap_hw
        adap_cov = evaluate_coverage(y_true_val, adap_lower, adap_upper)

        # Per-territory stratified
        terr_hw, per_terr_q = per_territory_intervals(
            cal_residuals, cal_territories, val_territories, alpha
        )
        terr_lower = val_pred - terr_hw
        terr_upper = val_pred + terr_hw
        terr_cov = evaluate_coverage(y_true_val, terr_lower, terr_upper)

        intervals[pct] = {
            "alpha": alpha,
            "std_quantile": q,
            "std_lower": std_lower,
            "std_upper": std_upper,
            "std_coverage": std_cov,
            "adap_lower": adap_lower,
            "adap_upper": adap_upper,
            "adap_coverage": adap_cov,
            "terr_lower": terr_lower,
            "terr_upper": terr_upper,
            "terr_coverage": terr_cov,
            "terr_quantiles": per_terr_q,
        }

    return {
        "val_pred": val_pred,
        "val_dev_pred": val_dev_pred,
        "y_true_val": y_true_val,
        "val_locations": val["location"].values,
        "val_territories": val_territories,
        "cal_residuals": cal_residuals,
        "cal_dev_preds": cal_dev_preds,
        "cal_territories": cal_territories,
        "ridge_alpha": ridge_full.alpha_,
        "n_cal": len(cal_residuals),
        "n_folds": len(fold_masks),
        "r2": r2_score(y_true_val, val_pred),
        "intervals": intervals,
    }


# ── Main pipeline ──────────────────────────────────────────────────


def main():
    data_dir = Path("data")
    t0 = time.time()

    print("=" * 70)
    print("CONFORMAL PREDICTION INTERVALS")
    print("Distribution-free intervals with coverage guarantees")
    print("=" * 70)

    # ── Load data ──
    print("\nLoading data...")
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)
    type_cols = add_election_type_onehot(df)

    # ── Candidate-slate presence (LOO-selected mask; see mask_renorm_eval.py) ──
    elections_cache = data_dir / "baseline_cache" / "elections.parquet"
    presence = None
    if elections_cache.exists():
        print("Building candidate-slate presence table...")
        _el = pd.read_parquet(
            elections_cache,
            columns=[
                "metric_type", "location", "election_type",
                "date_float", "party", "candidate",
            ],
        )
        presence = build_slate_presence(_el)
        presence["date_float"] = presence["date_float"].round(5)
    else:
        print("  (elections cache absent — slate mask disabled)")

    def present_for(frame, block):
        """Bool array (aligned to frame rows): does `block` field a candidate?
        Abstention is never masked; unknown (loc, election) defaults to present."""
        if presence is None or block == "Abstention":
            return np.ones(len(frame), dtype=bool)
        key = frame[["location", "election_type", "date_float"]].copy()
        key["date_float"] = key["date_float"].round(5)
        col = f"present_{block}"
        merged = key.merge(
            presence[["location", "election_type", "date_float", col]],
            on=["location", "election_type", "date_float"],
            how="left",
        )
        return merged[col].fillna(True).to_numpy(dtype=bool)
    # ── National estimates for 2024 (raw polls; abstention from participation poll) ──
    print("\nSetting up national estimates...")
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)

    val_est = {
        b: float(
            poll_feats[
                np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
                & (poll_feats["election_type"] == VAL_TYPE)
            ][f"poll_{b}"].iloc[0]
        )
        for b in TARGET_BLOCKS
    }
    val_est["Abstention"] = abs_pred

    print(f"  2024 national estimates (raw polls + gap model):")
    for tc in TARGET_COLS:
        print(f"    {tc:20s}: {val_est[tc]:.2f}")

    # ── LOO national estimates for realistic calibration ──
    print("\nBuilding LOO national estimates for calibration...")
    loo_nat_ests = {}
    for _, pf_row in poll_feats.iterrows():
        etype = pf_row["election_type"]
        edate = round(float(pf_row["date_float"]), 3)
        if edate > VAL_DATE - 0.1:
            continue
        loo_nat_ests[(etype, edate)] = {
            b: float(pf_row[f"poll_{b}"]) for b in TARGET_BLOCKS
        }

    # Add gap model LOO for abstention
    nm = national_means.sort_values("date_float").reset_index(drop=True)
    train_nm = nm[nm["date_float"] < VAL_DATE - 0.1].copy()
    train_nm_gap = train_nm.copy()
    train_nm_gap["gap_years"] = train_nm_gap["date_float"].diff()
    train_nm_gap = train_nm_gap.dropna(subset=["gap_years"]).reset_index(drop=True)
    X_gap = train_nm_gap[["gap_years"]].values
    y_gap = train_nm_gap["Abstention"].values
    for i in range(len(train_nm_gap)):
        mask = np.ones(len(train_nm_gap), dtype=bool)
        mask[i] = False
        lr = LinearRegression().fit(X_gap[mask], y_gap[mask])
        pred_abs = float(lr.predict(X_gap[[i]])[0])
        etype = train_nm_gap.iloc[i]["election_type"]
        edate = round(float(train_nm_gap.iloc[i]["date_float"]), 3)
        key = (etype, edate)
        if key in loo_nat_ests:
            loo_nat_ests[key]["Abstention"] = pred_abs
    # First election (no gap) → use actual
    for _, row in train_nm.iterrows():
        key = (row["election_type"], round(float(row["date_float"]), 3))
        if key in loo_nat_ests and "Abstention" not in loo_nat_ests[key]:
            loo_nat_ests[key]["Abstention"] = row["Abstention"]

    # ── Build datasets ──
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    df_v1 = df.dropna(subset=demo_indicators)
    df_v1_2lag = df_v1.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)

    df_legi = df[df["election_type"] == VAL_TYPE].copy()
    df_legi_v1 = df_legi.dropna(subset=demo_indicators)
    df_legi_v1_2 = df_legi_v1.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)

    nd_ct = dev_lag1 + dev_lag2 + type_cols
    nd_legi = dev_lag1 + dev_lag2

    datasets = {
        "ct_v1_2": df_v1_2lag,
        "legi_v1_2": df_legi_v1_2,
    }
    feat_maps = {
        "ct": (demo_indicators, nd_ct, demo_indicators + nd_ct, national_means),
        "legi": (demo_indicators, nd_legi, demo_indicators + nd_legi, national_means),
    }

    # ── Run conformal for each block ──
    print(f"\n{'=' * 70}")
    print("CONFORMAL INTERVALS PER BLOCK")
    print(f"{'=' * 70}")

    all_results = {}
    output_frames = []

    for tc in TARGET_COLS:
        ridge_name, data_key, feat_key, cfg = BEST_RIDGE[tc]
        demo_cols, nd_cols, all_cols, nm = feat_maps[feat_key]
        data = datasets[data_key]
        cfg = dict(cfg)
        cfg["n_demo"] = len(demo_cols)

        train, val = split_tv(data)
        ok_tr = train[all_cols].notna().all(axis=1)
        ok_v = val[all_cols].notna().all(axis=1)
        train_clean = train[ok_tr].copy()
        val_clean = val[ok_v].copy()

        print(f"\n{'─' * 60}")
        print(f"  {ABBR[tc]} ({tc}) — {ridge_name}")
        print(
            f"  train={len(train_clean):,} val={len(val_clean):,} feat={len(all_cols)}"
        )

        t1 = time.time()

        present_train = present_for(train_clean, tc)
        present_val = present_for(val_clean, tc)
        n_masked = int((~present_val).sum())
        if n_masked:
            print(f"  slate mask: {n_masked:,} val BVs have no {ABBR[tc]} candidate")

        # Run with Bayesian national estimates (realistic calibration)
        res = run_conformal_for_block(
            tc,
            train_clean,
            val_clean,
            all_cols,
            demo_cols,
            val_est,
            nm,
            cfg,
            loo_national_ests=loo_nat_ests,
            present_train=present_train,
            present_val=present_val,
        )

        elapsed = time.time() - t1
        all_results[tc] = res

        print(
            f"  R² = {res['r2']:.4f}  "
            f"({res['n_cal']:,} cal residuals, {res['n_folds']} folds, "
            f"{elapsed:.0f}s)"
        )
        print(
            f"  Cal residual stats: "
            f"mean={np.mean(res['cal_residuals']):.2f}, "
            f"std={np.std(res['cal_residuals']):.2f}, "
            f"median|r|={np.median(np.abs(res['cal_residuals'])):.2f}"
        )

        for pct, iv in sorted(res["intervals"].items()):
            sc = iv["std_coverage"]
            ac = iv["adap_coverage"]
            tc_cov = iv["terr_coverage"]
            print(
                f"  {pct}% standard:  "
                f"coverage={sc['coverage']:.3f}  "
                f"mean_width={sc['mean_width']:.2f}pp  "
                f"q={iv['std_quantile']:.2f}pp"
            )
            print(
                f"  {pct}% adaptive:  "
                f"coverage={ac['coverage']:.3f}  "
                f"mean_width={ac['mean_width']:.2f}pp  "
                f"median_width={ac['median_width']:.2f}pp"
            )
            print(
                f"  {pct}% terr-strat:"
                f"coverage={tc_cov['coverage']:.3f}  "
                f"mean_width={tc_cov['mean_width']:.2f}pp  "
                f"q_per_class: "
                + ", ".join(
                    f"{c[0]}={iv['terr_quantiles'][c]:.1f}"
                    for c in ("mainland", "DOM", "polynesia", "corsica", "abroad")
                )
            )

        # Per-territory coverage at 90% (the most useful level)
        iv90 = res["intervals"][90]
        terr_arr = res["val_territories"]
        y = res["y_true_val"]
        print(f"  90% terr-strat coverage by class:")
        for cls in ("mainland", "DOM", "polynesia", "corsica", "abroad"):
            m = terr_arr == cls
            if m.sum() < 2:
                continue
            cov = float(
                np.mean(
                    (y[m] >= iv90["terr_lower"][m]) & (y[m] <= iv90["terr_upper"][m])
                )
            )
            w = float(np.mean(iv90["terr_upper"][m] - iv90["terr_lower"][m]))
            print(f"    {cls:12s} n={int(m.sum()):5d}  cov={cov:.3f}  width={w:.1f}pp")

        # Build output frame for this block
        iv90 = res["intervals"][90]
        iv95 = res["intervals"][95]
        iv80 = res["intervals"][80]
        block_df = pd.DataFrame(
            {
                "location": res["val_locations"],
                "block": tc,
                "prediction": res["val_pred"],
                "actual": res["y_true_val"],
                "residual": res["y_true_val"] - res["val_pred"],
                # Territory-stratified intervals (terr_*): each territory class gets its own
                # quantile of PAST-election LOO residuals (sparse classes fall back to the
                # global quantile). This is what the site describes, and it calibrates the
                # particular territories (DOM/Corse/étranger/Polynésie) on themselves rather
                # than letting mainland set their width. Calibration never touches 2024.
                "lower_80": iv80["terr_lower"],
                "upper_80": iv80["terr_upper"],
                "lower_90": iv90["terr_lower"],
                "upper_90": iv90["terr_upper"],
                "lower_95": iv95["terr_lower"],
                "upper_95": iv95["terr_upper"],
            }
        )
        output_frames.append(block_df)

    # ── Summary table ──
    print(f"\n{'=' * 70}")
    print("SUMMARY: Conformal Interval Coverage and Width")
    print(f"{'=' * 70}")

    print(
        f"\n  {'Block':15s} {'R²':>6s}  "
        f"{'90% cov':>8s} {'90% width':>10s}  "
        f"{'95% cov':>8s} {'95% width':>10s}"
    )
    print("  " + "-" * 65)
    for tc in TARGET_COLS:
        r = all_results[tc]
        iv90 = r["intervals"][90]
        iv95 = r["intervals"][95]
        print(
            f"  {ABBR[tc]:15s} {r['r2']:6.4f}  "
            f"{iv90['adap_coverage']['coverage']:8.3f} "
            f"{iv90['adap_coverage']['mean_width']:9.2f}pp  "
            f"{iv95['adap_coverage']['coverage']:8.3f} "
            f"{iv95['adap_coverage']['mean_width']:9.2f}pp"
        )

    # ── Save output ──
    output = pd.concat(output_frames, ignore_index=True)
    out_path = data_dir / "predictions_with_intervals.csv"
    output.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  Saved {len(output):,} rows to {out_path}")

    # ── Useful stats for the party ──
    print(f"\n{'=' * 70}")
    print("ACTIONABLE STATISTICS FOR POLITICAL STRATEGY")
    print(f"{'=' * 70}")

    for tc in TARGET_COLS:
        r = all_results[tc]
        pred = r["val_pred"]
        iv90 = r["intervals"][90]

        print(f"\n  {tc}:")
        print(f"    Mean predicted: {np.mean(pred):.1f}%")
        print(f"    Std predicted:  {np.std(pred):.1f}%")
        print(
            f"    Mean 90% interval width: {iv90['adap_coverage']['mean_width']:.1f}pp"
        )

        # Threshold analysis: BVs where block > X%
        for threshold in [30, 40, 50]:
            n_above = np.sum(pred > threshold)
            pct_above = 100.0 * n_above / len(pred)
            # Confidence: BVs where LOWER bound > threshold
            n_confident = np.sum(iv90["adap_lower"] > threshold)
            if n_above > 0:
                print(
                    f"    BVs > {threshold}%: {n_above:,} ({pct_above:.1f}%)  "
                    f"confidently > {threshold}%: {n_confident:,}"
                )

    # ── Battleground BVs (narrow margins) ──
    print(f"\n  Battleground BVs (prediction within ±3pp of key thresholds):")
    for tc in TARGET_COLS:
        r = all_results[tc]
        pred = r["val_pred"]
        national = val_est[tc]
        near_national = np.abs(pred - national) < 3.0
        print(
            f"    {ABBR[tc]}: {np.sum(near_national):,} BVs within "
            f"±3pp of national avg ({national:.1f}%)"
        )

    print(f"\n  Total time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
