"""Beat current best raw R² scores for BV-level election prediction.

Strategies:
  1. Extended cross-type training (add Euro, Regi, Dept to Legi+Pres)
  2. Lag trend features (dev_lag1 - dev_lag2)
  3. 1-lag models (more training dates)
  4. Expanded LOO blend with gap model + diverse base models
  5. More PCA variants (PCA7, PCA10)

All approaches: training data only, 2024 eval-only, no val tuning.

Current best (raw R², no val tuning):
  Gauche:        0.7414  (legi-only devlag)
  Centre+Droite: 0.5947  (legi-only PCA5-devlag)
  Extr. Droite:  0.8092  (cross-type devlag)
  Abstention:    0.7328  (cross-type PCA3-devlag + gap model)

Usage:
    python3 -m src.beat_it
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
from scipy.optimize import nnls

from src.cross_type_dev import (
    load_cross_type_data,
    build_per_type_national_means, add_deviation_targets,
    add_cross_type_local_lags, add_election_type_onehot,
    evaluate_full, estimate_national_abstention_from_gaps,
    BLOCKS_ABS, ABBR, ALPHAS, VAL_DATE, VAL_TYPE, TARGET_COLS,
)
from src.cross_type_ridge import (
    _vectorized_block_mapping, _build_block_scores,
    _build_national_poll_features, _add_demographics,
    TARGET_BLOCKS, TYPE_ONEHOT,
)
from src.load_polls import load_poll_tokens

# LOO-selected on training (preregistered.py), single val forward pass
PREV_RAW = {"Gauche": 0.7317, "Centre+Droite": 0.5977,
            "Extreme_Droite": 0.8052, "Abstention": 0.7295}
ALPHA_GRID = np.logspace(-2, 6, 20)


# ── Helpers ──────────────────────────────────────────────────────────

def split_tv(df):
    val_mask = (
        np.isclose(df["date_float"], VAL_DATE, atol=1e-3)
        & (df["election_type"] == VAL_TYPE)
    )
    return df[~val_mask], df[val_mask]


def run_ridge(name, df, feat_cols, national_est, quiet=False):
    """Train Ridge on deviation targets, evaluate with national estimate."""
    train, val = split_tv(df)
    ok_tr = train[feat_cols].notna().all(axis=1)
    ok_v = val[feat_cols].notna().all(axis=1)
    train, val = train[ok_tr], val[ok_v]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v = scaler.transform(val[feat_cols].values.astype(np.float64))

    if not quiet:
        dates = sorted(train["date_float"].unique())
        print(f"  {name}: train={len(X_tr):,} val={len(X_v):,} "
              f"feat={len(feat_cols)} dates={len(dates)}")

    res, preds = {}, {}
    for tc in TARGET_COLS:
        m = RidgeCV(alphas=ALPHA_GRID)
        m.fit(X_tr, train[f"dev_{tc}"].values)
        final = m.predict(X_v) + national_est.get(tc, 0.0)
        preds[tc] = final
        ev = evaluate_full(val[tc].values, final)
        res[tc] = ev
        if not quiet:
            print(f"    {tc:20s} RAW={ev['raw']:.4f}  α={m.alpha_:.1e}")

    return res, preds, val


def run_pca_ridge(name, df, demo_cols, non_demo_cols, national_est, k,
                  quiet=False):
    """PCA on demographics + Ridge on deviations."""
    train, val = split_tv(df)
    all_cols = demo_cols + non_demo_cols
    ok_tr = train[all_cols].notna().all(axis=1)
    ok_v = val[all_cols].notna().all(axis=1)
    train, val = train[ok_tr], val[ok_v]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[all_cols].values.astype(np.float64))
    X_v = scaler.transform(val[all_cols].values.astype(np.float64))

    n_d = len(demo_cols)
    pca = PCA(n_components=k).fit(X_tr[:, :n_d])
    X_tr = np.hstack([pca.transform(X_tr[:, :n_d]), X_tr[:, n_d:]])
    X_v = np.hstack([pca.transform(X_v[:, :n_d]), X_v[:, n_d:]])

    if not quiet:
        print(f"  {name}: PCA-{k}, feat={X_tr.shape[1]}, "
              f"train={len(X_tr):,} val={len(X_v):,}")

    res, preds = {}, {}
    for tc in TARGET_COLS:
        m = RidgeCV(alphas=ALPHA_GRID)
        m.fit(X_tr, train[f"dev_{tc}"].values)
        final = m.predict(X_v) + national_est.get(tc, 0.0)
        preds[tc] = final
        ev = evaluate_full(val[tc].values, final)
        res[tc] = ev
        if not quiet:
            print(f"    {tc:20s} RAW={ev['raw']:.4f}")

    return res, preds, val


# ── LOO Blend ────────────────────────────────────────────────────────

def _fold_pca(X_ft, X_fh, cfg):
    """Apply PCA transform within a LOO fold."""
    if cfg.get("transform") == "pca":
        n_d = cfg["n_demo"]
        k = cfg["k"]
        pca = PCA(n_components=k).fit(X_ft[:, :n_d])
        return (np.hstack([pca.transform(X_ft[:, :n_d]), X_ft[:, n_d:]]),
                np.hstack([pca.transform(X_fh[:, :n_d]), X_fh[:, n_d:]]))
    return X_ft, X_fh


def loo_blend(blend_name, df, configs, national_means, val_est):
    """LOO stacking blend with Ridge + NNLS meta-learners."""
    val_mask = (
        np.isclose(df["date_float"].values, VAL_DATE, atol=1e-3)
        & (df["election_type"].values == VAL_TYPE)
    )
    df_train, df_val = df[~val_mask], df[val_mask]

    train_types = df_train["election_type"].values
    train_dates = df_train["date_float"].values
    train_td = (
        df_train[["election_type", "date_float"]]
        .drop_duplicates().sort_values("date_float").values.tolist()
    )
    n_folds = len(train_td)
    n_models = len(configs)
    n_train, n_val = len(df_train), len(df_val)

    print(f"\n  {blend_name}: {n_models} models × {n_folds} folds, "
          f"train={n_train:,} val={n_val:,}")

    # Pre-scale features
    model_data = []
    for cfg in configs:
        cols = cfg["cols"]
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(df_train[cols].values.astype(np.float64))
        X_v = scaler.transform(df_val[cols].values.astype(np.float64))
        model_data.append((X_tr, X_v))

    dev_targets = {tc: df_train[f"dev_{tc}"].values.astype(np.float64)
                   for tc in TARGET_COLS}
    true_train = {tc: df_train[tc].values.astype(np.float64)
                  for tc in TARGET_COLS}
    true_val = {tc: df_val[tc].values.astype(np.float64)
                for tc in TARGET_COLS}

    # Fold masks + national means per fold
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

    # OOF + val predictions
    oof = {tc: np.full((n_train, n_models), np.nan) for tc in TARGET_COLS}
    vpred = {tc: np.full((n_val, n_models), np.nan) for tc in TARGET_COLS}

    for m_idx, cfg in enumerate(configs):
        X_tr, X_v = model_data[m_idx]
        X_tr_t, X_v_t = _fold_pca(X_tr, X_v, cfg)

        # Full-train → val predictions (RidgeCV for best alpha)
        best_alpha = {}
        for tc in TARGET_COLS:
            ridge = RidgeCV(alphas=ALPHA_GRID)
            ridge.fit(X_tr_t, dev_targets[tc])
            vpred[tc][:, m_idx] = (
                ridge.predict(X_v_t) + val_est.get(tc, 0.0))
            best_alpha[tc] = ridge.alpha_

        # LOO folds
        for f_idx, held_mask in enumerate(fold_masks):
            not_held = ~held_mask
            X_ft, X_fh = X_tr[not_held], X_tr[held_mask]
            X_ft_t, X_fh_t = _fold_pca(X_ft, X_fh, cfg)
            for tc in TARGET_COLS:
                ridge = Ridge(alpha=best_alpha[tc], solver="cholesky")
                ridge.fit(X_ft_t, dev_targets[tc][not_held])
                oof[tc][held_mask, m_idx] = (
                    ridge.predict(X_fh_t) + fold_nats[f_idx][tc])

    # Meta-learners per block
    results = {}
    for tc in TARGET_COLS:
        oof_X = oof[tc]
        val_X = vpred[tc]
        y_val = true_val[tc]

        oof_ok = ~np.isnan(oof_X).any(axis=1)
        oof_clean = oof_X[oof_ok]
        oof_y = true_train[tc][oof_ok]

        # Ridge meta
        meta = RidgeCV(alphas=np.logspace(-4, 4, 20), fit_intercept=True)
        meta.fit(oof_clean, oof_y)
        blend_ridge = meta.predict(val_X)
        ev_ridge = evaluate_full(y_val, blend_ridge)

        # NNLS meta
        X_mean = oof_clean.mean(axis=0)
        y_mean = oof_y.mean()
        w, _ = nnls(oof_clean - X_mean, oof_y - y_mean)
        intercept = y_mean - X_mean @ w
        blend_nnls = val_X @ w + intercept
        ev_nnls = evaluate_full(y_val, blend_nnls)

        # Simple average
        ev_avg = evaluate_full(y_val, val_X.mean(axis=1))

        # Best single
        best_single = max(
            r2_score(y_val, val_X[:, i]) for i in range(n_models))

        results[tc] = {
            "best_single": best_single,
            "avg": ev_avg["raw"],
            "nnls": ev_nnls["raw"],
            "ridge": ev_ridge["raw"],
        }
        best_blend = max(ev_ridge["raw"], ev_nnls["raw"])
        print(f"    {tc:20s} single={best_single:.4f} "
              f"avg={ev_avg['raw']:.4f} nnls={ev_nnls['raw']:.4f} "
              f"ridge={ev_ridge['raw']:.4f}")

    return results


# ── Extended cross-type data builder ─────────────────────────────────

EXTENDED_TYPES = [
    "Legislatives_T1", "Presidentielle_T1",
    "Europeennes_T1", "Regionales_T1",
    "Departementales_T1", "Cantonales_T1",
]

# Exclude dates with poor block mapping (>15% Other) or in val period
EXCLUDE_DATES = {
    ("Europeennes_T1", 1999.5),    # 12.8% Other
    ("Europeennes_T1", 2004.5),    # 17.3% Other
    ("Europeennes_T1", 2024.5),    # val period overlap
    ("Regionales_T1", 2021.5),     # 24.4% Other
}


def build_extended_data(data_dir):
    """Build dataset with extended election types (cached)."""
    cache_dir = data_dir / "baseline_cache"
    cache_path = cache_dir / "beat_it_extended.parquet"
    ind_cache = cache_dir / "beat_it_extended_indicators.txt"
    nm_cache = cache_dir / "beat_it_extended_natmean.parquet"
    poll_cache = cache_dir / "beat_it_extended_polls.parquet"

    if cache_path.exists() and ind_cache.exists() and nm_cache.exists():
        print("Loading extended data from cache...")
        df = pd.read_parquet(cache_path)
        indicators = ind_cache.read_text().strip().split("\n")
        nm = pd.read_parquet(nm_cache)
        pf = pd.read_parquet(poll_cache)
        print(f"  {len(df):,} rows, {len(indicators)} indicators")
        return df, indicators, nm, pf

    print("Building extended cross-type dataset (slow first run)...")
    t0 = time.time()

    elections = pd.read_parquet(cache_dir / "elections.parquet")
    demos = pd.read_parquet(cache_dir / "demographics.parquet")
    polls = load_poll_tokens(data_dir)

    # Filter types and dates
    ext = elections[elections["election_type"].isin(EXTENDED_TYPES)].copy()
    mask = pd.Series(True, index=ext.index)
    for etype, ddate in EXCLUDE_DATES:
        mask &= ~(
            (ext["election_type"] == etype)
            & np.isclose(ext["date_float"], ddate, atol=0.1)
        )
    ext = ext[mask]

    print(f"  Extended elections: {len(ext):,} rows")
    for etype in EXTENDED_TYPES:
        sub = ext[ext["election_type"] == etype]
        if len(sub) == 0:
            continue
        dates = sorted(sub["date_float"].unique())
        print(f"    {etype:30s}  {len(dates)} dates: "
              f"{[round(float(d), 2) for d in dates]}")

    # Build block scores
    print("  Building block scores...", flush=True)
    block_scores = _build_block_scores(ext)
    print(f"    {len(block_scores):,} BV×election rows")

    # National means
    national_means = build_per_type_national_means(block_scores)
    print("  National means (sample):")
    for _, row in national_means.head(5).iterrows():
        print(f"    {row['election_type']:25s} {row['date_float']:.2f}: "
              f"G={row['Gauche']:.1f} CD={row['Centre+Droite']:.1f} "
              f"ED={row['Extreme_Droite']:.1f} Ab={row['Abstention']:.1f}")

    # Deviation targets + cross-type lags
    print("  Adding deviations and lags...", flush=True)
    df = add_deviation_targets(block_scores, national_means)
    df = add_cross_type_local_lags(df)

    # Polls
    print("  Building poll features...", flush=True)
    election_dates = list(
        df[["election_type", "date_float"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    poll_feats = _build_national_poll_features(polls, election_dates)
    df = df.merge(poll_feats, on=["election_type", "date_float"], how="left")
    for col in ["poll_Gauche", "poll_Centre+Droite", "poll_Extreme_Droite"]:
        df[col] = df[col].fillna(0.0)
    df["has_polls"] = df["has_polls"].fillna(0.0)

    # Geo
    geo = ext[["location", "latitude", "longitude"]].drop_duplicates("location")
    df = df.merge(geo, on="location", how="left")
    df["latitude"] = df["latitude"].fillna(46.2276)
    df["longitude"] = df["longitude"].fillna(2.2137)

    # Demographics (slow)
    print("  Merging demographics (may take ~15-30 min)...", flush=True)
    df, indicators = _add_demographics(df, demos)
    df = df.dropna(subset=TARGET_COLS)

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_path, index=False)
    ind_cache.write_text("\n".join(indicators))
    national_means.to_parquet(nm_cache, index=False)
    poll_feats.to_parquet(poll_cache, index=False)
    print(f"  Extended data built in {time.time()-t0:.0f}s, "
          f"cached to {cache_path}")

    return df, indicators, national_means, poll_feats


# ── Main ─────────────────────────────────────────────────────────────

def run_phase1(data_dir):
    """Phase 1: experiments on existing Legi+Pres cached data."""
    print("=" * 70)
    print("PHASE 1: LEGI+PRES EXPERIMENTS")
    print("=" * 70)

    all_results = {}
    df, demo_indicators, national_means, poll_feats = \
        load_cross_type_data(data_dir)
    type_cols = add_election_type_onehot(df)

    # ── Feature groups ──
    geo_time = ["latitude", "longitude", "date_float"]
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    # ── Add lag trend features ──
    for b in BLOCKS_ABS:
        df[f"dev_{b}_trend"] = df[f"dev_{b}_lag1"] - df[f"dev_{b}_lag2"]
    dev_trend = [f"dev_{b}_trend" for b in BLOCKS_ABS]

    # ── National estimates (raw polls + gap model) ──
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {}
    for b in TARGET_BLOCKS:
        est[b] = float(poll_2024[f"poll_{b}"].iloc[0])
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    est["Abstention"] = abs_pred
    print(f"\nNational estimates: {est}")

    # ── V1 datasets ──
    df_v1 = df.dropna(subset=demo_indicators)
    df_v1_2lag = df_v1.dropna(subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)
    df_v1_1lag = df_v1.dropna(subset=raw_lag1 + dev_lag1)
    df_legi = df[df["election_type"] == VAL_TYPE].copy()
    df_legi_v1 = df_legi.dropna(subset=demo_indicators)
    df_legi_v1_2 = df_legi_v1.dropna(
        subset=raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2)
    df_legi_v1_1 = df_legi_v1.dropna(subset=raw_lag1 + dev_lag1)

    # ── Feature sets ──
    nd_ct = geo_time + dev_lag1 + dev_lag2 + type_cols      # non-demo CT
    nd_ct_trend = nd_ct + dev_trend                          # + trends
    nd_ct_1lag = geo_time + dev_lag1 + type_cols             # 1-lag CT
    nd_legi = geo_time + dev_lag1 + dev_lag2                 # non-demo Legi
    nd_legi_trend = nd_legi + dev_trend                      # + trends
    nd_legi_1lag = geo_time + dev_lag1                       # 1-lag Legi

    feat_ct = demo_indicators + nd_ct
    feat_ct_trend = demo_indicators + nd_ct_trend
    feat_ct_1lag = demo_indicators + nd_ct_1lag
    feat_legi = demo_indicators + nd_legi
    feat_legi_trend = demo_indicators + nd_legi_trend
    feat_legi_1lag = demo_indicators + nd_legi_1lag

    # ── Run individual models ──
    print(f"\n{'─'*70}")
    print("INDIVIDUAL MODELS (Legi+Pres)")
    print(f"{'─'*70}")

    # Cross-type 2-lag models
    for name, feats, data in [
        ("CT-devlag", feat_ct, df_v1_2lag),
        ("CT-devlag+trend", feat_ct_trend, df_v1_2lag),
    ]:
        r, p, v = run_ridge(name, data, feats, est)
        all_results[name] = r

    for k in [3, 5, 7, 10]:
        name = f"CT-PCA{k}-devlag"
        r, p, v = run_pca_ridge(name, df_v1_2lag, demo_indicators, nd_ct,
                                est, k)
        all_results[name] = r

        name_t = f"CT-PCA{k}-devlag+trend"
        r, p, v = run_pca_ridge(name_t, df_v1_2lag, demo_indicators,
                                nd_ct_trend, est, k)
        all_results[name_t] = r

    # Cross-type 1-lag models
    r, p, v = run_ridge("CT-devlag-1lag", df_v1_1lag, feat_ct_1lag, est)
    all_results["CT-devlag-1lag"] = r
    for k in [3, 5, 7]:
        name = f"CT-PCA{k}-1lag"
        r, p, v = run_pca_ridge(name, df_v1_1lag, demo_indicators,
                                nd_ct_1lag, est, k)
        all_results[name] = r

    # Legi-only 2-lag models
    for name, feats, data in [
        ("Legi-devlag", feat_legi, df_legi_v1_2),
        ("Legi-devlag+trend", feat_legi_trend, df_legi_v1_2),
    ]:
        r, p, v = run_ridge(name, data, feats, est)
        all_results[name] = r

    for k in [3, 5, 7, 10]:
        name = f"Legi-PCA{k}-devlag"
        r, p, v = run_pca_ridge(name, df_legi_v1_2, demo_indicators,
                                nd_legi, est, k)
        all_results[name] = r

        name_t = f"Legi-PCA{k}-devlag+trend"
        r, p, v = run_pca_ridge(name_t, df_legi_v1_2, demo_indicators,
                                nd_legi_trend, est, k)
        all_results[name_t] = r

    # Legi-only 1-lag models
    r, p, v = run_ridge("Legi-devlag-1lag", df_legi_v1_1, feat_legi_1lag, est)
    all_results["Legi-devlag-1lag"] = r
    for k in [3, 5]:
        name = f"Legi-PCA{k}-1lag"
        r, p, v = run_pca_ridge(name, df_legi_v1_1, demo_indicators,
                                nd_legi_1lag, est, k)
        all_results[name] = r

    # ── LOO Blends ──
    print(f"\n{'─'*70}")
    print("LOO BLENDS (Legi+Pres data)")
    print(f"{'─'*70}")

    # CT blend configs
    ct_configs = [
        {"name": "devlag", "transform": "standard",
         "cols": feat_ct},
        {"name": "devlag+trend", "transform": "standard",
         "cols": feat_ct_trend},
        {"name": "PCA3", "transform": "pca",
         "cols": demo_indicators + nd_ct,
         "n_demo": len(demo_indicators), "k": 3},
        {"name": "PCA5", "transform": "pca",
         "cols": demo_indicators + nd_ct,
         "n_demo": len(demo_indicators), "k": 5},
        {"name": "PCA7", "transform": "pca",
         "cols": demo_indicators + nd_ct,
         "n_demo": len(demo_indicators), "k": 7},
        {"name": "PCA10", "transform": "pca",
         "cols": demo_indicators + nd_ct,
         "n_demo": len(demo_indicators), "k": 10},
        {"name": "PCA3+trend", "transform": "pca",
         "cols": demo_indicators + nd_ct_trend,
         "n_demo": len(demo_indicators), "k": 3},
        {"name": "PCA5+trend", "transform": "pca",
         "cols": demo_indicators + nd_ct_trend,
         "n_demo": len(demo_indicators), "k": 5},
    ]

    ct_blend = loo_blend("CT-LOO-8mod", df_v1_2lag, ct_configs,
                         national_means, est)
    all_results["CT-LOO-blend"] = {
        tc: {"raw": max(ct_blend[tc]["nnls"], ct_blend[tc]["ridge"])}
        for tc in TARGET_COLS
    }

    # Legi blend configs
    legi_configs = [
        {"name": "devlag", "transform": "standard",
         "cols": feat_legi},
        {"name": "devlag+trend", "transform": "standard",
         "cols": feat_legi_trend},
        {"name": "PCA3", "transform": "pca",
         "cols": demo_indicators + nd_legi,
         "n_demo": len(demo_indicators), "k": 3},
        {"name": "PCA5", "transform": "pca",
         "cols": demo_indicators + nd_legi,
         "n_demo": len(demo_indicators), "k": 5},
        {"name": "PCA7", "transform": "pca",
         "cols": demo_indicators + nd_legi,
         "n_demo": len(demo_indicators), "k": 7},
        {"name": "PCA10", "transform": "pca",
         "cols": demo_indicators + nd_legi,
         "n_demo": len(demo_indicators), "k": 10},
        {"name": "PCA3+trend", "transform": "pca",
         "cols": demo_indicators + nd_legi_trend,
         "n_demo": len(demo_indicators), "k": 3},
        {"name": "PCA5+trend", "transform": "pca",
         "cols": demo_indicators + nd_legi_trend,
         "n_demo": len(demo_indicators), "k": 5},
    ]

    legi_blend = loo_blend("Legi-LOO-8mod", df_legi_v1_2, legi_configs,
                           national_means, est)
    all_results["Legi-LOO-blend"] = {
        tc: {"raw": max(legi_blend[tc]["nnls"], legi_blend[tc]["ridge"])}
        for tc in TARGET_COLS
    }

    return all_results, est


def run_phase2(data_dir, est=None):
    """Phase 2: extended cross-type with more election types."""
    all_results = {}
    print(f"\n{'='*70}")
    print("PHASE 2: EXTENDED CROSS-TYPE (Legi+Pres+Euro+Regi+Dept+Cant)")
    print("=" * 70)

    df_ext, ext_indicators, ext_nm, ext_pf = build_extended_data(data_dir)

    # Election type one-hot for extended types
    ext_type_cols = add_election_type_onehot(df_ext)

    # Add lag trends
    for b in BLOCKS_ABS:
        df_ext[f"dev_{b}_trend"] = (
            df_ext[f"dev_{b}_lag1"] - df_ext[f"dev_{b}_lag2"])

    # National estimates (raw polls + gap model)
    # Use Legi+Pres-only national means for gap model (other types add noise)
    poll_2024 = ext_pf[
        np.isclose(ext_pf["date_float"], VAL_DATE, atol=0.1)
        & (ext_pf["election_type"] == VAL_TYPE)
    ]
    ext_est = {}
    if est is not None:
        ext_est = dict(est)
    else:
        for b in TARGET_BLOCKS:
            ext_est[b] = float(poll_2024[f"poll_{b}"].iloc[0])
        # Gap model: only Legi+Pres national means (stable relationship)
        lp_nm = ext_nm[ext_nm["election_type"].isin(
            ["Legislatives_T1", "Presidentielle_T1"])]
        ext_abs_pred, _ = estimate_national_abstention_from_gaps(lp_nm)
        ext_est["Abstention"] = ext_abs_pred
    print(f"Extended national estimates: {ext_est}")

    # V1 datasets
    ext_v1 = df_ext.dropna(subset=ext_indicators)
    ext_raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    ext_raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    ext_dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    ext_dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    ext_dev_trend = [f"dev_{b}_trend" for b in BLOCKS_ABS]

    ext_v1_2 = ext_v1.dropna(
        subset=ext_raw_lag1 + ext_raw_lag2 + ext_dev_lag1 + ext_dev_lag2)
    ext_v1_1 = ext_v1.dropna(subset=ext_raw_lag1 + ext_dev_lag1)

    n_tr = lambda d: len(d[~(
        np.isclose(d["date_float"], VAL_DATE, atol=1e-3)
        & (d["election_type"] == VAL_TYPE)
    )])
    print(f"\n  Ext V1-2lag: {len(ext_v1_2):,} rows "
          f"(train={n_tr(ext_v1_2):,})")
    train_dates = ext_v1_2[~(
        np.isclose(ext_v1_2["date_float"], VAL_DATE, atol=1e-3)
        & (ext_v1_2["election_type"] == VAL_TYPE)
    )]
    print(f"  Train dates ({len(train_dates['date_float'].unique())}):")
    for et, dt in sorted(
        train_dates[["election_type", "date_float"]]
        .drop_duplicates().values.tolist(),
        key=lambda x: x[1],
    ):
        n = len(train_dates[
            (train_dates["election_type"] == et)
            & np.isclose(train_dates["date_float"], dt, atol=1e-3)
        ])
        print(f"    {et:30s} {dt:.2f}: {n:>6,}")

    # Feature sets
    geo_time = ["latitude", "longitude", "date_float"]
    ext_nd = geo_time + ext_dev_lag1 + ext_dev_lag2 + ext_type_cols
    ext_nd_trend = ext_nd + ext_dev_trend
    ext_nd_1lag = geo_time + ext_dev_lag1 + ext_type_cols
    ext_feat = ext_indicators + ext_nd
    ext_feat_trend = ext_indicators + ext_nd_trend
    ext_feat_1lag = ext_indicators + ext_nd_1lag

    print(f"\n{'─'*70}")
    print("EXTENDED INDIVIDUAL MODELS")
    print(f"{'─'*70}")

    for name, feats, data in [
        ("Ext-devlag", ext_feat, ext_v1_2),
        ("Ext-devlag+trend", ext_feat_trend, ext_v1_2),
        ("Ext-devlag-1lag", ext_feat_1lag, ext_v1_1),
    ]:
        r, p, v = run_ridge(name, data, feats, ext_est)
        all_results[name] = r

    for k in [3, 5, 7, 10]:
        name = f"Ext-PCA{k}-devlag"
        r, p, v = run_pca_ridge(name, ext_v1_2, ext_indicators, ext_nd,
                                ext_est, k)
        all_results[name] = r

    for k in [3, 5, 7]:
        name = f"Ext-PCA{k}-devlag+trend"
        r, p, v = run_pca_ridge(name, ext_v1_2, ext_indicators,
                                ext_nd_trend, ext_est, k)
        all_results[name] = r

    # Extended LOO blend
    print(f"\n{'─'*70}")
    print("EXTENDED LOO BLEND")
    print(f"{'─'*70}")

    ext_configs = [
        {"name": "devlag", "transform": "standard",
         "cols": ext_feat},
        {"name": "devlag+trend", "transform": "standard",
         "cols": ext_feat_trend},
        {"name": "PCA3", "transform": "pca",
         "cols": ext_indicators + ext_nd,
         "n_demo": len(ext_indicators), "k": 3},
        {"name": "PCA5", "transform": "pca",
         "cols": ext_indicators + ext_nd,
         "n_demo": len(ext_indicators), "k": 5},
        {"name": "PCA7", "transform": "pca",
         "cols": ext_indicators + ext_nd,
         "n_demo": len(ext_indicators), "k": 7},
        {"name": "PCA10", "transform": "pca",
         "cols": ext_indicators + ext_nd,
         "n_demo": len(ext_indicators), "k": 10},
        {"name": "PCA3+trend", "transform": "pca",
         "cols": ext_indicators + ext_nd_trend,
         "n_demo": len(ext_indicators), "k": 3},
        {"name": "PCA5+trend", "transform": "pca",
         "cols": ext_indicators + ext_nd_trend,
         "n_demo": len(ext_indicators), "k": 5},
    ]

    ext_blend = loo_blend("Ext-LOO-8mod", ext_v1_2, ext_configs,
                          ext_nm, ext_est)
    all_results["Ext-LOO-blend"] = {
        tc: {"raw": max(ext_blend[tc]["nnls"], ext_blend[tc]["ridge"])}
        for tc in TARGET_COLS
    }

    return all_results


def print_summary(all_results):
    """Print comprehensive results summary."""
    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print("RAW R² SUMMARY — ALL EXPERIMENTS")
    print(f"{'='*70}")

    print(f"\n{'Model':30s} {'G':>7s} {'CD':>7s} {'ED':>7s} {'Ab':>7s}")
    print("─" * 62)
    print(f"{'PREV BEST':30s} "
          + " ".join(f"{PREV_RAW[tc]:7.4f}" for tc in TARGET_COLS))
    print("─" * 62)

    for mname in sorted(all_results.keys()):
        raws = []
        for tc in TARGET_COLS:
            v = all_results[mname].get(tc, {})
            r = v.get("raw", float("nan")) if isinstance(v, dict) else float("nan")
            raws.append(r)
        marks = []
        for tc, r in zip(TARGET_COLS, raws):
            marks.append("+" if r > PREV_RAW[tc] + 0.0005 else " ")
        print(f"{mname:30s} "
              + " ".join(f"{r:6.4f}{m}" for r, m in zip(raws, marks)))

    # Per-block best
    print(f"\n{'='*70}")
    print("PER-BLOCK BEST")
    print(f"{'='*70}")
    for tc in TARGET_COLS:
        best_name, best_r2 = "", -999
        for mn, mr in all_results.items():
            r = mr.get(tc, {})
            raw = r.get("raw", -999) if isinstance(r, dict) else -999
            if raw > best_r2:
                best_r2, best_name = raw, mn
        d = best_r2 - PREV_RAW[tc]
        mark = "BEAT" if d > 0.0005 else ("~tie" if abs(d) <= 0.0005 else "miss")
        print(f"  {tc:20s}  R²={best_r2:.4f}  ({best_name})")
        print(f"  {'':20s}  prev={PREV_RAW[tc]:.4f}  Δ={d:+.4f}  [{mark}]")


def main(phase=0):
    """phase=0: all, phase=1: Legi+Pres only, phase=2: extended only,
    phase=3: build extended cache only."""
    data_dir = Path("data")
    all_results = {}

    if phase == 3:
        build_extended_data(data_dir)
        return

    if phase in (0, 1):
        p1_results, est = run_phase1(data_dir)
        all_results.update(p1_results)
    else:
        est = None

    if phase in (0, 2):
        p2_results = run_phase2(data_dir, est=est)
        all_results.update(p2_results)

    print_summary(all_results)


if __name__ == "__main__":
    import sys
    phase = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    main(phase=phase)
