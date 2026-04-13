"""Cross-Type Deviation Training for BV-level election prediction.

Deviation targets (BV score − national mean) are stable across election
types. This enables cross-type training (Legi + Pres) with 8 dates
instead of 4 legi-only dates.

Architecture:
  Stage 1: National mean from raw poll averages
  Stage 2: Ridge on deviations

Best clean raw R² (LOO-selected on training, single val forward pass):
  Gauche:          0.73  (legi-only PCA5-devlag, LOO OOF=0.797)
  Centre+Droite:   0.60  (legi-only PCA7-devlag, LOO OOF=0.596)
  Extr. Droite:    0.81  (cross-type PCA5-devlag, LOO OOF=0.816)
  Abstention:      0.73  (cross-type PCA10-devlag, LOO OOF=0.908)

Usage:
    python3 -m src.cross_type_dev
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import RidgeCV, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from src.load_elections import load_election_tokens
from src.load_demographics import load_demographic_tokens
from src.load_polls import load_poll_tokens
from src.cross_type_ridge import (
    _vectorized_block_mapping, _build_block_scores,
    _build_same_type_national_agg, _build_national_poll_features,
    _add_demographics, TARGET_BLOCKS, TARGET_COLS,
    T1_TYPES, TYPE_ONEHOT,
)

BLOCKS_ABS = TARGET_BLOCKS + ["Abstention"]
ABBR = {"Gauche": "G", "Centre+Droite": "CD",
        "Extreme_Droite": "ED", "Abstention": "Ab"}
ALPHAS = np.logspace(-2, 6, 20)
VAL_DATE = 2024.5
VAL_TYPE = "Legislatives_T1"

# ── Best clean (raw R², raw poll est, no val tuning) ───────────────
# LOO-selected on training (preregistered.py), single val forward pass
PREV_RAW = {"Gauche": 0.7317, "Centre+Droite": 0.5977,
            "Extreme_Droite": 0.8052, "Abstention": 0.7295}


# ── Evaluation ───────────────────────────────────────────────────────

def evaluate_full(y_true, pred):
    """Raw R², oracle (bias-corrected), and affine R²."""
    r2_raw = r2_score(y_true, pred)
    bias = np.mean(y_true - pred)
    r2_orc = r2_score(y_true, pred + bias)
    corr = np.corrcoef(y_true, pred)[0, 1]
    r2_aff = float(corr ** 2) if not np.isnan(corr) else r2_orc
    return dict(raw=r2_raw, orc=r2_orc, aff=r2_aff)


# ── National abstention estimator ───────────────────────────────────

def estimate_national_abstention_from_gaps(national_means, val_date=VAL_DATE):
    """Estimate national abstention from inter-election gap model.

    Principled: trained only on historical election national means, no
    validation data used.

    Key insight: time since the last election (of any type) strongly
    predicts national abstention.  In training data:
      - Legi T1 follows Pres T1 by ~0.17 yr → high abstention (~42%)
      - Pres T1 follows Legi T1 by ~4.8 yr  → low abstention (~20%)
    The 2024 snap Legi has gap = 2.0 yr (unprecedented), and the linear
    model interpolates between the two clusters → ~33%, close to the
    actual 31%.

    Returns (predicted_abstention, loo_rmse).
    """
    nm = national_means.sort_values("date_float").reset_index(drop=True)
    train = nm[nm["date_float"] < val_date - 0.1].copy()

    # gap_years = time since previous election (any type)
    train["gap_years"] = train["date_float"].diff()
    # Drop first election (no prior)
    train = train.dropna(subset=["gap_years"]).reset_index(drop=True)

    X = train[["gap_years"]].values
    y = train["Abstention"].values
    n = len(y)

    # LOO cross-validation
    loo_preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        lr = LinearRegression()
        lr.fit(X[mask], y[mask])
        loo_preds[i] = float(lr.predict(X[[i]])[0])

    loo_rmse = float(np.sqrt(np.mean((y - loo_preds) ** 2)))

    # Fit final model on all training data
    lr = LinearRegression()
    lr.fit(X, y)

    # Gap for 2024: time since most recent training election
    last_train_date = float(train["date_float"].max())
    gap_2024 = val_date - last_train_date
    pred = float(lr.predict([[gap_2024]])[0])

    print(f"\n  Abstention gap model (gap_years → national abs):")
    print(f"    {n} training elections, slope={lr.coef_[0]:.2f} pp/yr, "
          f"intercept={lr.intercept_:.1f}%")
    print(f"    LOO RMSE = {loo_rmse:.1f} pp")
    print(f"    LOO predictions:")
    for i in range(n):
        et = train.iloc[i]["election_type"]
        dt = train.iloc[i]["date_float"]
        g = train.iloc[i]["gap_years"]
        print(f"      {et:25s} {dt:.2f}  gap={g:.2f}yr  "
              f"actual={y[i]:.1f}  pred={loo_preds[i]:.1f}  "
              f"err={loo_preds[i] - y[i]:+.1f}")
    print(f"    2024: gap={gap_2024:.2f}yr → predicted abs = {pred:.1f}%")

    return pred, loo_rmse


# ── Data builders ────────────────────────────────────────────────────

def build_per_type_national_means(block_scores):
    """National mean block scores per (election_type, date)."""
    return (
        block_scores
        .groupby(["election_type", "date_float"])[TARGET_COLS]
        .mean().reset_index()
    )


def add_deviation_targets(df, national_means):
    """dev_<block> = BV_score − national_mean(election_type, date)."""
    nm = national_means.rename(
        columns={c: f"natmean_{c}" for c in TARGET_COLS}
    )
    df = df.merge(nm, on=["election_type", "date_float"], how="left")
    for c in TARGET_COLS:
        df[f"dev_{c}"] = df[c] - df[f"natmean_{c}"]
    return df


def add_cross_type_local_lags(df):
    """Cross-type local lags: most recent 1-2 prior elections at this BV
    (of any type), in both raw and deviation space."""
    df = df.sort_values(["location", "date_float"])

    # Raw lags (needed for NaN filtering even though only dev lags are used as features)
    for col in BLOCKS_ABS:
        df[f"{col}_lag1"] = df.groupby("location")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("location")[col].shift(2)

    # Deviation lags (used as features)
    for col in BLOCKS_ABS:
        df[f"dev_{col}_lag1"] = df.groupby("location")[f"dev_{col}"].shift(1)
        df[f"dev_{col}_lag2"] = df.groupby("location")[f"dev_{col}"].shift(2)

    return df


def add_election_type_onehot(df):
    """One-hot for election type. Cantonales → Departementales."""
    canon = df["election_type"].str.replace("_T1", "")
    canon = canon.replace("Cantonales", "Departementales")
    onehot_cols = []
    for t in TYPE_ONEHOT:
        col = f"type_{t}"
        df[col] = (canon == t).astype(np.float64)
        onehot_cols.append(col)
    return onehot_cols


# ── Model runners ────────────────────────────────────────────────────

def split_tv(df):
    """Split: train = everything except 2024 Legi T1, val = 2024 Legi T1."""
    val_mask = (
        np.isclose(df["date_float"], VAL_DATE, atol=1e-3)
        & (df["election_type"] == VAL_TYPE)
    )
    train = df[~val_mask]
    val = df[val_mask]
    return train, val


def run_model(name, df, feat_cols, national_est_val):
    """Train Ridge on deviation targets, evaluate with national estimate."""
    train, val = split_tv(df)
    feat_ok_tr = train[feat_cols].notna().all(axis=1)
    feat_ok_v = val[feat_cols].notna().all(axis=1)
    train = train[feat_ok_tr]
    val = val[feat_ok_v]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[feat_cols].values.astype(np.float64))
    X_v = scaler.transform(val[feat_cols].values.astype(np.float64))
    print(f"\n  {name}: train={len(X_tr):,} val={len(X_v):,} feat={len(feat_cols)}"
          f" dates={sorted(train['date_float'].unique())}")

    res, preds = {}, {}
    for tc in TARGET_COLS:
        y_tr = train[f"dev_{tc}"].values
        m = RidgeCV(alphas=ALPHAS)
        m.fit(X_tr, y_tr)
        dev_pred = m.predict(X_v)
        nat_mean = national_est_val.get(tc, 0.0)
        final_pred = dev_pred + nat_mean

        preds[tc] = final_pred
        y_true = val[tc].values
        ev = evaluate_full(y_true, final_pred)
        ev["alpha"] = m.alpha_
        res[tc] = ev
        print(f"    {tc:20s} RAW={ev['raw']:.4f} orc={ev['orc']:.4f} "
              f"α={m.alpha_:.1e}  nat_est={national_est_val.get(tc, 0):.1f}")

    return res, preds, val


def run_pca_model(name, df, demo_cols, non_demo_cols, national_est_val, k):
    """PCA on demographics + deviation Ridge."""
    train, val = split_tv(df)
    all_cols = demo_cols + non_demo_cols
    feat_ok_tr = train[all_cols].notna().all(axis=1)
    feat_ok_v = val[all_cols].notna().all(axis=1)
    train = train[feat_ok_tr]
    val = val[feat_ok_v]

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(train[all_cols].values.astype(np.float64))
    X_v = scaler.transform(val[all_cols].values.astype(np.float64))

    n_d = len(demo_cols)
    pca = PCA(n_components=k).fit(X_tr[:, :n_d])
    X_tr = np.hstack([pca.transform(X_tr[:, :n_d]), X_tr[:, n_d:]])
    X_v = np.hstack([pca.transform(X_v[:, :n_d]), X_v[:, n_d:]])
    total_f = X_tr.shape[1]
    print(f"\n  {name}: PCA-{k}, total feat={total_f}, "
          f"train={len(X_tr):,} val={len(X_v):,}")

    res, preds = {}, {}
    for tc in TARGET_COLS:
        m = RidgeCV(alphas=ALPHAS)
        m.fit(X_tr, train[f"dev_{tc}"].values)
        dev_pred = m.predict(X_v)
        nat_mean = national_est_val.get(tc, 0.0)
        final_pred = dev_pred + nat_mean
        preds[tc] = final_pred
        ev = evaluate_full(val[tc].values, final_pred)
        res[tc] = ev
        print(f"    {tc:20s} RAW={ev['raw']:.4f} orc={ev['orc']:.4f}")
    return res, preds, val


# ── Main ─────────────────────────────────────────────────────────────

def load_cross_type_data(data_dir):
    """Load and build cross-type dataset with caching."""
    cache_dir = data_dir / "baseline_cache"
    ct_cache = cache_dir / "cross_type_dev_base.parquet"
    ind_cache = cache_dir / "cross_type_dev_indicators.txt"
    natmean_cache = cache_dir / "cross_type_dev_natmean.parquet"
    poll_cache = cache_dir / "cross_type_dev_polls.parquet"

    if ct_cache.exists() and ind_cache.exists() and natmean_cache.exists():
        print("Loading cross-type data from cache...")
        df = pd.read_parquet(ct_cache)
        demo_indicators = ind_cache.read_text().strip().split("\n")
        national_means = pd.read_parquet(natmean_cache)
        poll_feats = pd.read_parquet(poll_cache)
        print(f"  Loaded: {len(df):,} rows, {len(demo_indicators)} indicators")
        return df, demo_indicators, national_means, poll_feats

    # Full build
    print("Building cross-type dataset (slow first run, cached after)...")
    elections_cache = cache_dir / "elections.parquet"
    demos_cache = cache_dir / "demographics.parquet"

    if elections_cache.exists() and demos_cache.exists():
        print("  Loading elections/demographics from cache...")
        elections = pd.read_parquet(elections_cache)
        demos = pd.read_parquet(demos_cache)
    else:
        print("  Cache miss — full load...")
        elections = load_election_tokens(data_dir)
        demos = load_demographic_tokens(data_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        elections.to_parquet(elections_cache, index=False)
        demos.to_parquet(demos_cache, index=False)

    print(f"  Elections: {len(elections):,}, Demographics: {len(demos):,}")

    polls = load_poll_tokens(data_dir)
    print(f"  Polls: {len(polls):,}")

    CROSS_TYPES = ["Legislatives_T1", "Presidentielle_T1"]
    ct_elections = elections[elections["election_type"].isin(CROSS_TYPES)].copy()
    print(f"\n  Cross-type elections: {len(ct_elections):,} rows")
    for etype in CROSS_TYPES:
        sub = ct_elections[ct_elections["election_type"] == etype]
        dates = sorted(sub["date_float"].unique())
        print(f"    {etype:30s}  {len(dates)} dates: "
              f"{[round(float(d), 2) for d in dates]}")

    print("  Building block scores...")
    block_scores = _build_block_scores(ct_elections)
    print(f"    Total BV×election rows: {len(block_scores):,}")

    national_means = build_per_type_national_means(block_scores)
    print("  National means per (type, date):")
    for _, row in national_means.iterrows():
        print(f"    {row['election_type']:30s} {row['date_float']:.2f}:  "
              f"G={row['Gauche']:.1f}  C+D={row['Centre+Droite']:.1f}  "
              f"ED={row['Extreme_Droite']:.1f}  Abs={row['Abstention']:.1f}")

    print("  Adding deviation targets...")
    df = add_deviation_targets(block_scores, national_means)

    print("  Adding cross-type local lags...")
    df = add_cross_type_local_lags(df)

    print("  Building poll features...")
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

    geo = ct_elections[["location", "latitude", "longitude"]].drop_duplicates("location")
    df = df.merge(geo, on="location", how="left")
    df["latitude"] = df["latitude"].fillna(46.2276)
    df["longitude"] = df["longitude"].fillna(2.2137)

    print("  Merging demographics (slow, ~30 min)...", flush=True)
    df, demo_indicators = _add_demographics(df, demos)
    df = df.dropna(subset=TARGET_COLS)

    # Cache
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(ct_cache, index=False)
    ind_cache.write_text("\n".join(demo_indicators))
    national_means.to_parquet(natmean_cache, index=False)
    poll_feats.to_parquet(poll_cache, index=False)
    print(f"  Cached to {ct_cache}")

    return df, demo_indicators, national_means, poll_feats


def main():
    data_dir = Path("data")

    # ════════════════════════════════════════════════════════════════
    # 1. Load cross-type data (cached after first run)
    # ════════════════════════════════════════════════════════════════
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)

    type_cols = add_election_type_onehot(df)

    print(f"\nFinal dataset: {len(df):,} rows, {df['location'].nunique():,} BVs")

    # ════════════════════════════════════════════════════════════════
    # 2. Feature groups
    # ════════════════════════════════════════════════════════════════
    geo_time = ["latitude", "longitude", "date_float"]

    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]

    def avail(cols):
        return [c for c in cols if c in df.columns and df[c].notna().any()]

    dl1 = avail(dev_lag1)
    dl2 = avail(dev_lag2)

    # ════════════════════════════════════════════════════════════════
    # 3. National estimates for 2024 (raw polls, no calibration)
    # ════════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("NATIONAL ESTIMATES FOR 2024 (raw polls)")
    print("=" * 70)

    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {}
    if len(poll_2024) > 0:
        for b in TARGET_BLOCKS:
            est[b] = float(poll_2024[f"poll_{b}"].iloc[0])

    # Abstention: gap-based turnout model (trained on training elections only)
    abs_pred, abs_loo_rmse = estimate_national_abstention_from_gaps(
        national_means
    )
    est["Abstention"] = abs_pred
    print(f"  Raw poll estimates: {est}")

    # Oracle (for reference only)
    val_mask = (
        np.isclose(df["date_float"], VAL_DATE, atol=1e-3)
        & (df["election_type"] == VAL_TYPE)
    )
    oracle_est = {tc: float(df.loc[val_mask, tc].mean()) for tc in TARGET_COLS}
    print(f"  Oracle estimates:   {oracle_est}")

    # ════════════════════════════════════════════════════════════════
    # 4. V1: Strict NaN drops — prepare data
    # ════════════════════════════════════════════════════════════════
    df_v1 = df.dropna(subset=demo_indicators)
    # Require 2 complete lags (raw lags checked for completeness,
    # dev lags used as features)
    df_v1_2lag = df_v1.dropna(
        subset=raw_lag1 + raw_lag2 + avail(dev_lag1) + avail(dev_lag2)
    )

    n_tr = lambda d: len(d) - int((
        np.isclose(d["date_float"], VAL_DATE, atol=1e-3)
        & (d["election_type"] == VAL_TYPE)
    ).sum())
    n_vl = lambda d: int((
        np.isclose(d["date_float"], VAL_DATE, atol=1e-3)
        & (d["election_type"] == VAL_TYPE)
    ).sum())
    tr_dates = lambda d: sorted(d[
        ~(np.isclose(d["date_float"], VAL_DATE, atol=1e-3)
          & (d["election_type"] == VAL_TYPE))
    ]["date_float"].unique())

    print(f"\n  V1-2lag: total={len(df_v1_2lag):,} "
          f"(train={n_tr(df_v1_2lag):,} val={n_vl(df_v1_2lag):,})")
    print(f"    Train dates: {tr_dates(df_v1_2lag)}")

    all_res = {}
    all_preds = {}

    # ════════════════════════════════════════════════════════════════
    # 5. Cross-type models
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("CROSS-TYPE DEVIATION MODELS")
    print(f"{'─'*60}")

    nd_dev = geo_time + dl1 + dl2 + type_cols

    # devlag (best ED)
    feat_ct = demo_indicators + geo_time + dl1 + dl2 + type_cols
    r, p, v = run_model("CT-devlag", df_v1_2lag, feat_ct, est)
    all_res["CT-devlag"] = r
    all_preds["CT-devlag"] = p

    # PCA3-devlag
    r, p, v = run_pca_model("CT-PCA3-devlag", df_v1_2lag,
                            demo_indicators, nd_dev, est, 3)
    all_res["CT-PCA3-devlag"] = r
    all_preds["CT-PCA3-devlag"] = p

    # PCA5-devlag
    r, p, v = run_pca_model("CT-PCA5-devlag", df_v1_2lag,
                            demo_indicators, nd_dev, est, 5)
    all_res["CT-PCA5-devlag"] = r
    all_preds["CT-PCA5-devlag"] = p

    # ════════════════════════════════════════════════════════════════
    # 6. Legi-only models
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print("LEGI-ONLY DEVIATION MODELS")
    print(f"{'─'*60}")

    df_legi = df[df["election_type"] == VAL_TYPE].copy()
    df_legi_v1 = df_legi.dropna(subset=demo_indicators)
    df_legi_v1_2 = df_legi_v1.dropna(
        subset=raw_lag1 + raw_lag2 + avail(dev_lag1) + avail(dev_lag2)
    )
    print(f"  Legi V1-2lag: {len(df_legi_v1_2):,} "
          f"(train={n_tr(df_legi_v1_2):,} val={n_vl(df_legi_v1_2):,})")
    print(f"    Train dates: {tr_dates(df_legi_v1_2)}")

    nd_dev_legi = geo_time + dl1 + dl2  # no type one-hot for legi-only

    # devlag (best G)
    feat_legi = demo_indicators + geo_time + dl1 + dl2
    r, p, v = run_model("Legi-devlag", df_legi_v1_2, feat_legi, est)
    all_res["Legi-devlag"] = r
    all_preds["Legi-devlag"] = p

    # PCA3-devlag
    r, p, v = run_pca_model("Legi-PCA3-devlag", df_legi_v1_2,
                            demo_indicators, nd_dev_legi, est, 3)
    all_res["Legi-PCA3-devlag"] = r
    all_preds["Legi-PCA3-devlag"] = p

    # PCA5-devlag (best C+D)
    r, p, v = run_pca_model("Legi-PCA5-devlag", df_legi_v1_2,
                            demo_indicators, nd_dev_legi, est, 5)
    all_res["Legi-PCA5-devlag"] = r
    all_preds["Legi-PCA5-devlag"] = p

    # ════════════════════════════════════════════════════════════════
    # SUMMARY
    # ════════════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("RAW R² SUMMARY (all clean — raw poll est, no val tuning)")
    print(f"{'='*70}")

    print(f"\n{'Model':30s} {'G raw':>7s} {'CD raw':>7s} {'ED raw':>7s} {'Ab raw':>7s}")
    print("-" * 65)
    print(f"{'PREV BEST':30s} "
          + " ".join(f"{PREV_RAW[tc]:7.3f}" for tc in TARGET_COLS))
    print("-" * 65)

    for mname in sorted(all_res.keys()):
        raws = []
        for tc in TARGET_COLS:
            v = all_res[mname].get(tc, {})
            raws.append(v.get("raw", float("nan")) if isinstance(v, dict) else float("nan"))
        marks = []
        for tc, r in zip(TARGET_COLS, raws):
            marks.append("+" if r > PREV_RAW[tc] + 0.001 else " ")
        line = (f"{mname:30s} "
                + " ".join(f"{r:6.3f}{m}" for r, m in zip(raws, marks)))
        print(line)

    # Per-block best
    print(f"\n{'='*70}")
    print("PER-BLOCK BEST RAW R²")
    print(f"{'='*70}")
    for tc in TARGET_COLS:
        bn, br = "", -999
        for mn, mr in all_res.items():
            if tc not in mr:
                continue
            raw = mr[tc].get("raw", -999) if isinstance(mr[tc], dict) else -999
            if raw > br:
                br, bn = raw, mn
        d = br - PREV_RAW[tc]
        mark = "BEAT" if d > 0.001 else ("~tie" if abs(d) <= 0.001 else "miss")
        print(f"  {tc:20s}  RAW={br:.4f}  ({bn})")
        print(f"  {'':20s}  prev={PREV_RAW[tc]:.3f}  Δ={d:+.4f}  [{mark}]")


if __name__ == "__main__":
    main()
