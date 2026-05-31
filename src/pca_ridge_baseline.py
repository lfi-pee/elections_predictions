"""PCA + Ridge Regression baseline for BV-level election prediction.

Uses demographic indicators + geo coords + lagged election block scores
with StandardScaler → PCA → Ridge to prevent overfitting.
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score

from src.load_elections import load_election_tokens
from src.load_demographics import load_demographic_tokens

LEFT = {
    "SOC",
    "COM",
    "VEC",
    "ECO",
    "EXG",
    "DVG",
    "FI",
    "NUP",
    "FG",
    "RDG",
    "DXG",
    "UG",
    "LO",
    "LCR",
    "GEN",
    "PRG",
    "LDVG",
    "LFI",
    "LCOM",
    "LECO",
    "LEXG",
    "LRDG",
    "LUG",
    "LVEC",
    "LUC",
    "LFG",
}
CENTER_RIGHT = {
    "UMP",
    "LR",
    "DVD",
    "REM",
    "ENS",
    "UDI",
    "MDM",
    "UDF",
    "CEN",
    "DVC",
    "NCE",
    "UDFD",
    "NC",
    "RPR",
    "RPF",
    "HOR",
    "LLR",
    "LREM",
    "LMDM",
    "LHOR",
    "LUDI",
    "LDVD",
    "LNC",
    "LUD",
    "LMAJ",
    "LDVC",
}
EXTREME_RIGHT = {
    "FN",
    "RN",
    "REC",
    "EXD",
    "MNR",
    "UXD",
    "DLF",
    "MPF",
    "LRN",
    "LREC",
    "LEXD",
    "LUXD",
}


def get_block(nuance: str) -> str:
    if pd.isna(nuance) or nuance == "":
        return "Other"
    if nuance in LEFT:
        return "Gauche"
    if nuance in CENTER_RIGHT:
        return "Centre+Droite"
    if nuance in EXTREME_RIGHT:
        return "Extreme_Droite"
    return "Other"


def _load_cached(data_dir: Path):
    """Load election & demographic tokens, caching processed DataFrames as parquet."""
    cache_dir = data_dir / "baseline_cache"
    elections_cache = cache_dir / "elections.parquet"
    demos_cache = cache_dir / "demographics.parquet"

    if elections_cache.exists() and demos_cache.exists():
        print("Loading from cache (data/baseline_cache/)...", flush=True)
        elections = pd.read_parquet(elections_cache)
        demos = pd.read_parquet(demos_cache)
        print(
            f"  Elections: {len(elections)} rows, Demographics: {len(demos)} rows",
            flush=True,
        )
        return elections, demos

    print(
        "Cache not found — running full load (this is slow, but only once)...",
        flush=True,
    )
    print("Loading election tokens...", flush=True)
    elections = load_election_tokens(data_dir)
    print("Loading demographic tokens...", flush=True)
    demos = load_demographic_tokens(data_dir)

    cache_dir.mkdir(parents=True, exist_ok=True)
    elections.to_parquet(elections_cache, index=False)
    demos.to_parquet(demos_cache, index=False)
    print(f"Cached to {cache_dir}/", flush=True)
    return elections, demos


def main():
    data_dir = Path("data")
    elections, demos = _load_cached(data_dir)

    # --- Filter to T1 legislatives only (cleaner comparison) ---
    legi_elections = elections[elections["election_type"] == "Legislatives_T1"].copy()

    unique_dates = sorted(legi_elections["date_float"].unique())
    print(f"\nLegislative T1 dates: {unique_dates}", flush=True)

    val_date = 2024.5  # 2024_legi_t1

    # --- Build block scores per BV per election ---
    print("Building block scores...", flush=True)
    legi_results = legi_elections[legi_elections["metric_type"] == "Result"].copy()
    legi_results["block"] = legi_results["party"].apply(get_block)

    block_scores = (
        legi_results.groupby(["location", "date_float", "block"])["value"]
        .sum()
        .unstack(fill_value=0.0)
        .reset_index()
    )

    target_blocks = ["Gauche", "Centre+Droite", "Extreme_Droite"]
    for b in target_blocks:
        if b not in block_scores.columns:
            block_scores[b] = 0.0

    # Sanity check
    known = block_scores[target_blocks].sum(axis=1)
    print(
        f"Block coverage: mean={known.mean():.1f}%, min={known.min():.1f}%, max={known.max():.1f}%",
        flush=True,
    )

    # --- Abstention ---
    abstentions = legi_elections[
        (legi_elections["metric_type"] == "Context")
        & (legi_elections["candidate"] == "Abstention")
    ][["location", "date_float", "value"]].rename(columns={"value": "Abstention"})

    targets = pd.merge(
        block_scores, abstentions, on=["location", "date_float"], how="inner"
    )

    # --- Build lagged features: previous election's block scores for same BV ---
    print("Building lagged election features...", flush=True)
    lag_cols = []
    for lag_n, lag_label in [(1, "lag1"), (2, "lag2")]:
        for b in target_blocks + ["Abstention"]:
            col_name = f"{b}_{lag_label}"
            lag_cols.append(col_name)
            targets[col_name] = np.nan

    # For each BV, sort by date and shift
    targets = targets.sort_values(["location", "date_float"])
    for b in target_blocks:
        targets[f"{b}_lag1"] = targets.groupby("location")[b].shift(1)
        targets[f"{b}_lag2"] = targets.groupby("location")[b].shift(2)
    targets["Abstention_lag1"] = targets.groupby("location")["Abstention"].shift(1)
    targets["Abstention_lag2"] = targets.groupby("location")["Abstention"].shift(2)

    # --- Demographic features ---
    all_indicators = demos["candidate"].unique().tolist()
    print(
        f"Demographic indicators ({len(all_indicators)}): {all_indicators}", flush=True
    )

    targets["commune"] = targets["location"].str.split("_").str[0]
    demos = demos.sort_values("availability_date")
    targets = targets.sort_values("date_float")

    out_df = targets.copy()
    for ind in all_indicators:
        df_ind = demos[demos["candidate"] == ind][
            ["location", "availability_date", "value"]
        ].copy()
        df_ind = df_ind.rename(columns={"location": "commune", "value": ind})
        df_ind = df_ind.sort_values("availability_date").dropna(
            subset=["availability_date"]
        )
        out_df = pd.merge_asof(
            out_df,
            df_ind,
            left_on="date_float",
            right_on="availability_date",
            by="commune",
            direction="backward",
        )
        if "availability_date" in out_df.columns:
            out_df = out_df.drop(columns=["availability_date"])

    # --- Geo coords ---
    geo_cols = legi_elections[["location", "latitude", "longitude"]].drop_duplicates(
        "location"
    )
    out_df = out_df.merge(geo_cols, on="location", how="left")
    out_df["latitude"] = out_df["latitude"].fillna(46.2276)
    out_df["longitude"] = out_df["longitude"].fillna(2.2137)

    print(f"Dataset shape before drops: {out_df.shape}", flush=True)
    # Drop indicators that are entirely NaN (unavailable for any vintage)
    available_indicators = [
        ind
        for ind in all_indicators
        if ind in out_df.columns and out_df[ind].notna().any()
    ]
    dropped = set(all_indicators) - set(available_indicators)
    if dropped:
        print(
            f"  Dropped {len(dropped)} all-NaN indicators: {sorted(dropped)}",
            flush=True,
        )
    all_indicators = available_indicators
    # Impute remaining NaN with column median
    for ind in all_indicators:
        out_df[ind] = out_df[ind].fillna(out_df[ind].median())
    out_df = out_df.dropna(subset=all_indicators)
    target_cols = ["Gauche", "Centre+Droite", "Extreme_Droite", "Abstention"]
    out_df = out_df.dropna(subset=target_cols)
    print(
        f"After NaN cleanup: {out_df.shape} ({len(all_indicators)} indicators)",
        flush=True,
    )
    alphas = np.logspace(-2, 6, 60)

    # ====================================================================
    # Model A: Demographics only (no lags)
    # ====================================================================
    print("\n" + "=" * 60, flush=True)
    print("MODEL A: Demographics + Geo only", flush=True)
    print("=" * 60, flush=True)

    demo_feature_cols = ["date_float", "latitude", "longitude"] + all_indicators
    _run_ridge(out_df, demo_feature_cols, target_cols, val_date, alphas)

    # ====================================================================
    # Model B: Demographics + 1-lag
    # ====================================================================
    lag1_cols = [f"{b}_lag1" for b in target_blocks] + ["Abstention_lag1"]
    df_lag1 = out_df.dropna(subset=lag1_cols)
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL B: Demographics + Geo + 1 Lag ({len(df_lag1)} rows)", flush=True)
    print("=" * 60, flush=True)

    feature_cols_b = demo_feature_cols + lag1_cols
    _run_ridge(df_lag1, feature_cols_b, target_cols, val_date, alphas)

    # ====================================================================
    # Model C: Demographics + 2-lag
    # ====================================================================
    lag2_cols = lag1_cols + [f"{b}_lag2" for b in target_blocks] + ["Abstention_lag2"]
    df_lag2 = out_df.dropna(subset=lag2_cols)
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL C: Demographics + Geo + 2 Lags ({len(df_lag2)} rows)", flush=True)
    print("=" * 60, flush=True)

    feature_cols_c = demo_feature_cols + lag2_cols
    _run_ridge(df_lag2, feature_cols_c, target_cols, val_date, alphas)

    # ====================================================================
    # Model D: Lag-only (no demographics) — how much do lags carry alone?
    # ====================================================================
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL D: Lag-only (no demographics)", flush=True)
    print("=" * 60, flush=True)
    feature_cols_d = lag1_cols
    _run_ridge(df_lag1, feature_cols_d, target_cols, val_date, alphas)

    # ====================================================================
    # Model E: Poly(2) features on Model B
    # ====================================================================
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL E: Poly(2) + Ridge on demographics + 1 lag", flush=True)
    print("=" * 60, flush=True)
    _run_poly_ridge(df_lag1, feature_cols_b, target_cols, val_date, alphas, degree=2)

    # ====================================================================
    # Model F: Elastic Net (L1+L2) — automatic feature selection
    # ====================================================================
    print(f"\n{'=' * 60}", flush=True)
    print(f"MODEL F: Elastic Net on demographics + 1 lag", flush=True)
    print("=" * 60, flush=True)
    _run_elastic_net(df_lag1, feature_cols_b, target_cols, val_date)


def _run_ridge(df, feature_cols, target_cols, val_date, alphas):
    """Train and evaluate Ridge with optional PCA."""
    train_df = df[
        (df["date_float"] < val_date)
        & ~np.isclose(df["date_float"], val_date, atol=1e-3)
    ]
    val_df = df[np.isclose(df["date_float"], val_date, atol=1e-3)]

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[target_cols]
    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = val_df[target_cols]

    print(
        f"  Train: {len(X_train)}, Val: {len(X_val)}, Features: {len(feature_cols)}",
        flush=True,
    )
    if len(X_val) == 0:
        print("  ERROR: 0 val samples", flush=True)
        return

    # Scaled Ridge (no PCA)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    for t_col in target_cols:
        model = RidgeCV(alphas=alphas)
        model.fit(X_tr, y_train[t_col])
        r2_tr = r2_score(y_train[t_col], model.predict(X_tr))
        r2_v = r2_score(y_val[t_col], model.predict(X_v))
        print(
            f"  {t_col}: Train R²={r2_tr:.4f}, Val R²={r2_v:.4f} (alpha={model.alpha_:.1f})",
            flush=True,
        )


def _run_poly_ridge(df, feature_cols, target_cols, val_date, alphas, degree=2):
    """Train Poly + Ridge."""
    train_df = df[
        (df["date_float"] < val_date)
        & ~np.isclose(df["date_float"], val_date, atol=1e-3)
    ]
    val_df = df[np.isclose(df["date_float"], val_date, atol=1e-3)]

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[target_cols]
    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = val_df[target_cols]

    print(f"  Train: {len(X_train)}, Val: {len(X_val)}", flush=True)
    if len(X_val) == 0:
        print("  ERROR: 0 val samples", flush=True)
        return

    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "poly",
                PolynomialFeatures(
                    degree=degree, interaction_only=False, include_bias=False
                ),
            ),
        ]
    )
    X_tr = pipe.fit_transform(X_train)
    X_v = pipe.transform(X_val)
    print(f"  Poly features: {X_tr.shape[1]}", flush=True)

    for t_col in target_cols:
        model = RidgeCV(alphas=alphas)
        model.fit(X_tr, y_train[t_col])
        r2_tr = r2_score(y_train[t_col], model.predict(X_tr))
        r2_v = r2_score(y_val[t_col], model.predict(X_v))
        print(
            f"  {t_col}: Train R²={r2_tr:.4f}, Val R²={r2_v:.4f} (alpha={model.alpha_:.1f})",
            flush=True,
        )


def _run_elastic_net(df, feature_cols, target_cols, val_date):
    """Train Elastic Net (L1+L2) with CV — automatic feature selection."""
    train_df = df[
        (df["date_float"] < val_date)
        & ~np.isclose(df["date_float"], val_date, atol=1e-3)
    ]
    val_df = df[np.isclose(df["date_float"], val_date, atol=1e-3)]

    X_train = train_df[feature_cols].values.astype(np.float64)
    y_train = train_df[target_cols]
    X_val = val_df[feature_cols].values.astype(np.float64)
    y_val = val_df[target_cols]

    print(
        f"  Train: {len(X_train)}, Val: {len(X_val)}, Features: {len(feature_cols)}",
        flush=True,
    )
    if len(X_val) == 0:
        print("  ERROR: 0 val samples", flush=True)
        return

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_v = scaler.transform(X_val)

    for t_col in target_cols:
        model = ElasticNetCV(
            l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99],
            n_alphas=50,
            cv=5,
            max_iter=10000,
        )
        model.fit(X_tr, y_train[t_col])
        r2_tr = r2_score(y_train[t_col], model.predict(X_tr))
        r2_v = r2_score(y_val[t_col], model.predict(X_v))
        n_nonzero = np.sum(model.coef_ != 0)
        print(
            f"  {t_col}: Train R²={r2_tr:.4f}, Val R²={r2_v:.4f} "
            f"(alpha={model.alpha_:.4f}, l1_ratio={model.l1_ratio_:.2f}, "
            f"nonzero={n_nonzero}/{len(feature_cols)})",
            flush=True,
        )


if __name__ == "__main__":
    main()
