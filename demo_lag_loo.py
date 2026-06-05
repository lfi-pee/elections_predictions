"""Experiment: do LAGGED / DIFFERENCED demographics get selected by LOO?

Today only the political *scores* are lagged (dev_<block>_lag1/2). Demographics
enter as contemporaneous levels. This tests two extra forms, on the same LOO
criterion the pre-registered model uses:
  - demo levels lagged ~5 yr (previous census vintage)
  - demo deltas = current vintage − previous vintage (local social *trend*)
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import numpy as np
import pandas as pd
from pathlib import Path

from src.cross_type_dev import (
    load_cross_type_data,
    add_election_type_onehot,
    BLOCKS_ABS,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.cross_type_dev import estimate_national_abstention_from_gaps
from src.preregistered import run_loo_and_val

LAG_YEARS = 5.0


def build_lagged_demos(data_dir: Path, demo_indicators: list[str]) -> pd.DataFrame:
    """Wide (commune, availability_date) demographics → merge_asof at date and
    date-LAG_YEARS to get current and lagged vintage per BV×election row."""
    demos = pd.read_parquet(data_dir / "baseline_cache" / "demographics.parquet")
    demos = demos[demos["candidate"].isin(demo_indicators)]
    wide = (
        demos.pivot_table(
            index=["location", "availability_date"],
            columns="candidate",
            values="value",
        )
        .reset_index()
        .rename(columns={"location": "commune"})
        .sort_values("availability_date")
    )
    return wide


def attach_demo_lags(df: pd.DataFrame, wide: pd.DataFrame, inds: list[str]):
    df = df.sort_values("date_float").copy()
    df["_lagdate"] = df["date_float"].astype("float64") - LAG_YEARS
    wide = wide.copy()
    wide["availability_date"] = wide["availability_date"].astype("float64")
    lag = wide.rename(columns={c: f"{c}__lag" for c in inds})
    df = pd.merge_asof(
        df,
        lag,
        left_on="_lagdate",
        right_on="availability_date",
        by="commune",
        direction="backward",
    )
    # Keep only rows that actually have a lagged vintage (date−5 ≥ demo start);
    # median-impute the few indicators sparse at that vintage (as the model does).
    df = df[df["availability_date"].notna()].copy()
    lag_cols, delta_cols = [], []
    for c in inds:
        lc, dc = f"{c}__lag", f"{c}__delta"
        if not df[lc].notna().any():
            continue
        df[lc] = df[lc].fillna(df[lc].median())
        df[dc] = df[c] - df[lc]
        lag_cols.append(lc)
        delta_cols.append(dc)
    print(f"  kept {len(lag_cols)}/{len(inds)} lagged indicators")
    return df, lag_cols, delta_cols


def run_compare(df, demo_indicators, national_means, est, extra_cols, pca_k, label):
    type_cols = [c for c in df.columns if c.startswith("type_")]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]

    need = demo_indicators + raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2 + extra_cols
    base = df.dropna(subset=need).copy()
    nd_ct = dev_lag1 + dev_lag2 + type_cols
    cfg = {"n_demo": len(demo_indicators)}
    if pca_k:
        cfg["pca_k"] = pca_k

    r_base = run_loo_and_val(
        "b", base, demo_indicators + nd_ct, est, national_means, dict(cfg)
    )
    r_ext = run_loo_and_val(
        "e", base, demo_indicators + nd_ct + extra_cols, est, national_means, dict(cfg)
    )

    pca_tag = f"PCA{pca_k}" if pca_k else "full"
    print(f"\n  [{label} | {pca_tag} | n={len(base):,} | +{len(extra_cols)} cols]")
    print(f"  {'block':14s} {'OOFΔ':>9s}   {'OOF base→ext':>20s}  selects(LOO)?")
    for tc in TARGET_COLS:
        ob, oe = r_base[tc]["oof_r2"], r_ext[tc]["oof_r2"]
        d = oe - ob
        sel = "YES" if d > 0.0005 else ("~tie" if abs(d) <= 0.0005 else "no")
        print(f"  {tc:14s} {d:+9.4f}   ({ob:.4f}→{oe:.4f})  {sel}")


def main():
    data_dir = Path("data")
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)
    add_election_type_onehot(df)

    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    est["Abstention"] = estimate_national_abstention_from_gaps(national_means)[0]

    print("Building lagged demographics (merge_asof)...")
    wide = build_lagged_demos(data_dir, demo_indicators)
    df, lag_cols, delta_cols = attach_demo_lags(df, wide, demo_indicators)
    print(f"  delta coverage: {df[delta_cols].notna().all(axis=1).mean() * 100:.1f}%")

    curated = [
        f"{c}__delta"
        for c in [
            "Taux_Chomage",
            "Pct_Cadres",
            "Pct_Ouvriers",
            "Pct_Immigres",
            "Pct_Sans_Diplome",
            "Pct_Bac_Plus_5",
            "Pct_HLM",
            "Pct_Retraites",
            "Pct_Age_60_Plus",
        ]
        if f"{c}__delta" in delta_cols
    ]
    print(
        f"\n{'=' * 70}\nC. CURATED deltas ({len(curated)} cols), full-demo\n{'=' * 70}"
    )
    run_compare(df, demo_indicators, national_means, est, curated, None, "curated-Δ")

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    z = StandardScaler().fit_transform(df[delta_cols].values.astype(np.float64))
    pcs = PCA(n_components=5).fit_transform(z)
    pc_cols = [f"deltaPC{i}" for i in range(5)]
    for i, col in enumerate(pc_cols):
        df[col] = pcs[:, i]
    print(f"\n{'=' * 70}\nD. delta-PCA (5 comps from {len(delta_cols)} Δ)\n{'=' * 70}")
    run_compare(df, demo_indicators, national_means, est, pc_cols, None, "delta-PCA5")


if __name__ == "__main__":
    main()
