"""Experiment: does DVF price/m² (per commune) get selected by the LOO criterion?

Adds commune-level median price/m² as a non-demo (non-PCA) Ridge feature and
compares LOO OOF R² with vs without it, reusing the pre-registered harness.

DVF starts 2014 → price exists for 2017/2022/2024 elections, NaN for
2002/2007/2012. Two scenarios:
  (A) all folds, price median-imputed (what the current pipeline would do)
  (B) only price-observed folds (2017/2022 train, 2024 val) — fair contrast
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import gzip
import numpy as np
import pandas as pd
from pathlib import Path

from src.cross_type_dev import (
    load_cross_type_data,
    add_election_type_onehot,
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
)
from src.preregistered import run_loo_and_val

DVF_YEARS = [2021, 2022, 2023, 2024]
DVF_DIR = Path("/tmp")
CACHE = Path("data/baseline_cache/dvf_price_per_commune.parquet")
RESIDENTIAL = {"Maison", "Appartement"}


def aggregate_dvf_year(year: int) -> pd.DataFrame:
    path = DVF_DIR / f"dvf_{year}.csv.gz"
    usecols = [
        "nature_mutation",
        "valeur_fonciere",
        "code_commune",
        "type_local",
        "surface_reelle_bati",
    ]
    with gzip.open(path, "rt") as fh:
        df = pd.read_csv(fh, usecols=usecols, low_memory=False)
    df = df[
        (df["nature_mutation"] == "Vente")
        & (df["type_local"].isin(RESIDENTIAL))
        & (df["surface_reelle_bati"] > 0)
        & (df["valeur_fonciere"] > 0)
    ].copy()
    df["ppm2"] = df["valeur_fonciere"] / df["surface_reelle_bati"]
    df = df[(df["ppm2"] >= 200) & (df["ppm2"] <= 30000)]
    agg = (
        df.groupby("code_commune")
        .agg(prix_m2=("ppm2", "median"), n_ventes=("ppm2", "size"))
        .reset_index()
    )
    agg["year"] = year
    return agg


def build_price_table() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    parts = [aggregate_dvf_year(y) for y in DVF_YEARS]
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(CACHE, index=False)
    return out


PRICE_FEATS = ["prix_static", "prix_growth", "prix_rel_dept", "log_nventes"]


def build_commune_features(price: pd.DataFrame) -> pd.DataFrame:
    """Engineer orthogonal commune-level price signals from the per-year table."""
    p = price.rename(columns={"code_commune": "commune"}).copy()
    wide = p.pivot_table(index="commune", columns="year", values="prix_m2")
    nv = p.groupby("commune")["n_ventes"].sum()

    feat = pd.DataFrame(index=wide.index)
    feat["prix_static_raw"] = wide.median(axis=1)
    feat["prix_static"] = np.log(feat["prix_static_raw"])
    # Gentrification: log price growth over the available window
    if 2021 in wide.columns and 2024 in wide.columns:
        feat["prix_growth"] = np.log(wide[2024]) - np.log(wide[2021])
    else:
        feat["prix_growth"] = np.nan
    feat["log_nventes"] = np.log(nv)
    feat = feat.reset_index()
    # Relative standing within department (first 2 chars of INSEE code)
    feat["dept"] = feat["commune"].str[:2]
    dept_med = feat.groupby("dept")["prix_static"].transform("median")
    feat["prix_rel_dept"] = feat["prix_static"] - dept_med
    return feat.drop(columns=["dept"])


def attach_price(df: pd.DataFrame, price: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    feat = build_commune_features(price)
    df = df.merge(feat, on="commune", how="left")
    return df


def run_compare(df, demo_indicators, national_means, est, price_feats, pca_k, label):
    type_cols = [c for c in df.columns if c.startswith("type_")]
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    raw_lag1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    raw_lag2 = [f"{b}_lag2" for b in BLOCKS_ABS]

    base = df.dropna(
        subset=demo_indicators + raw_lag1 + raw_lag2 + dev_lag1 + dev_lag2 + price_feats
    ).copy()

    nd_ct = dev_lag1 + dev_lag2 + type_cols
    feats_base = demo_indicators + nd_ct
    feats_price = demo_indicators + nd_ct + price_feats
    cfg = {"n_demo": len(demo_indicators)}
    if pca_k:
        cfg["pca_k"] = pca_k

    r_base = run_loo_and_val("base", base, feats_base, est, national_means, dict(cfg))
    r_price = run_loo_and_val(
        "price", base, feats_price, est, national_means, dict(cfg)
    )

    pca_tag = f"PCA{pca_k}" if pca_k else "full-demo"
    print(
        f"\n  [{label} | {pca_tag} | n={len(base):,} | feats={'+'.join(price_feats)}]"
    )
    print(
        f"  {'block':14s} {'OOFΔ':>8s} {'ValΔ':>8s}   "
        f"({'OOF b→p':>16s}) ({'Val b→p':>16s})  verdict"
    )
    for tc in TARGET_COLS:
        ob, op = r_base[tc]["oof_r2"], r_price[tc]["oof_r2"]
        vb, vp = r_base[tc]["val_r2"], r_price[tc]["val_r2"]
        do, dv = op - ob, vp - vb
        # real selection = OOF up AND Val not worse
        if do > 0.0005 and dv > -0.0002:
            verdict = "SELECT ✓"
        elif do > 0.0005:
            verdict = "OOF-only"
        elif abs(do) <= 0.0005:
            verdict = "~tie"
        else:
            verdict = "no"
        print(
            f"  {tc:14s} {do:+8.4f} {dv:+8.4f}   "
            f"({ob:.4f}→{op:.4f}) ({vb:.4f}→{vp:.4f})  {verdict}"
        )


def main():
    data_dir = Path("data")
    print("Building DVF price/m² per commune...")
    price = build_price_table()
    for y in DVF_YEARS:
        sub = price[price["year"] == y]
        print(
            f"  {y}: {len(sub):,} communes, median ppm² = {sub['prix_m2'].median():.0f} €"
        )

    df, demo_indicators, national_means, poll_feats = load_cross_type_data(data_dir)
    add_election_type_onehot(df)
    df = attach_price(df, price)

    from src.cross_type_ridge import TARGET_BLOCKS
    from src.cross_type_dev import estimate_national_abstention_from_gaps

    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    est["Abstention"] = abs_pred

    cov = df["prix_static_raw"].notna()
    print(f"\nStatic-price coverage: {cov.mean() * 100:.1f}% of BV×election rows")
    print(f"prix_growth coverage: {df['prix_growth'].notna().mean() * 100:.1f}%")

    # Engineered single features, full-demo (each its own shot)
    print(f"\n{'=' * 90}\nA. SINGLE ENGINEERED FEATURES (full-demo Ridge)\n{'=' * 90}")
    for f in PRICE_FEATS:
        run_compare(df, demo_indicators, national_means, est, [f], None, f)

    # Best orthogonal signals on the PCA-compressed config (price recovers
    # variance PCA discarded)
    print(f"\n{'=' * 90}\nB. ON PCA-COMPRESSED DEMO (k=5, k=7)\n{'=' * 90}")
    for k in (5, 7):
        run_compare(
            df, demo_indicators, national_means, est, ["prix_static"], k, "stat"
        )
        run_compare(
            df, demo_indicators, national_means, est, ["prix_growth"], k, "growth"
        )

    # All four together, full-demo and PCA7
    print(f"\n{'=' * 90}\nC. ALL FOUR PRICE FEATURES TOGETHER\n{'=' * 90}")
    run_compare(df, demo_indicators, national_means, est, PRICE_FEATS, None, "all")
    run_compare(df, demo_indicators, national_means, est, PRICE_FEATS, 7, "all")


if __name__ == "__main__":
    main()
