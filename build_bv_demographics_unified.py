"""Single unified demographic table — one row per bureau de vote.

Best-available resolution per indicator: IRIS-weighted value where it exists,
commune value otherwise. Uses the most recent vintage available per indicator
(latest `availability_date`). Adds mobility/churn indicators, a per-BV IRIS
coverage fraction, and the BV->IRIS concentration (hhi).

Output: data/baseline_cache/bv_demographics_unified.parquet  (one row / bv_key)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.iris_features import build_bv_iris_demographics
from src.mobility_features import build_commune_mobility, build_bv_iris_mobility


def _latest_per_location(tokens: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent (by availability_date) value per location × candidate."""
    tokens = tokens.reset_index(drop=True)
    idx = tokens.groupby(["location", "candidate"])["availability_date"].idxmax()
    return tokens.loc[idx, ["location", "candidate", "value"]]


def _wide(tokens: pd.DataFrame, key: str) -> pd.DataFrame:
    w = _latest_per_location(tokens).pivot(
        index="location", columns="candidate", values="value"
    )
    w.index.name = key
    return w


def main():
    data_dir = Path("data")

    commune = pd.read_parquet(data_dir / "baseline_cache" / "demographics.parquet")
    iris = build_bv_iris_demographics(data_dir, use_cache=True)
    mob_com = build_commune_mobility(data_dir, use_cache=True)
    mob_iris = build_bv_iris_mobility(data_dir, use_cache=True)
    weights = pd.read_parquet(data_dir / "geo" / "bv_iris_weights.parquet")

    # BV -> commune map and the BV->IRIS concentration (hhi is constant per bv_key)
    bv_keys = pd.Index(weights["bv_key"].unique(), name="bv_key")
    hhi = weights.drop_duplicates("bv_key").set_index("bv_key")["hhi"]
    bv_commune = pd.Series(bv_keys.str.split("_").str[0], index=bv_keys, name="commune")

    # Wide tables at each resolution
    com_wide = _wide(pd.concat([commune, mob_com]), "commune")  # commune-keyed
    iris_wide = _wide(pd.concat([iris, mob_iris]), "bv_key")  # bv-keyed

    indicators = sorted(set(com_wide.columns) | set(iris_wide.columns))

    # Assemble one row per BV: IRIS value where present, else commune value
    out = pd.DataFrame(index=bv_keys)
    out["commune"] = bv_commune
    com_for_bv = com_wide.reindex(bv_commune.values)
    com_for_bv.index = bv_keys
    iris_for_bv = iris_wide.reindex(bv_keys)

    n_from_iris = pd.Series(0, index=bv_keys)
    n_total = pd.Series(0, index=bv_keys)
    for ind in indicators:
        iv = (
            iris_for_bv[ind]
            if ind in iris_for_bv.columns
            else pd.Series(np.nan, index=bv_keys)
        )
        cv = (
            com_for_bv[ind]
            if ind in com_for_bv.columns
            else pd.Series(np.nan, index=bv_keys)
        )
        out[ind] = iv.where(iv.notna(), cv)
        present = out[ind].notna()
        n_total += present.astype(int)
        n_from_iris += (iv.notna() & present).astype(int)

    out["iris_coverage"] = (n_from_iris / n_total.replace(0, np.nan)).astype(np.float32)
    out["bv_iris_hhi"] = hhi.reindex(bv_keys).astype(np.float32)

    cache = data_dir / "baseline_cache" / "bv_demographics_unified.parquet"
    out.reset_index().to_parquet(cache, index=False)

    print(f"Unified BV demographics → {cache}")
    print(f"  {len(out):,} bureaux × {len(indicators)} indicators + metadata")
    print(f"  mean IRIS coverage: {out['iris_coverage'].mean():.1%} of indicators")
    print(
        f"  BVs >=50% IRIS-sourced: {(out['iris_coverage'] >= 0.5).mean():.1%}  "
        f"| commune-only fallback: {(out['iris_coverage'].fillna(0) == 0).mean():.1%}"
    )
    fully_missing = out[indicators].isna().all(axis=1).sum()
    print(f"  BVs with no demographics at all: {fully_missing:,}")
    print(f"\n  sample (first 3 BVs, 6 indicators):")
    cols = ["commune", "iris_coverage", "bv_iris_hhi"] + indicators[:6]
    print(out[cols].head(3).to_string())


if __name__ == "__main__":
    main()
