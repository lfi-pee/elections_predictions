"""IRIS-resolution demographics aggregated to BV level.

Reuses the validated commune indicator-builders in `load_demographics`
(identical INSEE variable names at IRIS granularity) by patching the file
reader to synthesise a `CODGEO` column from `IRIS`, then area/population-weights
each IRIS indicator onto its bureaux de vote via `data/geo/bv_iris_weights.parquet`.

Output token schema matches `load_demographic_tokens`, keyed by `bv_key`
(location), so it merges into the existing harness unchanged.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import src.load_demographics as ld

warnings.filterwarnings("ignore", category=FutureWarning)

_BUILDERS = (
    ld._load_census_activity_vintage,
    ld._load_census_education_vintage,
    ld._load_census_population_vintage,
    ld._load_census_housing_vintage,
    ld._load_census_families_vintage,
)


def _read_insee_iris(path: Path) -> pd.DataFrame | None:
    """Read an INSEE IRIS file and synthesise CODGEO from the IRIS key."""
    if not path.exists():
        return None
    frames: list[pd.DataFrame] = []
    try:
        if path.suffix in (".xlsx", ".xls"):
            xls = pd.ExcelFile(path)
            for sheet in xls.sheet_names:
                for skip in (0, 5):
                    try:
                        df = pd.read_excel(
                            xls, sheet_name=sheet, skiprows=skip, dtype={"IRIS": str}
                        )
                    except Exception:
                        continue
                    if "IRIS" in df.columns:
                        frames.append(df)
                        break
                if frames:
                    break
        else:
            for sep in (";", ",", "\t"):
                try:
                    df = pd.read_csv(
                        path, sep=sep, dtype={"IRIS": str}, low_memory=False
                    )
                except Exception:
                    continue
                if "IRIS" in df.columns:
                    frames.append(df)
                    break
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None

    if not frames:
        return None
    df = frames[0]
    df["CODGEO"] = (
        df["IRIS"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(9)
    )
    return df


def load_iris_tokens(data_dir: Path) -> pd.DataFrame:
    """Load IRIS census tokens (location = 9-digit IRIS code)."""
    iris_root = data_dir / "demographics" / "census_iris"
    if not iris_root.exists():
        raise FileNotFoundError(iris_root)

    orig_reader = ld._read_insee_file
    ld._read_insee_file = _read_insee_iris
    try:
        all_frames: list[pd.DataFrame] = []
        for vdir in sorted(
            d for d in iris_root.iterdir() if d.is_dir() and d.name.isdigit()
        ):
            vintage = int(vdir.name)
            date_float, avail = ld._census_dates(vintage)
            frames = [b(vdir, date_float, avail) for b in _BUILDERS]
            frames = [f for f in frames if len(f) > 0]
            if not frames:
                print(f"  IRIS {vintage}: no indicators (column mismatch?)")
                continue
            combined = pd.concat(frames, ignore_index=True)
            print(
                f"  IRIS {vintage}: {len(combined):,} tokens "
                f"({combined['candidate'].nunique()} ind × "
                f"{combined['location'].nunique():,} IRIS) [avail={avail:.1f}]"
            )
            all_frames.append(combined)
    finally:
        ld._read_insee_file = orig_reader

    return pd.concat(all_frames, ignore_index=True)


def aggregate_to_bv(iris_tokens: pd.DataFrame, weights: pd.DataFrame) -> pd.DataFrame:
    """Weighted-mean IRIS indicators onto BV keys, per vintage.

    Returns tokens with location = bv_key, matching the commune token schema.
    """
    w = weights[["bv_key", "code_iris", "weight"]].copy()
    w["code_iris"] = w["code_iris"].astype(str).str.zfill(9)

    out: list[pd.DataFrame] = []
    for (avail, cand), grp in iris_tokens.groupby(
        ["availability_date", "candidate"], sort=False
    ):
        vals = grp[["location", "value", "date_float"]].rename(
            columns={"location": "code_iris"}
        )
        m = w.merge(vals, on="code_iris", how="inner")
        m = m[m["value"].notna()]
        if m.empty:
            continue
        m["wv"] = m["weight"] * m["value"]
        agg = m.groupby("bv_key").agg(wv=("wv", "sum"), wsum=("weight", "sum"))
        agg = agg[agg["wsum"] > 0]
        out.append(
            pd.DataFrame(
                {
                    "date_float": np.float32(grp["date_float"].iloc[0]),
                    "availability_date": np.float32(avail),
                    "election_type": "",
                    "location": agg.index.values,
                    "candidate": cand,
                    "value": (agg["wv"] / agg["wsum"]).values,
                }
            )
        )
    return pd.concat(out, ignore_index=True)


def build_bv_iris_demographics(data_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    """Full pipeline: IRIS census → BV-weighted demographic tokens (cached)."""
    cache = data_dir / "baseline_cache" / "bv_iris_demographics.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)

    weights = pd.read_parquet(data_dir / "geo" / "bv_iris_weights.parquet")
    iris_tokens = load_iris_tokens(data_dir)
    bv_tokens = aggregate_to_bv(iris_tokens, weights)
    cache.parent.mkdir(parents=True, exist_ok=True)
    bv_tokens.to_parquet(cache, index=False)
    return bv_tokens


if __name__ == "__main__":
    toks = build_bv_iris_demographics(Path("data"), use_cache=False)
    print(
        f"\nBV-IRIS tokens: {len(toks):,} rows, "
        f"{toks['candidate'].nunique()} indicators, "
        f"{toks['location'].nunique():,} BV keys, "
        f"{toks['availability_date'].nunique()} vintages"
    )
