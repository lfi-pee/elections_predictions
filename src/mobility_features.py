"""Residential-mobility / churn indicators from the census.

Two untapped census signals, strong a-priori predictors of abstention
(recent movers are frequently mal-inscrits — registered elsewhere):

  IRAN (résidence antérieure, evol-struct-pop):
    Pct_Mobilite_Recente   = % pop 1y+ who changed dwelling in the last year
    Pct_Migrant_Commune    = % who arrived from another commune (IRAN3-7)
  ANEM (ancienneté d'emménagement, logement):
    Pct_Emmenage_0_2ans    = % households moved in within 2 years
    Pct_Emmenage_0_4ans    = % households moved in within 4 years

Builders share the `(vintage_dir, date_float, avail)` signature of the
existing census builders and reuse `load_demographics` helpers, so they run
unchanged over commune (`census/`) or, with the IRIS reader patch, IRIS files.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import src.load_demographics as ld
from src.iris_features import _read_insee_iris, aggregate_to_bv

MOBILITY_INDICATORS = [
    "Pct_Mobilite_Recente",
    "Pct_Migrant_Commune",
    "Pct_Emmenage_0_2ans",
    "Pct_Emmenage_0_4ans",
]


def _build_iran(vintage_dir: Path, date_float: float, avail: float) -> pd.DataFrame:
    path = ld._glob_first(vintage_dir, "*evol*struct*pop*", "*evol*", "*struct*pop*")
    if path is None:
        return ld._empty_df()
    df = ld._read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return ld._empty_df()
    yy = ld._detect_prefix(df)
    if yy is None:
        return ld._empty_df()
    codgeo = df["CODGEO"].astype(str)
    pop = ld._find_col(df, f"P{yy}_POP01P")
    iran1 = ld._find_col(df, f"P{yy}_POP01P_IRAN1")
    if not pop or not iran1:
        return ld._empty_df()
    pop_v = pd.to_numeric(df[pop], errors="coerce")
    iran1_v = pd.to_numeric(df[iran1], errors="coerce")
    frames = [
        ld._make_tokens(
            codgeo,
            "Pct_Mobilite_Recente",
            (1.0 - ld._safe_ratio(iran1_v, pop_v)) * 100.0,
            date_float,
            avail,
        )
    ]
    migr = None
    for n in (3, 4, 5, 6, 7):
        c = ld._find_col(df, f"P{yy}_POP01P_IRAN{n}")
        if c:
            v = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
            migr = v if migr is None else migr + v
    if migr is not None:
        frames.append(
            ld._make_tokens(
                codgeo,
                "Pct_Migrant_Commune",
                ld._safe_ratio(migr, pop_v) * 100.0,
                date_float,
                avail,
            )
        )
    return (
        pd.concat([f for f in frames if len(f) > 0], ignore_index=True)
        if frames
        else ld._empty_df()
    )


def _build_anem(vintage_dir: Path, date_float: float, avail: float) -> pd.DataFrame:
    path = ld._glob_first(vintage_dir, "*logement*", "*LOG*", "*log*")
    if path is None:
        return ld._empty_df()
    df = ld._read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return ld._empty_df()
    yy = ld._detect_prefix(df)
    if yy is None:
        return ld._empty_df()
    codgeo = df["CODGEO"].astype(str)
    denom_c = ld._find_col(df, f"P{yy}_RP") or ld._find_col(df, f"P{yy}_MEN")
    a02 = ld._find_col(df, f"P{yy}_MEN_ANEM0002")
    a24 = ld._find_col(df, f"P{yy}_MEN_ANEM0204")
    if not denom_c or not a02:
        return ld._empty_df()
    denom = pd.to_numeric(df[denom_c], errors="coerce")
    v02 = pd.to_numeric(df[a02], errors="coerce")
    frames = [
        ld._make_tokens(
            codgeo,
            "Pct_Emmenage_0_2ans",
            ld._safe_ratio(v02, denom) * 100.0,
            date_float,
            avail,
        )
    ]
    if a24:
        v24 = pd.to_numeric(df[a24], errors="coerce").fillna(0.0)
        frames.append(
            ld._make_tokens(
                codgeo,
                "Pct_Emmenage_0_4ans",
                ld._safe_ratio(v02.fillna(0.0) + v24, denom) * 100.0,
                date_float,
                avail,
            )
        )
    return (
        pd.concat([f for f in frames if len(f) > 0], ignore_index=True)
        if frames
        else ld._empty_df()
    )


def _load_mobility(census_subdir: str, data_dir: Path, iris: bool) -> pd.DataFrame:
    root = data_dir / "demographics" / census_subdir
    orig = ld._read_insee_file
    if iris:
        ld._read_insee_file = _read_insee_iris
    try:
        frames = []
        for vdir in sorted(
            d for d in root.iterdir() if d.is_dir() and d.name.isdigit()
        ):
            vintage = int(vdir.name)
            date_float, avail = ld._census_dates(vintage)
            for builder in (_build_iran, _build_anem):
                f = builder(vdir, date_float, avail)
                if len(f) > 0:
                    frames.append(f)
    finally:
        ld._read_insee_file = orig
    return pd.concat(frames, ignore_index=True) if frames else ld._empty_df()


def build_commune_mobility(data_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    cache = data_dir / "baseline_cache" / "commune_mobility.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    toks = _load_mobility("census", data_dir, iris=False)
    cache.parent.mkdir(parents=True, exist_ok=True)
    toks.to_parquet(cache, index=False)
    return toks


def build_bv_iris_mobility(data_dir: Path, use_cache: bool = True) -> pd.DataFrame:
    cache = data_dir / "baseline_cache" / "bv_iris_mobility.parquet"
    if use_cache and cache.exists():
        return pd.read_parquet(cache)
    iris_toks = _load_mobility("census_iris", data_dir, iris=True)
    weights = pd.read_parquet(data_dir / "geo" / "bv_iris_weights.parquet")
    bv = aggregate_to_bv(iris_toks, weights)
    cache.parent.mkdir(parents=True, exist_ok=True)
    bv.to_parquet(cache, index=False)
    return bv


if __name__ == "__main__":
    c = build_commune_mobility(Path("data"), use_cache=False)
    print(
        f"commune mobility: {len(c):,} tokens, {c['candidate'].nunique()} ind, "
        f"{c['location'].nunique():,} communes, vintages "
        f"{sorted(c['availability_date'].unique())}"
    )
    b = build_bv_iris_mobility(Path("data"), use_cache=False)
    print(
        f"BV-IRIS mobility: {len(b):,} tokens, {b['candidate'].nunique()} ind, "
        f"{b['location'].nunique():,} BV keys"
    )
