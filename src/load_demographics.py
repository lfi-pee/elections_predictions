"""Load INSEE demographic data as universal tokens.

Reads Census (activité, diplômes, population) and BPE (Base Permanente des
Équipements) commune-level data and converts each indicator × commune into
a DataToken with metric_type='Demographics'.

Only the most recent vintage of each source is loaded.  Each token carries
an ``availability_date`` (publication date) so the router can enforce
causality: a token is only visible to the model when predicting elections
that happen *after* the data was published.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# Latest available vintages
CENSUS_VINTAGE = 2021
BPE_VINTAGE = 2024

# date_float = what the data describes (vintage midpoint)
# Census vintage 2021 pools surveys from 2017–2021, centred ~2019.5
CENSUS_DATE_FLOAT = np.float32(CENSUS_VINTAGE - 1.5)
# BPE 2024 = situation as of Jan 1 2024
BPE_DATE_FLOAT = np.float32(BPE_VINTAGE)

# availability_date = when the data was actually published
# Census 2021 detailed commune stats: published ~June 2024
CENSUS_AVAILABILITY_DATE = np.float32(2024.5)
# BPE 2024: published July 2025
BPE_AVAILABILITY_DATE = np.float32(2025.5)


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date_float", "availability_date", "election_type", "location",
            "candidate", "party", "metric_type", "value",
            "latitude", "longitude",
        ]
    )


def _read_insee_file(path: Path) -> pd.DataFrame | None:
    """Read an INSEE data file (Excel or CSV) and return the data sheet."""
    if not path.exists():
        return None
    try:
        if path.suffix in (".xlsx", ".xls"):
            xls = pd.ExcelFile(path)
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name, dtype={"CODGEO": str})
                if "CODGEO" in df.columns:
                    return df
            return pd.read_excel(xls, sheet_name=0)
        else:
            for sep in [";", ",", "\t"]:
                try:
                    df = pd.read_csv(path, sep=sep, dtype={"CODGEO": str}, low_memory=False)
                    if "CODGEO" in df.columns:
                        return df
                except Exception:
                    continue
            return None
    except Exception as e:
        print(f"  Warning: could not read {path}: {e}")
        return None


def _find_col(df: pd.DataFrame, exact: str) -> str | None:
    """Find a column by exact name (case-insensitive)."""
    for col in df.columns:
        if col.upper() == exact.upper():
            return col
    return None


def _safe_ratio(num: pd.Series, denom: pd.Series) -> pd.Series:
    return num / denom.replace(0, np.nan)


def _make_tokens(
    codgeo: pd.Series,
    indicator: str,
    values: pd.Series,
    date_float: float,
    availability_date: float,
) -> pd.DataFrame:
    """Create token rows for a single indicator."""
    mask = values.notna() & np.isfinite(values)
    if not mask.any():
        return pd.DataFrame()
    return pd.DataFrame({
        "date_float": date_float,
        "availability_date": availability_date,
        "election_type": "",
        "location": codgeo[mask].values,
        "candidate": indicator,
        "party": "",
        "metric_type": "Demographics",
        "value": values[mask].astype(np.float32).values,
    })


def _glob_first(directory: Path, *patterns: str) -> Path | None:
    """Return the first file matching any of the glob patterns."""
    for pat in patterns:
        hits = sorted(directory.glob(pat))
        if hits:
            return hits[0]
    return None


def _load_census_activity(demo_dir: Path) -> pd.DataFrame:
    """P0: unemployment rate, % ouvriers, % cadres."""
    census_dir = demo_dir / "census"
    if not census_dir.exists():
        return _empty_df()

    path = _glob_first(census_dir, "*activ*", "*ACT*", "*activite*")
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = str(CENSUS_VINTAGE)[-2:]
    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    # Unemployment rate = CHOM / ACT * 100
    chom = _find_col(df, f"P{yy}_CHOM1564")
    act = _find_col(df, f"P{yy}_ACT1564")
    if chom and act:
        rate = _safe_ratio(
            pd.to_numeric(df[chom], errors="coerce"),
            pd.to_numeric(df[act], errors="coerce"),
        ) * 100.0
        frames.append(_make_tokens(codgeo, "Taux_Chomage", rate, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE))

    # CSP breakdown
    csp_series: dict[int, pd.Series] = {}
    for i in range(1, 7):
        col = _find_col(df, f"C{yy}_ACT_CSP{i}")
        if col:
            csp_series[i] = pd.to_numeric(df[col], errors="coerce")

    if len(csp_series) >= 2:
        total_csp = sum(csp_series.values())
        if 6 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Ouvriers",
                _safe_ratio(csp_series[6], total_csp) * 100.0, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE,
            ))
        if 3 in csp_series:
            frames.append(_make_tokens(
                codgeo, "Pct_Cadres",
                _safe_ratio(csp_series[3], total_csp) * 100.0, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE,
            ))

    if not frames:
        return _empty_df()
    print(f"  Census ACT: loaded {sum(len(f) for f in frames)} tokens from {path.name}")
    return pd.concat(frames, ignore_index=True)





def _load_census_education(demo_dir: Path) -> pd.DataFrame:
    """P1: % sans diplôme, % bac+5."""
    census_dir = demo_dir / "census"
    if not census_dir.exists():
        return _empty_df()

    path = _glob_first(census_dir, "*diplom*", "*DIPL*", "*form*", "*FOR*")
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = str(CENSUS_VINTAGE)[-2:]
    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    denom_col = _find_col(df, f"P{yy}_NSCOL15P")
    if not denom_col:
        return _empty_df()
    denom = pd.to_numeric(df[denom_col], errors="coerce")

    nodip = _find_col(df, f"P{yy}_NSCOL15P_DIPLMIN")
    if nodip:
        pct = _safe_ratio(pd.to_numeric(df[nodip], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Sans_Diplome", pct, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE))

    sup5 = _find_col(df, f"P{yy}_NSCOL15P_SUP5")
    if sup5:
        pct = _safe_ratio(pd.to_numeric(df[sup5], errors="coerce"), denom) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Bac_Plus_5", pct, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE))

    if not frames:
        return _empty_df()
    print(f"  Census FOR: loaded {sum(len(f) for f in frames)} tokens from {path.name}")
    return pd.concat(frames, ignore_index=True)


def _load_census_population(demo_dir: Path) -> pd.DataFrame:
    """P1: % age 18-24 (proxy 15-24), % age 60+, % immigrants."""
    census_dir = demo_dir / "census"
    if not census_dir.exists():
        return _empty_df()

    path = _glob_first(census_dir, "*pop*", "*POP*", "*evol*struct*")
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return _empty_df()

    yy = str(CENSUS_VINTAGE)[-2:]
    codgeo = df["CODGEO"].astype(str)
    frames: list[pd.DataFrame] = []

    pop_col = _find_col(df, f"P{yy}_POP")
    if not pop_col:
        return _empty_df()
    pop = pd.to_numeric(df[pop_col], errors="coerce")

    young = _find_col(df, f"P{yy}_POP1524")
    if young:
        pct = _safe_ratio(pd.to_numeric(df[young], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_18_24", pct, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE))

    e60 = _find_col(df, f"P{yy}_POP6074")
    e75 = _find_col(df, f"P{yy}_POP75P")
    if e60 and e75:
        elder = (pd.to_numeric(df[e60], errors="coerce")
                 + pd.to_numeric(df[e75], errors="coerce"))
        pct = _safe_ratio(elder, pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Age_60_Plus", pct, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE))

    imm = _find_col(df, f"P{yy}_POP_IMM")
    if imm:
        pct = _safe_ratio(pd.to_numeric(df[imm], errors="coerce"), pop) * 100.0
        frames.append(_make_tokens(codgeo, "Pct_Immigres", pct, CENSUS_DATE_FLOAT, CENSUS_AVAILABILITY_DATE))

    if not frames:
        return _empty_df()
    print(f"  Census POP: loaded {sum(len(f) for f in frames)} tokens from {path.name}")
    return pd.concat(frames, ignore_index=True)


def _load_bpe(demo_dir: Path, pop_data: pd.DataFrame | None = None) -> pd.DataFrame:
    """P2: equipment density per 1,000 inhabitants.

    BPE format: one row per (CODGEO, TYPEQU) with NB_EQUIP count.
    We aggregate selected equipment types and normalise by population.
    """
    bpe_dir = demo_dir / "bpe"
    if not bpe_dir.exists():
        return _empty_df()

    path = _glob_first(bpe_dir, "*bpe*", "*BPE*", "*.csv", "*.xlsx")
    if path is None:
        return _empty_df()

    df = _read_insee_file(path)
    if df is None:
        return _empty_df()

    # Normalise column names
    df.columns = [c.upper().strip() for c in df.columns]
    codgeo_col = "CODGEO" if "CODGEO" in df.columns else "DEPCOM" if "DEPCOM" in df.columns else None
    typequ_col = "TYPEQU" if "TYPEQU" in df.columns else None
    nb_col = "NB_EQUIP" if "NB_EQUIP" in df.columns else None

    if codgeo_col is None or typequ_col is None or nb_col is None:
        print(f"  BPE: missing required columns in {path.name}", flush=True)
        return _empty_df()

    df[codgeo_col] = df[codgeo_col].astype(str)
    df[nb_col] = pd.to_numeric(df[nb_col], errors="coerce").fillna(0)

    # Equipment codes of interest  →  indicator name
    EQUIP_INDICATORS: dict[str, list[str]] = {
        "BPE_Medecins_per_1k": ["D201"],
        "BPE_Pharmacies_per_1k": ["D301"],
        "BPE_Postes_per_1k": ["A206", "A504"],  # code changed over vintages
        "BPE_Supermarches_per_1k": ["B101", "B105"],
    }

    # Get population per commune for normalisation
    if pop_data is not None and len(pop_data) > 0:
        pop_lookup = pop_data.set_index("location")["value"].to_dict()
    else:
        pop_lookup = {}

    frames: list[pd.DataFrame] = []

    for indicator, codes in EQUIP_INDICATORS.items():
        subset = df[df[typequ_col].isin(codes)]
        if len(subset) == 0:
            continue
        counts = subset.groupby(codgeo_col)[nb_col].sum()

        if pop_lookup:
            # Normalise per 1,000 inhabitants
            pop_series = counts.index.map(lambda c: pop_lookup.get(c, np.nan))
            pop_series = pd.Series(pop_series.values, index=counts.index, dtype=np.float64)
            rate = (counts / pop_series * 1000.0).replace([np.inf, -np.inf], np.nan)
        else:
            # Fallback: raw count (will be less useful but still loads)
            rate = counts.astype(np.float64)

        codgeo_series = pd.Series(counts.index.values, index=counts.index)
        frames.append(_make_tokens(
            codgeo_series, indicator, rate, BPE_DATE_FLOAT, BPE_AVAILABILITY_DATE,
        ))

    if not frames:
        return _empty_df()
    print(f"  BPE: loaded {sum(len(f) for f in frames)} tokens from {path.name}", flush=True)
    return pd.concat(frames, ignore_index=True)


def _merge_geo_coords(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge latitude/longitude from geo lookup onto a token DataFrame."""
    coords_path = data_dir / "geo" / "location_coords.parquet"
    if coords_path.exists():
        coords = pd.read_parquet(coords_path)
        df = df.merge(coords[["location", "latitude", "longitude"]], on="location", how="left")
    else:
        df["latitude"] = np.float32(np.nan)
        df["longitude"] = np.float32(np.nan)
    df["latitude"] = df["latitude"].fillna(46.2276).astype(np.float32)
    df["longitude"] = df["longitude"].fillna(2.2137).astype(np.float32)
    return df


def load_demographic_tokens(data_dir: Path) -> pd.DataFrame:
    """Load all demographic data as universal tokens.

    Each token carries an ``availability_date`` (publication date) so the
    router can enforce temporal causality.  Returns an empty DataFrame
    gracefully if no demographic data files are present.
    """
    demo_dir = data_dir / "demographics"
    if not demo_dir.exists():
        print("  No demographics directory found, skipping.", flush=True)
        return _empty_df()

    census_frames = [
        _load_census_activity(demo_dir),
        _load_census_education(demo_dir),
        _load_census_population(demo_dir),
    ]
    census_frames = [f for f in census_frames if len(f) > 0]

    # Try to get population data for BPE normalisation
    pop_data = None
    for f in census_frames:
        pop_subset = f[f["candidate"] == "Pct_Age_18_24"]
        if len(pop_subset) > 0:
            # We don't have raw pop, but we can load it from the census file
            break

    # Load raw population counts for BPE normalisation
    pop_data = _load_census_pop_raw(demo_dir)

    bpe_frame = _load_bpe(demo_dir, pop_data)

    all_frames = census_frames + ([bpe_frame] if len(bpe_frame) > 0 else [])

    if not all_frames:
        print("  No demographic data files found, skipping.", flush=True)
        return _empty_df()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = _merge_geo_coords(combined, data_dir)
    combined.sort_values("availability_date", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    n_indicators = combined["candidate"].nunique()
    n_communes = combined["location"].nunique()
    print(f"  Loaded {len(combined)} demographic tokens "
          f"({n_indicators} indicators × {n_communes} communes)", flush=True)

    return combined


def _load_census_pop_raw(demo_dir: Path) -> pd.DataFrame | None:
    """Load raw population counts from census for BPE normalisation.

    Returns a DataFrame with columns (location, value) where value is
    the total population.
    """
    census_dir = demo_dir / "census"
    if not census_dir.exists():
        return None

    path = _glob_first(census_dir, "*pop*", "*POP*", "*evol*struct*")
    if path is None:
        return None

    df = _read_insee_file(path)
    if df is None or "CODGEO" not in df.columns:
        return None

    yy = str(CENSUS_VINTAGE)[-2:]
    pop_col = _find_col(df, f"P{yy}_POP")
    if not pop_col:
        return None

    pop = pd.to_numeric(df[pop_col], errors="coerce")
    codgeo = df["CODGEO"].astype(str)
    valid = pop.notna()
    return pd.DataFrame({
        "location": codgeo[valid].values,
        "value": pop[valid].values,
    })
