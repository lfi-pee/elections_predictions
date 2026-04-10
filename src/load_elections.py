from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

ELECTION_MONTH: dict[str, float] = {
    "pres_t1": 4.0,
    "pres_t2": 5.0,
    "legi_t1": 6.0,
    "legi_t2": 6.5,
    "muni_t1": 3.0,
    "muni_t2": 3.5,
    "euro_t1": 6.0,
    "regi_t1": 6.0,
    "regi_t2": 6.5,
    "dpmt_t1": 3.0,
    "dpmt_t2": 3.5,
    "cant_t1": 3.0,
    "cant_t2": 3.5,
}

ELECTION_TYPE_LABEL: dict[str, str] = {
    "pres": "Presidentielle",
    "legi": "Legislatives",
    "muni": "Municipales",
    "euro": "Europeennes",
    "regi": "Regionales",
    "dpmt": "Departementales",
    "cant": "Cantonales",
}


def parse_election_id(id_election: str) -> tuple[float, str]:
    parts = id_election.split("_")
    year = int(parts[0])
    type_tour = "_".join(parts[1:])
    month = ELECTION_MONTH.get(type_tour, 6.0)
    date_float = year + month / 12.0

    label = ELECTION_TYPE_LABEL.get(parts[1], parts[1])
    tour = parts[2].upper() if len(parts) > 2 else ""
    return date_float, f"{label}_{tour}" if tour else label


def _bv_candidate_tokens(data_dir: Path) -> pd.DataFrame:
    """Load candidate results at bureau de vote level (NO commune aggregation).

    Each row becomes a token with location = "{code_commune}_{code_bv}".
    """
    df = pd.read_parquet(
        data_dir / "elections" / "agregees" / "candidats_results.parquet",
        columns=[
            "id_election", "code_commune", "code_bv", "nom", "prenom",
            "nuance", "voix", "libelle_abrege_liste", "nom_tete_liste",
        ],
    )
    df["nuance"] = df["nuance"].fillna("").replace("", "NC")
    df["libelle_abrege_liste"] = df["libelle_abrege_liste"].fillna("")
    df["nom_tete_liste"] = df["nom_tete_liste"].fillna("")

    # Build BV-level location key
    df["location"] = df["code_commune"] + "_" + df["code_bv"]

    # Compute vote share per BV
    total = (
        df.groupby(["id_election", "location"], as_index=False)["voix"]
        .sum()
        .rename(columns={"voix": "total_voix"})
    )
    df = df.merge(total, on=["id_election", "location"])
    df["ratio"] = df["voix"] / df["total_voix"] * 100.0

    # Skip uncontested elections (only 1 candidate in BV)
    counts = df.groupby(["id_election", "location"]).size().rename("c_count")
    df = df.merge(counts, on=["id_election", "location"])
    df = df[df["c_count"] > 1].drop(columns="c_count")

    return df


def _bv_context_tokens(data_dir: Path) -> pd.DataFrame:
    """Load general results at bureau de vote level (abstentions, blancs).

    Each row becomes a token with location = "{code_commune}_{code_bv}".
    """
    df = pd.read_parquet(
        data_dir / "elections" / "agregees" / "general_results.parquet",
        columns=[
            "id_election", "code_commune", "code_bv",
            "inscrits", "abstentions", "blancs",
        ],
    )
    df["location"] = df["code_commune"] + "_" + df["code_bv"]
    df["ratio_abstentions"] = df["abstentions"] / df["inscrits"] * 100.0
    df["ratio_blancs"] = df["blancs"] / df["inscrits"] * 100.0
    return df


def _resolve_candidate_names(df: pd.DataFrame) -> pd.Series:
    """Build a candidate name series using a fallback chain.

    Priority:
      1. prenom + nom (present for 2008-2020, 2026_muni_t2, overseas)
      2. Cross-fill from T2: if a T1 entry has no name but a matching
         (code_commune, libelle_abrege_liste) exists in T2, use T2's name
      3. nom_tete_liste (head-of-list name, when available)
      4. UNKNOWN

    NOTE: libelle_abrege_liste (list name) is intentionally NOT used as a
    candidate name — it is a list label, not a person.  It is only used as
    a join key for the T2 cross-fill in step 2.
    """
    # Step 1: Direct name
    name = (df["prenom"].fillna("") + " " + df["nom"].fillna("")).str.strip()
    has_name = df["nom"].notna() & (df["nom"] != "")

    # Step 2: Cross-fill from T2 for T1 entries without names
    missing_mask = ~has_name
    if missing_mask.any() and "libelle_abrege_liste" in df.columns:
        # Build lookup from T2 entries that DO have names
        t2_mask = df["id_election"].str.endswith("_t2") & has_name
        if t2_mask.any():
            t2_lookup = (
                df.loc[t2_mask]
                .groupby(["code_commune", "libelle_abrege_liste"], dropna=False)
                .agg(nom_t2=("nom", "first"), prenom_t2=("prenom", "first"))
                .reset_index()
            )
            t2_lookup = t2_lookup.dropna(subset=["nom_t2"])
            t2_lookup["t2_name"] = (
                t2_lookup["prenom_t2"].fillna("") + " " + t2_lookup["nom_t2"].fillna("")
            ).str.strip()
            if len(t2_lookup) > 0:
                t2_map = t2_lookup.set_index(
                    ["code_commune", "libelle_abrege_liste"]
                )["t2_name"]
                keys = list(zip(
                    df["code_commune"].values,
                    df["libelle_abrege_liste"].values,
                ))
                t2_filled = pd.Series(
                    [t2_map.get(k, "") for k in keys],
                    index=df.index,
                )
                apply_mask = (
                    missing_mask
                    & df["id_election"].str.endswith("_t1")
                    & (t2_filled != "")
                )
                name.loc[apply_mask] = t2_filled.loc[apply_mask]

    # Recompute what's still missing after T2 cross-fill
    still_empty = name.eq("") | name.isna()

    # Step 3: Use nom_tete_liste if available
    if "nom_tete_liste" in df.columns:
        tete = df["nom_tete_liste"].fillna("").str.strip()
        name = name.where(~still_empty, tete)
        still_empty = name.eq("") | name.isna()

    # If completely missing, return unknown
    name = name.where(~still_empty, "unknown")

    # Normalize: strip accents, remove special chars, lowercase
    def _normalize(s: str) -> str:
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        s = s.lower()
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    name = name.map(_normalize)

    # Force any name that hasn't appeared at least 2 times to unknown
    counts = name.value_counts()
    valid_names = set(counts[counts >= 2].index)
    valid_names.add("unknown")
    name = name.where(name.isin(valid_names), "unknown")

    return name


def _build_candidate_arrays(df: pd.DataFrame) -> pd.DataFrame:
    unique_eids = df["id_election"].unique()
    dates_map = {}
    etypes_map = {}
    for eid in unique_eids:
        d, e = parse_election_id(eid)
        dates_map[eid] = d
        etypes_map[eid] = e

    dates = df["id_election"].map(dates_map).values.astype(np.float32)
    election_types = df["id_election"].map(etypes_map).values

    candidate_names = _resolve_candidate_names(df)

    return pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": df["location"].values,
            "candidate": candidate_names.values,
            "party": df["nuance"].fillna("").values,
            "metric_type": "Result",
            "value": df["ratio"].astype(np.float32).values,
        }
    )


def _build_context_arrays(df: pd.DataFrame) -> pd.DataFrame:
    unique_eids = df["id_election"].unique()
    dates_map = {}
    etypes_map = {}
    for eid in unique_eids:
        d, e = parse_election_id(eid)
        dates_map[eid] = d
        etypes_map[eid] = e

    dates = df["id_election"].map(dates_map).values.astype(np.float32)
    election_types = df["id_election"].map(etypes_map).values

    abstention = pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": df["location"].values,
            "candidate": "Abstention",
            "party": "",
            "metric_type": "Context",
            "value": df["ratio_abstentions"].astype(np.float32).values,
        }
    )
    blancs = pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": df["location"].values,
            "candidate": "Blancs",
            "party": "",
            "metric_type": "Context",
            "value": df["ratio_blancs"].astype(np.float32).values,
        }
    )
    return pd.concat([abstention, blancs], ignore_index=True)


def _merge_geo_coords(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge latitude/longitude from geo lookup onto a token DataFrame.

    Supports both BV-level locations ('01004_0002') and commune/national
    locations by checking bv_coords.parquet first, then location_coords.parquet.
    """
    geo_dir = data_dir / "geo"

    # 1. Try BV-level coords (id_brut_miom → lat/lon)
    bv_path = geo_dir / "bv_coords.parquet"
    if bv_path.exists():
        bv_coords = pd.read_parquet(bv_path, columns=["id_brut_miom", "latitude", "longitude"])
        bv_coords = bv_coords.rename(columns={"id_brut_miom": "location"})
        df = df.merge(bv_coords[["location", "latitude", "longitude"]], on="location", how="left")
    else:
        df["latitude"] = np.float32(np.nan)
        df["longitude"] = np.float32(np.nan)

    # 2. Fall back to commune-level coords for unmatched locations
    still_missing = df["latitude"].isna()
    if still_missing.any():
        commune_path = geo_dir / "location_coords.parquet"
        if commune_path.exists():
            commune_coords = pd.read_parquet(commune_path)
            # For BV locations like "01004_0002", extract commune code "01004"
            df["_commune_key"] = df["location"].str.split("_").str[0]
            commune_coords_renamed = commune_coords.rename(
                columns={"location": "_commune_key", "latitude": "_lat_c", "longitude": "_lon_c"}
            )
            df = df.merge(
                commune_coords_renamed[["_commune_key", "_lat_c", "_lon_c"]],
                on="_commune_key", how="left",
            )
            df["latitude"] = df["latitude"].fillna(df["_lat_c"])
            df["longitude"] = df["longitude"].fillna(df["_lon_c"])
            df.drop(columns=["_commune_key", "_lat_c", "_lon_c"], inplace=True)

    # Fallback for unmatched locations: center of France
    df["latitude"] = df["latitude"].fillna(46.2276).astype(np.float32)
    df["longitude"] = df["longitude"].fillna(2.2137).astype(np.float32)
    return df


def load_election_tokens(data_dir: Path) -> pd.DataFrame:
    """Load election results at bureau de vote granularity.

    Every BV in every election becomes its own set of tokens.
    Location keys are '{code_commune}_{code_bv}' (e.g. '01004_0002').
    Coordinates come from bv_coords.parquet (exact BV positions).
    """
    candidate_bv = _bv_candidate_tokens(data_dir)
    context_bv = _bv_context_tokens(data_dir)

    # Filter out small communes from Municipales (inscrits < 750)
    muni_mask = context_bv["id_election"].str.contains("muni")
    small_muni_mask = muni_mask & (context_bv["inscrits"] < 750)
    valid_bvs = context_bv[~small_muni_mask][["id_election", "location"]]

    candidate_bv = candidate_bv.merge(valid_bvs, on=["id_election", "location"])
    context_bv = context_bv.merge(valid_bvs, on=["id_election", "location"])

    candidate_df = _build_candidate_arrays(candidate_bv)
    context_df = _build_context_arrays(context_bv)

    combined = pd.concat([candidate_df, context_df], ignore_index=True)
    combined = _merge_geo_coords(combined, data_dir)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)

    n_locations = combined["location"].nunique()
    n_bv = combined["location"].str.contains("_").sum()
    print(f"  Election tokens: {len(combined)} ({n_locations} unique locations, "
          f"{n_bv} BV-level tokens)", flush=True)
    return combined
