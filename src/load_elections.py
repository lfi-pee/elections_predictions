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


def _aggregate_candidate_tokens(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        data_dir / "elections" / "agregees" / "candidats_results.parquet",
        columns=[
            "id_election", "code_commune", "nom", "prenom", "nuance", "voix",
            "libelle_abrege_liste", "nom_tete_liste",
        ],
    )
    df["nuance"] = df["nuance"].fillna("")
    df["nuance"] = df["nuance"].replace("", "NC")
    df["libelle_abrege_liste"] = df["libelle_abrege_liste"].fillna("")
    df["nom_tete_liste"] = df["nom_tete_liste"].fillna("")
    commune = df.groupby(
        ["id_election", "code_commune", "nom", "prenom", "nuance", "libelle_abrege_liste"],
        dropna=False,
        as_index=False
    ).agg(
        voix=("voix", "sum"),
        nom_tete_liste=("nom_tete_liste", "first"),
    )
    total = (
        commune.groupby(["id_election", "code_commune"], as_index=False)["voix"]
        .sum()
        .rename(columns={"voix": "total_voix"})
    )
    commune = commune.merge(total, on=["id_election", "code_commune"])
    commune["ratio"] = commune["voix"] / commune["total_voix"] * 100.0
    
    # NEW: Skip uncontested elections (only 1 candidate)
    counts = commune.groupby(["id_election", "code_commune"]).size().rename("c_count")
    commune = commune.merge(counts, on=["id_election", "code_commune"])
    commune = commune[commune["c_count"] > 1].drop(columns="c_count")
    
    return commune


def _aggregate_context_tokens(data_dir: Path) -> pd.DataFrame:
    df = pd.read_parquet(
        data_dir / "elections" / "agregees" / "general_results.parquet",
        columns=[
            "id_election",
            "code_commune",
            "libelle_commune",
            "inscrits",
            "abstentions",
            "blancs",
        ],
    )
    commune = df.groupby(["id_election", "code_commune"], as_index=False).agg(
        inscrits=("inscrits", "sum"),
        abstentions=("abstentions", "sum"),
        blancs=("blancs", "sum"),
    )
    commune["ratio_abstentions"] = commune["abstentions"] / commune["inscrits"] * 100.0
    commune["ratio_blancs"] = commune["blancs"] / commune["inscrits"] * 100.0
    return commune


def _resolve_candidate_names(commune_df: pd.DataFrame) -> pd.Series:
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
    name = (commune_df["prenom"].fillna("") + " " + commune_df["nom"].fillna("")).str.strip()
    has_name = commune_df["nom"].notna() & (commune_df["nom"] != "")

    # Step 2: Cross-fill from T2 for T1 entries without names
    missing_mask = ~has_name
    if missing_mask.any() and "libelle_abrege_liste" in commune_df.columns:
        # Build lookup from T2 entries that DO have names
        t2_mask = commune_df["id_election"].str.endswith("_t2") & has_name
        if t2_mask.any():
            t2_lookup = (
                commune_df.loc[t2_mask]
                .groupby(["code_commune", "libelle_abrege_liste"], dropna=False)
                .agg(nom_t2=("nom", "first"), prenom_t2=("prenom", "first"))
                .reset_index()
            )
            t2_lookup = t2_lookup.dropna(subset=["nom_t2"])
            t2_lookup["t2_name"] = (
                t2_lookup["prenom_t2"].fillna("") + " " + t2_lookup["nom_t2"].fillna("")
            ).str.strip()
            if len(t2_lookup) > 0:
                # Merge T2 names onto full dataframe by (code_commune, libelle_abrege_liste)
                t2_map = t2_lookup.set_index(
                    ["code_commune", "libelle_abrege_liste"]
                )["t2_name"]
                # Build a multi-index key for lookup
                keys = list(zip(
                    commune_df["code_commune"].values,
                    commune_df["libelle_abrege_liste"].values,
                ))
                t2_filled = pd.Series(
                    [t2_map.get(k, "") for k in keys],
                    index=commune_df.index,
                )
                # Only apply to T1 entries that are missing names
                apply_mask = (
                    missing_mask
                    & commune_df["id_election"].str.endswith("_t1")
                    & (t2_filled != "")
                )
                name.loc[apply_mask] = t2_filled.loc[apply_mask]

    # Recompute what's still missing after T2 cross-fill
    still_empty = name.eq("") | name.isna()

    # Step 3: Use nom_tete_liste if available
    if "nom_tete_liste" in commune_df.columns:
        tete = commune_df["nom_tete_liste"].fillna("").str.strip()
        name = name.where(~still_empty, tete)
        still_empty = name.eq("") | name.isna()

    # If completely missing, return unknown
    name = name.where(~still_empty, "unknown")

    # Normalize: strip accents, remove special chars, lowercase
    def _normalize(s: str) -> str:
        # NFKD decomposition then drop combining marks (accents)
        s = unicodedata.normalize("NFKD", s)
        s = "".join(c for c in s if not unicodedata.combining(c))
        # lowercase
        s = s.lower()
        # keep only alphanumeric and spaces
        s = re.sub(r"[^a-z0-9 ]", " ", s)
        # collapse multiple spaces
        s = re.sub(r"\s+", " ", s).strip()
        return s

    name = name.map(_normalize)

    # Force any name that hasn't appeared at least 2 times to unknown
    counts = name.value_counts()
    valid_names = set(counts[counts >= 2].index)
    
    # We always keep unknown valid so we don't accidentally map it to something else
    valid_names.add("unknown")
    
    name = name.where(name.isin(valid_names), "unknown")

    return name


def _build_candidate_arrays(commune_df: pd.DataFrame) -> pd.DataFrame:
    unique_eids = commune_df["id_election"].unique()
    dates_map = {}
    etypes_map = {}
    for eid in unique_eids:
        d, e = parse_election_id(eid)
        dates_map[eid] = d
        etypes_map[eid] = e
        
    dates = commune_df["id_election"].map(dates_map).values.astype(np.float32)
    election_types = commune_df["id_election"].map(etypes_map).values

    candidate_names = _resolve_candidate_names(commune_df)

    return pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": commune_df["code_commune"].values,
            "candidate": candidate_names.values,
            "party": commune_df["nuance"].fillna("").values,
            "metric_type": "Result",
            "value": commune_df["ratio"].astype(np.float32).values,
        }
    )


def _build_context_arrays(commune_df: pd.DataFrame) -> pd.DataFrame:
    unique_eids = commune_df["id_election"].unique()
    dates_map = {}
    etypes_map = {}
    for eid in unique_eids:
        d, e = parse_election_id(eid)
        dates_map[eid] = d
        etypes_map[eid] = e

    dates = commune_df["id_election"].map(dates_map).values.astype(np.float32)
    election_types = commune_df["id_election"].map(etypes_map).values

    abstention = pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": commune_df["code_commune"].values,
            "candidate": "Abstention",
            "party": "",
            "metric_type": "Context",
            "value": commune_df["ratio_abstentions"].astype(np.float32).values,
        }
    )
    blancs = pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": commune_df["code_commune"].values,
            "candidate": "Blancs",
            "party": "",
            "metric_type": "Context",
            "value": commune_df["ratio_blancs"].astype(np.float32).values,
        }
    )
    return pd.concat([abstention, blancs], ignore_index=True)


def _merge_geo_coords(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Merge latitude/longitude from geo lookup onto a token DataFrame."""
    coords_path = data_dir / "geo" / "location_coords.parquet"
    if coords_path.exists():
        coords = pd.read_parquet(coords_path)
        df = df.merge(coords[["location", "latitude", "longitude"]], on="location", how="left")
    else:
        df["latitude"] = np.float32(np.nan)
        df["longitude"] = np.float32(np.nan)
    # Fallback for unmatched locations: center of France
    df["latitude"] = df["latitude"].fillna(46.2276).astype(np.float32)
    df["longitude"] = df["longitude"].fillna(2.2137).astype(np.float32)
    return df


def load_election_tokens(data_dir: Path) -> pd.DataFrame:
    candidate_commune = _aggregate_candidate_tokens(data_dir)
    context_commune = _aggregate_context_tokens(data_dir)

    # Filter out small communes from Municipales (inscrits < 750)
    muni_mask = context_commune["id_election"].str.contains("muni")
    small_muni_mask = muni_mask & (context_commune["inscrits"] < 750)
    valid_communes = context_commune[~small_muni_mask][["id_election", "code_commune"]]

    candidate_commune = candidate_commune.merge(valid_communes, on=["id_election", "code_commune"])
    context_commune = context_commune.merge(valid_communes, on=["id_election", "code_commune"])

    candidate_df = _build_candidate_arrays(candidate_commune)
    context_df = _build_context_arrays(context_commune)

    combined = pd.concat([candidate_df, context_df], ignore_index=True)
    combined = _merge_geo_coords(combined, data_dir)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
