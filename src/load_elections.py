from __future__ import annotations

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
        columns=["id_election", "code_commune", "nom", "prenom", "nuance", "voix"],
    )
    commune = df.groupby(
        ["id_election", "code_commune", "nom", "prenom", "nuance"], as_index=False, dropna=False
    )["voix"].sum()
    total = (
        df.groupby(["id_election", "code_commune"], as_index=False)["voix"]
        .sum()
        .rename(columns={"voix": "total_voix"})
    )
    commune = commune.merge(total, on=["id_election", "code_commune"])
    commune["ratio"] = commune["voix"] / commune["total_voix"] * 100.0
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

    return pd.DataFrame(
        {
            "date_float": dates,
            "election_type": election_types,
            "location": commune_df["code_commune"].values,
            "candidate": (commune_df["prenom"] + " " + commune_df["nom"])
            .str.strip()
            .str.upper()
            .values,
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

    candidate_df = _build_candidate_arrays(candidate_commune)
    context_df = _build_context_arrays(context_commune)

    combined = pd.concat([candidate_df, context_df], ignore_index=True)
    combined = _merge_geo_coords(combined, data_dir)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
