from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _parse_date_to_float(date_str: str) -> float:
    parts = date_str.split("-")
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 6
    day = int(parts[2]) if len(parts) > 2 else 15
    return year + (month - 1 + day / 30.0) / 12.0


def _load_nsp_presidentielle(data_dir: Path) -> pd.DataFrame:
    path = (
        data_dir
        / "polls"
        / "presidentielle"
        / "2022"
        / "nsppolls_presidentielle_2022.csv"
    )
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    dates = df["fin_enquete"].apply(_parse_date_to_float).astype(np.float32)
    institute = df["nom_institut"].fillna("Unknown")
    tour_map = {"Premier tour": "T1", "Deuxième tour": "T2"}
    tour = df["tour"].map(tour_map).fillna("T1")
    return pd.DataFrame(
        {
            "date_float": dates,
            "election_type": "Presidentielle_" + tour,
            "location": "National",
            "candidate": df["candidat"].str.strip().str.upper().values,
            "party": df["parti"].fillna("").values,
            "metric_type": "Poll_" + institute,
            "value": pd.to_numeric(df["intentions"], errors="coerce")
            .astype(np.float32)
            .values,
        }
    ).dropna(subset=["value"])


def _load_nsp_regionales(data_dir: Path) -> pd.DataFrame:
    path = data_dir / "polls" / "regionales" / "nsppolls_regionales_2021.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    dates = df["fin_enquete"].apply(_parse_date_to_float).astype(np.float32)
    institute = df["nom_institut"].fillna("Unknown")
    tour_map = {"Premier tour": "T1", "Deuxième tour": "T2"}
    tour = df["tour"].map(tour_map).fillna("T1")
    return pd.DataFrame(
        {
            "date_float": dates,
            "election_type": "Regionales_" + tour,
            "location": df["region_name"].values,
            "candidate": df["tete_liste"].astype(str).str.strip().str.upper().values,
            "party": df["parti"].fillna("").values,
            "metric_type": "Poll_" + institute,
            "value": pd.to_numeric(df["intentions"], errors="coerce")
            .astype(np.float32)
            .values,
        }
    ).dropna(subset=["value"])


def _load_all_wiki_polls(data_dir: Path) -> pd.DataFrame:
    polls_dir = data_dir / "polls"
    frames: list[pd.DataFrame] = []

    for csv_path in sorted(polls_dir.rglob("*.csv")):
        if csv_path.name.startswith("nsppolls_"):
            continue

        lower_path = str(csv_path).lower()
        if "presidentielle" in lower_path:
            etype = "Presidentielle"
        elif "europeennes" in lower_path:
            etype = "Europeennes"
        elif "legislatives" in lower_path:
            etype = "Legislatives"
        elif "regionales" in lower_path:
            etype = "Regionales"
        elif "municipales" in lower_path:
            etype = "Municipales"
        elif "dpmt" in lower_path or "departementales" in lower_path:
            etype = "Departementales"
        elif "cant" in lower_path or "cantonales" in lower_path:
            etype = "Cantonales"
        else:
            etype = "Unknown"

        if (
            "second-tour" in lower_path
            or "t2" in lower_path
            or "tour_2" in lower_path
            or "deuxieme" in lower_path
        ):
            etype += "_T2"
        else:
            etype += "_T1"

        frame = _parse_wiki_poll_csv(csv_path, etype)
        if frame is not None and len(frame) > 0:
            frames.append(frame)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _parse_wiki_poll_csv(path: Path, election_type: str) -> pd.DataFrame | None:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    meta_cols = {
        "sondeur",
        "source",
        "date début",
        "date fin",
        "échantillon",
        "Sondeur",
        "Date",
        "Échantillon",
        "Polling firm",
        "Fieldwork date",
        "Sample size",
        "Abs.",
    }
    candidate_cols = [
        c for c in df.columns if c not in meta_cols and "Unnamed" not in c
    ]
    if not candidate_cols:
        return None

    date_col = next(
        (
            c
            for c in df.columns
            if "date" in c.lower() or "fieldwork" in c.lower() or "période" in c.lower()
        ),
        None,
    )
    institute_col = next(
        (c for c in ["sondeur", "Sondeur", "Polling firm"] if c in df.columns), None
    )

    rows: list[dict[str, object]] = []
    for _, row in df.iterrows():
        date_str = str(row.get(date_col, "")) if date_col else ""
        if not date_str or date_str == "nan" or len(date_str) < 4:
            continue
        try:
            date_float = _parse_date_to_float(date_str)
        except (ValueError, IndexError):
            continue

        institute = (
            str(row.get(institute_col, "Unknown")) if institute_col else "Unknown"
        )
        for candidate in candidate_cols:
            val = row.get(candidate)
            if pd.isna(val):
                continue
            s_val = (
                str(val)
                .replace(",", ".")
                .replace("%", "")
                .replace("<", "")
                .replace(">", "")
                .strip()
            )
            if "[" in s_val:
                s_val = s_val.split("[")[0]
            if not s_val or s_val == "-" or s_val.lower() == "nan":
                continue
            try:
                val_float = float(s_val)
            except (ValueError, TypeError):
                continue
            rows.append(
                {
                    "date_float": np.float32(date_float),
                    "election_type": election_type,
                    "location": "National",
                    "candidate": candidate.strip().upper(),
                    "party": "",
                    "metric_type": f"Poll_{institute}",
                    "value": np.float32(val_float),
                }
            )

    return pd.DataFrame(rows) if rows else None


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


def load_poll_tokens(data_dir: Path) -> pd.DataFrame:
    frames = [
        _load_nsp_presidentielle(data_dir),
        _load_nsp_regionales(data_dir),
        _load_all_wiki_polls(data_dir),
    ]
    frames = [f for f in frames if len(f) > 0]
    if not frames:
        return pd.DataFrame(
            columns=[
                "date_float",
                "election_type",
                "location",
                "candidate",
                "party",
                "metric_type",
                "value",
                "latitude",
                "longitude",
            ]
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = _merge_geo_coords(combined, data_dir)
    combined.sort_values("date_float", inplace=True)
    combined.reset_index(drop=True, inplace=True)
    return combined
