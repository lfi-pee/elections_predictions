from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.nuance_mapping import CITY_TO_CODE_COMMUNE, map_coalition_to_nuance


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
            # Fallback: try French date format ("30 juin 2024")
            date_float_maybe = _parse_french_date(date_str)
            if date_float_maybe is None:
                continue
            date_float = date_float_maybe

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


_FRENCH_MONTHS = {
    "janvier": 1, "février": 2, "mars": 3, "avril": 4,
    "mai": 5, "juin": 6, "juillet": 7, "août": 8,
    "septembre": 9, "octobre": 10, "novembre": 11, "décembre": 12,
}


def _parse_french_date(s: str) -> float | None:
    """Parse French survey date strings like '5-10 mars 2026' into a date float."""
    s = str(s).strip()
    if not s or s == "nan":
        return None
    year_match = re.search(r"(\d{4})", s)
    if not year_match:
        return None
    year = int(year_match.group(1))
    months_found: list[tuple[int, int]] = []
    for name, num in _FRENCH_MONTHS.items():
        for m in re.finditer(name, s, re.IGNORECASE):
            months_found.append((m.start(), num))
    if not months_found:
        return None
    months_found.sort()
    month = months_found[-1][1]
    days = [
        int(d)
        for d in re.findall(r"(\d{1,2})", s)
        if int(d) <= 31 and d != str(year)
    ]
    day = days[-1] if days else 15
    return year + (month - 1 + day / 30.0) / 12.0


def _clean_poll_value(raw: object) -> float | None:
    """Extract a numeric poll value from messy cell contents.

    Handles footnotes like '26[d]', French decimals '0,5', '<1', '—',
    and embedded candidate names like '32 Doucet'.
    """
    s = str(raw).strip()
    if not s or s in ("nan", "—", "-", "–"):
        return None
    # Strip footnote references e.g. [d], [e], [c]
    s = re.sub(r"\[[a-zA-Z0-9]+\]", "", s).strip()
    # "<1" or "<0,5" -> just take the number
    s = s.replace("<", "").replace(">", "").strip()
    # French decimal
    s = s.replace(",", ".").strip()
    # Strip embedded candidate names: "32 Doucet" -> "32"
    m = re.match(r"^([\d.]+)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _extract_city_from_filename(fname: str) -> str:
    """Extract city name from filenames like 'municipales_2026_paris_sondages_8.csv'."""
    m = re.match(r"municipales_\d{4}_(.+?)_sondages", fname)
    if m:
        return m.group(1).replace("_", " ").title()
    return "Unknown"


def _load_municipales_polls(data_dir: Path) -> pd.DataFrame:
    """Load all scraped Wikipedia municipales polling CSVs."""
    muni_dir = data_dir / "polls" / "municipales"
    if not muni_dir.exists():
        return pd.DataFrame()

    rows: list[dict[str, object]] = []

    for csv_path in sorted(muni_dir.glob("*.csv")):
        city_name = _extract_city_from_filename(csv_path.name)
        # Use code_commune if known, otherwise fallback to the city name
        city_location = CITY_TO_CODE_COMMUNE.get(city_name, city_name)
        
        fname_lower = csv_path.name.lower()

        # Determine round from table structure (even-indexed = T1, odd = T2
        # for most cities). T2 tables typically have fewer, coalesced columns.
        # We'll figure this out from whether the column set looks like T1 or T2.
        df = pd.read_csv(csv_path)
        if len(df) < 3:
            continue

        meta_cols = {"Source", "Date de réalisation", "Échantillon", "Autres"}
        candidate_cols = [
            c for c in df.columns
            if c not in meta_cols and "Unnamed" not in str(c)
        ]
        if not candidate_cols:
            continue

        # Determine election round. Heuristic: if columns contain coalition
        # strings with " - " separators AND <= 6 candidate cols, it's T2.
        coalition_cols = sum(1 for c in candidate_cols if " - " in c)
        etype = "Municipales_T2" if (coalition_cols > len(candidate_cols) * 0.5 and len(candidate_cols) <= 8) else "Municipales_T1"

        for _, row in df.iterrows():
            date_str = str(row.get("Date de réalisation", ""))
            date_float = _parse_french_date(date_str)
            if date_float is None:
                continue

            institute = str(row.get("Source", "Unknown"))
            if institute in ("nan", "Source", ""):
                institute = "Unknown"

            for cand_col in candidate_cols:
                val = _clean_poll_value(row.get(cand_col))
                if val is None:
                    continue
                # For municipales polls the column header IS the party/coalition
                # Use the first component as the primary party for the party field
                party_label = cand_col.strip()
                nuance_code = map_coalition_to_nuance(party_label)
                
                rows.append({
                    "date_float": np.float32(date_float),
                    "election_type": etype,
                    "location": city_location,
                    "candidate": cand_col.strip().upper(),
                    "party": nuance_code,
                    "metric_type": f"Poll_{institute}",
                    "value": np.float32(val),
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame()



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
    cache_path = data_dir / "baseline_cache" / "polls.parquet"
    if cache_path.exists():
        return pd.read_parquet(cache_path)

    frames = [
        _load_nsp_presidentielle(data_dir),
        _load_nsp_regionales(data_dir),
        _load_all_wiki_polls(data_dir),
        _load_municipales_polls(data_dir),
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

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(cache_path, index=False)

    return combined
