"""Build geo-coordinate lookup for all location types in the election data.

Fetches commune centroids from geo.api.gouv.fr, then derives centroids for
départements, régions, cantons, circonscriptions, and national level.

Also resolves historical commune codes (communes that were merged into
"communes nouvelles") using the etalab decoupage-administratif dataset.

Outputs:
    data/geo/commune_coords.parquet   — raw commune centroids
    data/geo/location_coords.parquet  — unified lookup (location → lat, lon)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests


FRANCE_CENTER_LAT = 46.2276
FRANCE_CENTER_LON = 2.2137

# Fallback for overseas voter codes (ZZ*) — distinct from France center
OVERSEAS_LAT = 0.0
OVERSEAS_LON = 0.0

# Département → Région mapping (2016 redrawing)
DEPT_TO_REGION: dict[str, str] = {
    # Auvergne-Rhône-Alpes
    "01": "Auvergne-Rhône-Alpes", "03": "Auvergne-Rhône-Alpes",
    "07": "Auvergne-Rhône-Alpes", "15": "Auvergne-Rhône-Alpes",
    "26": "Auvergne-Rhône-Alpes", "38": "Auvergne-Rhône-Alpes",
    "42": "Auvergne-Rhône-Alpes", "43": "Auvergne-Rhône-Alpes",
    "63": "Auvergne-Rhône-Alpes", "69": "Auvergne-Rhône-Alpes",
    "73": "Auvergne-Rhône-Alpes", "74": "Auvergne-Rhône-Alpes",
    # Bourgogne-Franche-Comté
    "21": "Bourgogne-Franche-Comté", "25": "Bourgogne-Franche-Comté",
    "39": "Bourgogne-Franche-Comté", "58": "Bourgogne-Franche-Comté",
    "70": "Bourgogne-Franche-Comté", "71": "Bourgogne-Franche-Comté",
    "89": "Bourgogne-Franche-Comté", "90": "Bourgogne-Franche-Comté",
    # Bretagne
    "22": "Bretagne", "29": "Bretagne", "35": "Bretagne", "56": "Bretagne",
    # Centre-Val de Loire
    "18": "Centre-Val de Loire", "28": "Centre-Val de Loire",
    "36": "Centre-Val de Loire", "37": "Centre-Val de Loire",
    "41": "Centre-Val de Loire", "45": "Centre-Val de Loire",
    # Corse
    "2A": "Corse", "2B": "Corse",
    # Grand Est
    "08": "Grand Est", "10": "Grand Est", "51": "Grand Est",
    "52": "Grand Est", "54": "Grand Est", "55": "Grand Est",
    "57": "Grand Est", "67": "Grand Est", "68": "Grand Est",
    "88": "Grand Est",
    # Hauts-de-France
    "02": "Hauts-de-France", "59": "Hauts-de-France",
    "60": "Hauts-de-France", "62": "Hauts-de-France",
    "80": "Hauts-de-France",
    # Île-de-France
    "75": "Île-de-France", "77": "Île-de-France",
    "78": "Île-de-France", "91": "Île-de-France",
    "92": "Île-de-France", "93": "Île-de-France",
    "94": "Île-de-France", "95": "Île-de-France",
    # Normandie
    "14": "Normandie", "27": "Normandie", "50": "Normandie",
    "61": "Normandie", "76": "Normandie",
    # Nouvelle-Aquitaine
    "16": "Nouvelle-Aquitaine", "17": "Nouvelle-Aquitaine",
    "19": "Nouvelle-Aquitaine", "23": "Nouvelle-Aquitaine",
    "24": "Nouvelle-Aquitaine", "33": "Nouvelle-Aquitaine",
    "40": "Nouvelle-Aquitaine", "47": "Nouvelle-Aquitaine",
    "64": "Nouvelle-Aquitaine", "79": "Nouvelle-Aquitaine",
    "86": "Nouvelle-Aquitaine", "87": "Nouvelle-Aquitaine",
    # Occitanie
    "09": "Occitanie", "11": "Occitanie", "12": "Occitanie",
    "30": "Occitanie", "31": "Occitanie", "32": "Occitanie",
    "34": "Occitanie", "46": "Occitanie", "48": "Occitanie",
    "65": "Occitanie", "66": "Occitanie", "81": "Occitanie",
    "82": "Occitanie",
    # Pays de la Loire
    "44": "Pays de la Loire", "49": "Pays de la Loire",
    "53": "Pays de la Loire", "72": "Pays de la Loire",
    "85": "Pays de la Loire",
    # Provence-Alpes-Côte d'Azur
    "04": "Provence-Alpes-Côte d'Azur", "05": "Provence-Alpes-Côte d'Azur",
    "06": "Provence-Alpes-Côte d'Azur", "13": "Provence-Alpes-Côte d'Azur",
    "83": "Provence-Alpes-Côte d'Azur", "84": "Provence-Alpes-Côte d'Azur",
    # DOM-TOM
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
    "974": "La Réunion", "976": "Mayotte",
}


def _extract_dept_from_commune(code_commune: str) -> str:
    """Extract département code from an INSEE commune code."""
    if code_commune.startswith("97") or code_commune.startswith("98"):
        return code_commune[:3]
    return code_commune[:2]


def fetch_commune_centroids() -> pd.DataFrame:
    """Fetch all commune centroids from the French government API."""
    url = "https://geo.api.gouv.fr/communes?fields=code,nom,centre&format=json"
    print(f"Fetching commune centroids from {url} ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for commune in data:
        code = commune.get("code", "")
        nom = commune.get("nom", "")
        centre = commune.get("centre")
        if centre and centre.get("coordinates"):
            lon, lat = centre["coordinates"]
            rows.append({"code_commune": code, "nom": nom, "latitude": lat, "longitude": lon})

    # Also fetch arrondissements municipaux (Paris, Lyon, Marseille)
    arr_url = "https://geo.api.gouv.fr/communes?type=arrondissement-municipal&fields=code,nom,centre"
    print(f"Fetching arrondissements municipaux from API ...")
    try:
        arr_resp = requests.get(arr_url, timeout=60)
        arr_resp.raise_for_status()
        arr_data = arr_resp.json()
        for commune in arr_data:
            code = commune.get("code", "")
            nom = commune.get("nom", "")
            centre = commune.get("centre")
            if centre and centre.get("coordinates"):
                lon, lat = centre["coordinates"]
                rows.append({"code_commune": code, "nom": nom, "latitude": lat, "longitude": lon})
        print(f"  Fetched {len(arr_data)} arrondissements municipaux.")
    except Exception as e:
        print(f"  Warning: could not fetch arrondissements: {e}")

    df = pd.DataFrame(rows)
    print(f"  Total commune centroids: {len(df)}.")
    return df


def fetch_historical_commune_mapping() -> dict[str, str]:
    """Fetch mapping of old commune codes → successor commune codes.

    Uses the etalab decoupage-administratif dataset which includes
    communes-déléguées (old communes merged into communes nouvelles)
    with their chefLieu (successor commune code).

    Returns:
        dict mapping old_code → successor_code
    """
    url = "https://unpkg.com/@etalab/decoupage-administratif@4.0.0/data/communes.json"
    print(f"Fetching historical commune mapping from etalab ...")
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    communes = resp.json()

    # Build mapping: old code → chefLieu (successor) for commune-deleguee types
    mapping: dict[str, str] = {}
    for c in communes:
        if c.get("type") == "commune-deleguee" and c.get("chefLieu"):
            mapping[c["code"]] = c["chefLieu"]

    print(f"  Found {len(mapping)} historical commune → successor mappings.")
    return mapping


def build_location_coords(data_dir: Path, commune_df: pd.DataFrame) -> pd.DataFrame:
    """Build a unified location → (lat, lon) lookup from commune centroids + election data."""
    records: list[dict] = []

    # 1. Communes — location key is code_commune
    for _, row in commune_df.iterrows():
        records.append({
            "location": row["code_commune"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
        })

    # Load election general results to get inscrits for weighting
    general_path = data_dir / "elections" / "agregees" / "general_results.parquet"
    if general_path.exists():
        gen = pd.read_parquet(general_path, columns=["code_commune", "inscrits"])
        gen = gen.groupby("code_commune", as_index=False)["inscrits"].sum()
    else:
        gen = pd.DataFrame(columns=["code_commune", "inscrits"])

    # Merge commune coords with inscrits for weighting
    commune_with_weight = commune_df.merge(gen, on="code_commune", how="left")
    commune_with_weight["inscrits"] = commune_with_weight["inscrits"].fillna(1.0)
    commune_with_weight["dept"] = commune_with_weight["code_commune"].apply(_extract_dept_from_commune)

    # 2. Départements — weighted average of commune centroids
    dept_groups = commune_with_weight.groupby("dept")
    dept_coords = {}
    for dept, grp in dept_groups:
        w = grp["inscrits"].values.astype(np.float64)
        total_w = w.sum()
        if total_w > 0:
            lat = (grp["latitude"].values * w).sum() / total_w
            lon = (grp["longitude"].values * w).sum() / total_w
        else:
            lat = grp["latitude"].mean()
            lon = grp["longitude"].mean()
        dept_coords[dept] = (lat, lon)
        records.append({"location": dept, "latitude": lat, "longitude": lon})

    # 3. Régions — weighted average of département centroids
    region_lats: dict[str, list] = {}
    region_lons: dict[str, list] = {}
    region_weights: dict[str, list] = {}
    for dept, (lat, lon) in dept_coords.items():
        region = DEPT_TO_REGION.get(dept)
        if region:
            dept_inscrits = commune_with_weight[commune_with_weight["dept"] == dept]["inscrits"].sum()
            region_lats.setdefault(region, []).append(lat)
            region_lons.setdefault(region, []).append(lon)
            region_weights.setdefault(region, []).append(dept_inscrits)

    for region in region_lats:
        w = np.array(region_weights[region], dtype=np.float64)
        total_w = w.sum()
        if total_w > 0:
            lat = (np.array(region_lats[region]) * w).sum() / total_w
            lon = (np.array(region_lons[region]) * w).sum() / total_w
        else:
            lat = np.mean(region_lats[region])
            lon = np.mean(region_lons[region])
        records.append({"location": region, "latitude": lat, "longitude": lon})

    # 4. Bureaux de vote — exact BV coordinates from bv_coords.parquet
    bv_path = data_dir / "geo" / "bv_coords.parquet"
    if bv_path.exists():
        bv_df = pd.read_parquet(bv_path, columns=["id_brut_miom", "latitude", "longitude"])
        bv_records = bv_df.rename(columns={"id_brut_miom": "location"}).to_dict("records")
        records.extend(bv_records)
        print(f"  Added {len(bv_records)} BV-level coordinates from bv_coords.parquet")

    # 5. Cantons & Circonscriptions from candidats_results.parquet
    cand_path = data_dir / "elections" / "agregees" / "candidats_results.parquet"
    if cand_path.exists():
        try:
            cand = pd.read_parquet(
                cand_path,
                columns=["code_commune", "code_canton", "code_circonscription"],
            )
        except Exception:
            cand = pd.DataFrame()

        if not cand.empty and "code_canton" in cand.columns:
            cand["dept"] = cand["code_commune"].apply(_extract_dept_from_commune)
            cand_with_coords = cand.merge(
                commune_df[["code_commune", "latitude", "longitude"]],
                on="code_commune", how="inner",
            )
            # Cantons
            if "code_canton" in cand_with_coords.columns:
                canton_grp = cand_with_coords.groupby(["dept", "code_canton"])
                for (dept, canton), g in canton_grp:
                    lat = g["latitude"].mean()
                    lon = g["longitude"].mean()
                    records.append({"location": f"{dept}_{canton}", "latitude": lat, "longitude": lon})

            # Circonscriptions
            if "code_circonscription" in cand_with_coords.columns:
                circo_grp = cand_with_coords.groupby(["dept", "code_circonscription"])
                for (dept, circo), g in circo_grp:
                    lat = g["latitude"].mean()
                    lon = g["longitude"].mean()
                    records.append({"location": f"{dept}_circo_{circo}", "latitude": lat, "longitude": lon})

    # 6. National
    records.append({
        "location": "National",
        "latitude": FRANCE_CENTER_LAT,
        "longitude": FRANCE_CENTER_LON,
    })

    result = pd.DataFrame(records)
    # Deduplicate — keep first occurrence (communes take priority)
    result = result.drop_duplicates(subset=["location"], keep="first")
    result["latitude"] = result["latitude"].astype(np.float32)
    result["longitude"] = result["longitude"].astype(np.float32)
    return result


def _resolve_historical_communes(
    result: pd.DataFrame, data_dir: Path,
    historical_mapping: dict[str, str],
    dept_coords: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    """Find election commune codes missing from the geo lookup and resolve them.

    Resolution strategies (in priority order):
    1. Geocoded historical commune coords (from geocode_bv.py)
    2. Geocoded ZZ consular city coords (from geocode_bv.py)
    3. chefLieu mapping — use the successor commune's coordinates
    4. Département centroid — use the weighted centroid of the département
    5. France center — for truly unknown codes
    """
    geo_dir = data_dir / "geo"

    # Load geocoded coordinates from geocode_bv.py outputs
    hist_coords: dict[str, tuple[float, float]] = {}
    hist_path = geo_dir / "historical_commune_coords.parquet"
    if hist_path.exists():
        df_hist = pd.read_parquet(hist_path)
        hist_coords = dict(
            zip(df_hist["code_commune"], zip(df_hist["latitude"], df_hist["longitude"]))
        )
        print(f"  Loaded {len(hist_coords)} geocoded historical commune coords")

    zz_coords: dict[str, tuple[float, float]] = {}
    zz_path = geo_dir / "zz_consular_coords.parquet"
    if zz_path.exists():
        df_zz = pd.read_parquet(zz_path)
        zz_coords = dict(
            zip(df_zz["code_commune"], zip(df_zz["latitude"], df_zz["longitude"]))
        )
        print(f"  Loaded {len(zz_coords)} geocoded ZZ consular coords")

    # Collect all commune codes appearing in election data
    cand_path = data_dir / "elections" / "agregees" / "candidats_results.parquet"
    gen_path = data_dir / "elections" / "agregees" / "general_results.parquet"
    election_codes: set[str] = set()
    if cand_path.exists():
        cand = pd.read_parquet(cand_path, columns=["code_commune"])
        election_codes |= set(cand["code_commune"].unique())
    if gen_path.exists():
        gen = pd.read_parquet(gen_path, columns=["code_commune"])
        election_codes |= set(gen["code_commune"].unique())

    mapped_locs = set(result["location"].unique())
    unmatched = election_codes - mapped_locs

    if not unmatched:
        print("  All election locations already have coordinates.")
        return result

    print(f"  {len(unmatched)} election locations missing coordinates, resolving...")

    # Build a quick lookup of existing coords
    coord_lookup: dict[str, tuple[float, float]] = dict(
        zip(result["location"], zip(result["latitude"], result["longitude"]))
    )

    new_records: list[dict] = []
    resolved_hist = 0
    resolved_zz = 0
    resolved_chef = 0
    resolved_dept = 0
    resolved_national = 0

    for code in sorted(unmatched):
        # Strategy 1: Geocoded historical commune coords
        if code in hist_coords:
            lat, lon = hist_coords[code]
            new_records.append({"location": code, "latitude": lat, "longitude": lon})
            resolved_hist += 1
            continue

        # Strategy 2: Geocoded ZZ consular city coords
        if code in zz_coords:
            lat, lon = zz_coords[code]
            new_records.append({"location": code, "latitude": lat, "longitude": lon})
            resolved_zz += 1
            continue

        # Strategy 3: chefLieu mapping
        successor = historical_mapping.get(code)
        if successor and successor in coord_lookup:
            lat, lon = coord_lookup[successor]
            new_records.append({"location": code, "latitude": lat, "longitude": lon})
            resolved_chef += 1
            continue

        # Strategy 4: Département centroid
        if not code.startswith("ZZ"):
            dept = _extract_dept_from_commune(code)
            if dept in dept_coords:
                lat, lon = dept_coords[dept]
                new_records.append({"location": code, "latitude": lat, "longitude": lon})
                resolved_dept += 1
                continue

        # Strategy 5: France center — last resort
        new_records.append({
            "location": code,
            "latitude": FRANCE_CENTER_LAT,
            "longitude": FRANCE_CENTER_LON,
        })
        resolved_national += 1

    print(f"    Resolved via geocoded historical: {resolved_hist}")
    print(f"    Resolved via geocoded ZZ consular: {resolved_zz}")
    print(f"    Resolved via successor commune: {resolved_chef}")
    print(f"    Resolved via département centroid: {resolved_dept}")
    print(f"    Resolved to France center (unknown): {resolved_national}")

    if new_records:
        extra = pd.DataFrame(new_records)
        extra["latitude"] = extra["latitude"].astype(np.float32)
        extra["longitude"] = extra["longitude"].astype(np.float32)
        result = pd.concat([result, extra], ignore_index=True)
        result = result.drop_duplicates(subset=["location"], keep="first")

    return result


def main():
    data_dir = Path("data")
    geo_dir = data_dir / "geo"
    geo_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Fetch commune centroids
    commune_df = fetch_commune_centroids()
    commune_path = geo_dir / "commune_coords.parquet"
    commune_df.to_parquet(commune_path, index=False)
    print(f"Saved {len(commune_df)} commune centroids to {commune_path}")

    # Step 2: Fetch historical commune mapping
    historical_mapping = fetch_historical_commune_mapping()

    # Step 3: Build unified location lookup
    location_df = build_location_coords(data_dir, commune_df)

    # Step 4: Build dept_coords lookup for fallback resolution
    commune_with_dept = commune_df.copy()
    commune_with_dept["dept"] = commune_with_dept["code_commune"].apply(_extract_dept_from_commune)
    dept_coords: dict[str, tuple[float, float]] = {}
    for dept, grp in commune_with_dept.groupby("dept"):
        dept_coords[dept] = (grp["latitude"].mean(), grp["longitude"].mean())

    # Step 5: Resolve historical/missing commune codes
    location_df = _resolve_historical_communes(
        location_df, data_dir, historical_mapping, dept_coords
    )

    location_path = geo_dir / "location_coords.parquet"
    location_df.to_parquet(location_path, index=False)
    print(f"Saved {len(location_df)} location coordinates to {location_path}")

    # Summary
    print("\n--- Summary ---")
    print(f"Communes:        {(location_df['location'].str.len() == 5).sum()}")
    print(f"National:        {(location_df['location'] == 'National').sum()}")
    print(f"Total locations: {len(location_df)}")

    # Verify: check remaining unmatched
    cand_path = data_dir / "elections" / "agregees" / "candidats_results.parquet"
    if cand_path.exists():
        cand = pd.read_parquet(cand_path, columns=["code_commune"])
        election_codes = set(cand["code_commune"].unique())
        still_unmatched = election_codes - set(location_df["location"].unique())
        print(f"Unmatched election locations remaining: {len(still_unmatched)}")

    # Spot-check Paris
    paris = location_df[location_df["location"] == "75056"]
    if not paris.empty:
        print(f"\nParis (75056): lat={paris.iloc[0]['latitude']:.4f}, lon={paris.iloc[0]['longitude']:.4f}")


if __name__ == "__main__":
    main()
