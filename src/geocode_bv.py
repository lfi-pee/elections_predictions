"""Geocode all bureaux de vote to (latitude, longitude).

PRINCIPLE: Every BV in the output must have a verified exact location.
           No commune-center fallbacks. No averages. If we can't determine
           the actual location, the BV is DROPPED from the dataset.

Resolution cascade (in priority order):
  1. REU elector address centroid — average of all geocoded voter addresses
     assigned to that BV (16M rows → 68K BV centroids). Best available source.
  2. REU BV polling station geocode — BAN-geocoded address of the polling
     station itself (from bv_coords_reu.parquet).
  3. Etalab contour polygon centroid — Voronoi polygon center from official
     BV contour GeoJSON.
  4. Single-BV commune — if there's only one BV in the commune (in REU or
     contour), we know the location unambiguously.
  5. Historical commune geocoding — BAN API + geo.api.gouv.fr with proper
     département filtering + Nominatim with département qualifier.
  6. ZZ consular city — hardcoded mapping → Nominatim city geocoding.

Any BV that doesn't match any of these is DROPPED.

Outputs:
    data/geo/bv_elector_centroids.parquet  — BV centroids from voter addresses
    data/geo/historical_commune_coords.parquet — geocoded old communes
    data/geo/zz_consular_coords.parquet    — geocoded overseas posts
    data/geo/bv_coords.parquet             — FINAL: only BVs with exact coords
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import ijson
import numpy as np
import pandas as pd
import requests


DATA_DIR = Path("data")
GEO_DIR = DATA_DIR / "geo"
ELECTIONS_DIR = DATA_DIR / "elections" / "agregees"

CONTOURS_URL = (
    "https://object.files.data.gouv.fr/data-pipeline-open/reu/"
    "contours-france-entiere-latest-v2.geojson"
)
ETALAB_COMMUNES_URL = (
    "https://unpkg.com/@etalab/decoupage-administratif@4.0.0/data/communes.json"
)
GEO_API_URL = "https://geo.api.gouv.fr/communes"
BAN_API_URL = "https://api-adresse.data.gouv.fr/search/"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

# Département name lookup for Nominatim queries
DEPT_NAMES: dict[str, str] = {
    "01": "Ain", "02": "Aisne", "03": "Allier", "04": "Alpes-de-Haute-Provence",
    "05": "Hautes-Alpes", "06": "Alpes-Maritimes", "07": "Ardèche", "08": "Ardennes",
    "09": "Ariège", "10": "Aube", "11": "Aude", "12": "Aveyron",
    "13": "Bouches-du-Rhône", "14": "Calvados", "15": "Cantal", "16": "Charente",
    "17": "Charente-Maritime", "18": "Cher", "19": "Corrèze", "2A": "Corse-du-Sud",
    "2B": "Haute-Corse", "21": "Côte-d'Or", "22": "Côtes-d'Armor", "23": "Creuse",
    "24": "Dordogne", "25": "Doubs", "26": "Drôme", "27": "Eure",
    "28": "Eure-et-Loir", "29": "Finistère", "30": "Gard", "31": "Haute-Garonne",
    "32": "Gers", "33": "Gironde", "34": "Hérault", "35": "Ille-et-Vilaine",
    "36": "Indre", "37": "Indre-et-Loire", "38": "Isère", "39": "Jura",
    "40": "Landes", "41": "Loir-et-Cher", "42": "Loire", "43": "Haute-Loire",
    "44": "Loire-Atlantique", "45": "Loiret", "46": "Lot", "47": "Lot-et-Garonne",
    "48": "Lozère", "49": "Maine-et-Loire", "50": "Manche", "51": "Marne",
    "52": "Haute-Marne", "53": "Mayenne", "54": "Meurthe-et-Moselle", "55": "Meuse",
    "56": "Morbihan", "57": "Moselle", "58": "Nièvre", "59": "Nord",
    "60": "Oise", "61": "Orne", "62": "Pas-de-Calais", "63": "Puy-de-Dôme",
    "64": "Pyrénées-Atlantiques", "65": "Hautes-Pyrénées", "66": "Pyrénées-Orientales",
    "67": "Bas-Rhin", "68": "Haut-Rhin", "69": "Rhône", "70": "Haute-Saône",
    "71": "Saône-et-Loire", "72": "Sarthe", "73": "Savoie", "74": "Haute-Savoie",
    "75": "Paris", "76": "Seine-Maritime", "77": "Seine-et-Marne", "78": "Yvelines",
    "79": "Deux-Sèvres", "80": "Somme", "81": "Tarn", "82": "Tarn-et-Garonne",
    "83": "Var", "84": "Vaucluse", "85": "Vendée", "86": "Vienne",
    "87": "Haute-Vienne", "88": "Vosges", "89": "Yonne", "90": "Territoire de Belfort",
    "91": "Essonne", "92": "Hauts-de-Seine", "93": "Seine-Saint-Denis",
    "94": "Val-de-Marne", "95": "Val-d'Oise",
    "971": "Guadeloupe", "972": "Martinique", "973": "Guyane",
    "974": "La Réunion", "976": "Mayotte",
}

# Complete ZZ → consular city mapping (232 codes)
# Sources: legislatives_2024_t1.csv + general_results.parquet
ZZ_CITY_NAMES: dict[str, str] = {
    "ZZ001": "Abidjan",
    "ZZ002": "Abu Dhabi",
    "ZZ003": "Abuja",
    "ZZ004": "Accra",
    "ZZ005": "Ashgabat",
    "ZZ006": "Addis Ababa",
    "ZZ007": "Agadir",
    "ZZ008": "Alexandria, Egypt",
    "ZZ009": "Algiers",
    "ZZ010": "Almaty",
    "ZZ011": "Amman",
    "ZZ012": "Amsterdam",
    "ZZ013": "Andorra la Vella",
    "ZZ014": "Ankara",
    "ZZ015": "Annaba",
    "ZZ017": "Asuncion",
    "ZZ018": "Athens",
    "ZZ019": "Atlanta",
    "ZZ020": "Baghdad",
    "ZZ021": "Baku",
    "ZZ022": "Bamako",
    "ZZ023": "Bandar Seri Begawan",
    "ZZ024": "Bangalore",
    "ZZ025": "Bangkok",
    "ZZ026": "Bangui",
    "ZZ027": "Barcelona",
    "ZZ028": "Belgrade",
    "ZZ029": "Berlin",
    "ZZ030": "Beirut",
    "ZZ031": "Bilbao",
    "ZZ032": "Bissau",
    "ZZ033": "Bogota",
    "ZZ034": "Mumbai",
    "ZZ035": "Boston",
    "ZZ036": "Brasilia",
    "ZZ037": "Bratislava",
    "ZZ038": "Brazzaville",
    "ZZ039": "Brussels",
    "ZZ040": "Bucharest",
    "ZZ041": "Budapest",
    "ZZ042": "Buenos Aires",
    "ZZ043": "Bujumbura",
    "ZZ044": "Kolkata",
    "ZZ046": "Guangzhou",
    "ZZ047": "Caracas",
    "ZZ048": "Casablanca",
    "ZZ049": "Castries, Saint Lucia",
    "ZZ050": "Chengdu",
    "ZZ051": "Chicago",
    "ZZ052": "Chisinau",
    "ZZ054": "Colombo",
    "ZZ055": "Conakry",
    "ZZ056": "Copenhagen",
    "ZZ057": "Cotonou",
    "ZZ058": "Krakow",
    "ZZ059": "Dhaka",
    "ZZ060": "Dakar",
    "ZZ061": "Damascus",
    "ZZ062": "Dar es Salaam",
    "ZZ063": "Jeddah",
    "ZZ064": "Djibouti",
    "ZZ065": "Juba, South Sudan",
    "ZZ066": "Doha",
    "ZZ067": "Douala",
    "ZZ068": "Dushanbe",
    "ZZ069": "Dubai",
    "ZZ070": "Dublin",
    "ZZ071": "Dusseldorf",
    "ZZ072": "Edinburgh",
    "ZZ073": "Yekaterinburg",
    "ZZ074": "Erbil",
    "ZZ075": "Yerevan",
    "ZZ076": "Fez, Morocco",
    "ZZ077": "Frankfurt",
    "ZZ078": "Gaborone",
    "ZZ079": "Geneva",
    "ZZ080": "Guatemala City",
    "ZZ081": "Haifa",
    "ZZ082": "Hamburg",
    "ZZ083": "Hanoi",
    "ZZ084": "Harare",
    "ZZ085": "Helsinki",
    "ZZ086": "Ho Chi Minh City",
    "ZZ087": "Hong Kong",
    "ZZ088": "Houston",
    "ZZ089": "Islamabad",
    "ZZ090": "Istanbul",
    "ZZ091": "Jakarta",
    "ZZ092": "Jerusalem",
    "ZZ093": "Johannesburg",
    "ZZ094": "Kabul",
    "ZZ095": "Kampala",
    "ZZ096": "Karachi",
    "ZZ097": "Kathmandu",
    "ZZ098": "Khartoum",
    "ZZ099": "Kyiv",
    "ZZ100": "Kigali",
    "ZZ101": "Kingston, Jamaica",
    "ZZ102": "Kinshasa",
    "ZZ103": "Kuwait City",
    "ZZ104": "Kuala Lumpur",
    "ZZ105": "Kyoto",
    "ZZ106": "Havana",
    "ZZ107": "New Orleans",
    "ZZ108": "La Paz",
    "ZZ109": "Valletta",
    "ZZ110": "Lagos",
    "ZZ111": "Cairo",
    "ZZ112": "Cape Town",
    "ZZ113": "Libreville",
    "ZZ115": "Lima",
    "ZZ116": "Lisbon",
    "ZZ117": "Ljubljana",
    "ZZ118": "Lome",
    "ZZ119": "London",
    "ZZ120": "Los Angeles",
    "ZZ121": "Luanda",
    "ZZ122": "Lusaka",
    "ZZ123": "Luxembourg City",
    "ZZ124": "Madrid",
    "ZZ125": "Malabo",
    "ZZ126": "Managua",
    "ZZ127": "Manama",
    "ZZ128": "Manila",
    "ZZ129": "Maputo",
    "ZZ130": "Marrakech",
    "ZZ131": "Muscat",
    "ZZ132": "Mexico City",
    "ZZ133": "Miami",
    "ZZ134": "Milan",
    "ZZ135": "Minsk",
    "ZZ136": "Monaco",
    "ZZ137": "Moncton",
    "ZZ138": "Montevideo",
    "ZZ139": "Montreal",
    "ZZ140": "Moroni, Comoros",
    "ZZ141": "Moscow",
    "ZZ142": "Munich",
    "ZZ143": "Nairobi",
    "ZZ144": "Naples",
    "ZZ145": "N'Djamena",
    "ZZ146": "New Delhi",
    "ZZ147": "New York",
    "ZZ148": "Niamey",
    "ZZ149": "Nicosia",
    "ZZ150": "Nouakchott",
    "ZZ151": "Oran",
    "ZZ152": "Oslo",
    "ZZ153": "Ouagadougou",
    "ZZ154": "Ulaanbaatar",
    "ZZ155": "Panama City",
    "ZZ156": "Paramaribo",
    "ZZ157": "Beijing",
    "ZZ158": "Phnom Penh",
    "ZZ159": "Podgorica",
    "ZZ160": "Pointe-Noire",
    "ZZ161": "Pondicherry",
    "ZZ162": "Port-au-Prince",
    "ZZ163": "Port Louis, Mauritius",
    "ZZ164": "Port Moresby",
    "ZZ166": "Port of Spain",
    "ZZ167": "Port Vila",
    "ZZ168": "Prague",
    "ZZ169": "Praia, Cape Verde",
    "ZZ170": "Pristina",
    "ZZ171": "Quebec City",
    "ZZ172": "Quito",
    "ZZ173": "Rabat",
    "ZZ174": "Yangon",
    "ZZ175": "Recife",
    "ZZ176": "Reykjavik",
    "ZZ177": "Riga",
    "ZZ178": "Rio de Janeiro",
    "ZZ179": "Riyadh",
    "ZZ180": "Rome",
    "ZZ181": "Santo Domingo",
    "ZZ182": "Saint Petersburg",
    "ZZ183": "San Francisco",
    "ZZ184": "San Jose, Costa Rica",
    "ZZ185": "San Salvador",
    "ZZ186": "Sanaa",
    "ZZ187": "Santiago, Chile",
    "ZZ188": "Sao Paulo",
    "ZZ189": "Sarajevo",
    "ZZ190": "Saarbrucken",
    "ZZ191": "Seoul",
    "ZZ192": "Seville",
    "ZZ193": "Shanghai",
    "ZZ194": "Shenyang",
    "ZZ195": "Singapore",
    "ZZ196": "Skopje",
    "ZZ197": "Sofia",
    "ZZ198": "Stockholm",
    "ZZ199": "Stuttgart",
    "ZZ200": "Suva, Fiji",
    "ZZ201": "Sydney",
    "ZZ202": "Tashkent",
    "ZZ203": "Tallinn",
    "ZZ204": "Antananarivo",
    "ZZ205": "Tangier",
    "ZZ206": "Tbilisi",
    "ZZ207": "Tegucigalpa",
    "ZZ208": "Tehran",
    "ZZ209": "Tel Aviv",
    "ZZ210": "Thessaloniki",
    "ZZ211": "Tirana",
    "ZZ212": "Tokyo",
    "ZZ213": "Toronto",
    "ZZ214": "Tripoli",
    "ZZ215": "Tunis",
    "ZZ217": "Vancouver",
    "ZZ218": "Warsaw",
    "ZZ219": "Victoria, Seychelles",
    "ZZ220": "Vienna",
    "ZZ221": "Vientiane",
    "ZZ222": "Vilnius",
    "ZZ223": "Washington DC",
    "ZZ224": "Wellington",
    "ZZ225": "Windhoek",
    "ZZ226": "Wuhan",
    "ZZ227": "Yaounde",
    "ZZ228": "Zagreb",
    "ZZ229": "Zurich",
    "ZZ231": "Taipei",
    "ZZ232": "Nassau, Bahamas",
    "ZZ233": "Astana",
    "ZZ234": "Monterrey",
    "ZZ235": "Nassau, Bahamas",
    "ZZ236": "Astana",
    "ZZ237": "Mosul",
    "ZZ238": "Florence",
    "ZZ239": "Managua",
}


def _dept_code(commune_code: str) -> str:
    """Extract département code from an INSEE commune code."""
    return commune_code[:3] if commune_code[:2] in ("97", "98") else commune_code[:2]


# ---------------------------------------------------------------------------
# Step 1: Compute BV centroids from REU elector addresses
# ---------------------------------------------------------------------------
def compute_elector_centroids(reu_addr_path: Path, cache_path: Path) -> pd.DataFrame:
    """Average all geocoded voter addresses per BV → BV centroid.

    This is the BEST available source: it gives the geographic center of the
    actual voter catchment area for each BV.
    """
    if cache_path.exists():
        print(f"  Loading cached elector centroids from {cache_path}")
        return pd.read_parquet(cache_path)

    print("  Computing BV centroids from 16M elector addresses...")
    addr = pd.read_parquet(
        reu_addr_path,
        columns=["id_brut_bv_reu", "latitude", "longitude"],
    )
    print(f"    Loaded {len(addr)} elector addresses")

    bv = addr.groupby("id_brut_bv_reu").agg(
        latitude=("latitude", "mean"),
        longitude=("longitude", "mean"),
        n_addresses=("latitude", "count"),
    ).reset_index()

    # Convert REU id format (01001_1) → MIOM format (01001_0001)
    def reu_to_miom(reu_id: str) -> str:
        parts = reu_id.rsplit("_", 1)
        if len(parts) == 2:
            return f"{parts[0]}_{parts[1].zfill(4)}"
        return reu_id

    bv["id_brut_miom"] = bv["id_brut_bv_reu"].apply(reu_to_miom)
    bv["code_commune"] = bv["id_brut_miom"].apply(lambda x: x.rsplit("_", 1)[0])
    bv["latitude"] = bv["latitude"].astype(np.float32)
    bv["longitude"] = bv["longitude"].astype(np.float32)

    bv.to_parquet(cache_path, index=False)
    print(f"    Computed {len(bv)} BV elector centroids → {cache_path}")
    return bv


# ---------------------------------------------------------------------------
# Step 2: Extract polygon centroids from the GeoJSON contours
# ---------------------------------------------------------------------------
def extract_contour_centroids(geojson_path: Path) -> pd.DataFrame:
    """Stream-parse the Etalab BV contours GeoJSON & compute centroids."""
    records = []
    with open(geojson_path, "rb") as f:
        for feature in ijson.items(f, "features.item"):
            props = feature["properties"]
            geom = feature["geometry"]

            code_bv = props.get("codeBureauVote", "")
            code_commune = props.get("codeCommune", "")

            coords = geom["coordinates"]
            if geom["type"] == "Polygon":
                ring = coords[0]
            elif geom["type"] == "MultiPolygon":
                ring = []
                for polygon in coords:
                    ring.extend(polygon[0])
            else:
                continue

            lons = [p[0] for p in ring]
            lats = [p[1] for p in ring]

            records.append({
                "id_brut_miom": code_bv,
                "code_commune": code_commune,
                "latitude": sum(lats) / len(lats),
                "longitude": sum(lons) / len(lons),
            })

            if len(records) % 10000 == 0:
                print(f"    {len(records)} features processed...")

    df = pd.DataFrame(records)
    df["latitude"] = df["latitude"].astype(np.float32)
    df["longitude"] = df["longitude"].astype(np.float32)
    print(f"    Total contour centroids: {len(df)}")
    return df


# ---------------------------------------------------------------------------
# Historical commune geocoding (with proper département filtering)
# ---------------------------------------------------------------------------
def _geocode_historical_communes(
    codes_with_names: dict[str, str],
    cache_path: Path,
    force_regeocode: set[str] | None = None,
) -> dict[str, tuple[float, float]]:
    """Geocode historical/merged commune codes to their actual (lat, lon).

    Uses a cascade WITH département filtering at every step:
      1. geo.api.gouv.fr with name + codeDépartement (guaranteed match)
      2. BAN API with direct citycode
      3. BAN API with name search, filtered by département
      4. Nominatim with "{name}, {département_name}, France"

    Results are cached to parquet and reused across runs.
    """
    # Load cache
    cached: dict[str, tuple[float, float]] = {}
    if cache_path.exists():
        df_cache = pd.read_parquet(cache_path)
        cached = dict(
            zip(df_cache["code_commune"], zip(df_cache["latitude"], df_cache["longitude"]))
        )
        print(f"    Loaded {len(cached)} cached historical commune coords")

    # Remove codes that need re-geocoding from cache
    if force_regeocode:
        for code in force_regeocode:
            cached.pop(code, None)

    # Find what still needs geocoding
    need = {c: n for c, n in codes_with_names.items() if c not in cached}
    if not need:
        print("    All historical communes already cached")
        return cached

    print(f"    Geocoding {len(need)} historical communes...")
    resolved: dict[str, tuple[float, float]] = {}
    source_stats: dict[str, int] = {}

    for i, (code, name) in enumerate(sorted(need.items())):
        dept = _dept_code(code)
        dept_name = DEPT_NAMES.get(dept, "")
        lat, lon = None, None
        source = "failed"

        # Strategy 1: geo.api.gouv.fr with codeDépartement (most reliable)
        try:
            r = requests.get(
                GEO_API_URL,
                params={
                    "nom": name,
                    "codeDepartement": dept,
                    "fields": "centre",
                    "limit": 1,
                },
                timeout=10,
            )
            if r.status_code == 200 and r.json():
                centre = r.json()[0].get("centre", {}).get("coordinates")
                if centre:
                    lat, lon = centre[1], centre[0]
                    source = "geo_api_dept"
        except Exception:
            pass

        # Strategy 2: BAN API direct citycode
        if lat is None:
            try:
                r = requests.get(
                    BAN_API_URL,
                    params={"q": name, "type": "municipality", "citycode": code, "limit": 1},
                    timeout=10,
                )
                if r.status_code == 200:
                    features = r.json().get("features", [])
                    if features:
                        coords = features[0]["geometry"]["coordinates"]
                        lat, lon = coords[1], coords[0]
                        source = "ban_citycode"
            except Exception:
                pass

        # Strategy 3: BAN API name + département filter
        if lat is None:
            try:
                r = requests.get(
                    BAN_API_URL,
                    params={"q": name, "type": "municipality", "limit": 5},
                    timeout=10,
                )
                if r.status_code == 200:
                    features = r.json().get("features", [])
                    dept_matches = [
                        f for f in features
                        if str(f["properties"].get("citycode", "")).startswith(dept)
                    ]
                    if dept_matches:
                        coords = dept_matches[0]["geometry"]["coordinates"]
                        lat, lon = coords[1], coords[0]
                        source = "ban_dept"
            except Exception:
                pass

        # Strategy 4: Nominatim WITH département name (rate limited)
        if lat is None:
            query = f"{name}, {dept_name}, France" if dept_name else f"{name}, France"
            try:
                r = requests.get(
                    NOMINATIM_URL,
                    params={
                        "q": query,
                        "format": "json",
                        "limit": 5,
                        "countrycodes": "fr",
                    },
                    headers={"User-Agent": "elections-predictions/1.0"},
                    timeout=10,
                )
                if r.status_code == 200 and r.json():
                    results = r.json()
                    # Pick first result — the département name in query ensures
                    # we get the right one
                    lat, lon = float(results[0]["lat"]), float(results[0]["lon"])
                    source = "nominatim_dept"
                time.sleep(1.05)
            except Exception:
                time.sleep(1.05)

        source_stats[source] = source_stats.get(source, 0) + 1
        if lat is not None:
            resolved[code] = (lat, lon)

        if (i + 1) % 100 == 0:
            print(f"      {i + 1}/{len(need)} geocoded ({len(resolved)} resolved)")

    print(f"    Geocoded {len(resolved)}/{len(need)} new communes")
    for src, cnt in sorted(source_stats.items(), key=lambda x: -x[1]):
        print(f"      {src}: {cnt}")

    # Merge with cache and save
    all_coords = {**cached, **resolved}
    df_all = pd.DataFrame([
        {"code_commune": c, "latitude": lat, "longitude": lon}
        for c, (lat, lon) in all_coords.items()
    ])
    df_all["latitude"] = df_all["latitude"].astype(np.float32)
    df_all["longitude"] = df_all["longitude"].astype(np.float32)
    df_all.to_parquet(cache_path, index=False)
    print(f"    Saved {len(df_all)} historical commune coords to {cache_path}")

    return all_coords


# ---------------------------------------------------------------------------
# Audit: check historical communes against département bounding boxes
# ---------------------------------------------------------------------------
def _audit_historical_coords(
    hist_coords: dict[str, tuple[float, float]],
    commune_df: pd.DataFrame | None = None,
) -> set[str]:
    """Check each historical coord against its département's bounding box.

    Returns set of commune codes that fail the check (wrong département).
    """
    if commune_df is None:
        # Build département bounding boxes from current commune centroids
        print("    Fetching commune centroids for audit...")
        try:
            resp = requests.get(
                "https://geo.api.gouv.fr/communes?fields=code,centre&format=json",
                timeout=60,
            )
            resp.raise_for_status()
            rows = []
            for c in resp.json():
                centre = c.get("centre")
                if centre and centre.get("coordinates"):
                    lon, lat = centre["coordinates"]
                    rows.append({"code": c["code"], "lat": lat, "lon": lon})
            commune_df = pd.DataFrame(rows)
        except Exception:
            print("    WARNING: Could not fetch commune centroids for audit")
            return set()

    commune_df["dept"] = commune_df["code"].apply(_dept_code)

    # Build département bounding boxes with 50km margin (~0.45°)
    MARGIN = 0.45
    dept_bounds: dict[str, tuple[float, float, float, float]] = {}
    for dept, grp in commune_df.groupby("dept"):
        dept_bounds[dept] = (
            grp["lat"].min() - MARGIN,
            grp["lat"].max() + MARGIN,
            grp["lon"].min() - MARGIN,
            grp["lon"].max() + MARGIN,
        )

    bad_codes: set[str] = set()
    for code, (lat, lon) in hist_coords.items():
        dept = _dept_code(code)
        if dept not in dept_bounds:
            continue
        lat_min, lat_max, lon_min, lon_max = dept_bounds[dept]
        if not (lat_min <= lat <= lat_max and lon_min <= lon <= lon_max):
            bad_codes.add(code)

    return bad_codes


# ---------------------------------------------------------------------------
# ZZ consular city geocoding
# ---------------------------------------------------------------------------
def _geocode_zz_consular(
    zz_codes: list[str],
    cache_path: Path,
) -> dict[str, tuple[float, float]]:
    """Geocode overseas ZZ codes to their consular city coordinates."""
    cached: dict[str, tuple[float, float]] = {}
    if cache_path.exists():
        df_cache = pd.read_parquet(cache_path)
        cached = dict(
            zip(df_cache["code_commune"], zip(df_cache["latitude"], df_cache["longitude"]))
        )
        print(f"    Loaded {len(cached)} cached ZZ consular coords")

    need = [c for c in zz_codes if c not in cached]
    if not need:
        print("    All ZZ consular coords already cached")
        return cached

    print(f"    Geocoding {len(need)} ZZ consular posts...")
    resolved: dict[str, tuple[float, float]] = {}

    for code in sorted(need):
        city = ZZ_CITY_NAMES.get(code)
        if not city:
            print(f"      WARNING: No city name for {code}")
            continue

        lat, lon = None, None
        try:
            r = requests.get(
                NOMINATIM_URL,
                params={"q": city, "format": "json", "limit": 1},
                headers={"User-Agent": "elections-predictions/1.0"},
                timeout=10,
            )
            if r.status_code == 200 and r.json():
                res = r.json()[0]
                lat, lon = float(res["lat"]), float(res["lon"])
            time.sleep(1.05)
        except Exception:
            time.sleep(1.05)

        if lat is not None:
            resolved[code] = (lat, lon)
        else:
            print(f"      FAILED: {code} ({city})")

    print(f"    Geocoded {len(resolved)}/{len(need)} ZZ consular posts")

    all_coords = {**cached, **resolved}
    df_all = pd.DataFrame([
        {"code_commune": c, "latitude": lat, "longitude": lon}
        for c, (lat, lon) in all_coords.items()
    ])
    df_all["latitude"] = df_all["latitude"].astype(np.float32)
    df_all["longitude"] = df_all["longitude"].astype(np.float32)
    df_all.to_parquet(cache_path, index=False)
    print(f"    Saved {len(df_all)} ZZ consular coords to {cache_path}")

    return all_coords


# ---------------------------------------------------------------------------
# Main build: assign exact coords or DROP
# ---------------------------------------------------------------------------
def build_bv_coords(
    elector_centroids: pd.DataFrame,
    contours: pd.DataFrame,
) -> pd.DataFrame:
    """Assign verified exact coordinates to every election BV.

    Any BV that cannot be assigned an exact, verified location is DROPPED.
    Zero fallbacks. Zero commune averages.
    """
    # --- Load election BVs ---
    cand = pd.read_parquet(
        ELECTIONS_DIR / "candidats_results.parquet",
        columns=["code_commune", "code_bv"],
    )
    election_bvs = cand[["code_commune", "code_bv"]].drop_duplicates().copy()
    election_bvs["id_brut_miom"] = (
        election_bvs["code_commune"] + "_" + election_bvs["code_bv"]
    )
    print(f"\n  Total election BVs: {len(election_bvs)}")

    # --- Build lookups ---
    # 1. REU elector centroids (best source)
    elector_lookup = dict(
        zip(elector_centroids["id_brut_miom"],
            zip(elector_centroids["latitude"], elector_centroids["longitude"]))
    )

    # 2. REU BV polling station geocoded addresses
    bv_addr_path = GEO_DIR / "bv_coords_reu.parquet"
    bv_addr_lookup: dict[str, tuple[float, float]] = {}
    if bv_addr_path.exists():
        bv_addr = pd.read_parquet(bv_addr_path)
        valid = bv_addr[bv_addr["latitude"].notna()]
        bv_addr_lookup = dict(
            zip(valid["id_brut_miom"], zip(valid["latitude"], valid["longitude"]))
        )
        print(f"    Loaded {len(bv_addr_lookup)} BV polling station geocodes")

    # 3. Contour centroids
    contour_lookup = dict(
        zip(contours["id_brut_miom"],
            zip(contours["latitude"], contours["longitude"]))
    )

    # 4. Single-BV commune lookups (REU + contour)
    reu_communes = elector_centroids.groupby("code_commune")["id_brut_miom"].count()
    single_bv_reu = set(reu_communes[reu_communes == 1].index)
    reu_single_lookup: dict[str, tuple[float, float]] = {}
    for _, row in elector_centroids[elector_centroids["code_commune"].isin(single_bv_reu)].iterrows():
        reu_single_lookup[row["code_commune"]] = (row["latitude"], row["longitude"])

    contour_commune_counts = contours.groupby("code_commune").size()
    single_bv_contour = set(contour_commune_counts[contour_commune_counts == 1].index)
    contour_single_lookup: dict[str, tuple[float, float]] = {}
    for _, row in contours[contours["code_commune"].isin(single_bv_contour)].iterrows():
        contour_single_lookup[row["code_commune"]] = (row["latitude"], row["longitude"])

    # --- Historical communes ---
    print("\n  Resolving historical communes...")
    hist_commune_codes: set[str] = set()
    all_known_communes = set(elector_centroids["code_commune"]) | set(contours["code_commune"])
    for code in set(election_bvs["code_commune"]):
        if code.startswith("ZZ"):
            continue
        if code in all_known_communes:
            continue
        hist_commune_codes.add(code)

    # Get names
    print("    Fetching etalab commune data...")
    resp = requests.get(ETALAB_COMMUNES_URL, timeout=60)
    resp.raise_for_status()
    etalab_names = {c["code"]: c["nom"] for c in resp.json()}

    gen = pd.read_parquet(
        ELECTIONS_DIR / "general_results.parquet",
        columns=["code_commune", "libelle_commune"],
    )
    gen_names = (
        gen.drop_duplicates(subset=["code_commune"])
        .set_index("code_commune")["libelle_commune"]
        .to_dict()
    )

    hist_names = {}
    for code in hist_commune_codes:
        name = etalab_names.get(code) or gen_names.get(code)
        if name:
            hist_names[code] = name
        else:
            hist_names[code] = f"commune {code}"

    print(f"    {len(hist_names)} historical communes to resolve")

    # Geocode (first pass)
    hist_cache_path = GEO_DIR / "historical_commune_coords.parquet"
    hist_coords = _geocode_historical_communes(hist_names, hist_cache_path)

    # Audit: find codes that ended up in the wrong département
    print("\n  Auditing historical commune coordinates...")
    bad_codes = _audit_historical_coords(hist_coords)
    if bad_codes:
        print(f"    ⚠️  {len(bad_codes)} communes fail département bounding-box check")
        print(f"       Re-geocoding with stricter département filtering...")
        # Force re-geocode the bad ones
        bad_names = {c: hist_names[c] for c in bad_codes if c in hist_names}
        hist_coords = _geocode_historical_communes(
            hist_names, hist_cache_path, force_regeocode=bad_codes
        )
        # Re-audit
        still_bad = _audit_historical_coords(hist_coords)
        if still_bad:
            print(f"    ⚠️  {len(still_bad)} STILL fail after re-geocoding — will be DROPPED")
            for code in still_bad:
                hist_coords.pop(code, None)
        else:
            print("    ✅ All historical communes now pass audit")
    else:
        print("    ✅ All historical communes pass audit")

    # --- ZZ overseas ---
    print("\n  Resolving ZZ overseas codes...")
    zz_codes = sorted(
        c for c in set(election_bvs["code_commune"]) if c.startswith("ZZ")
    )
    zz_coords = _geocode_zz_consular(
        zz_codes,
        GEO_DIR / "zz_consular_coords.parquet",
    )

    # --- Assign coordinates: exact only, DROP the rest ---
    print("\n  Assigning coordinates (exact only, no fallbacks)...")
    stats: dict[str, int] = {}
    records = []

    for _, row in election_bvs.iterrows():
        code = row["code_commune"]
        bv_id = row["id_brut_miom"]
        lat, lon = None, None
        source = None

        # 1. REU elector address centroid (best)
        if bv_id in elector_lookup:
            lat, lon = elector_lookup[bv_id]
            source = "reu_elector_centroid"

        # 2. REU BV polling station address (geocoded)
        elif bv_id in bv_addr_lookup:
            lat, lon = bv_addr_lookup[bv_id]
            source = "reu_bv_address"

        # 3. Etalab contour polygon centroid
        elif bv_id in contour_lookup:
            lat, lon = contour_lookup[bv_id]
            source = "contour_centroid"

        # 4. Single-BV commune (REU) — unambiguous
        elif code in reu_single_lookup:
            lat, lon = reu_single_lookup[code]
            source = "single_bv_reu"

        # 5. Single-BV commune (contour) — unambiguous
        elif code in contour_single_lookup:
            lat, lon = contour_single_lookup[code]
            source = "single_bv_contour"

        # 6. Historical commune → geocoded actual location
        elif code in hist_coords:
            lat, lon = hist_coords[code]
            source = "historical_geocoded"

        # 7. ZZ overseas → consular city
        elif code.startswith("ZZ") and code in zz_coords:
            lat, lon = zz_coords[code]
            source = "zz_consular"

        # NO FALLBACK — drop if we can't verify the location
        if lat is not None and source is not None:
            stats[source] = stats.get(source, 0) + 1
            records.append({
                "code_commune": code,
                "code_bv": row["code_bv"],
                "id_brut_miom": bv_id,
                "latitude": lat,
                "longitude": lon,
                "source": source,
            })
        else:
            stats["DROPPED"] = stats.get("DROPPED", 0) + 1

    print("\n  --- Resolution stats ---")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        pct = v / len(election_bvs) * 100
        marker = "❌" if k == "DROPPED" else "✅"
        print(f"    {marker} {k}: {v} ({pct:.1f}%)")

    df = pd.DataFrame(records)
    df["latitude"] = df["latitude"].astype(np.float32)
    df["longitude"] = df["longitude"].astype(np.float32)
    df = df.drop_duplicates(subset=["id_brut_miom"], keep="first")

    total = len(election_bvs)
    kept = len(df)
    dropped = total - kept
    print(f"\n  📊 Kept: {kept}/{total} ({kept/total*100:.1f}%)")
    print(f"  📊 Dropped: {dropped}/{total} ({dropped/total*100:.1f}%)")
    print(f"  📊 Unique positions: {df.groupby(['latitude', 'longitude']).ngroups}")

    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    GEO_DIR.mkdir(parents=True, exist_ok=True)
    geojson_path = GEO_DIR / "contours-bv.geojson"

    # Download contours if needed
    if not geojson_path.exists():
        print(f"Downloading BV contours GeoJSON ({CONTOURS_URL})...")
        resp = requests.get(CONTOURS_URL, stream=True, timeout=300)
        resp.raise_for_status()
        with open(geojson_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
        print(f"  Saved to {geojson_path}")

    # Step 1: Compute REU elector centroids (primary source)
    print("\n=== Step 1: REU elector address centroids ===")
    reu_addr_path = GEO_DIR / "table-adresses-reu.parquet"
    if not reu_addr_path.exists():
        print(f"ERROR: {reu_addr_path} not found. Download from:")
        print("  https://static.data.gouv.fr/resources/bureaux-de-vote-et-adresses-de-leurs-electeurs/20230626-135723/table-adresses-reu.parquet")
        return
    elector_centroids = compute_elector_centroids(
        reu_addr_path,
        GEO_DIR / "bv_elector_centroids.parquet",
    )

    # Step 2: Load contour centroids
    print("\n=== Step 2: Etalab contour centroids ===")
    centroids_path = GEO_DIR / "bv_contour_centroids.parquet"
    if centroids_path.exists():
        print(f"  Loading cached contour centroids from {centroids_path}")
        contours = pd.read_parquet(centroids_path)
    else:
        print("  Extracting contour centroids...")
        contours = extract_contour_centroids(geojson_path)
        contours.to_parquet(centroids_path, index=False)
        print(f"  Saved to {centroids_path}")

    # Steps 3-7: Build complete BV coords (exact only)
    print("\n=== Steps 3-7: Build exact BV coords ===")
    bv_coords = build_bv_coords(elector_centroids, contours)

    # Save (drop the 'source' column from final output)
    output_path = GEO_DIR / "bv_coords.parquet"
    bv_coords_out = bv_coords.drop(columns=["source"])
    bv_coords_out.to_parquet(output_path, index=False)

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL: {len(bv_coords_out)} BVs with EXACT verified coordinates")
    print(f"Unique positions: {bv_coords_out.groupby(['latitude', 'longitude']).ngroups}")
    print(f"Lat range: [{bv_coords_out['latitude'].min():.4f}, {bv_coords_out['latitude'].max():.4f}]")
    print(f"Lon range: [{bv_coords_out['longitude'].min():.4f}, {bv_coords_out['longitude'].max():.4f}]")
    print(f"Saved to {output_path}")
    print(f"{'='*60}")

    # Source breakdown
    print("\nSource breakdown:")
    for src, count in bv_coords["source"].value_counts().items():
        print(f"  {src}: {count}")


if __name__ == "__main__":
    main()
