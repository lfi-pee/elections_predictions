# Geo-Mapping: Lat/Long for Every Location

## Goal

Assign a `(latitude, longitude)` coordinate pair to **every single token** in the data — election results (at bureau de vote level), polls, and demographics — regardless of the geographic granularity of the `location` field. **Zero fallback to France center** — every token has at least a commune-level coordinate.

---

## Current State — ✅ ALL LOCATIONS MAPPED

### Location Types in the Token Pool

Election results use **bureau de vote** (BV) level locations. Polls and demographics use coarser granularities. All are mapped to coordinates.

#### Election Results (`candidats_results.parquet` + `general_results.parquet`)

The raw data contains **5 geographic columns**:

| Column | Unique values | Example | Used? |
|---|---|---|---|
| `code_commune` | **37,053** | `01160`, `75056` | Part of BV key |
| `code_departement` | **117** | `01`, `75`, `2A` | ❌ |
| `code_canton` | **79** | `08`, `01` | ❌ |
| `code_circonscription` | **21** | `04`, `05` | ❌ |
| `code_bv` (bureau de vote) | **2,226** strings (**76,836** unique (commune,bv) pairs) | `0001`, `0002` | ✅ |

`load_elections.py` keeps results at BV level. Location key = `"{code_commune}_{code_bv}"` (e.g. `"29019_0012"`) → **76,571 unique BV locations**. Coordinates come from exact BV positions in `bv_coords.parquet`.

#### Polls (`load_polls.py`)

| Granularity | Example `location` values | Token count | Source |
|---|---|---|---|
| **Country** | `"National"` | ~13,249 | Wiki polls, NSPPolls présidentielle |
| **Region** | `"Île-de-France"`, `"Bretagne"` (13 regions) | ~1,500 | NSPPolls régionales |
| **Commune** | `"75056"`, `"69123"` (~20 polled cities) | ~2,200 | Municipales polls |

#### Demographics (`load_demographics.py`)

| Granularity | Example `location` values | Token count |
|---|---|---|
| **Commune** | `"75056"`, `"29019"` (~36.7K) | ~2.17M |

The model **predicts** only **Result** tokens (BV level). All other tokens serve as **context** selected by the router.

---

## All Location Types

### 1. Bureaux de vote (~77,000) — PRIMARY

- **Key**: `"{code_commune}_{code_bv}"` (e.g. `"29019_0012"`)
- **Source**: **Exact BV coordinates** from `data/geo/bv_coords.parquet` — 72,795 BVs with verified positions.
- **Fallback**: For ~4,000 historical BVs not in `bv_coords.parquet`, the parent commune centroid is used.

#### BV Coverage & Confidence

| Source | Count | % | Confidence |
|---|---|---|---|
| REU elector address centroid | 66,417 | 86.4% | 🟢 **Best** — average of all geocoded voter addresses for each BV |
| Historical commune geocoded | 2,857 | 3.7% | 🟢 **High** — BAN/Nominatim with département filtering, all pass audit |
| REU BV polling station address | 2,119 | 2.8% | 🟢 **Exact** — BAN-geocoded address of the physical polling station |
| Single-BV commune (REU) | 1,001 | 1.3% | 🟢 **High** — only 1 BV in commune → unambiguous |
| ZZ consular city | 232 | 0.3% | 🟢 **Medium** — city-level precision for overseas consular posts |
| Contour polygon centroid | 167 | 0.2% | 🟢 **High** — Voronoi centroid from Etalab |
| Single-BV commune (contour) | 2 | 0.0% | 🟢 **High** — same as above |
| **Total kept** | **72,795** | **94.7%** | |
| ❌ Dropped (no exact coords) | 4,041 | 5.3% | N/A — old BV codes with no mapping to current data |

**70,036 unique coordinate positions** across 72,795 BVs.

#### What was dropped and why

The **4,041 dropped BVs** are historical BV codes from pre-2019 elections in communes that currently have multiple BVs, where the old BV numbering changed (e.g., old `06004_0003` → new `06004_0103`). No public dataset contains historical BV addresses — the REU only started in 2019. Rather than assign an imprecise commune-center average, these are excluded entirely.

### 2. Communes (~37,000)
- **Key**: `code_commune` (INSEE code, e.g. `"75056"`)
- **Source**: [geo.api.gouv.fr/communes](https://geo.api.gouv.fr/communes?fields=code,nom,centre&format=json) — bulk API call returns all ~35,000 communes at once.

### 3. Départements (~100)
- **Key**: `code_departement` (e.g. `"75"`, `"2A"`)
- **Source**: **Population-weighted average** of commune centroids within each département (weighted by `inscrits`).

### 4. Régions (~18)
- **Key**: Region name string (e.g. `"Île-de-France"`, `"Provence-Alpes-Côte d'Azur"`)
- **Source**: **Population-weighted average** of département centroids.

### 5. National
- **Key**: `"National"` string
- **Source**: Hard-coded geographic center of France: **(46.2276, 2.2137)**.

### 6. Cantons (~2,000+)
- **Key**: `(département, code_canton)` — canton codes are not globally unique.
- **Source**: Average of commune centroids within each (département, canton) pair.

### 7. Circonscriptions (~577)
- **Key**: `(département, code_circonscription)` — same as cantons.
- **Source**: Average of commune centroids within each (département, circonscription) pair.

---

## BV Geocoding Details (`src/geocode_bv.py`)

### Primary Source: REU Elector Address Centroids (86.4%)

The INSEE REU (Répertoire Électoral Unique) `table-adresses-reu.parquet` contains **16 million geocoded voter addresses**. Each address has `(latitude, longitude)` from the BAN (Base Adresse Nationale) and is tagged with its BV ID. By averaging all voter addresses per BV, we compute the **geographic center of the BV's actual voter catchment area** — strictly more precise than a Voronoi polygon centroid.

- **Input**: `data/geo/table-adresses-reu.parquet` (490 MB, 15,970,992 rows)
- **Output**: `data/geo/bv_elector_centroids.parquet` (68,830 BVs)
- **ID format conversion**: REU uses `01001_1`, MIOM uses `01001_0001` — zero-padded automatically

### Secondary Source: REU BV Polling Station Geocode (2.8%)

The `table-bv-reu.parquet` file contains the **physical address** of each polling station (e.g., "Salle des fêtes", "Mairie", "Espace 1500"). These are geocoded via the BAN API. Used for BVs that exist in the REU BV table but NOT in the address table (edge cases).

### Contour Polygon Centroids (0.2%)

For the few BVs that exist in the Etalab contour GeoJSON but not in the REU, the Voronoi polygon centroid is used.

### Historical Communes (3.7%)

French "communes nouvelles" (mergers) absorbed ~1,900 old communes between 1999–2022.

**Resolution cascade WITH département filtering (fixes the old 134 disambiguation errors):**

| Strategy | Count | What it does |
|---|---|---|
| geo.api.gouv.fr + codeDépartement | 302 | Guaranteed département match |
| BAN API name + département filter | 242 | Filtered by citycode prefix |
| Nominatim + département name | 1,325 | `"{name}, {département_name}, France"` |
| Failed | 3 | 3 communes not found by any method |
| **Total** | **1,869** | All pass département bounding-box audit |

Results cached to `data/geo/historical_commune_coords.parquet`.

### ZZ Overseas Codes (0.3%)

232 BVs for French citizens abroad at consular posts. Each ZZ code → consular city → Nominatim → city-level coordinates.

### Pipeline Steps

1. ✅ Compute REU elector centroids — average of 16M geocoded voter addresses per BV → `bv_elector_centroids.parquet` (68,830)
2. ✅ Load Etalab contour centroids → `bv_contour_centroids.parquet` (68,611)
3. ✅ REU BV polling station geocodes → `bv_coords_reu.parquet` (56,138)
4. ✅ Single-BV commune match — only 1 BV in commune → unambiguous
5. ✅ Historical commune geocoding — 1,869 with département-filtered resolution
6. ✅ ZZ consular city geocoding — 232 overseas BVs
7. ✅ Drop unresolvable — 4,041 old BV codes dropped

---

## Output Files

| File | Content | Rows |
|---|---|---|
| `data/geo/commune_coords.parquet` | `code_commune, nom, latitude, longitude` | ~35,000 |
| `data/geo/bv_coords.parquet` | `code_commune, code_bv, id_brut_miom, latitude, longitude` | 72,795 |
| `data/geo/bv_elector_centroids.parquet` | BV centroids from 16M voter addresses | 68,830 |
| `data/geo/bv_contour_centroids.parquet` | BV polygon centroids from Etalab | 68,611 |
| `data/geo/bv_coords_reu.parquet` | Geocoded BV polling station addresses | 56,138 |
| `data/geo/historical_commune_coords.parquet` | Geocoded historical/merged communes | 1,869 |
| `data/geo/zz_consular_coords.parquet` | Geocoded overseas consular cities | 232 |
| `data/geo/location_coords.parquet` | **Unified lookup**: location → (lat, lon) | ~107,937 |

### Schema: `bv_coords.parquet`

| Column | Type | Description |
|---|---|---|
| `code_commune` | str | INSEE commune code (e.g. `01001`) or ZZ code (e.g. `ZZ147`) |
| `code_bv` | str | Bureau de vote code (e.g. `0001`) |
| `id_brut_miom` | str | Composite key `{code_commune}_{code_bv}` |
| `latitude` | float32 | WGS84 latitude |
| `longitude` | float32 | WGS84 longitude |

### Schema: `location_coords.parquet`

| Column | Type | Description |
|---|---|---|
| `location` | str | Location key (BV, commune, département, region, or `"National"`) |
| `latitude` | float32 | WGS84 latitude |
| `longitude` | float32 | WGS84 longitude |

---

## Implementation: `src/build_geo_mapping.py`

Builds `location_coords.parquet` from all sources:

1. **Communes** — `geo.api.gouv.fr` bulk API → `commune_coords.parquet`
2. **Départements** — population-weighted average of commune centroids
3. **Régions** — population-weighted average of département centroids
4. **Bureaux de vote** — load `bv_coords.parquet` (72,795 entries)
5. **Cantons & Circonscriptions** — average of commune centroids from election data
6. **National** — hard-coded `(46.2276, 2.2137)`
7. **Historical communes** — resolve via geocoded historical coords, ZZ consular coords, successor mapping, or département centroid fallback

Total: **~107,937 entries** in the unified lookup.

---

## Coverage Guarantee

| Location type | Source | Coverage |
|---|---|---|
| Bureaux de vote | **Exact BV coords** (REU/contour/geocoded) | 94.7% exact, 5.3% commune centroid fallback |
| Communes | geo.api.gouv.fr API | 100% (official reference) |
| Départements | Derived from communes | 100% |
| Cantons | Derived from communes + election data | 100% of cantons in our data |
| Circonscriptions | Derived from communes + election data | 100% of circos in our data |
| Regions (poll data) | Derived from communes | 100% (names match exactly) |
| National (poll data) | Hard-coded | 100% |

**Result: Every single token in the pool has a (latitude, longitude) pair. Zero fallback to France center.**

---

## Data Sources

| Source | URL | Size | Content |
|---|---|---|---|
| **REU elector addresses** | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bureaux-de-vote-et-adresses-de-leurs-electeurs/) | 490 MB | 16M geocoded voter addresses with BV assignment |
| **Etalab BV contours** | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/proposition-de-contours-des-bureaux-de-vote/) | 644 MB | Voronoi polygon contours per BV (GeoJSON) |
| REU BV table | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/bureaux-de-vote-et-adresses-de-leurs-electeurs/) | 3.5 MB | Polling station addresses |
| Commune centroids | geo.api.gouv.fr | API | ~35K commune centroids |
| Historical commune mapping | [etalab/decoupage-administratif](https://unpkg.com/@etalab/decoupage-administratif@4.0.0/data/communes.json) | API | Old commune code → successor code |
| BAN geocoder | [api-adresse.data.gouv.fr](https://api-adresse.data.gouv.fr) | API | Municipality name → coordinates |
| Nominatim | [nominatim.openstreetmap.org](https://nominatim.openstreetmap.org) | API | Worldwide place name → coordinates |
