# Geo-Mapping Plan: Lat/Long for Every Location

## Goal

Attribute a `(latitude, longitude)` coordinate pair to **every single row** in our token data — both election results and polls — regardless of the geographic granularity of the `location` field.

---

## Current State of Locations

### Election Results (`candidats_results.parquet` + `general_results.parquet`)

The raw data contains **5 geographic columns**:

| Column | Unique values | Example | Currently used? |
|---|---|---|---|
| `code_commune` | **37,053** | `01160`, `75056` | ✅ Used as `location` in tokens |
| `code_departement` | **117** | `01`, `75`, `2A` | ❌ Not used |
| `code_canton` | **79** | `08`, `01` | ❌ Not used |
| `code_circonscription` | **21** | `04`, `05` | ❌ Not used |
| `code_bv` (bureau de vote) | **2,226** | `0001`, `0002` | ❌ Not used |

Right now, `load_elections.py` aggregates everything to **commune level** and uses `code_commune` as the `location` field → **37,053 unique locations**.

### Polls (`load_polls.py`)

| Source | Location field | Values |
|---|---|---|
| Wiki polls (presidentielle, euro, legi, etc.) | `location` | `"National"` |
| NSPPolls presidentielle | `location` | `"National"` |
| NSPPolls regionales | `location` | Region names: `"Île-de-France"`, `"Occitanie"`, `"Bretagne"`, etc. (13 unique) |

---

## All Location Types to Map

We need lat/long for every distinct geographic entity ever seen in the data:

### 1. Communes (~37,000)
- **Key**: `code_commune` (INSEE code, e.g. `"75056"`)
- **Source**: [geo.api.gouv.fr/communes](https://geo.api.gouv.fr/communes?fields=code,nom,centre&format=json) — returns `centre` as `{type: "Point", coordinates: [lon, lat]}`
- **Method**: Single bulk API call: `GET /communes?fields=code,centre&format=json` returns all ~35,000 communes at once
- **Fallback**: Download CSV from [data.gouv.fr "Données sur les communes"](https://www.data.gouv.fr/fr/datasets/donnees-sur-les-communes-de-france-metropolitaine/)

### 2. Départements (~100)
- **Key**: `code_departement` (e.g. `"75"`, `"2A"`)
- **Source**: The geo API does NOT return centroids for départements. We'll **compute** the centroid as the **population-weighted average** of all commune centroids within each département. This is better than a geometric centroid because it reflects the actual center of electoral weight.
- **Fallback**: Hard-code a small lookup table (~100 entries) with well-known centroids from Wikipedia/IGN.

### 3. Régions (~18)
- **Key**: Region name string (e.g. `"Île-de-France"`, `"Provence-Alpes-Côte d'Azur"`)
- **Source**: Same as départements — **population-weighted average** of commune centroids within each region. We can map commune→département→region using the `code_departement` prefix of `code_commune`, or use the API: `GET /regions?fields=code,nom` + `GET /departements?codeRegion=XX`.
- **Name matching**: The region names in the poll data (`"Île-de-France"`, `"Bretagne"`, etc.) match the API names exactly ✅

### 4. National
- **Key**: `"National"` string
- **Source**: Hard-coded centroid of France: **(46.2276, 2.2137)** (standard geographic center) or population-weighted center (~Paris area: **48.86, 2.35**). We should use the geographic center to avoid bias.

### 5. Cantons (~2,000+)
- **Key**: `code_canton` (e.g. `"08"`)
- **Note**: Canton codes are **relative to their département** (not globally unique). A canton `"08"` in département `"01"` is different from canton `"08"` in département `"75"`.
- **Source**: Compute centroid as average of commune centroids within each (département, canton) pair.

### 6. Circonscriptions (~577)
- **Key**: `code_circonscription` (e.g. `"04"`)
- **Note**: Like cantons, these are **relative to their département**.
- **Source**: Compute centroid as average of commune centroids within the (département, circonscription) pair.

### 7. Bureaux de vote (~70,000+)
- **Key**: `code_bv` (e.g. `"0001"`)
- **Note**: These are **relative to their commune**. Bureau de vote `"0001"` in commune `"75056"` is different from `"0001"` in commune `"13055"`.
- **Source**: Assign the **same coordinates as the parent commune**. Bureaux de vote are sub-commune divisions within the same town hall area — the commune centroid is precise enough for our purposes.
- **Alternative**: The data.gouv.fr dataset ["Bureaux de vote et contours"](https://www.data.gouv.fr/fr/datasets/bureaux-de-vote-et-contours-des-bureaux-de-vote/) provides exact BV contours as GeoJSON, but this is overkill for our model.

---

## Implementation Plan

### Step 1: Download commune centroids
```python
# Script: src/build_geo_mapping.py
# Fetch all ~35,000 commune centroids from the API
GET https://geo.api.gouv.fr/communes?fields=code,nom,centre&format=json
# Parse into: { code_commune: (lat, lon) }
# Save as: data/geo/commune_coords.parquet
```

### Step 2: Build derived centroids
From the commune centroids + the raw election data (which has `code_departement`, `code_canton`, `code_circonscription`):

```python
# For each département: weighted average of commune centroids (weight = inscrits)
# For each (département, canton): weighted average of commune centroids
# For each (département, circonscription): weighted average of commune centroids
# For each region: weighted average of département centroids
# National: (46.2276, 2.2137)
```

Save all into a single lookup file: `data/geo/location_coords.parquet`

### Step 3: Integrate into `load_elections.py`
Add `latitude` and `longitude` columns to the token DataFrame:

```python
def load_election_tokens(data_dir: Path) -> pd.DataFrame:
    # ... existing code ...
    # After building combined DataFrame:
    coords = pd.read_parquet(data_dir / "geo" / "location_coords.parquet")
    combined = combined.merge(coords, on="location", how="left")
    return combined
```

### Step 4: Integrate into `load_polls.py`
Map poll locations (region names, "National") to coordinates:

```python
# Region name → (lat, lon) from the region centroids computed in Step 2
# "National" → (46.2276, 2.2137)
```

### Step 5: Add to model features
Update `dataset.py` to include lat/lon as continuous features alongside `dates`:

```python
# In TokenPool.__init__:
self.latitude = df_sorted["latitude"].values.astype(np.float32)
self.longitude = df_sorted["longitude"].values.astype(np.float32)

# In TokenDataset.__getitem__:
token_dict["latitude"] = self.pool.latitude[sampled_idx]
token_dict["longitude"] = self.pool.longitude[sampled_idx]
```

---

## Output Files

| File | Content | Rows |
|---|---|---|
| `data/geo/commune_coords.parquet` | `code_commune, nom, latitude, longitude` | ~35,000 |
| `data/geo/location_coords.parquet` | `location, latitude, longitude` | ~37,100 (communes + depts + regions + national) |

---

## Coverage Guarantee

| Location type | Source | Coverage |
|---|---|---|
| Communes | geo.api.gouv.fr API | 100% (official reference) |
| Départements | Derived from communes | 100% |
| Cantons | Derived from communes + election data | 100% of cantons in our data |
| Circonscriptions | Derived from communes + election data | 100% of circos in our data |
| Bureaux de vote | Parent commune centroid | 100% |
| Regions (poll data) | Derived from communes | 100% (names match exactly) |
| National (poll data) | Hard-coded | 100% |

**Result: Every single row in the token data will have a (latitude, longitude) pair.**
