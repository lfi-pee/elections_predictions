# 🗳️ French Election Predictions

**Bureau de vote–level prediction of French election results using demographic data, historical voting patterns, and national polls.**

Predicts four block vote shares (0–100%) for every bureau de vote (BV) in France:

| Block | Description | Best R² |
|---|---|---|
| **Gauche** | Left-wing (SOC, COM, FI, NUP, ECO, VEC, EXG, DVG…) | **0.74** |
| **Centre+Droite** | Center and right (UMP, LR, REM, ENS, DVD, UDI…) | **0.61** |
| **Extrême Droite** | Far-right (FN, RN, REC, EXD…) | **0.80** |
| **Abstention** | Abstention rate | **0.74** |

Validated on the 2024 Législatives T1 (~69K BVs). No validation tuning of any kind — all model selection is done via Leave-One-Election-Out (LOO) on training data, with a single forward pass on the validation set.

---

## Architecture: Two-Stage Deviation Model

### Stage 1 — National Mean Estimation

Estimate the national average block score for the target election:

- **Vote blocks** (Gauche, Centre+Droite, Extrême Droite): raw poll averages from a 1-year window before the election.
- **Abstention**: a gap-based turnout model — linear regression from inter-election time gap to national abstention, fit on 9 training elections. Predicts 33.0% for 2024 (actual 31.0%, error 2.0pp).

### Stage 2 — BV Deviation Prediction

Predict how much each BV deviates from the national mean:

```
target_dev  = BV_block_score − national_mean(election_type, date)
prediction  = national_mean_estimate + Ridge(deviation_features)
```

Deviations are stable across elections and election types — a BV that's +10pp Gauche relative to national stays roughly +10pp regardless of year or election type. This enables cross-type training (Législatives + Présidentielles + more).

### Feature Vector (~60 dimensions)

| Feature Group | Dims | Notes |
|---|---|---|
| Demographics (census) | 52 | Last available vintage via `merge_asof`. Strict NaN drop (V1). |
| Deviation lags 1–2 | 8 | `BV_lag − national_mean(lag_election)`. Cross-type: most recent prior election. |
| Election type one-hot | 6 | For cross-type models only. |

### Model

StandardScaler → optional PCA (3–10 components on demographics) → Ridge Regression (RidgeCV with LOO, α ∈ [10⁻², 10⁶]).

---

## Repository Structure

```
├── README.md                   # This file
├── algorithm.md                # Full algorithm design, results, and lessons learned
├── baseline.md                 # Ridge baseline evaluation (V1/V2 comparison)
├── election_data.md            # Election data dictionary (56 elections, 1999–2026)
├── demographic_data.md         # Demographic data dictionary (census, état civil, Sirene)
├── polls_data.md               # Polls data dictionary (presidential, legislative, euro…)
├── mapping.md                  # Geo-mapping: lat/lon for every BV, commune, département
│
├── download_elections.py       # Download raw election files from data.gouv.fr
│
├── src/
│   ├── load_elections.py       # Load & normalize 56 elections into unified DataFrame
│   ├── load_demographics.py    # Load 17 census vintages → commune-level indicators
│   ├── load_polls.py           # Load & normalize polls across all election types
│   ├── nuance_mapping.py       # Map party nuances → {Gauche, Centre+Droite, Extr. Droite}
│   ├── download_demographics.py # Download census data from INSEE
│   ├── geocode_bv.py           # Geocode ~73K bureaux de vote (REU, BAN, Nominatim)
│   ├── build_geo_mapping.py    # Build unified location → (lat, lon) lookup
│   │
│   ├── pca_ridge_baseline.py   # V1/V2 Ridge baseline (demographics + lags)
│   ├── cross_type_ridge.py     # Cross-type Ridge with block mapping
│   ├── cross_type_dev.py       # Core deviation model (Legi+Pres cross-type)
│   ├── beat_it.py              # Extended cross-type (6 election types, experiments)
│   ├── preregistered.py        # Pre-registered model selection (LOO → single val pass)
│   ├── conformal.py            # Conformal prediction intervals (80/90/95% coverage)
│   │
│   ├── bayesian_polls.py       # Bayesian poll aggregation
│   ├── residual_boost.py       # Ridge + XGB residual boosting
│   ├── gp_residual_boost.py    # Gaussian Process residual boosting
│   └── shap_waterfall.py       # SHAP feature importance analysis
│
├── data/                       # Data directory (gitignored)
│   ├── elections/              # Raw & aggregated election results
│   │   └── agregees/           # BV-level parquet (68MB + 154MB)
│   ├── demographics/           # Census data (17 vintages)
│   ├── polls/                  # Polls by election type and year
│   ├── geo/                    # Geocoding outputs (BV coords, commune coords)
│   └── baseline_cache/         # Cached processed DataFrames
│
├── plots/                      # Generated plots (SHAP, etc.)
└── tests/                      # Test suite
```

---

## Data

### Elections

**56 elections** (1999–2026) at bureau de vote level from [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/donnees-des-elections-agregees/), covering Présidentielles, Législatives, Européennes, Régionales, Départementales, Cantonales, and Municipales. ~3.16M rows across `general_results.parquet` (participation) and `candidats_results.parquet` (per-candidate votes).

See [`election_data.md`](election_data.md) for the complete data dictionary.

### Demographics

**52 indicators** from the INSEE Census (2009–2022, 17 vintages) at commune level:
- Socioprofessional categories (ouvriers, cadres, employés…)
- Education levels (sans diplôme through bac+5)
- Employment by sector (agriculture, industrie, construction, tertiaire)
- Age brackets, immigration rate, unemployment
- Housing (propriétaires, HLM, logements vacants…)
- Household structure and marital status

See [`demographic_data.md`](demographic_data.md) for the complete data dictionary.

### Polls

Voting intention polls scraped from Wikipedia and [nsppolls](https://github.com/nsppolls/nsppolls) for all election types (2002–2027). Used for Stage 1 national mean estimation.

See [`polls_data.md`](polls_data.md) for the complete data dictionary.

### Geocoding

**72,795 BVs** geocoded with verified coordinates (94.7% from exact sources: REU elector address centroids, BV polling station addresses, Etalab contour centroids). Zero fallback to France center.

See [`mapping.md`](mapping.md) for full geocoding methodology.

---

## Quick Start

### 1. Download Data

```bash
# Download election result files
python download_elections.py

# Download census data from INSEE
python -m src.download_demographics
```

> **Note:** First-time data loading takes ~30 min and caches processed DataFrames to `data/baseline_cache/`. Subsequent runs load in seconds.

### 2. Run the Baseline

```bash
# Ridge baseline with demographics + lagged block scores
python -m src.pca_ridge_baseline
```

### 3. Run Pre-registered Model Selection

```bash
# LOO model selection on training → single forward pass on validation
python3 -u -m src.preregistered
```

### 4. Generate Prediction Intervals

```bash
# Conformal prediction intervals (80%, 90%, 95% coverage)
python3 -u -m src.conformal
```

This outputs `data/predictions_with_intervals.csv` — per-BV predictions with adaptive conformal intervals for all four blocks.

---

## Key Results

Models selected by best Leave-One-Election-Out R² on training data. No validation feedback enters the selection or fitting pipeline.

| Block | R² | Model | Training Config |
|---|---|---|---|
| Gauche | **0.74** | PCA5-devlag | Legi-only, V1, 4 train dates |
| Centre+Droite | **0.61** | PCA7-devlag | Legi-only, V1, 4 train dates |
| Extrême Droite | **0.80** | PCA5-devlag | Cross-type (Legi+Pres), V1, 8 train dates |
| Abstention | **0.74** | PCA10-devlag | Cross-type (Legi+Pres), V1, 8 train dates |

### Conformal Prediction Intervals

Distribution-free intervals with finite-sample coverage guarantees:

| Block | 90% Coverage | 90% Width | 95% Coverage | 95% Width |
|---|---|---|---|---|
| Gauche | ≥ 90% | ~14pp | ≥ 95% | ~18pp |
| Centre+Droite | ≥ 90% | ~16pp | ≥ 95% | ~20pp |
| Extrême Droite | ≥ 90% | ~12pp | ≥ 95% | ~16pp |
| Abstention | ≥ 90% | ~13pp | ≥ 95% | ~16pp |

---

## Key Design Decisions

1. **Deviation targets + deviation lags.** The single biggest innovation — removes national-level bias from both target and lags, making Ridge learn stable local deviations.

2. **No validation tuning.** All model selection uses LOO on training elections. The validation set is touched exactly once per model in a single forward pass. See the [strict rule](algorithm.md#absolute-rule-never-tune-on-validation-data).

3. **Cross-type training in deviation space.** Training on Legi + Pres T1 doubles the training dates (4 → 8). Best for Extrême Droite. Extended 6-type (20 dates) is best for Abstention.

4. **Gap-based turnout model for Abstention.** No direct poll exists for abstention. A linear regression from inter-election gap to national abstention, fit on 9 training elections, correctly interpolates the unprecedented 2024 snap election.

5. **PCA on demographics.** Compresses 52 correlated census indicators to 3–10 orthogonal components. Prevents Ridge overfitting with limited training elections.

See [`algorithm.md`](algorithm.md) for the complete algorithm design, feature innovation log, and exhaustive list of what doesn't work.

---

## Documentation

| Document | Description |
|---|---|
| [`algorithm.md`](algorithm.md) | Full algorithm design, results table, feature innovations, and failures log |
| [`baseline.md`](baseline.md) | Ridge baseline evaluation (V1 vs V2, 29 vs 47 indicators) |
| [`election_data.md`](election_data.md) | Election data dictionary — 56 elections, sources, schemas |
| [`demographic_data.md`](demographic_data.md) | Census data dictionary — indicators, vintages, caveats |
| [`polls_data.md`](polls_data.md) | Polls data dictionary — all election types and sources |
| [`mapping.md`](mapping.md) | Geocoding methodology — BV coordinates pipeline |

---

## License

Data sourced from [data.gouv.fr](https://www.data.gouv.fr) (Licence Ouverte / Open License), [INSEE](https://www.insee.fr), and [nsppolls](https://github.com/nsppolls/nsppolls).
