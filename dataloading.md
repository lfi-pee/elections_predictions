# Universal Token DataLoader Pipeline

## Overview

The dataloading pipeline builds a single **unified token pool** (~32.9M tokens) from all data sources (elections, polls, demographics) and provides it to the model as a GPU-resident `PoolCache`. Context selection is no longer performed by the DataLoader — it is handled entirely by the model's **full-pool router** on the GPU.

The `TokenDataset` only decides:
1. Which election group to predict (the target)
2. Which tokens in that group are masked vs revealed
3. Whether to include other-location results from the same election

## 1. Raw Data Normalization & Ingestion

The system unifies three major streams of raw data under an identical schema during startup, creating an in-memory `TokenPool` (~32.9M tokens).

### 1.1 Elections Data (`src/load_elections.py`)
- **Source**: Parquet files (`candidats_results.parquet` and `general_results.parquet`).
- **Granularity**: **Bureau de vote (BV) level** — no commune aggregation. Each BV × election × candidate produces one result token. Location keys are `"{code_commune}_{code_bv}"` (e.g. `"29019_0012"` = BV 12 in Brest). ~76,571 unique BV locations. Every result is expressed as a percentage (0–100), never absolute counts.
- **Token Instantiation**:
  - `Result` tokens: candidate scores (candidate name → `candidate`, party → `party`, `metric_type="Result"`)
  - `Context` tokens: environmental stats (e.g., `Abstention`, `Blancs`, `metric_type="Context"`)
- **Geo-coordinates**: BV-level lat/lon from `bv_coords.parquet` (exact REU elector centroids / contour polygons). Fallback to commune centroid from `location_coords.parquet` for unmatched BVs.

### 1.2 Polls Data (`src/load_polls.py`)
- **Source**: Various raw `.csv` files in `data/polls/`.
- **Parsing**: Ingests all available CSV datasets across all elections (Presidential, Legislative, Regional, European, etc.). Dirty strings (`[1]`, `<`, `%`, empty cells) are dynamically cleaned. All historical data is preserved, including low-quality institutes, because self-attention can implicitly weigh token reliability.
- **Token Instantiation**: `metric_type=Poll_{InstituteName}`, `location="National"` (or region name for regional polls).
- **Candidate Resolution**: For municipales, poll party codes (e.g., `LFI`, `LREM`) are mapped to actual candidate names via `NUANCE_EQUIVALENCES` in `src/nuance_mapping.py`, expanding coalition-level nuances to individual party matches.

### 1.3 Demographics Data (`src/load_demographics.py`)
- **Source**: INSEE Census (Activité, Diplômes, Population, Logement, Familles-Ménages) commune-level data in `data/demographics/census/{vintage}/`.
- **Vintages**: All available vintages are loaded automatically: **Census 2006–2022** (17 vintage directories — 2006–2008 have incompatible schemas for some themes). ~8M+ tokens total.
- **Token Instantiation**: `metric_type="Demographics"`, indicator name (e.g., `Taux_Chomage`, `Pct_Sans_Diplome`, `Pct_HLM`) stored in `candidate` field. `election_type` and `party` fields empty.
- **Normalization**: Census ratios scaled to [0, 100]. Column name differences across vintages (e.g., `P10_POP1529` vs `P21_POP1524`, `NSCOL15P_DIPL0` vs `NSCOL15P_DIPLMIN`) are handled by auto-detection with fallback chains.
- **Publication-date causality**: Each vintage token carries two dates:
  - `date_float`: The date the data **describes** — Census vintage Y → Y−1.5 (centre of 5-year survey window).
  - `availability_date`: The date the data was **published** — Census Y → Y+3.5 (~June Y+3).
  The router uses `availability_date` for temporal filtering — a token is invisible to the model when predicting elections that happen before its publication date. The model's embedding uses `date_float` as its temporal coordinate.
- **Indicators** (up to 30 per vintage): ACT theme (12): `Taux_Chomage`, `Pct_Ouvriers`, `Pct_Cadres`, `Pct_Employes`, `Pct_Prof_Intermediaires`, `Pct_Agriculteurs`, `Pct_Artisans`, `Pct_Emploi_Agriculture/Industrie/Construction/Tertiaire`, `Pct_Retraites`. FOR theme (6): `Pct_Sans_Diplome`, `Pct_CAP_BEP`, `Pct_Bac`, `Pct_Bac_Plus_2/3_4/5`. POP theme (6): `Pct_Age_0_14/18_24/30_44/45_59/60_Plus`, `Pct_Immigres`. LOG theme (4): `Pct_Proprietaires`, `Pct_HLM`, `Pct_Locataires`, `Pct_Logements_Vacants`. FAM theme (2): `Pct_Menages_Seuls`, `Pct_Familles_Monoparentales`. Actual count varies by vintage due to column availability.
- **Scale**: ~2.17M tokens across 13 census vintages, ~36,700 communes, spanning 2007.5–2020.5 (date_float) / 2012.5–2025.5 (availability)

## 2. TokenPool and PoolCache

### TokenPool (CPU, `src/dataset.py`)
Holds the full token pool sorted by **`availability_date`** (for causality enforcement):
- All string fields (election_type, location, candidate, party, metric_type) are hashed to integer indices via MD5 → `num_buckets` (50,000).
- `dates` (float32) = `availability_date` — when data was published. Pool is sorted on this.
- `reference_dates` (float32) = `date_float` — what the data describes. Used as model input.
- `value` (float32), `latitude` (float32), `longitude` (float32).
- Precomputed `election_groups`: list of `(anchor_date, result_indices_array)` tuples. `anchor_date` uses the **reference date** (actual election date, not availability).

### PoolCache (GPU, `src/dataset.py`)
Mirrors all TokenPool data as contiguous GPU tensors for fast indexed access:
- `dates` (availability) and `reference_dates` (model input) — both on GPU.
- `election_type`, `location`, `candidate`, `party`, `metric_type`, `values`, `latitude`, `longitude`.
- `val_only_mask`: boolean tensor marking 2026 municipal result tokens for suppression during training routing.
- `key_cache`: `(N, 16)` float16 tensor of pre-computed L2-normalized key projections, rebuilt every 100 training steps.
- `gather_tokens(indices)`: returns a dict with `"dates"` pointing to **reference_dates**, so the model sees the true temporal coordinate.

## 3. TokenDataset and Sampling

The `TokenDataset` operates at the **election group** level. Each `__getitem__` call returns:
```python
{
    "anchor_date": float32,             # absolute date of the target election
    "masked_pool_indices": int64[],     # pool indices of masked candidate tokens
    "unmasked_pool_indices": int64[],   # pool indices of revealed tokens (if any)
}
```

### Masking Strategy
- **75% of samples**: Single-location only (predict one commune's results)
  - Of these, **75%**: all candidates fully masked (blind prediction)
  - Of these, **25%**: 1–2 candidates revealed (conditional scoring)
- **25% of samples**: Multi-location context — includes up to 5 result tokens from other locations at the same election date (cross-location signal)

### Data Splits
All splits share the **same unified pool** — no separate train/val pools. The splits differ only in which election groups they iterate over:

| Split | Groups | Description |
|---|---|---|
| Train | ~1.63M | All non-2026-muni BV-level groups, minus dev |
| Dev | ~86K | 5% random sample of train groups (fixed seed=42) |
| Val | ~35K | 2026 Municipales BV-level result groups only |

## 4. Collation (`collate_token_sets`)

The collate function produces padded batch tensors:

```python
anchor_dates:    (B,)        # float32, absolute target dates
target_indices:  (B, T_max)  # long, pool indices of all target tokens
target_masked:   (B, T_max)  # bool, True for masked positions
target_padding:  (B, T_max)  # bool, True for padding positions
```

Target tokens = masked candidates + unmasked candidates (concatenated). Padding fills shorter samples to `T_max` (the maximum target count in the batch).

## 5. Data Flow Summary

```
build_dataloaders()
├── load_election_tokens()       → election DataFrame (~32.5M rows, BV-level, availability_date = date_float)
├── load_poll_tokens()           → poll DataFrame (~1.8M rows, availability_date = date_float)
├── _resolve_poll_candidates()   → match poll parties to election candidates (extracts commune from BV locations)
├── load_demographic_tokens()    → demo DataFrame (~8M+ rows, availability_date = publication date)
├── concat + ensure availability_date + sort by availability_date → combined DataFrame (~32.9M+ rows)
├── TokenPool(combined)          → CPU pool sorted by availability_date, with reference_dates
├── PoolCache(pool, GPU)         → GPU-resident cache: dates (avail), reference_dates (model), val_only_mask
├── Split election_groups        → train / dev / val
└── Return: train_dl, dev_dl, val_dl, pool, pool_cache

TokenDataset.__getitem__()
├── Pick election group → (anchor_date [reference], result_indices)
├── Apply masking strategy → masked_indices, unmasked_indices
└── Return dict of indices

Forward Pass (model.py)
├── Gather target tokens from PoolCache (gets reference_dates as "dates")
├── Compute anchor query from masked targets
├── Score full pool via key cache matmul
├── Mask future tokens: searchsorted on availability_date (not reference_date)
├── Top-K select context tokens
├── Combine targets + context, convert reference_dates to relative dates
├── Embed (identity + value w/ masking) → LayerNorm
├── STE re-scoring for gradient flow
├── Transformer encoder
└── Value head → predictions
```
