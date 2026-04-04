# Universal Token DataLoader Pipeline

## Overview

The dataloading pipeline builds a single **unified token pool** (~16.8M tokens) from all data sources (elections, polls, demographics) and provides it to the model as a GPU-resident `PoolCache`. Context selection is no longer performed by the DataLoader — it is handled entirely by the model's **full-pool router** on the GPU.

The `TokenDataset` only decides:
1. Which election group to predict (the target)
2. Which tokens in that group are masked vs revealed
3. Whether to include other-location results from the same election

## 1. Raw Data Normalization & Ingestion

The system unifies three major streams of raw data under an identical schema during startup, creating an in-memory `TokenPool` (~16.8M tokens).

### 1.1 Elections Data (`src/load_elections.py`)
- **Source**: Parquet files (`candidats_results.parquet` and `general_results.parquet`).
- **Granularity**: Raw data at `bureau-de-vote` level (~30M records), aggregated to **commune-level**. Every result is expressed as a percentage (0–100), never absolute counts, to ensure geographic size invariance.
- **Token Instantiation**:
  - `Result` tokens: candidate scores (candidate name → `candidate`, party → `party`, `metric_type="Result"`)
  - `Context` tokens: environmental stats (e.g., `Abstention`, `Blancs`, `metric_type="Result"`)
- **Geo-coordinates**: Latitude/longitude attached via the geo mapping table (see `mapping.md`).

### 1.2 Polls Data (`src/load_polls.py`)
- **Source**: Various raw `.csv` files in `data/polls/`.
- **Parsing**: Ingests all available CSV datasets across all elections (Presidential, Legislative, Regional, European, etc.). Dirty strings (`[1]`, `<`, `%`, empty cells) are dynamically cleaned. All historical data is preserved, including low-quality institutes, because self-attention can implicitly weigh token reliability.
- **Token Instantiation**: `metric_type=Poll_{InstituteName}`, `location="National"` (or region name for regional polls).
- **Candidate Resolution**: For municipales, poll party codes (e.g., `LFI`, `LREM`) are mapped to actual candidate names via `NUANCE_EQUIVALENCES` in `src/nuance_mapping.py`, expanding coalition-level nuances to individual party matches.

### 1.3 Demographics Data (`src/load_demographics.py`)
- **Source**: INSEE Census (Activité, Diplômes, Population) and BPE (Base Permanente des Équipements) commune-level data in `data/demographics/`.
- **Vintage**: Only the **latest vintage** of each source is loaded: Census 2021, BPE 2024.
- **Token Instantiation**: `metric_type="Demographics"`, indicator name (e.g., `Taux_Chomage`, `BPE_Medecins_per_1k`) stored in `candidate` field. `election_type` and `party` fields empty.
- **Normalization**: Census ratios scaled to [0, 100]. BPE counts normalised per 1,000 inhabitants using Census population.
- **Publication-date causality**: Each token carries two dates:
  - `date_float`: The date the data **describes** (e.g., 2019.5 for Census 2021 which pools 2017–2021 surveys).
  - `availability_date`: The date the data was **published** and thus usable (e.g., 2024.5 for Census 2021 published June 2024; 2025.5 for BPE 2024 published July 2025).
  The router uses `availability_date` for temporal filtering — a token is invisible to the model when predicting elections that happen before its publication date. The model's embedding uses `date_float` as its temporal coordinate.
- **Census indicators**: `Taux_Chomage`, `Pct_Ouvriers`, `Pct_Cadres`, `Pct_Sans_Diplome`, `Pct_Bac_Plus_5`, `Pct_Age_18_24`, `Pct_Age_60_Plus`, `Pct_Immigres`
- **BPE indicators**: `BPE_Medecins_per_1k`, `BPE_Pharmacies_per_1k`, `BPE_Postes_per_1k`, `BPE_Supermarches_per_1k`

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
- `key_cache`: `(N, 64)` float16 tensor of pre-computed L2-normalized key projections, rebuilt every 100 training steps.
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
| Train | ~1.08M | All non-2026-muni groups, minus dev |
| Dev | ~57K | 5% random sample of train groups (fixed seed=42) |
| Val | ~7.1K | 2026 Municipales result groups only |

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
├── load_election_tokens()       → election DataFrame (~14.5M rows, availability_date = date_float)
├── load_poll_tokens()           → poll DataFrame (~1.8M rows, availability_date = date_float)
├── _resolve_poll_candidates()   → match poll parties to election candidates
├── load_demographic_tokens()    → demo DataFrame (~420K rows, availability_date = publication date)
├── concat + ensure availability_date + sort by availability_date → combined DataFrame (~16.8M+ rows)
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
