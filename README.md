# French Elections Predictions: Universal Masked Set Transformer

This repository implements a deep-learning system for analyzing and predicting French political elections. Instead of feature engineering or manual aggregations, it flattens the entire universe of French election data (20+ years of municipal, legislative, regional, European, and presidential results + polls + demographics) into ~32.9M atomic tokens at **bureau de vote** granularity and trains a **Universal Masked Set Transformer** (UMST) via masked self-supervised learning.

## Architecture

The system is built on a permutation-invariant Set Transformer with a **full-pool learnable router**.

Key architectural concepts (detailed in `archi.md`):
- **Raw Data as Tokens**: Every data point becomes a structured token: `(date_float, availability_date, election_type, location, candidate, party, metric_type, value, latitude, longitude)`. `date_float` is the reference date (what the data describes); `availability_date` is the publication date (when it became usable). The router enforces temporal causality using `availability_date`.
- **Zero Feature Engineering**: No manual candidate-party alignment or aggregation. String fields are hashed into embedding buckets. The model discovers all relationships via self-attention.
- **Full-Pool Router**: A learned router scores the entire ~32.9M token pool against each target query using a pre-computed key cache on GPU, selecting the top-256 most relevant context tokens for each prediction.
- **Geo-Aware**: Every token carries latitude/longitude coordinates. Election tokens use exact BV positions (from REU elector centroids and Etalab contour polygons), enabling the router to learn fine-grained spatial relationships.
- **KL Divergence Loss**: Within each election group (same type/location/date), candidate logits are passed through softmax to produce a vote share distribution. Loss is KL divergence between predicted and true distributions, computed only on masked positions.

## Data Structure

| Directory | Content | Documentation |
|---|---|---|
| `data/elections/` | Historical results (parquet, bureau de vote level) | `election_data.md` |
| `data/polls/` | Polling data (CSV, national/regional) | `polls_data.md` |
| `data/demographics/` | INSEE Census indicators (multi-vintage) | `demographic_data.md` |
| `data/geo/` | BV coords + commune centroids + derived coordinates | `mapping.md` |

## Codebase

| File | Description |
|---|---|
| `src/model.py` | `UniversalMaskedSetTransformer`, `LearnableRouter`, `TokenEmbedding` |
| `src/dataset.py` | `TokenPool` (CPU), `PoolCache` (GPU), `TokenDataset`, `collate_token_sets` |
| `src/dataloader.py` | `build_dataloaders()`, unified pool construction, poll candidate resolution |
| `src/train.py` | Training loop with EMA, cosine annealing, entropy regularization, KL divergence loss |
| `src/eval.py` | Evaluation with per-split metric breakdown |
| `src/visualize_trajectories.py` | Trajectory prediction visualizations |
| `src/load_elections.py` | Election data ingestion at BV level → token DataFrame |
| `src/load_polls.py` | Poll data ingestion → token DataFrame |
| `src/load_demographics.py` | Demographic data ingestion → token DataFrame |
| `src/build_geo_mapping.py` | BV + commune lat/lon download + derived centroids |
| `src/geocode_bv.py` | BV-level geocoding (REU, contours, historical) |
| `src/nuance_mapping.py` | Political nuance → party equivalence mapping |

## Documentation

| Document | Content |
|---|---|
| `archi.md` | Full architecture specification (embedding, router, transformer, loss, training config) |
| `dataloading.md` | Data pipeline: token pool, PoolCache, dataset sampling, collation |
| `sampling_plan.md` | Router architecture: full-pool scoring, key cache, top-K selection, entropy regularization |
| `eval.md` | Evaluation methodology and metrics |
| `vizs.md` | Visualization concepts and use cases |
| `mapping.md` | Geo-mapping: BV geolocation methodology, all location types, coverage |
| `election_data.md` | Election data sources and coverage |
| `polls_data.md` | Polling data sources and coverage |
| `demographic_data.md` | Demographic data sources and priority |

## Training

```bash
# Start training
cd /home/veesion/elections_predictions
nohup python3 -m src.train >> training.log 2>&1 &

# Monitor with TensorBoard
tensorboard --logdir runs --bind_all
```

## Scale

- **Token pool**: ~32.9M tokens (BV-level elections + polls + demographics)
- **Unique locations**: ~76,571 BVs + ~36.7K communes/regions/national
- **Model**: ~1.5M parameters (`d_model=48, nhead=4, num_layers=4, d_router=16`)
- **GPU memory**: ~18GB peak on NVIDIA TITAN RTX 24GB
- **Training**: ~50.9K batches per epoch (batch_size=32)
