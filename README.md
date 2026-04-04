# French Elections Predictions: Universal Masked Set Transformer

This repository implements a deep-learning system for analyzing and predicting French political elections. Instead of feature engineering or manual aggregations, it flattens the entire universe of French election data (20+ years of municipal, legislative, regional, European, and presidential results + polls + demographics) into ~16.8M atomic tokens and trains a **Universal Masked Set Transformer** (UMST) via masked self-supervised learning.

## Architecture

The system is built on a permutation-invariant Set Transformer with a **full-pool learnable router**.

Key architectural concepts (detailed in `archi.md`):
- **Raw Data as Tokens**: Every data point becomes a structured token: `(date_float, availability_date, election_type, location, candidate, party, metric_type, value, latitude, longitude)`. `date_float` is the reference date (what the data describes); `availability_date` is the publication date (when it became usable). The router enforces temporal causality using `availability_date`.
- **Zero Feature Engineering**: No manual candidate-party alignment or aggregation. String fields are hashed into embedding buckets. The model discovers all relationships via self-attention.
- **Full-Pool Router**: A learned router scores the entire 16.8M token pool against each target query using a pre-computed key cache on GPU, selecting the top-256 most relevant context tokens for each prediction.
- **Geo-Aware**: Every token carries latitude/longitude coordinates, enabling the router and transformer to learn spatial relationships and discover geographically distant "twin" communes.
- **KL Divergence Loss**: Within each election group (same type/location/date), candidate logits are passed through softmax to produce a vote share distribution. Loss is KL divergence between predicted and true distributions, computed only on masked positions.

## Data Structure

| Directory | Content | Documentation |
|---|---|---|
| `data/elections/` | Historical results (parquet, commune-level) | `election_data.md` |
| `data/polls/` | Polling data (CSV, national/regional) | `polls_data.md` |
| `data/demographics/` | INSEE census + BPE indicators | `demographic_data.md` |
| `data/geo/` | Commune centroids and derived coordinates | `mapping.md` |

## Codebase

| File | Description |
|---|---|
| `src/model.py` | `UniversalMaskedSetTransformer`, `LearnableRouter`, `TokenEmbedding` |
| `src/dataset.py` | `TokenPool` (CPU), `PoolCache` (GPU), `TokenDataset`, `collate_token_sets` |
| `src/dataloader.py` | `build_dataloaders()`, unified pool construction, poll candidate resolution |
| `src/train.py` | Training loop with EMA, cosine annealing, entropy regularization, KL divergence loss |
| `src/eval.py` | Evaluation with per-split metric breakdown |
| `src/visualize_trajectories.py` | Trajectory prediction visualizations |
| `src/load_elections.py` | Election data ingestion → token DataFrame |
| `src/load_polls.py` | Poll data ingestion → token DataFrame |
| `src/load_demographics.py` | Demographic data ingestion → token DataFrame |
| `src/build_geo_mapping.py` | Commune lat/lon download + derived centroids |
| `src/nuance_mapping.py` | Political nuance → party equivalence mapping |

## Documentation

| Document | Content |
|---|---|
| `archi.md` | Full architecture specification (embedding, router, transformer, loss, training config) |
| `dataloading.md` | Data pipeline: token pool, PoolCache, dataset sampling, collation |
| `sampling_plan.md` | Router architecture: full-pool scoring, key cache, top-K selection, entropy regularization |
| `eval.md` | Evaluation methodology and metrics |
| `vizs.md` | Visualization concepts and use cases |
| `mapping.md` | Geo-mapping plan for lat/lon assignment |
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

- **Token pool**: ~16.8M tokens (elections + polls + demographics)
- **Model**: ~15.4M parameters
- **GPU memory**: ~4.5GB peak on NVIDIA TITAN RTX 24GB
- **Training**: ~33.7K batches per epoch (batch_size=32)
