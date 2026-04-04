# Election Prediction: Universal Masked Set Transformer

## The Core Concept

We abandon all feature engineering, aggregation, and manual alignment between datasets. Instead of creating structured feature vectors, we flatten the entire universe of French election data (results, polls, and demographics) into atomic, raw data points (~16.8M tokens).

The network takes an **arbitrary subset of the entire data universe** as a Set of Tokens and is tasked with predicting the remaining, masked subset. A **learnable full-pool router** scores the entire 16.8M token pool against each target query and selects the top-K most relevant context tokens automatically.

## 1. Raw Data as Tokens (Zero Feature Engineering)

Every single row from our raw datasets (whether it's an election result in a specific village, a national poll, or a demographic statistic) is converted into a universal atomic token.

There is no summing, no averaging, and no manual mapping of candidates to parties. We simply encode the text.

A token consists of five parts:
1. **Time (dual dates)**:
   - `date_float` (reference date): The date the data **describes**, represented as a continuous float (fractional years, e.g. `2022.33`). Stored as absolute values in the pool and converted to **relative dates** (offset from the anchor/target date) before entering the transformer.
   - `availability_date` (publication date): When the data was actually **published** and became usable. The pool is sorted by this date, and the router's `searchsorted` uses it to enforce strict temporal causality — a token is invisible to the model when predicting elections that happen before its publication. For election/poll data: `availability_date == date_float`. For demographics: `availability_date` is typically 1–3 years after `date_float`.
2. **Textual Context**: Who, what party, what election, where, and what metric type (passed as raw strings, hashed into embedding buckets). Candidates receive a dual encoding combining their specific identity and their party affiliation.
3. **Geo-Coordinates**: Latitude and longitude of the token's location, normalized around France's center `(46.5, 2.5)` and projected via a learned linear layer. This enables the router and transformer to learn spatial relationships.
4. **Continuous Value**: The actual number (e.g., vote percentage, turnout %, poll intent %, demographic ratio).

For example, raw rows from different datasets become tokens:
- `[22.27, "Presidentielle_T1", "Brest", "MACRON Emmanuel", "LREM", "Result", 27.8, 48.39, -4.49]` (availability_date = 22.27)
- `[24.43, "Europeennes", "National", "BARDELLA Jordan", "RN", "Poll_Ifop", 31.5, 46.23, 2.21]` (availability_date = 24.43)
- `[19.50, "", "Brest", "Taux_Chomage", "", "Demographics", 6.8, 48.39, -4.49]` (availability_date = 24.50, Census 2021 published June 2024)
- `[24.00, "", "Brest", "BPE_Medecins_per_1k", "", "Demographics", 1.2, 48.39, -4.49]` (availability_date = 25.50, BPE 2024 published July 2025)

### Token Embedding

The identity part of the embedding (96 dimensions) combines several learned projections:

```python
token_identity = Linear(relative_date)          # (1 → 96)
               + StringEmbedding(election_type)  # (96)
               + StringEmbedding(candidate)      # (3, zero-padded to 96)
               + StringEmbedding(party)          # (96)
               + StringEmbedding(metric_type)    # (96)
               + Linear(lat_norm, lon_norm)       # (2 → 96)
```

The candidate embedding is intentionally small (3 dims, zero-initialized) to prevent the model from memorizing candidate identities too strongly — it should learn party and structural dynamics instead.

The value part (32 dimensions) is either:
- `Linear(value)` for unmasked tokens
- A learned `[MASK]` parameter for masked tokens

The final token embedding is `Concat(identity, value)` → `LayerNorm` → `(B, K, 128)`.

## 2. Full-Pool Router

The model uses a **learnable router** that scores the entire 16.8M token pool to find the most relevant context for each prediction target. This replaces the earlier random-sampling approach.

### How It Works

1. **Pre-computed Key Cache** (`PoolCache`): All 16.8M token identity embeddings are projected through a learned `key_proj` layer (96 → 64), L2-normalized, and stored as float16 on GPU (~2.2GB). This cache is rebuilt every 100 training steps to reflect updated weights.

2. **Anchor Query**: For each batch sample, the masked target tokens' identity embeddings are pooled and projected through `query_proj` (96 → 64), L2-normalized into a single query vector.

3. **Full-Pool Scoring**: Brute-force matmul `(B, 64) × (64, 16.8M) → (B, 16.8M)` scores. Future tokens (published after the anchor date) are masked via `searchsorted` on the `availability_date`-sorted pool. Val-only tokens (2026 municipal results) are suppressed during training.

4. **Top-K Selection**: The top `K=256` scoring tokens are selected as context (minus the number of target tokens, which are force-included).

5. **Gradient Flow via STE**: After selection, the selected tokens are **re-scored from scratch** using current (not cached) weights. This Straight-Through Estimator trick ensures gradients flow through `query_proj`, `key_proj`, and `temperature` despite the discrete top-K selection.

### Router Warm-up

For the first 500 training steps, the router is bypassed and context tokens are selected randomly. This lets the transformer learn basic representations before the router starts specializing.

### Entropy Regularization

The router's concentration is regularized toward a target normalized entropy of ~0.4 (the "goldilocks" regime). This prevents both uniform attention (no selectivity) and degenerate collapse (attending to a single token).

## 3. Omni-Directional Masked SSL Training

We train using a pure Masked Self-Supervised Learning (SSL) objective across the entire unified dataset.

1. **Input Construction**: The dataset selects an election group (a set of candidates in a specific location at a specific date) as the prediction target. The full-pool router then selects the most relevant context tokens from the entire universe.
2. **Masking**: The target candidates' `Value` is replaced with a learned `[MASK]` token. Some targets may be partially revealed (conditional prediction).
3. **Prediction**: The transformer processes the combined set (targets + selected context) and outputs a scalar logit per token. Logits within the same election group are passed through `softmax` to produce a probability distribution over candidates. Loss is computed via **KL Divergence** between the predicted distribution and the true normalized vote shares, **only on masked positions**.

### Loss Function: KL Divergence over Election Groups

Tokens are grouped by `(election_type, location, date)`. Within each group, the true values are normalized to sum to 1 (vote share distribution). The model's logits are passed through `log_softmax`, and KL divergence is computed:

```python
# For each election group g:
targets_g = true_values_g / true_values_g.sum()  # normalize to distribution
log_probs_g = log_softmax(logits_g)
loss_g = (targets_g * (log(targets_g) - log_probs_g)).sum()  # KL divergence
```

Only masked tokens contribute to the loss. This ensures: (a) the loss has a zero lower bound (when predictions = targets), and (b) the model learns relative vote shares within each race rather than absolute percentages.

## 4. Simulating Future Scenarios

Because the model is trained to predict *any* masked part of its input from *any* other part, it natively supports complex "what-if" scenario simulations without any architectural changes.

- **Baseline Prediction**: Feed historical tokens + demographics. Add target tokens for future candidates with values `[MASK]`ed. The model outputs softmax-normalized predicted vote shares.
- **Conditional Scenario ("If candidate X gets 30%")**: Unmask candidate X's token and hardcode its value. Leave others masked. The transformer propagates this constraint via self-attention, reshaping the remaining candidates' predicted shares.
- **Conditioning on Abstention**: Unmask the abstention token and set it to a hypothetical value. The model predicts candidate scores conditioned on that turnout environment.

## 5. Architectural Details

### Model Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `d_model` | 128 | Token embedding dimension (96 identity + 32 value) |
| `nhead` | 4 | Transformer attention heads |
| `num_layers` | 4 | Transformer encoder layers |
| `d_router` | 64 | Router projection dimension |
| `top_k` | 256 | Total selected tokens (targets + context) |
| `num_buckets` | 50,000 | Hash embedding table size for string fields |
| `router_warmup_steps` | 500 | Steps before router activates |

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | AdamW (lr=1e-3, weight_decay=0.01, fused=True) |
| Scheduler | CosineAnnealingWarmRestarts (T_0=5000, T_mult=2, eta_min=1e-5) |
| EMA | Exponential Moving Average (decay=0.999) |
| Batch size | 32 |
| Gradient clipping | max_norm=1.0 |
| Entropy regularization | λ=0.05, target=0.4 |
| Key cache rebuild | Every 100 steps (~0.5s per rebuild) |
| Early stopping | patience=10 epochs on dev loss |

### Output Layer

The value head outputs a single scalar per token: `nn.Linear(128, 1)`. These logits are not probabilities on their own — they become a probability distribution only when passed through `softmax` over all candidates within the same election group.

### Feature Scaling and Layer Normalization

- **Geo Scaling**: Latitude centered on 46.5, longitude on 2.5, both divided by 5.0 before linear projection.
- **Input Normalization**: `LayerNorm` applied to the combined token embedding before the transformer.
- **Pre-Layer Normalization (Pre-LN)**: Transformer layers use `norm_first=True` for training stability.

## 6. Data Scale

| Component | Size |
|---|---|
| Total token pool | ~16.8M tokens |
| Election result tokens | ~14.5M |
| Poll tokens | ~1.8M |
| Demographic tokens | ~0.5M |
| Val-only tokens (2026 muni results) | ~20K |
| Train election groups | ~1.08M |
| Dev election groups | ~57K |
| Val election groups (2026 Municipales) | ~7.1K |
| Unique communes | ~37K |
| GPU PoolCache memory | ~0.6GB (features) + ~2.2GB (key cache float16) |
| Peak training GPU memory | ~4.5GB on NVIDIA TITAN RTX 24GB |
| Model parameters | ~15.4M |

Demographic token count will grow significantly if/when multi-vintage loading is enabled (up to ~4.5M tokens for 16 Census vintages + 18 BPE vintages × ~35K communes).

## 7. File Structure

```
src/
├── model.py              # UniversalMaskedSetTransformer, LearnableRouter, TokenEmbedding
├── dataset.py            # TokenPool, PoolCache, TokenDataset, collate_token_sets
├── dataloader.py         # build_dataloaders(), load_all_tokens(), poll candidate resolution
├── train.py              # Training loop, EMA, entropy regularization, loss computation
├── eval.py               # Evaluation script
├── visualize_trajectories.py  # Trajectory prediction visualizations
├── load_elections.py     # Election data ingestion → token DataFrame
├── load_polls.py         # Poll data ingestion → token DataFrame
├── load_demographics.py  # Demographic data ingestion → token DataFrame
├── build_geo_mapping.py  # Commune lat/lon download + derived centroids
└── nuance_mapping.py     # Political nuance → party equivalence mapping
```
