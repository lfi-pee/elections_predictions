# Architectural Plan: Full-Pool Learnable Router

**Status: IMPLEMENTED (v2 — Full-Pool)**

The goal is to intelligently select the most relevant context tokens for the `UniversalMaskedSetTransformer` by scoring the **entire 16.8M token pool** via a pre-computed key cache, replacing the earlier random-sampling and small-pool top-K approaches.

## Implemented Architecture

### 1. Data Pipeline (`src/dataset.py`)

The `TokenDataset` no longer samples context — it only returns target election group info (pool indices for masked and unmasked tokens). All context selection is delegated to the GPU-side router.

- **`PoolCache`**: Holds all 16.8M token features as contiguous GPU tensors (~0.6GB) + pre-computed key cache (~2.2GB float16). Stores both `dates` (availability_date, for causality) and `reference_dates` (date_float, for model input). Provides `gather_tokens(indices)` for O(1) lookups, returning `reference_dates` as `"dates"` so the model sees the true temporal coordinate.
- **`TokenDataset.__getitem__`**: Returns `{anchor_date, masked_pool_indices, unmasked_pool_indices}`.
- **`collate_token_sets`**: Batches into `(anchor_dates, target_indices, target_masked, target_padding)`.

### 2. The Learnable Router (`src/model.py` — `LearnableRouter`)

A lightweight module (~28K parameters) that scores the full pool:

* **Step A: Pre-computed Key Cache.** `build_key_cache()` processes all 16.8M tokens in chunks of 100K, computes identity embeddings → `key_proj` (96→64) → L2-normalize → store as float16 on GPU. ~0.5s to rebuild.
* **Step B: Target Anchor Query.** Extract masked-target identity embeddings, average-pool → `query_proj` (96→64) → L2-normalize → single `(B, 64)` query per sample.
* **Step C: Full-Pool Scoring.** Brute-force matmul: `(B, 64) × (64, 16.8M)` → `(B, 16.8M)` cosine similarity scores, divided by a learnable temperature.
* **Step D: Masking.** Future tokens masked via `searchsorted` on `availability_date`-sorted pool (publication date, not reference date). Val-only tokens (2026 muni results) masked during training. Target tokens masked (force-included separately).
* **Step E: Top-K Selection.** `top_k=256` total selected tokens (targets + context). Context budget = 256 - (max targets in batch).
* **Step F: STE Gradient Flow.** Selected tokens are re-scored using current weights (not stale cache) via multiplicative Straight-Through Estimator. Context tokens are weighted by `softmax(live_scores) × n_context`, creating differentiable gradients through `query_proj`, `key_proj`, and `temperature`.

### 3. Router Warm-up
For the first 500 training steps, the router is bypassed and context tokens are selected randomly. Uses `randint` (sampling with replacement) when there aren't enough past tokens before the anchor date. This lets the transformer learn basic representations before the router starts specializing.

### 4. Key Cache Refresh
The cache is rebuilt every **100 training steps** (~0.5s per rebuild, negligible overhead). This keeps selection quality fresh while amortizing the cost of recomputing 16.8M key projections.

---

## Key Router Metric: Normalized Entropy

**`Router/entropy_reg_loss`** (logged to TensorBoard) tracks how close the router's concentration is to the target:

- Entropy regularization pushes normalized entropy toward **0.4** (the "goldilocks" regime).
- **→ 1.0** = uniform distribution (no selectivity, equivalent to random sampling)
- **→ 0.0** = degenerate collapse (one token gets all attention)
- **0.4** = selective but diverse context selection

Additional logged metrics:
- `Router/temperature` — learned temperature parameter
- `Router/selected_mean_time_delta` — average |Δt| of selected tokens (temporal reach)
- `Router/selected_mean_geo_dist_km` — average geographic distance from anchor (spatial reach)
- `Router/cache_rebuild_seconds` — time to rebuild key cache
- `Router/warmup` — 1.0 during warmup, 0.0 after

---

## Design Decisions

1. **top_k = 256**: Aggressive filtering (from 16.8M). Up from the earlier 32, because full-pool routing finds higher-quality tokens that deserve more representation.
2. **Identity-only routing**: Router scores based on *what* a token is (type, location, time, party, geography), not its value. Prevents information shortcuts.
3. **Cosine similarity + temperature**: Bounded scoring (−1 to +1 before temperature) prevents logit drift that plagued the earlier dot-product router.
4. **d_router = 64**: Small projection keeps the `(B, 64) × (64, 16.8M)` matmul cheap (~2.15GB transient).
5. **Unified pool**: Train and eval use the same pool and the same routing mechanism. No more distribution shift from random vs learned selection.
6. **float16 cache**: Halves memory for 16.8M × 64 keys (2.2GB vs 4.4GB).

---

## Evolution History

| Version | Pool Size | Selection | Top-K | Status |
|---|---|---|---|---|
| v0 | 1024 random | Random uniform | 1024 (no routing) | Replaced |
| v1 | 4096 random | Learnable dot-product router | 32 | Replaced |
| **v2** | **16.8M (full pool)** | **Cosine key cache + brute-force matmul** | **256** | **Current** |
