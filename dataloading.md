# Universal Token DataLoader Pipeline

## Overview

The dataloading pipeline is designed around a core philosophy: the model must discover connections natively across any arbitrary subset of the universe. To achieve this, data is conceptually unified into a flat pool of `DataToken` objects, and sampling is purely **timestamp-centric**.

Data is not grouped sequentially by commune or by election cycle; rather, it is fetched via a stochastic temporal window that captures heterogeneous contexts across the entire dataset.

## 1. Raw Data Normalization & Ingestion

The system unifies two major streams of raw data under an identical schema (`DataToken` format) during startup, creating an in-memory `TokenPool` (approx. 14 million tokens).

### 1.1 Elections Data (`src/load_elections.py`)
- **Source**: Parquet files (`candidats_results.parquet` and `general_results.parquet`).
- **Granularity Transformation**: Raw data is structured at the `bureau-de-vote` level (~30 million records). It is aggregated to the **commune-level**. **Crucially, every single election result is always expressed as a percentage (0-100), never as absolute vote counts.** `voix` (votes) and `inscrits` (registered voters) are mathematically fused into ratios. We do this because absolute counts would break geographical size invariance: a 52% margin means the exact same political dynamic in Paris as in a 300-person village. Exposing absolute numbers would force the model to waste parameters memorizing population density rather than learning multi-dimensional political trajectories.
- **Token Instantiation**:
  - `Result` tokens wrap specific candidate scores (Candidate name mapped to `candidate`, Party mapped to `party`).
  - `Context` tokens represent environmental statistics (e.g., `Abstention` or `Blancs` ratios).

### 1.2 Polls Data (`src/load_polls.py`)
- **Source**: Various raw `.csv` files inside `data/polls/`.
- **Parsing Flexibility**: The module ingests **all** available CSV datasets across all elections (Presidential, Legislative, Regional, European, etc.). We do not filter out secondary tours, nor do we discard dirty data blocks. The parser dynamically isolates dirty strings (`[1]`, `<`, `%`, empty cells) to recover all underlying floats. We train on the full unvarnished spectrum of historical data, including low-quality institutes, because an attention-based Set Transformer structure is mathematically capable of implicitly weighing lower-reliability `Poll_{InstituteName}` tokens during contextualization. Discarding rows deletes signal; preserving them lets self-attention find the noise bounds.
- **Token Instantiation**: All poll rows project into the same `DataToken` space. They encode their geographic scope via `location="National"` (or regional scope) and their structural nature via `metric_type=Poll_{InstituteName}`.

## 2. Timestamp-Centric Sampling Architecture

The `TokenDataset` operates by casting stochastic temporal windows over the sorted unified `TokenPool`.

### Step-by-Step Sample Construction (`__getitem__`)
1. **Temporal Anchoring**: A random anchor point in time is uniformly selected across the entire dataset’s life span (e.g., a timestamp float between ~1999.0 and ~2026.0).
2. **Context Windowing**: Using binary search (`bisect`), all tokens whose timestamps fall within a configurable `window_half_years` (e.g., `±0.5` years) are extracted. This pool crosses all communes, polling entities, and abstract contexts dynamically.
3. **Sub-Sampling & Target Masking**: The logic applies specific rules to simulate predictive scenarios uniformly across all subsets:
   - **75% of samples**: Focus purely on predicting a single election location without cross-location leakage. Among these, **75%** have all anchor candidates fully masked (for full prediction), while **25%** have 1 or 2 candidates revealed to learn conditional outcome scoring.
   - **25% of samples**: Inject context from the exact same election at another location (up to 5 candidate results maximum).
4. **Context Stratification**: The remaining `max_seq_len` token slots are filled by perfectly balancing **50% past election context** and **50% polling/other context** parameters. If one pool falls short, the other uses the remaining budget.
5. **Temporal Invariant Shift**: During training, a uniform random temporal shift (e.g., `[-10.0, +10.0]` years) is applied across all dates within the sampled sequence simultaneously. This forces the model to encode inter-token temporal distances relative to each other instead of memorizing an absolute date reference frame.

## 3. Data Flow & Interface

- **`TokenDataset`**: Maintains the global token pool. Returns a tuple of `(tokens, masked_indices, target_tensor)`.
- **Masking Routine**: At the dataset level, a boolean mask vector is constructed denoting which token values are hidden (e.g. `15%`). The true continuous values are dynamically binned into `100` discrete classes outputted via the target tensor to feed Cross Entropy classification mechanisms.
- **Collate Function (`collate_token_sets`)**: Injects identical length boundary limits to dynamically padded `[PAD]` sequences while propagating valid masking matrices. Returns `(batch_tokens, batch_masks, targets, padding_mask)`.
