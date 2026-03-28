# French Elections Predictions: Universal Masked Set Transformer

This repository contains a modern, deep-learning approach to analyzing and predicting French political elections. Instead of relying on traditional feature engineering, manual aggregations, or rigid party alignments, it flattens the entire universe of French election data (20 years of municipal, legislative, regional, European, and presidential results + polling/demographics) into an atomic, permutation-invariant sequence of tokens trained via Masked Set Modeling.

## Architecture

The system is built on a **Universal Masked Set Transformer** (UMST) operating under a Set-to-Distribution paradigm, deeply inspired by masked self-supervised architectures (like JEPA).

Key architectural concepts (as defined in `archi.md`):
- **Raw Data as Tokens**: Every data point is converted into a structured text token (`date_float`, `election_type`, `location`, `candidate`, `party`, `metric`) paired with a scalar value. 
- **Zero Feature Engineering**: The model ingests data directly; there are no manual alignments between polls and results. Candidate identities and party affiliations are distinct embeddings, allowing the model to naturally learn inter-candidate alignment.
- **Temporal Invariance**: Years are converted to floats with random reference frame shifting to encourage the model to learn relative, continuous temporal relationships rather than memorizing fixed years.
- **100-Bin Output Classification**: Instead of a noisy single continuous regression output, the `[0, 100]` value space interval maps to a 100-bin output layer natively providing distributions over expected vote allocations using Cross-Entropy loss.

## Data Structure

The project dynamically relies on scraping and processing public election and polling data:
- `data/elections/`: Contains granular historical results. (Documented in `election_data.md`)
- `data/polls/`: Contains historical and recent polling data. (Documented in `polls_data.md`)

## Codebase

- `src/token.py`: Definition of the atomic `DataToken`.
- `src/dataset.py` & `src/dataloader.py`: Handling randomized context lengths up to `max_seq_len`, temporal shifting noise for training invariance, and padding targets logic.
- `src/model.py`: Definition of the custom `TokenEmbedding` string hash buckets alongside the native Transformer processing.
- `src/train.py`: Unified 100-bin Cross-Entropy training framework iterating on extracted masked candidate tokens.
