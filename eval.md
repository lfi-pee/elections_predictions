# Evaluation Script (`src/eval.py`)

The evaluation script measures how well the `UniversalMaskedSetTransformer` can predict election outcomes by conditioning on all available historical context via the full-pool router.

## How it Works

1. **Data Loading & PoolCache**
   The script loads all tokens into a unified `TokenPool` and builds a `PoolCache` on GPU, identical to the training pipeline. This gives the model access to the entire 16.8M token universe.

2. **Key Cache Construction**
   The model's `rebuild_key_cache()` is called to pre-compute L2-normalized key projections for the full pool, enabling the router to score all tokens.

3. **Target Identification**
   The script identifies target election groups (e.g., all 2026 Municipales communes). For each group, the target tokens are provided with their values `[MASK]`ed.

4. **Full-Pool Routing**
   For each batch, the model's router:
   - Computes an anchor query from the masked target identity embeddings
   - Scores the entire pool via matmul against the key cache
   - Masks future tokens and selects the top-K most relevant context
   - No manual context windowing or size limits

5. **Prediction & Scoring**
   The model outputs a scalar logit per token. Logits within the same election group are passed through `softmax` to produce a vote share distribution. Metrics:
   - **KL Divergence**: Primary loss metric (matches training objective)
   - **MAE**: Mean Absolute Error between predicted and true vote share per candidate
   - **Winner Accuracy**: Whether the predicted winner (argmax of softmax) matches the true winner

   Metrics are broken down by:
   - Election type (Presidentielle, Legislatives, Municipales, etc.)
   - Polled vs unpolled candidates
   - Combined granularity (e.g., `Municipales_unpolled`)

6. **EMA Weights**
   Evaluation uses the Exponential Moving Average (EMA) of model weights for stable, smoothed predictions.

## Usage

```bash
cd /home/veesion/elections_predictions
python3 -m src.eval [options]
```

**Key parameters:**
- `--model`: Path to trained checkpoint (default: `best_model.pth` — contains EMA weights)
- Model architecture must match: `d_model=128, nhead=4, num_layers=4, d_router=64, top_k=256`

## Relation to Training Eval

During training, evaluation happens at two levels:
1. **Per-step**: Every 10 steps, one dev batch and one val batch are evaluated and logged to TensorBoard
2. **Per-epoch**: Full dev and val set evaluation using EMA weights, with per-split metric breakdown

The standalone `eval.py` is for detailed post-training analysis with configurable targets.
