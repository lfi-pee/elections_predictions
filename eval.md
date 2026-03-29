# Evaluation Script (`eval.py`)

The evaluation script is designed to measure how well the `UniversalMaskedSetTransformer` model can predict a "future" scenario by strictly conditioning it on a predefined contextual past.

Because the model architecture is omni-directional and stateless, doing evaluation requires intentionally masking information and restricting the context window instead of iterating across a sequence layout.

## How it Works

1. **Target Identification**
   The script specifies a single "target" election as the ultimate goal (by default, `2024_legi_t1` corresponding to the 1st Round of the 2024 Legislative Elections). It determines the exact continuous `date_float` for that election.

2. **Temporal Context Framing** 
   To mimic predicting the future, the script scopes the universe of data down to a restricted window. It grabs all available data tokens (polls, past election results, context items like abstention) from `target_date - 1.0` year to the `target_date` strictly prior.

3. **Masking the Targets**
   The true result tokens for the specific targeted election are extracted from the dataset. These tokens are placed in the sequence with their numeric `<value>` explicitly hidden via the `[MASK]` token.

4. **Omni-Directional Inference**
   The context tokens (unmasked) and target tokens (masked but maintaining identity and structural information like location/candidate) are shuffled together into a set and passed through the `UniversalMaskedSetTransformer`.

5. **Prediction & Scoring**
   The Self-Supervised Learning (SSL) task requires predicting the 100-bin classification probability distribution for the masked target values.
   - **Argmax**: Simply selects the discrete 1% bin with the highest confidence probability.
   - **Expected Value**: Multiplies the distribution by the 0-100 values to get a continuously smooth expected percentage.

   Finally, the predictions are compared uniformly against the true ground percentages, outputting the continuous **Mean Absolute Error (MAE)** and **RMSE**. 

## Usage

```bash
python3 src/eval.py [options]
```

**Options:**
- `--model`: Path to trained checkpoint (default: `best_model.pth`).
- `--target`: The string ID for the targeted evaluation election (default: `2024_legi_t1`). 
- `--context`: The size of the context window looking backwards in continuous half-years (default: `0.5`, equating to 1 full year).
- `--seq-len`: The combined sequence length limit (default: `1024`). The evaluation explicitly caps target elements to 25% of the frame limit to ensure adequate contextual backing.
