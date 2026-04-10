# Overfitting Investigation Findings

This document summarizes the recent investigation into the Universal Masked Set Transformer's inference behavior, the resolution of the "flat prediction" bug, and the subsequent discovery of severe overfitting.

## 1. The Context Starvation Bug (Resolved)

**The Problem:**
Initially, the visualization pipeline produced completely flat prediction trajectories over time, effectively collapsing to a static ~8% prior for all candidates regardless of the available polling or historical context. This suggested the model wasn't using the temporal signals at all.

**Root Cause:**
In `src/visualize_trajectories.py`, inference was incorrectly batching multiple locations (e.g., 50 communes) into a single forward pass. This resulted in roughly 1016 target tokens per batch. Because the `LearnableRouter` operates under a strict budget constraint (`top_k = 256`), the calculation `max(1, top_k - num_targets)` forced the context budget to exactly 1 token. During inference, the model was entirely starved of predictive context, causing the flatline behavior. Training survived because batches consisted of a single election location group, allowing a healthy context budget of ~245 tokens.

**The Fix:**
The visualization script was refactored to perform isolated, per-location inference matching the training data constraints. Predictions are now aggregated into national weighted averages post-inference.

## 2. Post-Fix Results: Confirmation of Learning

With the context budget restored during inference, visualizations now show that the model **is actively learning from context**:

*   **Temporal Dynamics Restored:** Tracked candidates (e.g., Le Pen) now exhibit sloped trajectories over time. The model is clearly updating its predictions as the anchor date approaches election day. But the update amplitude is very tiny.
*   **Strong In-Sample Performance (Train Set - Legislatives 2022):** The model significantly outperforms the "Naive Baseline" (party prior).
    *   *Baseline MEA (Party Prior):* ~10.3 percentage points (pp)
    *   *Model MEA:* ~2.0 pp
    *   *Takeaway:* The model is successfully extracting patterns and mapping context to outcomes on the data it has seen.

## 3. The Overfitting Problem

While the model performs exceptionally well on the training data, evaluation on the validation set reveals critical generalization failures.

*   **Catastrophic Overfitting (Val Set - Legislatives 2024):**
    *   *Baseline MEA (Party Prior):* ~12.0 pp
    *   *Model MEA:* ~33.0 pp
*   **Analysis:** The model is not just failing to generalize; its predictions on unseen elections are significantly worse than naive guessing. The 33pp error vs 2pp training error represents a massive generalization gap. The model is likely memorizing the training elections (retrieving specific, idiosyncratic tokens) rather than learning structural, causal relationships between polling/demographics and election outcomes.

## 4. Recommended Next Steps

For the next session, focus should shift entirely from architecture debugging to **model regularization and representation tracking**:

1.  **Analyze Router Attention:**
    *   Inspect what tokens the `LearnableRouter` is actually selecting in both train and validation splits.
    *   Is it latching onto "memorized" demographic tokens or specific historical results that uniquely identify a training commune, or is it grabbing general polling statistics?
2.  **Add Strict Regularization:**
    *   The model capacity is likely too high given the dataset's variance.
    *   Implement heavy **Dropout** in the transformer layers and embeddings.
    *   Increase **Weight Decay** / L2 regularization.
    *   Temporarily reduce the hidden state dimension or the number of attention heads to forcefully restrict its ability to memorize.
3.  **Evaluate Loss & Architecture Tweaks:**
    *   *Candidate Embeddings:* Check if initializing `candidate_emb` with small random values instead of zeros prevents premature collapse.
    *   *Router Entropy Constraint:* Check if the entropy regularization on the router is inadvertently forcing hard, deterministic selections of specific identifiers.
    *   *Loss Function Shift:* As discussed earlier, consider transitioning from softmax-over-groups translation invariance to direct continuous value regression (e.g., Sigmoid + L1 loss) to enforce stricter bounds on candidate share predictions.

## Reference Files
*   **Fix implemented in:** `src/visualize_trajectories.py`
*   **Visualizations output:** `/home/veesion/elections_predictions/visualizations/`
*   **Key Train Plot:** `viz_1a_ghost_error_distribution_train.png`
*   **Key Val Plot:** `viz_1a_ghost_error_distribution_val.png`

## 5. Update: Latest Training Checkpoint Analysis

New visualizations have been generated using the latest training checkpoint (`epoch 0, step 10000`). The goal was to verify if the prediction was still "constant in time" or if the model successfully learned temporal dynamics after fixing the context allocation and running the training for a while.

You can inspect the new trajectories here:
*   ![Tracked Candidate - Train](/home/veesion/elections_predictions/visualizations/viz_1b_tracked_candidate_national_train.png)
*   ![Ghost Candidate - Train](/home/veesion/elections_predictions/visualizations/viz_1a_ghost_candidate_national_train.png)
*   ![Ghost Candidate - Val](/home/veesion/elections_predictions/visualizations/viz_1a_ghost_candidate_national_val.png)

*(See all results in the `/home/veesion/elections_predictions/visualizations` folder).*

Initial observations from the `epoch0_step10000` visualizations indicated that the trajectories remained mostly flat, and the router metrics (`Router/selected_mean_time_delta`) revealed that it was consistently selecting context tokens from approximately **12 years in the past**. This anomalous behavior led to the discovery of the Temporal Aliasing Bug.

## 6. The Temporal Aliasing Bug (The 12-Year Shift)

**The Problem:**
Even after fixing the context starvation bug, the model's router was failing to fetch recent context, opting instead to retrieve data from roughly 12 years prior. This resulted in poor, quasi-static trajectories and catastrophic overfitting on out-of-distribution absolute dates (like Validation 2024).

**Root Cause:**
The sinusoidal temporal identity embeddings (`TokenEmbedding._embed_identity`) were computed using `torch.sin(date)` and `torch.sin(date / 5.0)`, where `date` is measured in years. The term `torch.sin(date)` possesses a natural period of $2\pi \approx 6.28$ years. Therefore, exactly two periods correspond to $\approx 12.56$ years. 

To the router's query/key MLPs, a context token from 12.5 years ago generated nearly the identical high-frequency embedding as a token from today. Because the lower-frequency term wasn't powerful enough to break this symmetry, the router learned to cluster contemporary queries with heavily aliased data from ~12 years prior. Worse, when evaluated on validation data from novel unseen years (e.g., 2024), the embeddings shifted wildly across these aliases, causing the MLPs to output random garbage, completely explaining the catastrophic generalization gap.

**The Fix:**
The `src/model.py` embeddings have been refactored to use explicitly scaled periods that natively span the entire 20-year history of the dataset without perfect integer aliasing:
*   **Period 1:** 40 years ($\omega_1 = 2\pi / 40.0$) - Guaranteed zero aliasing inside the dataset envelope.
*   **Period 2:** 8 years ($\omega_2 = 2\pi / 8.0$) - Finer temporal resolution while missing the catastrophic 12-year boundary.

The stagnant training run has been terminated, and a fresh run was launched (`python3 src/train.py > training_regularized.log 2>&1 &`). Tensorboard should now show `Router/selected_mean_time_delta` stabilizing to accurate, recent timestamps.
