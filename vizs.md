# Visualizing the Universal Masked Set Transformer for Political Strategy

This document outlines key visualizations and use cases to demonstrate the power of the Set Transformer election prediction model to a political party. Unlike traditional polling models, this algorithm is uniquely omni-directional, stateless, and naturally outputs a 100-bin probability distribution.

The primary goal of these visualizations is to prove that this model is not just a poll aggregator, but a **simulation engine** capable of vastly outperforming naive polling averages by understanding structural relationships in the electorate.

Use the best checkpoint and eval it with real data.

## 1a. The "Ghost Candidate" Trajectory (Zero-Shot Prediction)

*   **Concept:** Proving the model can accurately forecast a specific candidate's localized performance even when no direct polling data exists for them in that region.
*   **Visualization:** **Converging Line Chart with Confidence Bands**
    *   **X-axis:** Time (months out from the election).
    *   **Y-axis:** Expected Vote Percentage.
    *   **Visual:** A solid line representing the model's prediction for the "ghost" candidate (e.g., Jean-Jacques Gaultier in the 2022 Legislative Election). Surrounding the line is a shaded area representing the model's uncertainty. As time passes and the model processes *other* national context tokens (competitor polls, macro trends), the line converges almost perfectly onto the true historical outcome.
*   **Baseline Comparison:** Include a flat, horizontal line representing the **Naive Polling Baseline** (which, for an unpolled candidate, defaults to their party's historical average previous scores). The widening gap between the rigid baseline and our model's responsive, converging line visually proves the algorithmic superiority.
*   **The Pitch:** *"We don't need a poll for every candidate. The algorithm infers their exact score purely from the vacuum left by the shifts of other candidates and regional history, whereas a standard model remains completely blind until explicitly told."*

## 1b. The "Tracked Candidate" Trajectory (Data-Rich Prediction)

*   **Concept:** Demonstrating that even when direct polling *is* available, our model outperforms a simple poll average by contextualizing the noisy polls within the broader political environment.
*   **Visualization:** **Signal vs. Noise Smoothing Line Chart** (implemented in `src/visualize_trajectories.py`)
    *   **X-axis:** Time (months out from the election).
    *   **Y-axis:** Expected Vote Percentage.
    *   **Visual:** Display scattered data points for individual published polls. Our model's prediction is a smooth, stable line that intelligently cuts through the noise, naturally dampening outlier polls (because it knows structurally they don't make sense) and adjusting gracefully based on other macroscopic tokens.
*   **Baseline Comparison:** Overlay a jagged line representing a **Naive Rolling Average Baseline**. The baseline violently spikes and dips with every new, noisy poll release (overreacting to outliers). Our model's line is much smoother, converging onto the final outcome faster and with less volatility by incorporating structural priors rather than just mathematical averages.
*   **The Pitch:** *"Even when you have polling data, raw averages are highly susceptible to noise and outliers. Our algorithm acts as an intelligent filter, combining the raw polling signal with underlying structural truths to give you a stable, accurate trajectory."*

## 2. The Contextual Ripple Effect (Counterfactual Simulator)

*   **Concept:** Highlighting that the model treats data as unstructured "tokens," meaning we can easily inject hypothetical tokens to test "What-If" scenarios.
*   **Visualization:** **Interactive Dynamic Tornado/Waterfall Chart**
    *   **Visual:** Start with the baseline expected outcome for the party's candidate. Below it, show real-time adjustments as specific hypothetical "tokens" are added to the set context:
        *   *What if abstention rises by 5%?* (Shows a -1.2% impact)
        *   *What if an allied third-party candidate drops out?* (Shows a +4.5% surge)
*   **Baseline Comparison:** A naive model only responds to direct polling bumps. If a hypothetical change isn't explicitly captured in a poll, a naive model shows a `0.0%` impact. The chart makes this explicit by contrasting our dynamic contextual response against the baseline's static inability to run complex environmental simulations.
*   **The Pitch:** *"Stop guessing how a national event will impact a local race. We can inject any hypothetical scenario into the model and see exactly how the shockwave mathematically propagates to your specific candidate."*

## 3. The Resolution Enhancer (National-to-Local Translation)

*   **Concept:** Political parties often only have the budget for high-level national or regional polls. The model can map these macro trends onto micro-geographies (communes / bureaux de vote) using structural baseline similarity.
*   **Visualization:** **Side-by-Side Animated Choropleth Maps**
    *   **Visual:** The left map shows chunky, uniformly colored blocks representing broad regional polling averages. The right map is highly granular (commune level). When "Predict" is triggered, the granular map instantly fills in with high-resolution, localized expected vote shares.
*   **Baseline Comparison:** The left map *is* the visual representation of the naive baseline. It assumes every commune in a region votes exactly like the regional average. The right map highlights the model's superiority by differentiating a wealthy urban center from a rural outskirt within the exact same polling region.
*   **The Pitch:** *"You buy the cheap, broad national polls. A standard model applies them blindly. Our algorithm acts as a high-resolution lens, fracturing that overarching trend into exact, tailored predictions for every single polling station."*

## 4. The "Information Necessity" Curve (Proving Efficiency)

*   **Concept:** Demonstrating that the Set Transformer is incredibly data-efficient and doesn't need a massive, continuous timeline of polls to be accurate.
*   **Visualization:** **Cumulative Accuracy Scatter Plot**
    *   **X-axis:** Number of context tokens provided (e.g., 5 prior election results, 20 polls, 100 polls).
    *   **Y-axis:** Overall Mean Absolute Error (MAE) compared to ground truth.
    *   **Visual:** Our model's error rate plummets dramatically with just a tiny handful of context tokens.
*   **Baseline Comparison:** Overlay a second line representing a traditional tracking/regression model. The traditional model's MAE drops much slower, or flatlines early because it cannot handle missing interaction variables efficiently. It clearly shows our model reaching 95% of maximum accuracy with a mere fraction of the data required by the baseline.
*   **The Pitch:** *"Traditional models need continuous, expensive polling cycles. Our Set Transformer achieves maximum accuracy with a fraction of the data footprint, saving you polling budget while outperforming the standard tracker."*

## 5. The Risk & Volatility Matrix 

*   **Concept:** Even though the model outputs a single softmax-normalized vote share (not a full probability distribution per candidate), we can estimate uncertainty by running **Monte Carlo context perturbations**: varying which context tokens the router selects (via temperature scaling or subsampling the top-K) and observing the variance of predictions across runs.
*   **Visualization:** **Ridge Plot (Joyplot) of Prediction Variance**
    *   **X-axis:** Expected Vote Percentage (0-100%).
    *   **Y-axis:** Snapshots in time (6 months out, 3 months out, 1 week out).
    *   **Visual:** At 6 months out, the distribution of Monte Carlo predictions is wide (high context-sensitivity). As election day approaches and more direct data becomes available, the distribution narrows into a sharp peak.
*   **Baseline Comparison:** A standard polling average provides a single point estimate with a static +/- 3% margin of error. Our Monte Carlo envelope dynamically adapts to the actual information landscape.
*   **The Pitch:** *"A standard average gives you a single number and a boilerplate margin of error. We show you how stable our prediction actually is given the available data. A wide spread means the race is volatile and worth investing; a narrow peak means the electorate is locked in."*

## 6. Election Cross-Pollination Transferability

*   **Concept:** The architecture is stateless, meaning it can easily map patterns from one *type* of political event to a completely different one (e.g., using European election dynamics to predict a localized Municipal election).
*   **Visualization:** **Sankey Diagram (Flow Chart)**
    *   **Visual:** Nodes representing seemingly unrelated past data (e.g., "European Polls 2024"). Thick flowing bands connect these into a central prediction node for the target "Legislative Election 2024".
*   **Baseline Comparison:** The baseline approach is entirely disconnected. Traditional tracking models cannot cross-pollinate election types (they can't mathematically blend a municipal outcome into a presidential poll). Show the baseline relying *only* on the thin, isolated trickle of target-specific data, while our model gulps insights from every available historical stream.
*   **The Pitch:** *"Voters behave in interconnected ways. Because our algorithm is omni-directional, it intelligently borrows insights from completely different types of elections to predict the blind spots in yours."*
