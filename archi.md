# Election Prediction: Universal Masked Set Transformer

## The Core Concept

We abandon all feature engineering, aggregation, and manual alignment between datasets. Instead of creating structured feature vectors, we flatten the entire universe of French election data (results, polls, and demographics) into atomic, raw data points. 

The network takes an **arbitrary subset of the entire data universe** as a Set of Tokens and is tasked with predicting the remaining, masked subset. 

## 1. Raw Data as Tokens (Zero Feature Engineering)

Every single row from our raw datasets (whether it's an election result in a specific village, a national poll, or a demographic statistic) is converted into a universal atomic token. 

There is no summing, no averaging, and no manual mapping of candidates to parties. We simply encode the text.

A token consists of three parts:
1. **Time**: The date represented as a continuous float (e.g., fractional years). To ensure the network learns relative temporal distances rather than memorizing absolute dates, the reference year ($t=0$) is either set to 2000 or randomly shifted for each sample during training (making the network invariant to the absolute date reference frame).
2. **Textual Context**: Who, what party, what election, and where (passed as raw strings). Candidates receive a dual encoding combining their specific identity and their party affiliation (if it exists).
3. **Continuous Value**: The actual number (e.g., vote percentage, turnout %, poll intent %).

For example, raw rows from different datasets become tokens:
- `[22.27, "Presidentielle_T1", "Brest", "MACRON Emmanuel", "LREM", "Result", 27.8]`
- `[22.27, "Presidentielle_T1", "Brest", "Abstention", "", "Context", 26.0]`
- `[24.43, "Europeennes", "National", "BARDELLA Jordan", "RN", "Poll_Ifop", 31.5]`
- `[24.43, "Europeennes", "Brest", "Unemployment_Rate", "", "Demographics", 6.8]`

### Dual Encoding & Time
We encode categorical text and temporal distance. The candidate receives a dual embedding of both their identity and their political party, forcing the network to learn both personal and partisan dynamics.

```python
# 'date_float' is standardized to a random reference year per sample during training
token_identity = Linear(date_float) + \
                 StringEmbedding("Presidentielle_T1") + \
                 StringEmbedding("Brest") + \
                 StringEmbedding("MACRON Emmanuel") + \
                 StringEmbedding("LREM") + \
                 StringEmbedding("Result")

# The final token is the identity plus the scalar value
token = Concat(token_identity, Linear(Value)) 
```
*(If the value is masked, we provide a learned `[MASK]` token instead of `Linear(Value)`).*

## 2. Omni-Directional Masked SSL Training

We train using a pure Masked Self-Supervised Learning (SSL) objective across the entire unified dataset. 

1. **Input Construction**: We sample an arbitrary set of data tokens (e.g., all available historical records, polls, and demographics for a given subset of the universe).
2. **Masking**: We randomly apply a `[MASK]` to the numerical `Value` of an arbitrary subset of tokens. We can mask candidate scores, abstention rates, or even past polling numbers.
3. **Prediction**: The Deep Set network (Transformer with global self-attention) processes the entire set and predicts the missing values. Loss is calculated against the true raw numeric values.

Because the self-attention mechanism is permutation-invariant and global, the network must figure out on its own that a target token with `Candidate="MACRON Emmanuel"` in 2027 should attend to historical tokens with the identical text string in 2022, or to poll tokens from the same timeframe. We do absolutely no manual engineered correspondences.

## 3. Simulating Future Scenarios

Because the model is trained to predict *any* masked part of its input from *any* other part, it natively supports complex "what-if" scenario simulations without any architectural changes.

To predict the 2027 Presidential election:
- **Baseline Prediction**: We feed the set of historical tokens, plus 2027 demographics and 2027 polls. We add Target tokens for the 2027 candidates with their values `[MASK]`ed. The model outputs the predicted election results.
- **Conditional Scenario ("If candidate X gets 30%")**: We unmask Candidate X's target token and hardcode its value to `30.0`. We leave the other candidates masked. The self-attention matrix immediately propagates this constraint, and the network predicts how the remaining candidates' scores reshape around this fixed point.
- **Conditioning on Abstention**: We unmask the 2027 Abstention token and set it to a hypothetical value. The model predicts the candidate scores precisely conditioned on that turnout environment.

## 4. Architectural Details & Training Scope

### Dynamic Context Sizing
The architecture places no restrictions on the size or type of context fed to the model at train or inference time. It is built to accept any arbitrary subset. To handle the $O(N^2)$ computational limit of attention when provided with a massive context during training, the `TokenDataset` automatically limits the context via random uniform sampling to a specified `max_seq_len` (e.g., 1024 tokens). This allows the architecture to ingest arbitrary lengths seamlessly while remaining safely within VRAM bounds.

### 100-Bin Output Classification
Instead of predicting a single continuous scalar via MSE Regression (which forces the network to average out divergent multi-modal scenarios), the architecture maps the `[0, 100]` value space of election percentages into 100 discrete classification bins.

The transformer's value head outputs a 100-dimensional probability distribution (`nn.Linear(d_model, 100)`), and the model is trained via Cross-Entropy loss. This allows the network to natively model uncertainty, split alliances, and express non-linear multimodal probability distributions.
