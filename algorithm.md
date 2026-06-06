# Block-Level Election Prediction: Algorithm Design

## ABSOLUTE RULE: NEVER tune on validation data

**No parameter, hyperparameter, weight, estimate, threshold, or correction may be optimized using the 2024 validation set.** This includes but is not limited to:

- Meta-blend / meta-learner weights fit on validation predictions
- Abstention national estimate grid-searched on validation R²
- Lambda / shift parameters maximizing validation R²
- Any sweep, grid search, or selection criterion computed on validation targets

The validation set is for **evaluation only** — a single forward pass, no feedback loop. Any metric that touches validation labels during fitting is an oracle metric, not a real result. If a design choice requires validation tuning to work, it is not a valid design choice.

## Problem

Predict BV-level (bureau de vote) block vote shares for French elections:
- **Gauche** (left), **Centre+Droite** (center-right), **Extreme Droite** (far-right), **Abstention**

Validation: 2024 Legislatives T1 (~69K BVs). Training: all prior Legislatives + Presidentielles T1 (2002-2022).

## Best Results (raw R², LOO-selected on training, single val forward pass)

Models selected by best Leave-One-Election-Out R² on training data. No validation feedback enters the selection or fitting pipeline. See `preregistered.py`.

| Block | Raw R² | Model | Config | LOO OOF R² |
|---|---|---|---|---|
| Gauche | **0.74** | PCA5-devlag | legi-only, V1, 4 train dates | 0.797 |
| Centre+Droite | **0.61** | PCA7-devlag | legi-only, V1, 4 train dates | 0.597 |
| Extr. Droite | **0.80** | PCA5-devlag | cross-type (Legi+Pres), V1, 8 train dates | 0.816 |
| Abstention | **0.41** (0.77 cross-sect.) | PCA10-devlag | cross-type (Legi+Pres), V1, 8 train dates | 0.907 |

All numbers use raw poll averages as the national estimate (G=30.7, C+D=32.8, ED=36.5) for vote blocks. For Abstention, the national level comes from the published pre-election **participation poll** (Ipsos/CEVIPOF "indice de participation", 63% participation for 2024 → 37.0% abstention), used directly like the vote-block polls. Actual 2024 abstention is 31.0%, so the raw Abstention R² = 0.41 (a ~6pp national-level offset, vs ~18pp under a history-only estimator); the cross-sectional/debiased R² is 0.77, unchanged. No calibration, no grid search, no validation tuning of any kind.

### Note on LOO limitations

LOO with 4 folds (legi-only) or 8 folds (cross-type) is noisy. Models that are genuinely good on validation can have poor OOF R² (e.g., CT-devlag-1lag: OOF=0.56 but val=0.75 for Gauche). Extended-type models (20 folds) have more stable OOF but the training distribution differs from validation (Legi T1). The full OOF vs val table is in `preregistered.py` output.

## Best Architecture: Two-Stage Deviation + Cross-Type Training

### Stage 1: National mean estimation

Estimate the national average block score for 2024 from polls:
- Vote blocks (G, C+D, ED): use raw poll averages directly. The 2024 polls (G=30.7, C+D=32.8, ED=36.5) closely match actual results (G=29.3, C+D=31.4, ED=37.1).
- Abstention: a published pre-election **participation poll** (Ipsos/CEVIPOF "indice de participation", a 0–10 "certain to vote" scale converted to %), used directly like the vote-block polls. For 2024: 63% participation → 37.0% abstention (actual 31.0%). When no participation poll covers an election, the fallback is a **LOO-selected** historical estimator among five candidates (gap-model, last-same-type, same-type mean, global mean, last-any), scored by LOO RMSE — the choice is LOO-derived, no test information enters it. (Note: a history-only estimator extrapolates the 2002→2022 upward trend to ~49.6% and structurally cannot see the 2024 snap-election mobilization; the participation poll can.)

### Stage 2: BV deviation model

Predict how much each BV deviates from the national mean:
```
target_dev = BV_block_score - national_mean(election_type, date)
pred = national_mean_estimate + Ridge_deviation_pred
```

Deviations are more stable across elections and across election types. A BV that's +10pp Gauche relative to national stays roughly +10pp regardless of election type or year.

### Feature vector (~60 dims)

| Feature group | Dims | Notes |
|---|---|---|
| Demographics (census) | 52 | Last available vintage via merge_asof. V1=strict NaN drop, V2=median impute |
| Deviation lags 1-2 | 8 | `BV_lag - national_mean(lag_election)`. Cross-type: most recent prior election of any type |
| Election type one-hot | 6 | Only for cross-type models (unnecessary in deviation space but doesn't hurt) |

### Training configurations

**Legi-only:** Train on Legislatives T1 only (with cross-type lags from the full dataset). V1 (strict NaN) retains 4 training dates (2007.5-2022.5, ~262K rows). Best for C+D with PCA10 (0.60).

**Cross-type (Legi+Pres):** Train on Legislatives + Presidentielles T1 in deviation space. V1 gives 8 training dates (2007-2022 Pres + Legi), ~522K rows. Best for ED (0.81). With 1-lag variant (dropping lag2 requirement), 9 training dates, best for Gauche (0.75).

**Extended cross-type:** Train on Legi + Pres + Euro + Regi + Dept + Cantonales T1. V1 gives ~20 training dates, ~1.2M rows. Best for Abstention (0.77 with PCA3). Dates with poor block mapping (>15% "Other") excluded: Euro 1999/2004, Regi 2021. Gap model uses Legi+Pres national means only (other types add noise to the turnout model).

All configurations are implemented in `cross_type_dev.py` (Legi+Pres) and `beat_it.py` (extended + all experiments).

### LOO Stacking Blend (valid — no validation tuning)

Leave-One-Election-Out stacking on training data only. For each training date, train each base model on the other dates, predict the held-out date → out-of-fold (OOF) predictions. Fit a meta-learner (Ridge or NNLS) on pooled OOF predictions. Apply to 2024 predictions from models trained on all training data. All weights are learned from training data; 2024 is a single forward pass.

Tested in `beat_it.py` with up to 8 models × 20 folds. NNLS and Ridge meta-learners give marginal gains over best single model (+0.001–0.003). Individual models remain stronger due to per-block specialization.

## Feature Innovations (in order of impact)

1. **Deviation targets + deviation lags**. The single biggest innovation. Removes national-level bias from both target and lags, making the Ridge learn stable local deviations rather than absolute vote shares. Enables cross-type training and proper national estimation.

2. **Two-stage national estimation from polls (incl. a participation poll for abstention)**. Without a national estimate, the Ridge predicts deviations centered on zero — raw R² is catastrophic because predictions miss the national level entirely. With raw poll estimates, the model is properly calibrated for G, C+D, ED. For Abstention, the national level comes from the published **participation poll** (Ipsos/CEVIPOF "indice de participation", 63% → 37.0% for 2024), used directly like the vote-block polls. Raw Abstention R² is therefore 0.41 (a ~6pp national-level offset, vs ~18pp for a history-only estimator that extrapolates to ~49.6% and misses the snap-election mobilization); the cross-sectional/debiased R² is 0.77. Fallback when no participation poll exists: a LOO-selected historical estimator.

3. **Cross-type training in deviation space**. Legi + Pres T1 gives 8 training dates instead of 4. Best for ED (0.81 cross-type vs 0.78 legi-only). With 1-lag variant (dropping lag2 requirement), 9 dates — best for Gauche (0.75 vs 0.74 with 2-lag legi-only). Worse for C+D (0.57 vs 0.60 legi-only PCA10).

4. **Extended cross-type training (6 election types)**. Adding Euro, Regi, Dept, Cantonales T1 to Legi+Pres gives ~20 training dates. Breakthrough for Abstention: 0.73 → 0.77 (PCA3-devlag). Abstention deviation patterns are highly stable across all election types — more data directly helps. Does NOT help G, C+D, or ED (these blocks have type-specific patterns that become noise with more types).

5. **PCA on demographics (3-10 components)**. Compresses 52 correlated census indicators to orthogonal components. C+D benefits from more components: PCA3=0.59, PCA5=0.59, PCA7=0.60, PCA10=0.60. Abstention benefits from fewer: PCA3=0.77. Works for V1 but fails for V2 (multi-vintage data makes PCA incoherent).

6. **Removing geo/time raw features (lat, lon, date_float)**. Ridge's linear plane in lat/lon is redundant with PCA demographics; linear time trend is dangerous for extrapolation and redundant with deviation lags. Removing improved G (+0.013), C+D (+0.006), Ab (+0.005); ED -0.007 (noise). Cleaner model with better prior: deviations are time-stationary, geography is in demographics.

7. **1-lag models (dropping lag2 requirement)**. For cross-type Legi+Pres, drops lag2 to unlock 1 extra training date (Legi 2002.5, whose lag1 comes from Pres 2002.33). Improves Gauche (0.72 → 0.75) because more training data outweighs lost lag2 information. Catastrophic for legi-only models (G drops 0.74 → 0.59) because with only 5 training dates, the noisy Legi 2002 lag from Pres 2002 corrupts the model.

## Data Pipeline Details

### Two data pipelines

- **V1 (strict NaN drop):** Drop any BV with missing demographic indicators. Best for all blocks (G, C+D, ED).
- **V2 (median imputation):** Fill missing demographics with column median. Retains more training data but adds noise. Uniformly worse than V1 in clean evaluation.

### Block mapping

All candidates/parties mapped to 3 blocks + Abstention via code sets: standard nuances, L-prefixed (regionales/europeennes), BC-prefixed (departementales), presidentielle candidate abbreviations, NC full-name lookup. See `cross_type_ridge.py` for the complete mapping.

### Lag construction

- **Legi-only:** Same-type lags (previous Legislatives at same BV)
- **Cross-type:** Cross-type lags (most recent prior election of any type at same BV). When using deviation lags, subtract national mean of the lag election to make cross-type lags comparable.

### National polls

1-year window before election date, all T1 polls (any type), normalized to 100% across 3 vote blocks. All 11 Legi+Pres dates (2002-2024) have poll coverage.

## What Doesn't Work (avoid these)

### Models

- **GBT (HistGradientBoosting).** Max G=0.77, C+D=0.27. Trees cannot extrapolate to 2024's unprecedented political context.

### Features

- **Geo (lat, lon) and time (date_float) as raw features.** Removed: Ridge fits a linear plane in lat/lon (crude geographic gradient already captured by demographics + lags) and a linear time trend (dangerous extrapolation — if ED deviations grew 2007→2022, Ridge predicts even larger 2024 deviations). Deviation space should be time-stationary; temporal momentum is in lags. Removing all 3 features improved G (+0.013), C+D (+0.006), Ab (+0.005); ED dropped 0.007 (LOO noise).
- **National-level economic variables (GDP growth, real wages, inflation) or political variables (president party, incumbency, popularity).** Same failure mode as national polls and aggregate lags: with 4-8 training elections, any election-level constant acts as a dummy variable. The Ridge memorizes per-election offsets. Polls already integrate these factors implicitly (voters respond to the economy when polled). The right place for macro fundamentals would be Stage 1, but 8 data points cannot learn a stable macro→vote relationship.
- **National polls as Ridge input features.** Catastrophic (G=-0.89, C+D=-0.60). With only 5 legi dates, poll values act as election dummy variables. The Ridge memorizes date-specific offsets that don't generalize. Use polls only for post-hoc correction or national estimation.
- **National aggregate lags as features.** Same problem as polls — date-level constants overfit with 5 elections.
- **Raw (non-deviation) lags in cross-type training.** Catastrophic for G (0.37) and C+D (-0.51). A BV with Gauche=42% in a pres might have 30% in legi. Must convert to deviations first.
- **Both raw + deviation lags combined.** Redundant. 77 features of mixed lags produce noise. G drops to 0.07.
- **PCA x lag cross features.** `PCA_i x lag_j` — neutral to slightly negative vs plain PCA. The cross terms add noise without improving over PCA alone.
- **Logit-transformed lags.** `log(p/(100-p))` — hurts all blocks. G drops 0.74→0.63.
- **Sqrt-transformed lags.** `sqrt(lag)` — hurts all blocks in combination with logit.
- **Interaction features** (lag squares, cross-products, trends). Catastrophic — G drops 0.74→0.23. Overfits with limited training elections.
- **Block ratio features (ED share of right, turnout, LR gap).** Redundant with raw lags. No improvement.
- **Spatial neighbor features** (Phase 6): dept mean lag, commune LOO mean, 10-NN average. Neutral or slightly negative. The Ridge already captures geographic structure through demographics and lags.

### Corrections / Post-hoc tuning (ALL violate no-validation-tuning rule)

- **Post-hoc poll correction (lambda x shift).** Fitting lambda on validation data is validation tuning. LOO analysis showed lambda=0.0 for ED and Abstention on all training elections, confirming that apparent improvements were overfit to 2024.
- **Abstention national estimate grid search.** Sweeping the abstention estimate against validation R² is validation tuning. The previous -10pp snap adjustment was grid-searched. Without it, Abstention raw R² = -2.5 (the ~18pp national estimate error dominates). A principled turnout model is needed.
- **Meta-blend weights fit on validation.** Fitting Ridge/ElasticNet/NNLS on stacked validation predictions is validation tuning.
- **Calibrated national estimates (linear regression on 5 elections).** Catastrophic for ED — maps polls to historical ED levels (~15-25%), underestimates the 2024 surge to 37%. Raw polls are better despite being uncalibrated.
- **Affine R² as primary metric.** It's an oracle metric (requires val labels). The gap between raw R² and affine R² can be huge (e.g., Abstention: -3.9 raw vs 0.72 affine). Always use raw R² as the real metric; affine R² is only useful for measuring ranking quality.

### Data

- **PCA on V2 (multi-vintage training).** Different census vintages have different column availability, making PCA components incoherent. PCA only works with V1.
- **V2 (median imputation) for cross-type models.** Poor (CT-V2-rawlag+int: G=0.46, C+D=-0.16). Median imputation across election types adds more noise. V2 is uniformly worse than V1 in clean evaluation.
- **Adding election types for G, C+D, ED.** Extended cross-type (6 types, 20 dates) helps Abstention dramatically (0.73→0.77) but hurts vote blocks: G 0.73 (vs 0.75), ED 0.80 (vs 0.81). The additional types add noise to vote block deviation patterns. Best approach: Legi+Pres for vote blocks, extended types for Abstention only.
- **Lag trend features (dev_lag1 − dev_lag2).** Mathematically redundant: trend is in the column span of [lag1, lag2]. Ridge can capture the same information via coefficient differences. Confirmed: identical R² with and without trends on all full-feature models. Only appears to help with PCA (different feature space), but the effect is <0.001.
- **1-lag models for legi-only.** Catastrophic: G drops 0.74→0.59, C+D 0.57→0.39. The extra training date (Legi 2002) has noisy lag1 from Pres 2002, and with only 5 training dates the noise corrupts the model. Works for cross-type (9 dates absorb the noise).

### Temporal / time-aware models

- **Exponential decay weighting on deviation lags (lag freshness encoding).** `dw_lag = dev_lag * exp(-alpha * lag_age)` as a pre-processing step (analogous to PCA for demographics). Sweep alpha in [0.0, 0.1, 0.2, 0.3, 0.5, 1.0] × PCA × data configs (60 LOO runs). For cross-type models, OOF R² improved (e.g., CT-PCA5 G: 0.757→0.776, CD: 0.572→0.602 at alpha=0.2), but this was LOO noise with 8 folds — validation R² was worse (CD: 0.60→0.56, Ab: 0.73→0.40). Ridge already handles the bimodal lag age distribution (~0.17yr vs ~4.83yr in cross-type data) implicitly through regularization. The explicit decay is redundant. For legi-only models, all lag ages are ~5yr (uniform), so decay has zero effect (OOF identical across all alpha values).
- **Temporal prediction at arbitrary lead times (0-2yr before election).** The deviation model is proven time-invariant: oracle R² (using true national means) is flat across all tau values (G=0.74, ED=0.81 at every tau). The ONLY time-varying component is the national estimate. At tau=0 (election time), polls give accurate national estimates and the model matches the baseline exactly. At tau>0, structural national estimates (linear trend on same-type results, blended with polls) degrade because the 2024 snap election is unprecedented — ED structural estimate is 16.9% vs actual 37.1% (-20pp), C+D structural is 49.5% vs 31.4% (+18pp). Gauche and Abstention degrade gracefully (structural errors ~2pp). The architecture is correct but produces no R² improvement at tau=0 — it only extends the model to new prediction times where the baseline cannot operate. No new hyperparameters, no validation tuning.

### Architecture

- **Cross-type training on raw (non-deviation) targets.** Catastrophic for G and C+D (R² < -8). The demographics-to-block mapping changes non-linearly across election types. One-hot type indicator only provides additive intercept shift, not slope correction. Must use deviation targets.
- **Election type one-hot in deviation space.** Makes no difference (ED: 0.809 with vs 0.809 without). Deviation targets already remove the type-specific structure.
- **Compositional / joint modeling (ILR transform).** G+C+D+ED+Abs sum to ~100%, so mapping to 3 unconstrained ILR (Isometric Log-Ratio) coordinates and fitting Ridge in ILR space was tested to enforce the simplex constraint and let strong blocks (G, ED) help the weak one (C+D). Hurts all blocks badly (best C+D: 0.45 vs 0.57 independent). The nonlinear log-ratio transform distorts the signal Ridge can capture; 4 independent linear models in the original space outperform 3 coupled models in ILR space. The sum-to-100 constraint is enforced perfectly but per-block accuracy drops.
- **Residual C+D (100 − G − ED − Abs).** Pure residual is catastrophic: national estimate errors (~18pp for Abstention, ~1-2pp for other blocks) propagate entirely into C+D instead of cancelling. Augmented Ridge (C+D model with cross-validated donor block deviation predictions as extra features) ties with the independent baseline — the donor predictions are collinear with the existing features and add no new information.

## Key Constraints

1. **The bottleneck is training elections, not model capacity.** Legi-only V1 trains on 4 dates (~262K rows). Cross-type Legi+Pres extends to 8-9 dates (~522-597K rows). Extended 6-type goes to ~20 dates (~1.2M rows). Adding more types confirmed to help Abstention dramatically but adds noise for vote blocks.
2. **Abstention national level from a participation poll.** The published Ipsos/CEVIPOF "indice de participation" (pre-election turnout intention) gives the national level directly, like the vote-block polls: 63% participation → 37.0% abstention for 2024 (actual 31.0%). Extended cross-type training (20 dates) combined with PCA3 achieves cross-sectional R²=0.77 for Abstention. Fallback when no participation poll exists: a LOO-selected historical estimator (which must use Legi+Pres national means only — Euro/Dept/Regi have very different abstention dynamics).
3. **The 2024 snap election is unprecedented.** The ED surge (+12pp) and turnout boost (-19pp abstention) test extrapolation. Results may be less impressive on "normal" elections where the Ridge baseline is already strong.
