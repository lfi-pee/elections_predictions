# Préconisations: where to take this model next

Based on:
- Settled experiments in [`algorithm.md`](algorithm.md) (residual XGB, GP, splits, transforms, etc.)
- Three Stage-2 mods tested in this session: voter-weighting, Centre/Droite split, commune random-effect
- LOO + 2024-val residual diagnosis (slices by département / inscrits / |dev_lag1| / k-means demographic cluster)

The headline: **the remaining error is not a model-capacity problem.** Adding nonlinear capacity, intra-block splits, or generic shrinkage produces gains within LOO noise. Real headroom is in three concrete places below.

---

## 1. The current model is bottlenecked by special territories

Across **all four blocks**, the same départements dominate the worst-residual list:

| Dept | Description | G err | CD err | ED err | Ab err |
|---|---|---|---|---|---|
| 98 | Polynesia / Wallis | -11pp | -9pp | **-25pp** | **-19pp** |
| 2A / 2B | Corsica | -10 / -18 | -9 | -9 / -13 | -9 / -11 |
| 97 | DOM (Antilles, Réunion, Guyane, Mayotte) | -12 | +6 | -6 | +2 |
| ZZ | French abroad | +8 | — | -5 | +3 |
| 65, 09, 48, 52 | Pyrénées / Lozère ultra-rural | -7 to -14 | -10 to -16 | -7 | -4 |

Magnitudes are extreme: ED in dept 98 predicts ~27% vs actual ~2%. **These ~3,500 BVs (~5% of validation) account for a disproportionate share of total RMSE.**

### Recommendation 1 (highest leverage, lowest cost)
Add explicit indicators `is_DOM`, `is_Corsica`, `is_abroad`, `is_micro_rural`. Fit Ridge as today; the indicators absorb the systematic offsets. Expected: +0.005 to +0.02 on R² for vote blocks, possibly more for ED. If even that doesn't suffice, consider a **separate model for these zones**, accepting that DOM-TOM may need different demographic predictors entirely.

---

## 2. Heteroscedasticity is real but `inscrits` weighting is too aggressive

Within-decile R² by BV size (smallest → largest):
- Abstention: **0.20 → 0.86** (largest gap)
- Centre+Droite: 0.40 → 0.79
- Extreme Droite: 0.65 → 0.85
- Gauche: 0.63 → 0.86

Naive weighting by `inscrits` was tested in this session — it **hurt Abstention by 3.5pp** because it down-weighted the smallest BVs to near-zero, removing coverage of low-population demographic patterns the model genuinely needs.

### Recommendation 2
Use `sqrt(inscrits)` weights, not `inscrits`. This compresses the weight ratio from ~10,000:1 to ~100:1, retains rural pattern coverage, and still down-weights the noisiest observations. Test on Abstention specifically.

---

## 3. Stage-1 (national mean) errors dominate "regime change" elections

LOO OOF R² per training election reveals what the deviation model *cannot* fix:

| | G | CD | ED | Ab |
|---|---|---|---|---|
| 2007 (Sarkozy) | 0.79 | **0.41** | **0.09** | 0.37 |
| 2017 (Macron) | **0.61** | 0.54 | 0.70 | 0.68 |
| 2022 | 0.72 | 0.57 | 0.72 | 0.78 |
| 2024 (val) | 0.74 | 0.61 | 0.80 | 0.74 |

Each block has its own "broken" historical election where lags from the prior election no longer transfer (Sarkozy unified the right; Macron split the left; FN was at its 2007 low ebb after the 2002 surge). 2024, by contrast, is the **easiest** year — it builds directly on stable 2022 patterns.

### Recommendation 3
Stage 1 — not Stage 2 — is the lever for handling future regime changes. The work in `bayesian_polls.py` (commit 1ffd648) is the right direction: propagate poll uncertainty into prediction intervals so we *know* when we're predicting blindly. No Stage-2 trick will fix a 20pp Stage-1 miss; the model needs to express that uncertainty rather than hide it.

---

## 4. The C/D split hypothesis is dead

Tested in this session: predicting Centre and Droite as separate sub-blocks and summing was **-0.006 vs baseline**. The diagnosis confirms why: CD's R²=0.61 is not driven by intra-block heterogeneity that a finer grain would resolve. It's driven by 2007 being unforecastable (per-election R²=0.41) and by the same special-territory + small-BV issues that hit every block.

**Don't revisit this without strong new evidence.** The Ridge with 60 features can already represent opposite-signed coefficients within a single target column.

---

## 5. The commune-RE win was largely a special-territory effect in disguise

Empirical-Bayes shrinkage of training residuals by commune helped vote blocks (+0.013 G, +0.025 CD, +0.019 ED) but hurt Abstention (-0.023). Cross-referencing with the residual-by-département table: the worst-residual départements are exactly the special territories, and the commune RE was effectively learning persistent commune-level offsets there.

**Implication:** explicit territory indicators (Reco 1) should subsume most of the commune-RE gain *and* avoid the Abstention regression. The commune RE is a generic patch over a specific structural problem.

---

## What is *not* worth pursuing further

Settled by experiments in this codebase or in `algorithm.md`:

- ❌ Tree models on residuals (XGB, GP) — no nonlinear demographic→residual signal
- ❌ Cross/interaction features — overfit on 4–8 train dates
- ❌ Logit / sqrt lag transforms
- ❌ Spatial neighbor features (dept mean, k-NN BV); also ❌ a learned-length-scale Gaussian kernel smoother on the Ridge OOF *residual field* over lat/lon (2026-05; script not kept). LOO picked the largest ℓ (50 km, grid max → mildest correction) for **every** block and OOF R² dropped in all four (G 0.797→0.788, CD 0.597→0.589, ED 0.816→0.810, Ab 0.907→0.904). Val deltas were inconsistent in sign and within the noise band (G +0.006, ED +0.017, but CD −0.024, Ab −0.038). self+nbr ≈ nbr-only to ~0.0003 → no residual per-BV persistence left; the dev lags already absorb the local/spatial signal. The deviation residual is spatially white.
- ❌ Compositional / ILR joint modeling
- ❌ Macro fundamentals (GDP, inflation, popularity) as features
- ❌ Voter-weighted Ridge (this session)
- ❌ Centre/Droite split (this session)
- ❌ Cluster-then-shrink on demographics (XGB-residual experiment already settled this)
- ❌ Non-inscrits / registration-rate feature (2026-05; **−0.001 to 0 LOO on every block in its production config — abandoned. See investigation below for the three evaluation errors that briefly made it look positive.**)

---

## Suggested ordering

1. ~~**`is_DOM`, `is_Corsica`, `is_abroad`, `is_micro_rural` indicators**~~ — **investigated 2026-05; rejected. See "Investigation: non-mainland fixes" below.**
2. ~~**`sqrt(inscrits)` weights**~~ — **investigated 2026-05; clean negative on every block. See below.**
3. **Tighter Stage-1 with poll-uncertainty propagation into intervals** — already in flight (`bayesian_polls.py`); finish and verify the uncertainty reaches conformal intervals. *This is now the only remaining principled lever.*
4. **Stop here for point R².** Further Stage-2 work has diminishing returns. Spend the time on prediction-interval calibration and on documenting failure modes per dept / per block, since those are what an end-user actually needs to know.

The model is genuinely close to the ceiling that 4–20 training elections allow. Diminishing returns from here on.

---

## Investigation: non-mainland fixes (2026-05-04)

A full pass through Recos 1, 2, and several variants of "do something different for non-mainland BVs" produced **zero validatable gains**. The investigation also surfaced a structural limit on what training data can validate. Documenting in detail so this isn't re-run hopefully.

### What was tried

All experiments use the same 4 LOO-selected configs from `preregistered.py`. Decision protocol: a rule is applied iff its LOO OOF R² beats the unmodified baseline on training. **No val tuning.**

| Variant | Mechanism | LOO verdict | Val Δ vs PREV (uniform) |
|---|---|---|---|
| Territory indicators | `is_DOM`, `is_Corsica`, `is_abroad`, `is_polynesia` as Ridge features | Selected models unchanged (LOO ±0.001) → no LOO gate to fail, applied | G +0.005, CD +0.012, ED −0.002, Ab **−0.016** |
| BV-level persistence | For non-mainland BVs: `pred = BV_lag1` | Refused unanimously (Δ ≈ −0.02 to −0.08 every block) | G +0.046, CD −0.022, ED +0.004, Ab −0.047 |
| BV-level dev-persistence | For non-mainland BVs: `pred = nat_2024 + dev_lag1` | Refused unanimously (Δ ≈ −0.01 to −0.03) | G +0.046, CD −0.020, ED +0.027, Ab +0.016 |
| `sqrt(inscrits)` weighting | Ridge with sample weights = √inscrits | Refused unanimously (Δ ≈ 0.000 to −0.003) | G +0.001, CD −0.007, ED ≈ 0, Ab −0.007 |
| Per-territory Stage-1 | `nat_X` per territory; mainland=poll, non-mainland=last-known same-type territory mean (territory-level persistence) | Refused unanimously (Δ ≈ −0.02 every block) | G **−0.075**, CD −0.034, ED −0.047, Ab **−0.323** |

The territory-indicator regression on Abstention is *the same mechanism* as every other failed variant: the territories' deviations are **not stationary** across regimes. A coefficient learned on 2002–2022 is wrong-signed for the 2024 snap.

### The dev-persistence "gain" is not a real gain

BV-level dev-persistence shows a +0.07 aggregate val improvement (3 of 4 blocks beat baseline on this single 2024 forward pass). **This is overfitting to one election. The rule does not generalize and should not be applied.**

The evidence:
- LOO refuses it on every training fold for every block, by Δ ≈ −0.01 to −0.08. Across 4–8 training years × 4 blocks (16–32 independent tests), the rule consistently *loses*.
- The val "win" is on n=1 election. With 4–8 LOO folds rejecting and one val pass accepting, the simplest explanation is sampling noise on a regime-shifted year, not a real structural improvement.
- The pretty story ("dev-persistence works because val is a snap election with short same-type lag") is post-hoc rationalization. There is no training-data evidence that the rule generalizes to *any* future snap election; it just happened to align with 2024's specific deviations.

**Lag-gap regimes available for training vs val:**

| Source | Lag-gap structure | Same-type as target? |
|---|---|---|
| Legi-only training (G, CD selected) | All folds: ~5-yr gap (Legi→Legi) | Yes |
| Cross-type training (ED, Ab selected) | 5 folds at ~0.17yr (Pres→Legi same year), 4 folds at ~4.83yr | **No** (Pres↔Legi) |
| **Val (2024 Legi snap)** | **2-yr gap, same-type** (Legi 2022 → Legi 2024) | **Yes** |

The 2-yr same-type combination **does not exist anywhere in training**. The cycle structure forbids it: regular Legi cycles are 5 years, and the cross-type 0.17yr pairs are different-type.

For non-mainland BVs, the same-type-ness of the lag is decisive — Pres→Legi candidates differ enough that BV-level deviations don't transfer (this is why BV-level persistence rmse for ED in special territories is ~23 in cross-type LOO folds even at 0.17yr gap). Val is same-type, so the deviations *do* transfer, and persistence wins. **Training has zero same-type short-gap folds. LOO is structurally blind to the regime val tests.**

### Why the obvious workarounds don't workaround

- **Restrict LOO to short-gap folds only** — the only short-gap folds are cross-type (different-type), so they don't represent val's regime either.
- **Construct same-type 2-yr fold synthetically** — would require off-cycle election types (Cant 2015 → Legi 2017 ≈ 2.25yr, but Cantonales have known block-mapping noise per [algorithm.md](algorithm.md)).
- **Build an "equivalent feature" LOO can validate** — tested two: territory indicators (rejected, val regression on Ab) and per-territory Stage-1 with territory-level persistence (rejected on LOO, large val regression). Aggregating to territory level reduces variance from BV noise but **cannot reduce variance from the regime shift itself** — DOM Abstention 2022→2024 dropped ~10pp because the snap turnout boost reached the territories too. Only Polynesia ED, which sits at ~2-5% historically and is genuinely decoupled, is structurally stable enough to benefit, and that's not enough cells to make a model viable.
- **Reframe the rule as parameter-free architecture** — devpers has zero learnable parameters, but applying it because val confirms is still val-informed and was rejected as such.

### Conclusion

**Dev-persistence is just bad — it doesn't generalize.** The +0.07 val gain is a single-election fluke; LOO's unanimous refusal is the trustworthy signal. Stop trying to recover it. No "equivalent feature," no clever LOO redesign, no extra training data short of multiple snap elections (which won't exist) makes this a real gain. The dataset cannot validate any Stage-2 fix targeting the snap regime, because the snap regime is fundamentally OOD relative to all available training and validation data. Treat the +0.07 as a warning about how much R² noise a single OOD election can produce, not as a missed opportunity.

The remaining principled work is Stage-1 uncertainty (Reco 3) — that direction doesn't depend on the missing regime. The bayesian_polls.py work in commit 1ffd648 is the right next step.

### Files touched, then reverted

- `src/cross_type_dev.py`, `src/preregistered.py` — territory indicators added then reverted (no diff in git after revert).
- `src/territory_persistence_exp.py` — investigation script, retained for reproducibility. Re-running it should produce the same negative result. Do not pursue further variants of this without new training data covering a 2-yr same-type lag regime.

---

## Investigation: non-inscrits / registration-rate feature (2026-05-29)

**Question:** does giving the model the *non-registered* population (eligible adults absent from the rolls — distinct from abstention, which is registered-but-didn't-vote) improve prediction?

**Answer: no.** A commune-level registration-rate feature gives **−0.001 to 0 LOO OOF R²** on every block *in its production model config*. Abandoned. (Three wrong intermediate conclusions were reached before this — they are kept below as a cautionary record of how to evaluate features wrong.)

### The feature

A commune-level **registration ratio** `reg_ratio = inscrits / pop_18+`:
- `inscrits`: BV inscrits (`general_results.parquet`) summed to commune, per election.
- `pop_18+`: INSEE census, approximated as `POP − POP0014 − 0.2·POP1529`. Census vintage matched to election year, floored at 2010 (earliest struct-pop file).
- Spread: mean 0.92, std 0.146. **Coverage ~95 %** per election type (see Error 3).

### Final result — LOO OOF R² per block, in its production config

| Bloc | Production config | Δ LOO (base → +reg_ratio) |
|---|---|---|
| Gauche | Legi-PCA5 | −0.0000 |
| Centre+Droite | Legi-PCA7 | −0.0001 |
| Extreme_Droite | CT-PCA5 | −0.0003 |
| Abstention | CT-PCA10 | −0.0010 |

Nothing clears the ±0.001 noise band; all lean slightly negative. The feature is not redundant in the linear sense (grouped-CV R² of `reg_ratio ~ demographics = 0.41`, so 59 % is "new" variance), but that new variance carries **no marginal predictive signal for the targets once the model has the full cross-type training set.**

### The three errors made before reaching that answer (cautionary)

This investigation reversed its conclusion three times. Each reversal was a textbook evaluation mistake; documented so they are recognised faster next time.

1. **Selected on the test set.** First pass compared base vs +reg on the *single 2024 val forward pass* and concluded "it hurts" (Ab −0.009 to −0.013). That is feature selection on the held-out year — exactly what this whole document forbids. The val pass is n=1 OOD noise; the decision metric is LOO OOF R² on training only.

2. **Read LOO on a silently-truncated dataset → false positive.** Switching to the LOO gate showed a *consistent +0.004 (Ab) / +0.002 (ED)* across 5 configs — looked robust, was written up as a real gain ("worth pursuing"). It was an artifact: the dataset had silently lost rows (Error 3), collapsing the "cross-type" configs to Legi-only. On Legi-only data, starved of the Presidentielle folds, `reg_ratio` *looked* useful.

3. **A float32/float64 join bug masqueraded as 52.9 % coverage.** `df.date_float` is float32 (Pres = 2002.3334); `reg.date_float` was float64 (2002.3333…). The merge silently dropped **every Presidentielle row** (the .5 Legislatives dates are exactly representable in both dtypes, so they joined; the .333 dates did not). This both (a) made coverage look like 52.9 % when it is really ~95 %, and (b) caused Error 2 by turning cross-type into Legi-only. Fix: cast `reg.date_float` to float32 before the join. With Pres restored, CT configs go 320k → 573k rows and the +0.004 evaporates to −0.001.

**Takeaways:** never read a feature delta without first asserting row count and per-fold coverage are unchanged vs base; a "robust across configs" gain is meaningless if all configs silently share the same truncation; and merging on floats across dtypes is a silent-data-loss trap.

### How it was proven (to reproduce — scripts were deleted)

The whole test reuses the existing pipeline; no new infra needed.

1. **Build the feature.** Sum `inscrits` to commune from `inscrits_lookup.parquet`; divide by `pop_18+` from `data/demographics/census/{vintage}/*evol-struct-pop*` (`POP − POP0014 − 0.2·POP1529`), reading pre-2017 `.xls`/`.xlsx` via `load_demographics._read_insee_file` (needs `xlrd`/`openpyxl` in the venv). Match census vintage to election year, floored at 2010. **Cast `date_float` to float32 before merging onto the model df** — this is the join that was broken.
2. **Sanity-gate the merge before reading any score.** Assert per-(election_type, date) coverage of the new column is uniform (~95 %) and that the row count fed to each model config is *identical* to the base run. (Skipping this is how the Presidentielle rows vanished and produced the false positive.)
3. **Score on LOO OOF R², never on 2024.** Call `preregistered.run_loo_and_val(name, df_subset, base_feats, est, national_means, cfg)` and again with `base_feats + ["reg_ratio"]` (appended after the demo block so PCA's `n_demo` is unchanged), on the *same* `reg_ratio`-present row subset for both. Compare `["oof_r2"]` per block.
4. **Use each block's production config**, not a fixed one: G→Legi-PCA5, CD→Legi-PCA7, ED→CT-PCA5, Ab→CT-PCA10. The CT (cross-type) configs are decisive for ED/Ab — that is where the Legi-only mirage disappears.
5. Result: Δ LOO = −0.0000 / −0.0001 / −0.0003 / −0.0010 (G/CD/ED/Ab). Cross-checked that the +0.004 mirage only survives when Presidentielle rows are absent (CT n drops 573k→320k) and that `reg_ratio ~ demographics` grouped-CV R² = 0.41 (new variance exists, but carries no marginal signal).

Abandoned on merit (null LOO at full coverage), **not** on coverage — coverage was fine. No tracked source files were modified; `xlrd`/`openpyxl` remain in the venv.

---

## 6. European elections — loaded into the γ machinery; observed differential still open (2026-05-30)

**Status.** The Europeans were **already in the raw data** (`general_results.parquet`:
`1999/2004/2009/2014/2019/2024_euro_t1`; `elections.parquet`: 11.5 M `Europeennes_T1` token rows) —
only absent from the *model* cache `cross_type_dev_base.parquet`. They are now wired into the
mobilization machinery via a dedicated γ panel `data/baseline_cache/gamma_panel.parquet`
(`movability_turnout._ensure_panel` → `cross_type_ridge._build_block_scores`, three T1 types),
**without touching the production model** (still trained on legi+présid only). Outcomes: a third γ
regime (euro 23.9 %, between legi 39.3 % and présid 12.3 %), a third curve on the site, and a
3-election abstention floor. **Open item:** the *observed* Pres−Eur differential as a validation
layer (below).

**Motivation (client feedback, end-of-campaign GOTV use):** the deliverable's Band 2 now
distinguishes **structural** abstention (chronic non-voters) from **conjunctural** abstention
(people who vote in high-salience contests and skip the rest) — the latter being the mobilizable
target. Today the split is approximated by each bureau's historical abstention floor
(`report_data.attach_abst_floor`), measured on **only two election types present in the data:
`Legislatives_T1` and `Presidentielle_T1`** (verified in `cross_type_dev_base.parquet`). There are
**no European elections loaded.**

**The idea (Bompard):** `Gauche(Présidentielle 2022) − Gauche(Européennes 2024)` at bureau level
isolates the **Left voters present at the high-turnout contest but absent at the low-turnout one** —
i.e. the conjunctural Left reservoir, *measured rather than modelled* (γ is an estimated slope; this
is an observed differential, a stronger proof for the client).

**Action.** Load **Européennes 2024** (and Présidentielle 2022) at bureau level via
`download_elections.py` / `load_elections.py`, key on `id_brut_miom` (100 % join with the master,
as the existing two types already achieve). Then:
1. recompute `abst_floor` over three election types (sharper structural/conjunctural split);
2. compute the Pres−Eur Left differential per bureau as a **measured** conjunctural reservoir, to
   cross-check the γ-based `mv` and to surface in the deployment panel.

**Caveat to respect.** Cantonales-style block-mapping noise does not apply to Européennes (clean
list-level Left/RN/macronist mapping), but turnout regimes differ — keep the γ curve **per election
type** (`MOVABILITY.md` §15); do not pool European transitions into the legislative curve.

## 7. IRIS sub-commune resolution & residential-mobility features (2026-06-06)

**The single largest untried data lever — sub-commune demographic resolution — and the
last untapped census theme (residential churn) were both tested. Neither is a bankable
gain. Confirms the §1–3 headline: the ceiling is training-election count + Stage-1, not
cross-sectional demographic detail.**

### What was built (kept; `src/iris_features.py`, `src/mobility_features.py`,
`iris_experiment.py`, `mobility_experiment.py`)
- IRIS census (vintages 2013–2022, avail 2016.5–2025.5) loaded by reusing the 52 validated
  commune indicator-builders verbatim (patched reader synthesises `CODGEO` from `IRIS`),
  then area/population-weighted onto BVs via the **pre-existing** `data/geo/bv_iris_weights.parquet`
  (68,623 BVs). Today demographics are joined at **commune** level, so every BV in a commune
  shares one demographic vector — IRIS gives genuine within-commune resolution.
- Residential mobility: IRAN (résidence antérieure → recent-mover / arrived-from-other-commune
  shares) and ANEM (ancienneté d'emménagement → short-tenure shares). 4 commune indicators,
  full coverage back to 2009.5.
- All evaluated through the **unchanged** pre-registered LOO harness (`run_loo_and_val`),
  CT Legi+Pres 2-lag V1, PCA∈{none,5,7,10}, LOO-OOF selection + single 2024 val pass.

### Results (val R² of LOO-selected model; baseline = commune, same harness)
| design | G | CD | ED | Ab |
|---|---|---|---|---|
| commune_full (8 dates) | 0.7179 | 0.5630 | 0.8020 | 0.4102 |
| IRIS-fallback level (full) | 0.7192 | 0.5707 | 0.8033 | 0.4145 |
| commune + within-commune Δ (full) | 0.7283 | 0.5624 | 0.8031 | 0.4217 |
| commune + mobility (full) | 0.7173 | 0.5703 | 0.8057 | 0.4155 |
| commune_recent (≥2017, 4 dates) | 0.6793 | 0.5199 | 0.8098 | 0.3698 |
| IRIS-fallback (recent, same rows) | 0.6803 | 0.5040 | 0.8094 | 0.3737 |

### Verdict — not bankable, but informative
- **On the OOF selection metric (the honest one), full-sample deltas are all ±0.002 = noise.**
  The strict no-val-tuning rule therefore cannot *select* any IRIS/mobility variant, even
  where val nudges up (delta_full: G val +0.010, Ab val +0.012 at flat OOF; IRIS-fallback CD
  val +0.008).
- **Resolution genuinely carries signal** — isolated on the same recent rows, IRIS resolution
  lifts the *training-honest* OOF for CD (+0.022, 0.574→0.595) and G (+0.011). But with only
  4 recent training dates the OOF→val transfer is unstable (CD val −0.016 there). Real signal,
  drowned by the fold-count noise floor that §1–3 already identified.
- **Mobility is redundant** with existing demographics (HLM %, renters %, single-person
  households already proxy churn): every OOF delta ≤0.001.

### Implication
The IRIS pipeline is worth keeping for a **product** reason, not an R² one: it gives BVs within
a commune distinct demographic profiles (neighbourhood-level GOTV targeting), which the
commune join cannot. But for point R² it changes nothing the lag features don't already capture.
Do not pursue finer cross-sectional demographics further. The remaining levers are unchanged:
Stage-1 / turnout-intention microdata for the abstention national level, and more elections
(time only).

---

## 8. Européennes as extra training folds (2026-06-06)

**The "more elections raises the ceiling" lever was tested in its sharpest form — add
européennes as deviation-model folds — and rejected. Adding euro folds degrades
out-of-sample prediction of legislative/presidential elections in every block. The
ceiling is fold count *of the same regime*, not fold count.**

### Motivation
§1–3 name training-election count as the binding constraint. The existing `Ext-*` configs
(`beat_it.py`) bundle euro with régionales+départementales+cantonales, never isolating euro.
Européennes are the one extra scrutin that looks like a national bipolar list contest
(single national constituency), so a priori they should transfer where hyper-local scrutins
don't. Tested in isolation (`europeennes_experiment.py`, `europeennes_oof.py`).

### What was built (kept)
- `europeennes_experiment.py`: Legi+Pres vs Legi+Pres+Euro through the pre-registered
  PCA-grid LOO harness. Euro 1999/2004 excluded (>15% unmapped 'Other'); 2024 excluded
  (val overlap). Euro folds 2009/2014/2019, ~201k BV rows, block-mapping 93–96% (clean).
  Confound-controlled: both conditions scored on the **same** 2024 val rows (69,359;
  location intersection), since euro folds change which 2024 BVs have complete 2-lags.
- `europeennes_oof.py`: fold-fair OOF — for each held-out Legi+Pres fold, base trains on
  (LP folds \ f), euro trains on (LP folds \ f) + euro folds, both predict the SAME held
  rows. The only cross-condition-comparable OOF.

### Results
| | G | CD | ED | Ab |
|---|---|---|---|---|
| **2024 val** (same rows) base→euro | 0.718→0.731 | 0.563→0.604 | 0.802→0.808 | 0.408→0.440 |
| **Fair OOF** (held-out LP folds) base→euro | 0.758→0.689 | 0.572→0.534 | 0.816→0.789 | 0.907→0.876 |

The 2024 val improves in every block (+0.01 to +0.04) — tempting. But the fair OOF, pooled
over 2002–2022 LP folds, is **worse in every block** (G −0.069, CD −0.038, ED −0.026,
Ab −0.031). Base-column OOF reproduces the pre-registered baseline exactly → harness sound.

### Verdict — not bankable; single-year val luck
- 2024's legislatives were the snap election Macron called **by dissolving after the June
  2024 européennes** — an unusually euro-like legislative. Euro folds helped that one year
  while degrading the LOO average. This is precisely the single-year overfit the
  pre-registered OOF rule guards against; selection on OOF rejects euro for all four blocks.
- Mechanism: européennes are a second-order election (~50% abstention, list-based,
  protest-vote dynamics) — a different regime. Adding their rows biases the Ridge
  coefficients away from the legislative/presidential regime. Confirms the §6 γ-per-scrutin
  finding and explains why the bundled Ext-* configs never won selection.

### Implication
The ceiling is training-election count **within the target regime** (Legislatives + the
closely-coupled Presidentielle), not raw election count. No additional French scrutin is
regime-compatible enough to add as a fold. This closes the "more elections" data lever for
point R². The only remaining principled levers are unchanged: Stage-1 / turnout-intention
microdata (abstention national level) and prediction-interval calibration.

### Follow-up: is the degradation a slope-pooling artifact? (no)

Hypothesis: the harm comes from one shared coefficient vector (type one-hot shifts only the
intercept; slopes are pooled), so euro rows drag the Legi+Pres slopes. Tested by giving euro
its own slopes — features interacted with a euro indicator, held Legi+Pres rows read the
Legi+Pres-anchored slopes (`europeennes_typed_oof.py`). Fair OOF on common held-out LP folds:

| | base | pooled | typed | typed−base | recovered |
|---|---|---|---|---|---|
| G | 0.759 | 0.684 | 0.693 | −0.066 | +0.009 |
| CD | 0.572 | 0.534 | 0.534 | −0.037 | +0.001 |
| ED | 0.816 | 0.789 | 0.794 | −0.022 | +0.004 |
| Ab | 0.907 | 0.876 | 0.878 | −0.029 | +0.002 |

Typing recovers only +0.001..+0.009 (≤~12% of the gap); typed stays well below base in every
block. **Not a pooling artifact.** The residual degradation is the cross-type lag channel:
euro folds also rewrite the LP rows' lags (a 2022 legislative's lag2 becomes the 2019
*européenne* deviation), and a second-order-election deviation does not map to a legislative
deviation. The slope interaction separates euro *rows* but cannot un-feed euro deviations from
the *lag inputs* of legi/pres rows. Both channels for "use européennes" are closed: as training
observations (hurts even typed) and as cross-type lags (degrades LP features). Confirms the
regime-mismatch reading; the lever stays rejected.

### Decomposition: which channel? (lag channel harmful; rows channel is an OOF win for CD)

Isolated euro-as-rows (channel A) from euro-as-lags (channel B) by rebuilding lags so
européennes are training rows but never a lag source (`europeennes_lagiso_oof.py`, LP-sourced
lags via strict-backward merge_asof). Fair OOF on held-out Legi+Pres folds (the task-correct
selection metric); the 2024 Legi forward pass (`europeennes_rows_val.py`) is reported as a
single unbiased number, NOT used for selection:

| | base | A: fair-OOF (selector) | A: 2024-val (report only) | B: lag channel (fair-OOF) |
|---|---|---|---|---|
| G | 0.758 | +0.005 | −0.024 | −0.074 |
| CD | 0.572 | +0.039 raw / +0.016 selected | −0.013 | −0.051 |
| ED | 0.816 | +0.002 | +0.010 | −0.029 |
| Ab | 0.907 | −0.003 | +0.023 | −0.028 |

**Channel B (euro injected into the cross-type lags) is robustly harmful** — −0.03..−0.08,
every block, both configs. A 2022 legislative's lag2 becomes the 2019 européenne deviation,
the wrong predictor for a legislative. Exclude euro as a lag source, always.

**Channel A (euro as extra training rows, lags kept LP-sourced) is OOF-SELECTED for CD**
(+0.039 raw / +0.016 under per-block-best-config selection, on a clean same-sample fair-OOF),
marginally positive for G/ED, tie/neg for Ab. By the pre-registered rule (select on OOF; the
test does not get to un-select), this is a legitimate Centre+Droite improvement. The 2024 val
(−0.013) disagrees, but a single test election is the noisiest possible signal and the protocol
forbids letting it override the OOF decision — earlier text that called channel A "noise" on
the basis of the val was a methodological error (test leakage into selection).

**Conclusion (corrected).** The user's "the model mishandles types" hypothesis was right about
the mechanism — it is the lag construction, not slope-pooling. With euro kept out of the lags,
euro-as-rows is OOF-positive for CD. Open scope before shipping: this fair-OOF compared against
an LP-only baseline on common rows, not against the production CD champion in `preregistered.py`.
To act on it, add "Legi+Pres+euro-rows, LP-sourced lags" as a candidate and select on the
held-out-LP OOF (NOT the harness's all-fold OOF, which for a mixed-regime candidate wrongly
scores held-out-européenne prediction). The §8 headline (naive euro folds rejected) stands; the
sharpened result is that the rejection was the lag channel, and a CD-only rows gain survives OOF.

### Selection head-to-head: euro-rows vs CT vs Legi-only (overturns the CD claim)

The channel-A "CD +0.039" was measured against a CT baseline held out over Legi+Pres folds —
not against the actual candidates on the task metric. Proper head-to-head (`europeennes_select.py`):
all three designs scored by held-out-**legislative** OOF (folds 2012/2017/2022 — the only legi
folds with 2 prior legi lags for the legi-only design), identical common held rows. Best config
per design:

| Block | Legi-only | CT (Legi+Pres) | Euro-rows | winner | euro−best-other |
|---|---|---|---|---|---|
| G | 0.7195 | **0.7652** | 0.7588 | CT | −0.006 |
| CD | 0.5571 | **0.6003** | 0.5998 | CT | −0.0004 (tie) |
| ED | 0.6830 | 0.7378 | **0.7402** | Euro | +0.002 |
| Ab | 0.7629 | 0.7758 | **0.7867** | Euro | +0.011 |

**The CD win does not survive.** Against the proper CT baseline, CD goes to CT; euro-rows ties
(−0.0004). The earlier +0.039 was the sample-size confound again (euro compared on a different
held-out fold set). **And Ab/ED euro edges are not robust:** Abstention is +0.011 here
(held-out-legislative) but −0.003 in the held-out-LP run — it flips sign with the held-out
definition. At 3 common legi folds every euro delta is ±0.01, inside the fold-count noise floor.

**Final verdict (OOF grounds, not test).** No design robustly beats CT on the task-correct OOF;
euro-rows trades ±0.01 by block and flips sign across legitimate held-out fold sets. Européennes
are not bankable — and this conclusion is reached *on the OOF selection metric itself*, not by
letting the 2024 test un-select anything. The §1–3 fold-count ceiling holds.

**Side-finding (separate thread):** CT (Legi+Pres) dominates Legi-only on ALL four blocks here
(CD 0.600 vs 0.557, G 0.765 vs 0.720). The production per-block champion is *legi-only* for CD/G
— possibly a sample-size-confound artifact in the original selection. Worth a clean re-selection
of CT vs legi-only on identical rows (more promising than européennes), but 3 folds is too thin
to act on without the full harness.

---

## 9. CT vs Legi-only — confound-free re-selection (2026-06-06)

**The §8 side-finding ("CT dominates Legi-only") was an ARTIFACT and is reversed. On a faithful
identical-rows comparison, Legi-only ≥ CT on the task-correct OOF for all four blocks — the
production legi-only choice for CD/G is vindicated. A real confound exists but runs the OTHER
way: production's CT OOF for ED/Ab is inflated by easy held-out presidential folds.**

The §8 `europeennes_select.py` "CT dominates" used SAME-TYPE lags for legi-only
(`prepare_condition([Legi])`) — not the production candidate. Production legi-only uses
CROSS-TYPE lags (a legi row's lag1 = the same-year présidentielle), differing from CT ONLY in
training rows. Reproduced faithfully (`select_ct_vs_legi.py`) on the production base, scored on
held-out-LEGISLATIVE folds (2007/2012/2017/2022), identical rows:

| block | legi-only OOF | CT OOF | Δ(CT−legi) | 2024-val Δ(CT−legi) |
|---|---|---|---|---|
| G | 0.796 (PCA5) | 0.761 | −0.036 | −0.018 |
| CD | 0.597 (PCA7) | 0.549 | −0.047 | −0.035 |
| ED | 0.794 (PCA5) | 0.774 | −0.021 | +0.038 |
| Ab | 0.804 (PCA5) | 0.786 | −0.018 | +0.020 |

(legi-only reproduces the docstring OOF: G 0.797, CD 0.596 → harness sound.)

- **Legi-only wins OOF on every block.** Adding presidential ROWS to training hurts held-out-
  legislative prediction. The production legi-only champion for CD/G is correct, not a confound.
- **The real confound, opposite direction.** Production picked CT for ED/Ab on an OOF (~0.816 ED)
  that `run_loo_and_val` inflates by also holding out easy presidential folds. On held-out-
  legislative only, CT-ED = 0.774 < legi-only 0.794. Task-correct OOF prefers legi-only for
  ED/Ab too.
- **OOF↔val tension for ED/Ab.** The 4-fold OOF prefers legi-only; the 2024 val prefers CT
  (ED +0.038, Ab +0.020). By select-on-OOF the pick is legi-only, but a 4-fold OOF contradicted
  by the single most-recent year is not a confident basis to flip a working model. FLAG, do not
  act without the full harness.
- **Mechanism.** Cross-type LAGS help (same-year pres is a strong legi lag1); cross-type training
  ROWS hurt (pres demo→deviation patterns differ from legi). Use pres as a lag source, not as a
  training observation — the same lag-vs-rows split found for européennes in §8.

**Net:** no new bankable point-R² lever; my CT lead was wrong. Ceiling unchanged.

**ACTED (2026-06-06):** per "use what LOO dictates", switched the deployed ED and Abstention
models from CT to **legi-only PCA5** in `conformal.BEST_RIDGE` and `shap_waterfall.BEST_MODELS`
(G/CD were already legi-only). `predictions_with_intervals.csv` regenerated. Selection follows
the task-correct held-out-legislative OOF (legi-only > CT for ED/Ab). Recorded trade-off: this
*lowers* 2024-val R² for ED (0.804→0.766) and Ab (0.415→0.395) — the single test year prefers
CT — but per the pre-registered rule the LOO selects and the test does not override. Full site
rebuild (`report_build`: geojsons, SHAP, provenance, figures) still pending confirmation.
