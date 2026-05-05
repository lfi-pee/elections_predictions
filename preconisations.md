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
- ❌ Spatial neighbor features (dept mean, k-NN BV)
- ❌ Compositional / ILR joint modeling
- ❌ Macro fundamentals (GDP, inflation, popularity) as features
- ❌ Voter-weighted Ridge (this session)
- ❌ Centre/Droite split (this session)
- ❌ Cluster-then-shrink on demographics (XGB-residual experiment already settled this)

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

### Why the val signal exists but cannot be taken

BV-level dev-persistence shows a real **+0.07 aggregate val gain** (3 of 4 blocks beat baseline). LOO refuses it cleanly. The gap is structural, not noise.

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

For non-mainland BVs, **the dataset cannot validate any Stage-2 fix that targets the snap regime**. The +0.07 val gain from dev-persistence is real and structurally explainable, but it's overfitting in the only sense that matters for this codebase. Stop trying.

The remaining principled work is Stage-1 uncertainty (Reco 3) — that direction doesn't depend on the missing regime. The bayesian_polls.py work in commit 1ffd648 is the right next step.

### Files touched, then reverted

- `src/cross_type_dev.py`, `src/preregistered.py` — territory indicators added then reverted (no diff in git after revert).
- `src/territory_persistence_exp.py` — investigation script, retained for reproducibility. Re-running it should produce the same negative result. Do not pursue further variants of this without new training data covering a 2-yr same-type lag regime.
