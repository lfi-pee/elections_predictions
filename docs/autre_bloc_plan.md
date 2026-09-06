# Build plan — modelling the off-axis "Autre" bloc (LOO-gated)

Status: **Phase A COMPLETE 2026-09-06 → WEAK signal, borderline NO-GO.** Measured
on the correct metric (out-of-fold **share R²**) through the PRODUCTION model
(`cross_type_dev`) LOO: Autre R² = **0.141**, far below the four modelled blocs
(G 0.76, CD 0.60, ED 0.82, Abstention 0.79). Positive (not noise) but a weak bloc.
See §9c for the definitive result; §9b (crude circo R²≈0.43) was optimistic.
Author context: follows the LOO experiment of 2026-09-05 (see bottom).

> **TL;DR (definitive).** Judge by OOF **share R²**, not seats. Production-model
> LOO: Autre R²=0.14 vs 0.60–0.82 for the real blocs. Autre is weakly predictable —
> real signal (the persistent strongholds) but well below the modelled blocs, so a
> published Autre seat would be low-confidence (wide conformal band). Not a slam
> dunk; the invasive 4-pole build is hard to justify on R²=0.14. Judgment call.

## 1. Goal

Let the forecast **publish** the handful of seats currently grayed out as "non
publiable" because an off-axis force (Corsican nationalists, DOM autonomists,
local parties) dominates — by modelling that force as an explicit 4th electoral
bloc **Autre**, instead of leaving it as an unmodelled residual and refusing to
call the seat.

Success = these seats get an honest winner call + winnability score that clears
the project's **LOO** gate, **without** degrading the 567 seats that already work.

## 2. What the LOO experiment established (evidence, not aspiration)

Leave-one-election-out across {2017, 2022, 2024} legislative folds, using the
**production** `cross_type_ridge` bloc mapping (not the reduced
`backtest_2024_seats._bloc`, which is 2024-centric and caused a taxonomy
artifact in the first pass):

| Metric (OOF) | Value |
|---|---|
| National "Autre" share, per cycle | 3.1% / 3.3% / 1.8% (stable, genuine residual) |
| Autre-decisive seats recovered, 3-bloc → 4-bloc | 0/24 → **7/24** (naive persistence) |
| All-seats winner calls, 3-bloc vs 4-bloc | 1027 vs 1029 (no degradation) |
| False positives (Autre called on a real G/CD/ED seat) | 9/1707 = **0.53%** |
| Autre share OOF MAE | 2.2–2.8 pp (in-band with model halfwidths ±6–9.7) |

**Reading:** safe (no harm to mainstream seats, tiny FP rate), but a *naive*
persistence estimator only recovers ~29% of decisive seats. The upside is
concentrated in genuinely persistent strongholds (Corsica). A smarter estimator
is what decides whether this is worth the invasive runoff changes → **Phase A is
the go/no-go gate.**

## 3. Architecture today (what a 4th bloc touches)

- **Target / deviations**: `cross_type_ridge` predicts per-bureau deviations for
  Gauche / Centre+Droite / Extreme_Droite; `forecast_2027` emits `dG/dCD/dED` and
  already a 4th component `dAB` (abstention). **Autre would mirror the `dAB`
  plumbing** — the pipeline already carries a non-{G,CD,ED} quantity end-to-end.
- **Winner logic**: `winnability_2027.seat_winner(g, cd, ed, ab, …)` runs a
  bespoke 2nd-round runoff among **three** poles — qualification at 12.5%/turnout,
  front-républicain désistement, ED→left/CD barrage transfers. This is the piece
  with no notion of a 4th pole. **This is the real work.**
- **Uncertainty**: `conformal.py` derives intervals from LOO residuals for the 3
  blocs; winnability 1–5 (`score_circo`) reads them.
- **Serving**: `report_data_2027` writes `circo.json` (`dG/dCD/dED/dAB`, `r24*`)
  and `winnability_distribution` loops `seat_winner`.
- **Guardrail**: `coverage_2027` marks a seat non-publishable when off-axis vote
  exceeds the model's halfwidth. A seat now *modelled* for Autre must flip to
  publishable (its coverage effectively → ~100%).
- **JS parity**: `report_app/2027/js/{winnability,map,panel,config}.js` recompute
  the same model; `test_parity_2027` enforces Python↔JS equality. A 4th bloc =
  parallel JS changes + parity update.

## 4. Design decisions (the parts that need a human call)

**D1 — Autre share estimator (Phase A).** Define `Autre = 100 − (G+CD+ED)` of
expressed, as an explicit modelled share. Naive persistence (mean of other
folds) = 7/24. Candidate improvements to test under LOO:
  - per-territory incumbency persistence (Autre carried at the *incumbent's* prior
    share, not a cross-fold mean that dilutes across regime changes);
  - compose with the existing **attribution table** for partisan flips (William,
    Rimane, Tjibaou are G, not Autre — the table already says so);
  - shrink toward 0 outside the ~6 territories with a real off-axis tradition.

**D2 — Autre in the runoff (`seat_winner`).** DECIDED (2026-09-05): model Autre as
a **"sticky incumbent" anti-RN pole** — others barrage to it vs ED, it rarely
transfers out — and **let LOO arbitrate**. Deliberately few knobs to avoid
overfitting the ~24-seat sample; LOO winner accuracy must come out monotonically ≥
the 3-bloc baseline or the runoff rules are rejected. No per-territory
hand-tuning.

**D4 — Scope. DECIDED (2026-09-05): the principled, no-hardcode solution.** Do NOT
enumerate "off-axis territories" (that is itself a hardcode and would trip the
project's no-hardcode ethos / `test_no_hardcoded_2027`). Instead model Autre
**everywhere**, but with **LOO-validated shrinkage** so its predicted share
self-collapses to ≈0 wherever there is no persistent off-axis signal. The data
decides where Autre is real — no curated list, no arbitrary boundary. This keeps
the low false-positive behaviour of a restricted scope while staying data-driven.

**D3 — Corsica.** Modelling Autre *supersedes* the deliberate `NA` for the four
Corsica seats (that was a stopgap precisely because there was no 4th bloc).
Confirm this is intended (it is the point of the exercise).

## 5. Files to touch (concrete)

| File | Change |
|---|---|
| `src/cross_type_ridge.py` | Add Autre as a modelled output (mirror `dAB`); mapping already yields off-axis→Autre. |
| `src/forecast_2027.py` | Emit `dAutre` in `predictions_2027.csv`. |
| `src/conformal.py` | LOO conformal residuals for Autre → interval. |
| `src/winnability_2027.py` | `seat_winner`/`score_circo` gain a 4th pole + D2 transfer rules. |
| `src/report_data_2027.py` | Serve `dAutre`, `r24Autre`; `winnability_distribution` passes Autre. |
| `src/coverage_2027.py` | Modelled-Autre seats become publishable; re-scope threshold/marking. |
| `src/attribution_2027.py` | `cov_avant/apres` semantics revisited (Autre now modelled, not only measured). |
| `report_app/2027/js/{winnability,map,panel,config}.js` | 4-bloc winner/colour/panel + `test_parity_2027` update. |
| `src/backtest_2024_firstround.py`, `backtest_2024_seats.py` | 4-way majority/winner (rappel only). |

## 6. Validation gates (pre-registered, LOO — **not** 2024)

1. **Autre share**: LOO OOF MAE in-band (≤ ~3 pp, already met at 2.6).
2. **No degradation**: LOO all-seats winner accuracy on the mainstream seats ≥
   current 3-bloc baseline (1027/1731 in the scouting run — reproduce as the
   formal baseline first).
3. **Materiality (Phase A gate)**: smarter estimator must recover materially more
   than 7/24 (propose threshold ≥ ~12/24) to justify Phases B–D.
4. **FP rate** on real G/CD/ED seats ≤ ~1%.
5. `test_parity_2027`, `test_no_hardcoded_2027`, `test_coverage_2027` green.
6. 2024 backtest reported as **rappel only** — a single test year does not
   override the LOO (per the pre-registered rule, `conformal.py:75-76`).

## 7. Sequencing (phased, each phase is a stop/go)

- **Phase A — estimator + LOO (no serving).** Build the Autre share model, run
  LOO, apply gate #3. *If it doesn't beat 7/24 materially → stop, keep gray-out,
  document.* ← cheapest decisive step.
- **Phase B — `seat_winner` 4th pole + LOO winner revalidation** (gates #2, #4).
- **Phase C — conformal intervals for Autre** (gate #1 extended to intervals).
- **Phase D — serving + JS parity + coverage-guardrail flip** (gate #5).
- **Phase E — full `rebuild_2027.sh` green**, 2024 rappel printed.

## 8. Sign-off status

1. D2 runoff behaviour — **DECIDED: sticky incumbent, LOO arbitrates** (§4 D2).
2. D3 — modelling Autre supersedes the Corsica `NA` — **assumed yes** (it is the
   point of the exercise); flag if not.
3. Phase A materiality threshold — **proposed ≥12/24**; open to a different bar.
4. Scope — **DECIDED: principled no-hardcode shrinkage, modelled everywhere** (§4 D4).

---

## 9. Phase A (2026-09-06)

### 9a. First pass — WRONG METRIC (retracted)
Judged the bloc by seat-winner recovery under LOO: naive mean 7/24, persistent
floor 4/24, consistency shrink 4/24; only 1 of 10 decisive seats (ZB-04) is
Autre-persistent, the rest flips/emergences already reassigned to Gauche by the
attribution table. This led to a NO-GO — **but seat recovery is not how this
project validates.** Retained only as a note on why seats mislead: winner-calls in
~10 knife-edge runoffs are thresholded and noisy, and miss that Autre is predictable
in magnitude.

### 9b. Corrected — OOF SHARE R² (the project's actual gate)
Voter-weighted, pooled across LOO folds, crude LOO-mean estimator:

| Bloc | weighted OOF R² | unweighted R² | MAE | share sd |
|---|---|---|---|---|
| G | 0.615 | 0.577 | 5.39pp | 11.5pp |
| CD | −0.140 | −0.071 | 12.85pp | 13.9pp |
| ED | −0.082 | −0.002 | 11.23pp | 12.9pp |
| **AUTRE** | **0.430** | **0.458** | 2.27pp | 6.0pp |

**Autre is predictable OOF (R²≈0.43)** — better than CD/ED under the same estimator.
The negative CD/ED values reflect the crude cross-fold mean being a poor estimator
for volatile blocs (the production model uses lag + national-swing features and does
far better); so 0.43 is a **lower bound** on Autre's R² inside the real model.

Adding Autre does **not** degrade G/CD/ED: those shares are already computed with
Autre in the denominator (they sum to <100 in the strongholds), so making Autre
explicit leaves the other three unchanged.

### 9c. DEFINITIVE — production-model LOO share R² (`src/autre_oof.py`)
Autre promoted to a block via the module globals `cross_type_dev` reads
(`BLOCKS_ABS`/`TARGET_COLS`), so production's own functions build `dev_Other` + its
full lag machinery; base rebuilt (cache restored after). LOO = leave-one-cross-type-
election-out, scored per-BV on held **legislative** folds, oracle national mean:

| block | OOF share R² | MAE | share sd |
|---|---|---|---|
| Gauche | 0.759 | 5.72pp | 16.5pp |
| Centre+Droite | 0.597 | 7.49pp | 16.7pp |
| Extreme_Droite | 0.822 | 4.47pp | 14.6pp |
| Abstention | 0.792 | 3.84pp | 11.6pp |
| **Other (Autre)** | **0.141** | 2.89pp | 6.6pp |

The four modelled blocs land at plausible production R² (0.60–0.82) — this validates
the harness. **Autre = 0.141: positive (real signal — the persistent strongholds)
but ~4–5× weaker than any modelled bloc.** The crude circo-level 0.43 (§9b) was
optimistic; the faithful BV-level production LOO is the number to trust.

**Decision (go/no-go gate = "R² comparable to the other blocs"): NOT cleared.**
Autre is a *weak* bloc. Modelling it is defensible (R²>0, not fabrication) and beats
treating it as an unmodelled residual — but a published Autre seat would carry a wide
conformal band / low winnability confidence, exactly where it matters (the strong-
holds). The invasive 4-pole runoff + serving + JS-parity build is hard to justify on
R²=0.14. Recommendation: **do not build now**; if revisited, first raise Autre's R²
(territory-aware features) and re-clear this gate. Keep attribution table + gray-out.

### Appendix — LOO experiment provenance (2026-09-05)
Bureau-level `candidats_results.parquet` (2017/2022/2024 `_legi_t1`), mapped to
circo via `bv_master.parquet`; blocs via production `attribution_2027.block_sets()`;
Autre = residual; LOO = leave-one-legislative-election-out, mean estimator.
First pass used `backtest_2024_seats._bloc` (2024-centric) and produced a spurious
50% Autre in 2017 (REM/FN/NUP unmapped) — corrected to the production mapping.
