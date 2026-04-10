# Tabular Baseline Evaluation

Tabular baselines to benchmark the `UniversalMaskedSetTransformer` on BV-level election prediction.

## Experimental Setup

### Data Splits
*   **Validation:** 2024 Legislative T1 (`date_float = 2024.5`).
*   **Training:** All preceding Legislative T1 elections (2002–2022).

### Input Features ($X$)
*   **Demographic indicators (up to 47):** Census-derived, commune-level, temporally causal (`availability_date` < election date). Includes CSP breakdown (ouvriers, cadres, employés, professions intermédiaires, agriculteurs, artisans), education levels (sans diplôme, BEPC, CAP/BEP, bac, bac+2, bac+3/4, bac+5), employment by sector (agriculture, industrie, construction, tertiaire), age brackets (0-14, 18-24, 30-44, 45-59, 60+, 75+), unemployment rate, immigration rate, retiree rate, student rate, other inactive rate, housing (propriétaires, HLM, locataires, logements vacants, maisons, petits/grands logements, logement gratuit), household structure (ménages seuls, familles monoparentales, couples avec/sans enfants, familles nombreuses), marital status (célibataires, mariés, divorcés, veufs, pacsés, union libre).
*   **Geo coordinates:** Latitude/longitude per BV.
*   **Timestamp:** Continuous float (e.g. `2022.5`).
*   **Lagged block scores:** Same-BV block scores and abstention from the 1 or 2 prior Legislative T1 elections.

### Output Targets ($Y$)
Four regression targets (0–100%) per BV:
1.  **Gauche**: Sum of left-wing vote shares (SOC, COM, FI, NUP, ECO, VEC, EXG, DVG, UG, FG, etc.).
2.  **Centre+Droite**: Sum of center and right vote shares (UMP, LR, REM, ENS, DVD, UDI, MDM, HOR, etc.).
3.  **Extrême Droite**: Sum of extreme right vote shares (FN, RN, REC, EXD, MNR, DLF, MPF, etc.).
4.  **Abstention**: Abstention rate.

### Model
*   **Architecture:** StandardScaler + Ridge Regression (RidgeCV with LOO, `alpha` in `[10⁻², 10⁶]`).
*   **Script:** [`src/pca_ridge_baseline.py`](src/pca_ridge_baseline.py)

### Data Caching
*   First run calls the full loaders (~30 min) and saves processed DataFrames to `data/baseline_cache/{elections,demographics}.parquet`.
*   Subsequent runs load from cache in seconds.
*   Delete `data/baseline_cache/` to force a rebuild (e.g. after raw data changes).

### Evaluation
*   **Metric:** R² (coefficient of determination) on the validation set per target.
*   **Baseline vs. Network:** Aggregate the transformer's candidate-level predictions into the same 3 political blocks + abstention, then compare R².

## Baseline Results

### V1: 29 Demographic Indicators (strict NaN dropping)

**Script:** [`src/pca_ridge_baseline.py`](src/pca_ridge_baseline.py)

Validated on 2024 Legislative T1. Uses 29 demographic indicators (CSP breakdown, education levels, employment sectors, age brackets, unemployment, immigration, retirees, housing, household structure). Rows with any NaN demographic dropped.

| Model | Features | Gauche R² | Centre+Droite R² | Extr. Droite R² | Abstention R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Demo only** | 32 (29 demo + geo + time) | 0.28 | -0.34 | -0.54 | -3.35 |
| **B: Demo + 1 lag** | 36 | **0.70** | **0.64** | 0.65 | -3.82 |
| **C: Demo + 2 lags** | 40 | **0.74** | 0.40 | 0.59 | -4.78 |
| **D: Lag only** | 4 (lag scores only) | 0.71 | 0.63 | **0.67** | -4.08 |
| **E: Poly(2) + 1 lag** | 702 (poly features) | 0.72 | 0.57 | 0.57 | -3.94 |

### V2: 47 Demographic Indicators (median imputation)

Added 22 new indicators: BEPC diploma, student rate, other inactive rate, age 75+, housing type (maisons), dwelling size (petits/grands logements), free housing, couples avec/sans enfants, familles nombreuses, marital status (célibataires, mariés, divorcés, veufs, pacsés, union libre). NaN indicators imputed with column median; 5 all-NaN indicators dropped (heating types, old housing stock, overcrowding — column names vary across vintages).

| Model | Features | Gauche R² | Centre+Droite R² | Extr. Droite R² | Abstention R² |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **A: Demo only** | 50 (47 demo + geo + time) | 0.30 | -0.49 | -0.69 | -3.60 |
| **B: Demo + 1 lag** | 54 | **0.71** | 0.57 | 0.63 | -3.75 |
| **C: Demo + 2 lags** | 58 | **0.74** | 0.56 | **0.64** | -3.94 |
| **D: Lag only** | 4 (lag scores only) | 0.65 | -0.34 | 0.12 | -4.61 |
| **E: Poly(2) + 1 lag** | 1539 (poly features) | 0.72 | 0.28 | 0.45 | -3.23 |

### V1 → V2 Comparison

| Model | Gauche Δ | Centre+Droite Δ | Extr. Droite Δ | Abstention Δ |
| :--- | :--- | :--- | :--- | :--- |
| **A: Demo only** | +0.02 | -0.15 | -0.15 | -0.25 |
| **B: Demo + 1 lag** | +0.01 | -0.07 | -0.02 | +0.07 |
| **C: Demo + 2 lags** | 0.00 | **+0.16** | **+0.05** | +0.84 |
| **D: Lag only** | -0.06 | -0.97 | -0.55 | -0.53 |
| **E: Poly(2) + 1 lag** | 0.00 | -0.29 | -0.12 | +0.71 |

### Analysis

**Key findings:**

*   **V1: Lagged block scores are the dominant signal.** Model D (lag-only, no demographics) achieves 0.71 R² on Gauche, 0.63 on Centre+Droite, and 0.67 on Extrême Droite with just 4 features. A BV's past voting pattern is the strongest predictor of its future vote.
*   **V1: Demographics add marginal value on top of lags.** Model B vs D shows minimal improvement (Gauche 0.70 vs 0.71, Centre+Droite 0.64 vs 0.63, Extrême Droite 0.65 vs 0.67). The 29 demographic indicators are mostly redundant once lags are included.
*   **V2: Median imputation changes the data distribution.** Switching from strict NaN dropping to median imputation retains more BV rows (405K vs fewer) but introduces noise. This particularly hurts lag-only models (D: Centre+Droite 0.63→-0.34, Extr. Droite 0.67→0.12) which can't compensate without demographic features.
*   **V2: Demographics + 2 lags benefits most from expanded features.** Model C improves on Centre+Droite (+0.16) and Extr. Droite (+0.05). The second lag now helps rather than hurts Centre+Droite (0.40→0.56), likely because the larger training set from median imputation gives the model enough data to learn from both lags.
*   **V2: Demo-only (A) degrades on Centre+Droite and Extr. Droite.** The extra indicators add noise without lag anchoring — demographics alone predict the left better than the right.
*   **Poly(2) features overfit in both versions.** More features yield worse results than simpler models, particularly on Centre+Droite.
*   **Abstention is structurally unpredictable** from local features alone — all models in both versions show massive negative R². The 2024 snap election caused a national turnout shift (+10pp vs 2022) that no local demographic or historical pattern can explain. This is a global-context signal that only a cross-BV architecture can capture.

**Conclusion:** The best baseline is V1 Model B (29 demo + 1 lag) for Centre+Droite (0.64) and Extr. Droite (0.65), and V2 Model C (47 demo + 2 lags) for Gauche (0.74). The Universal Masked Set Transformer must exceed R² of 0.64–0.74 across all blocks to justify its architectural complexity. The key open challenge is **abstention prediction**, where the transformer's cross-BV routing mechanism should provide unique value by capturing national-level turnout dynamics that are invisible to purely local models.
