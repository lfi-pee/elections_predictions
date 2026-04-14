"""Bayesian Poll Aggregation: house-effect correction + recency weighting.

Replaces raw poll averages with corrected national estimates that:
1. Remove systematic per-institute biases (house effects) with shrinkage
2. Weight recent polls more heavily (exponential decay)
3. Filter out "Résultats" rows (actual results in poll tables) and unknowns

All parameters estimated via LOO on training elections — no validation
data used.  The decay parameter lambda is selected by minimizing LOO
RMSE across training elections.

Architecture (per election to predict):
  1. Estimate house effects from OTHER training elections (LOO)
  2. Correct each poll: corrected = raw - house_effect[institute, block]
  3. Weight by recency: w = exp(-lambda * years_before_election)
  4. Weighted average → national estimate per block
  5. Renormalize to 100% across 3 vote blocks

Usage:
    python3 -u -m src.bayesian_polls
"""
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import numpy as np
import pandas as pd
from pathlib import Path

from src.load_polls import load_poll_tokens
from src.cross_type_ridge import _poll_token_to_block, TARGET_BLOCKS
from src.cross_type_dev import (
    load_cross_type_data, estimate_national_abstention_from_gaps,
    TARGET_COLS, VAL_DATE, VAL_TYPE, ABBR,
)

LAMBDA_GRID = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0]
SHRINKAGE_STRENGTH = 2.0  # James-Stein prior strength for house effects


# ── Poll processing ─────────────────────────────────────────────────

def build_poll_block_shares(polls, window_start, window_end):
    """Convert raw poll tokens to per-poll per-block shares.

    Returns DataFrame with columns:
        date_float, institute, Gauche, Centre+Droite, Extreme_Droite
    Each row is one poll instance (institute x date), normalized to 100%.
    """
    p = polls.copy()
    p["block"] = p.apply(
        lambda r: _poll_token_to_block(
            str(r.get("party", "")), str(r.get("candidate", ""))),
        axis=1,
    )
    p["institute"] = p["metric_type"].str.replace("Poll_", "", n=1)

    mask = (
        (p["location"] == "National")
        & p["block"].isin(TARGET_BLOCKS)
        & ~p["election_type"].str.contains("T2", na=False)
        & (p["date_float"] >= window_start)
        & (p["date_float"] <= window_end)
        & ~p["institute"].isin(["Résultats", "Unknown"])
    )
    w = p[mask].copy()
    if len(w) == 0:
        return pd.DataFrame()

    # Sum within (date, institute, block) then normalize to 100%
    per_poll = (
        w.groupby(["date_float", "institute", "block"])["value"]
        .sum().reset_index()
    )
    totals = per_poll.groupby(["date_float", "institute"])["value"] \
        .transform("sum")
    per_poll["share"] = per_poll["value"] / totals * 100.0

    wide = per_poll.pivot_table(
        index=["date_float", "institute"],
        columns="block", values="share",
    ).reset_index()

    for b in TARGET_BLOCKS:
        if b not in wide.columns:
            wide[b] = np.nan
    return wide.dropna(subset=TARGET_BLOCKS)


def _raw_poll_average(polls, target_date, window=1.0):
    """Replicate the existing raw-average pipeline for comparison."""
    shares = build_poll_block_shares(
        polls, target_date - window, target_date)
    if len(shares) == 0:
        return {b: np.nan for b in TARGET_BLOCKS}
    avg = {b: float(shares[b].mean()) for b in TARGET_BLOCKS}
    total = sum(avg.values())
    return {b: v / total * 100.0 for b, v in avg.items()} if total > 0 \
        else avg


# ── House effects ───────────────────────────────────────────────────

def estimate_house_effects(polls, national_means, val_date,
                           exclude_date=None, window=1.0):
    """Estimate per-institute per-block biases from training elections.

    For each training election E:
      bias_i_b = mean(poll_share[i,b]) - actual[E,b]
    Then average across elections with James-Stein shrinkage toward zero.

    exclude_date: float or None.  If set, exclude that election (for LOO).
    """
    train_nm = national_means[
        national_means["date_float"] < val_date - 0.1
    ].copy()
    if exclude_date is not None:
        train_nm = train_nm[
            ~np.isclose(train_nm["date_float"], exclude_date, atol=0.05)
        ]

    records = []
    for _, row in train_nm.iterrows():
        edate = float(row["date_float"])
        shares = build_poll_block_shares(polls, edate - window, edate)
        if len(shares) == 0:
            continue

        # Average per institute across poll dates for this election
        inst_avg = shares.groupby("institute")[TARGET_BLOCKS].mean()

        for inst in inst_avg.index:
            for block in TARGET_BLOCKS:
                records.append({
                    "institute": inst,
                    "block": block,
                    "bias": inst_avg.loc[inst, block] - row[block],
                })

    if not records:
        return {}

    df = pd.DataFrame(records)
    effects = {}
    for (inst, block), g in df.groupby(["institute", "block"]):
        n = len(g)
        raw_bias = g["bias"].mean()
        # James-Stein shrinkage: fewer elections → more shrinkage toward 0
        effects[(inst, block)] = raw_bias * (n / (n + SHRINKAGE_STRENGTH))

    return effects


# ── Bayesian national estimate ──────────────────────────────────────

def estimate_national_bayesian(polls, target_date, house_effects,
                               decay_lambda, window=1.0):
    """Bias-corrected, recency-weighted national estimate for vote blocks.

    Returns dict {block_name: estimated_share}, normalized to 100%.
    """
    shares = build_poll_block_shares(
        polls, target_date - window, target_date)
    if len(shares) == 0:
        return {b: np.nan for b in TARGET_BLOCKS}
    shares = shares.copy()

    # Correct house effects
    if house_effects:
        for block in TARGET_BLOCKS:
            shares[block] = shares[block] - shares["institute"].map(
                lambda inst, b=block: house_effects.get((inst, b), 0.0)
            )

    # Recency weights
    years_before = target_date - shares["date_float"].values
    weights = np.exp(-decay_lambda * np.maximum(years_before, 0.0))

    result = {}
    for block in TARGET_BLOCKS:
        result[block] = float(
            np.average(shares[block].values, weights=weights))

    # Renormalize to 100%
    total = sum(result.values())
    if total > 0:
        result = {b: v / total * 100.0 for b, v in result.items()}

    return result


# ── LOO lambda selection ────────────────────────────────────────────

def loo_select_lambda(polls, national_means, val_date, window=1.0):
    """LOO selection of decay_lambda on training elections.

    For each lambda x each training election:
      1. Estimate house effects from other elections (LOO)
      2. Predict national shares with corrected + weighted polls
      3. Record squared errors vs actual

    Returns (best_lambda, best_rmse, raw_rmse, per_lambda_results, per_election).
    """
    train_nm = national_means[
        national_means["date_float"] < val_date - 0.1
    ].copy()

    # Pre-compute raw averages for comparison
    raw_se = []
    for _, row in train_nm.iterrows():
        edate = float(row["date_float"])
        raw = _raw_poll_average(polls, edate, window)
        for b in TARGET_BLOCKS:
            if not np.isnan(raw.get(b, np.nan)):
                raw_se.append((raw[b] - row[b]) ** 2)
    raw_rmse = np.sqrt(np.mean(raw_se)) if raw_se else np.nan

    results = {lam: [] for lam in LAMBDA_GRID}
    per_election = {lam: [] for lam in LAMBDA_GRID}

    for _, row in train_nm.iterrows():
        edate = float(row["date_float"])
        etype = row["election_type"]
        actual = {b: float(row[b]) for b in TARGET_BLOCKS}

        house_fx = estimate_house_effects(
            polls, national_means, val_date,
            exclude_date=edate, window=window,
        )

        for lam in LAMBDA_GRID:
            pred = estimate_national_bayesian(
                polls, edate, house_fx, lam, window,
            )
            block_errors = {}
            for b in TARGET_BLOCKS:
                err = pred[b] - actual[b]
                results[lam].append(err ** 2)
                block_errors[b] = err
            per_election[lam].append({
                "election_type": etype, "date_float": edate,
                **{f"err_{b}": block_errors[b] for b in TARGET_BLOCKS},
            })

    best_lam, best_rmse = None, np.inf
    for lam in LAMBDA_GRID:
        rmse = np.sqrt(np.mean(results[lam]))
        if rmse < best_rmse:
            best_lam, best_rmse = lam, rmse

    return best_lam, best_rmse, raw_rmse, results, per_election


# ── LOO national estimates for conformal calibration ────────────────

def get_loo_national_estimates(polls, national_means, val_date,
                               best_lambda, window=1.0):
    """Get LOO national estimates for each training election.

    Used by conformal.py to calibrate intervals with realistic national
    estimate error (not oracle actual means).

    Returns dict: {(election_type, date_float_rounded): {block: est}}
    """
    train_nm = national_means[
        national_means["date_float"] < val_date - 0.1
    ].copy()

    loo_ests = {}
    for _, row in train_nm.iterrows():
        edate = float(row["date_float"])
        etype = row["election_type"]

        house_fx = estimate_house_effects(
            polls, national_means, val_date,
            exclude_date=edate, window=window,
        )
        pred = estimate_national_bayesian(
            polls, edate, house_fx, best_lambda, window,
        )
        loo_ests[(etype, round(edate, 3))] = pred

    return loo_ests


# ── Convenience function for pipeline integration ───────────────────

def get_bayesian_estimates(data_dir):
    """Load data, estimate house effects + lambda, return 2024 estimates.

    Drop-in replacement for the raw poll averages in preregistered.py.

    Returns:
        est: dict {block: float} for all 4 blocks (G, C+D, ED, Abstention)
        info: dict with lambda, house effects, RMSE comparison
    """
    _, _, national_means, _ = load_cross_type_data(data_dir)
    polls = load_poll_tokens(data_dir)

    best_lam, best_rmse, raw_rmse, _, _ = loo_select_lambda(
        polls, national_means, VAL_DATE,
    )
    house_fx = estimate_house_effects(polls, national_means, VAL_DATE)
    bayes_est = estimate_national_bayesian(
        polls, VAL_DATE, house_fx, best_lam,
    )

    abs_pred, abs_loo_rmse = estimate_national_abstention_from_gaps(
        national_means)
    bayes_est["Abstention"] = abs_pred

    info = {
        "lambda": best_lam,
        "house_effects": house_fx,
        "bayesian_rmse": best_rmse,
        "raw_rmse": raw_rmse,
        "abs_loo_rmse": abs_loo_rmse,
        "national_means": national_means,
        "polls": polls,
    }

    return bayes_est, info


# ── Main ────────────────────────────────────────────────────────────

def main():
    data_dir = Path("data")

    print("=" * 70)
    print("BAYESIAN POLL AGGREGATION")
    print("House-effect correction + recency weighting")
    print("All parameters from LOO on training — no validation tuning")
    print("=" * 70)

    # Load data
    _, _, national_means, _ = load_cross_type_data(data_dir)
    polls = load_poll_tokens(data_dir)

    # ── LOO lambda selection ──
    print("\n── LOO selection of decay lambda ──")
    best_lam, best_rmse, raw_rmse, results, per_election = \
        loo_select_lambda(polls, national_means, VAL_DATE)

    print(f"\n  {'Lambda':>8s}  {'RMSE (pp)':>10s}  {'vs raw':>8s}")
    print("  " + "-" * 32)
    for lam in LAMBDA_GRID:
        rmse = np.sqrt(np.mean(results[lam]))
        delta = rmse - raw_rmse
        mark = " <--" if lam == best_lam else ""
        print(f"  {lam:8.1f}  {rmse:10.2f}  {delta:+8.2f}{mark}")
    print(f"  {'raw avg':>8s}  {raw_rmse:10.2f}")
    print(f"\n  Selected lambda = {best_lam}")
    print(f"  Bayesian LOO RMSE = {best_rmse:.2f} pp  "
          f"(raw = {raw_rmse:.2f} pp,  "
          f"delta = {best_rmse - raw_rmse:+.2f} pp)")

    # ── Per-election LOO breakdown ──
    print(f"\n── Per-election LOO errors (lambda={best_lam}) ──")
    print(f"  {'Election':30s} {'G err':>7s} {'CD err':>7s} {'ED err':>7s}")
    print("  " + "-" * 55)
    for rec in per_election[best_lam]:
        label = f"{rec['election_type']} {rec['date_float']:.2f}"
        print(f"  {label:30s} "
              f"{rec['err_Gauche']:+7.2f} "
              f"{rec['err_Centre+Droite']:+7.2f} "
              f"{rec['err_Extreme_Droite']:+7.2f}")

    # ── House effects ──
    print(f"\n── Estimated house effects (top biases) ──")
    house_fx = estimate_house_effects(polls, national_means, VAL_DATE)
    bias_list = [(k, v) for k, v in house_fx.items()]
    bias_list.sort(key=lambda x: abs(x[1]), reverse=True)
    print(f"  {'Institute':25s} {'Block':20s} {'Bias (pp)':>10s}")
    print("  " + "-" * 58)
    for (inst, block), bias in bias_list[:20]:
        print(f"  {inst:25s} {block:20s} {bias:+10.2f}")

    # ── 2024 estimates ──
    print(f"\n{'='*70}")
    print("2024 NATIONAL ESTIMATES")
    print(f"{'='*70}")
    bayes_est = estimate_national_bayesian(
        polls, VAL_DATE, house_fx, best_lam)
    raw_est = _raw_poll_average(polls, VAL_DATE)

    # Actual (from national_means)
    val_mask = np.isclose(national_means["date_float"], VAL_DATE, atol=0.1)
    actual = {b: float(national_means.loc[val_mask, b].iloc[0])
              for b in TARGET_BLOCKS} if val_mask.any() else {}

    print(f"\n  {'Block':20s} {'Raw':>8s} {'Bayesian':>8s} {'Actual':>8s} "
          f"{'|Raw err|':>9s} {'|Bay err|':>9s}")
    print("  " + "-" * 60)
    for b in TARGET_BLOCKS:
        r_err = abs(raw_est[b] - actual.get(b, np.nan))
        b_err = abs(bayes_est[b] - actual.get(b, np.nan))
        better = " <--" if b_err < r_err - 0.01 else ""
        print(f"  {b:20s} {raw_est[b]:8.2f} {bayes_est[b]:8.2f} "
              f"{actual.get(b, np.nan):8.2f} {r_err:9.2f} {b_err:9.2f}"
              f"{better}")

    total_raw = sum(abs(raw_est[b] - actual[b]) for b in TARGET_BLOCKS)
    total_bay = sum(abs(bayes_est[b] - actual[b]) for b in TARGET_BLOCKS)
    print(f"\n  Total |error|:  raw={total_raw:.2f} pp  "
          f"bayesian={total_bay:.2f} pp")

    # Abstention (gap model, unchanged)
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    bayes_est["Abstention"] = abs_pred

    print(f"\n  Final estimates (Bayesian + gap model):")
    for tc in TARGET_COLS:
        print(f"    {tc:20s}: {bayes_est[tc]:.2f}")


if __name__ == "__main__":
    main()
