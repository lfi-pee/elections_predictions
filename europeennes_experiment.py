"""Européennes as extra training folds: do they lift the deviation model?

The ceiling identified in preconisations.md §1-3 is the *number of training
elections*, not demographic detail. Européennes are the one untapped scrutin
that behaves like a national bipolar list contest (single national
constituency, party lists, national mood) — so unlike régionales /
départementales / cantonales (hyper-local, already bundled into the Ext-*
configs) they should transfer to the cross-type deviation model.

This isolates européennes: Legi+Pres (baseline) vs Legi+Pres+Euro, same
pre-registered PCA-grid LOO-OOF selection, single 2024 Legi T1 val pass.
  - Euro 1999/2004 excluded (>15% unmapped 'Other' block, per beat_it).
  - Euro 2024 excluded (val period).
  - Cross-type local lags are rebuilt per condition, so the baseline never sees
    a européenne in its lags (clean isolation of the treatment).
  - Confound control ([[loo-harness-sample-size-confound]]): adding euro folds
    changes which 2024 BVs have complete 2-lags, so both conditions are
    evaluated on the SAME val rows (location intersection). The cross-condition
    verdict is val R² on identical rows; OOF only selects PCA-k within a
    condition.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from src.cross_type_dev import (
    add_cross_type_local_lags,
    add_deviation_targets,
    add_election_type_onehot,
    build_per_type_national_means,
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
)
from src.cross_type_ridge import (
    _add_demographics,
    _build_block_scores,
    _build_national_poll_features,
    TARGET_BLOCKS,
)
from src.load_polls import load_poll_tokens
from iris_experiment import select_and_report

EURO = "Europeennes_T1"
TYPES = ["Legislatives_T1", "Presidentielle_T1", EURO]
EXCLUDE = {(EURO, 1999.5), (EURO, 2004.5), (EURO, VAL_DATE)}


def build_augmented_base(
    data_dir: Path,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Block-level Legi+Pres+Euro with demographics + deviations (no lags).

    Lags are intentionally NOT built here — each condition rebuilds its own
    cross-type lags over its own row subset. Cached after first (slow) run.
    """
    cache_dir = data_dir / "baseline_cache"
    base_cache = cache_dir / "euro_aug_base.parquet"
    ind_cache = cache_dir / "euro_aug_indicators.txt"
    nm_cache = cache_dir / "euro_aug_natmean.parquet"
    if base_cache.exists() and ind_cache.exists() and nm_cache.exists():
        print("Loading euro-augmented base from cache...")
        df = pd.read_parquet(base_cache)
        indicators = ind_cache.read_text().strip().split("\n")
        national_means = pd.read_parquet(nm_cache)
        return df, indicators, national_means

    print("Building euro-augmented base (slow first run, cached after)...")
    elections = pd.read_parquet(cache_dir / "elections.parquet")
    demos = pd.read_parquet(cache_dir / "demographics.parquet")
    # Cached parquets carry pandas StringDtype on location; _add_demographics
    # derives the commune key as object dtype, so align both to object.
    elections["location"] = elections["location"].astype(object)
    demos["location"] = demos["location"].astype(object)

    ext = elections[elections["election_type"].isin(TYPES)].copy()
    keep = pd.Series(True, index=ext.index)
    for etype, ddate in EXCLUDE:
        keep &= ~(
            (ext["election_type"] == etype)
            & np.isclose(ext["date_float"], ddate, atol=0.1)
        )
    ext = ext[keep]

    block_scores = _build_block_scores(ext)
    if "Other" in block_scores.columns:
        for etype in TYPES:
            sub = block_scores[block_scores["election_type"] == etype]
            for ddate in sorted(sub["date_float"].round(2).unique()):
                d = sub[np.isclose(sub["date_float"], ddate, atol=0.1)]
                print(
                    f"    {etype:20s} {ddate:.2f}: mapped="
                    f"{d[TARGET_BLOCKS].sum(axis=1).mean():.1f}% "
                    f"other={d['Other'].mean():.1f}%  ({len(d):,} BV)"
                )

    national_means = build_per_type_national_means(block_scores)
    df = add_deviation_targets(block_scores, national_means)

    geo = ext[["location", "latitude", "longitude"]].drop_duplicates("location")
    df = df.merge(geo, on="location", how="left")
    df["latitude"] = df["latitude"].fillna(46.2276)
    df["longitude"] = df["longitude"].fillna(2.2137)

    print("  Merging demographics (slow, ~15-30 min)...", flush=True)
    df, indicators = _add_demographics(df, demos)
    df = df.dropna(subset=TARGET_COLS)

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(base_cache, index=False)
    ind_cache.write_text("\n".join(indicators))
    national_means.to_parquet(nm_cache, index=False)
    print(f"  Cached to {base_cache}")
    return df, indicators, national_means


def prepare_condition(
    df_base: pd.DataFrame, demo_indicators: list[str], types: list[str]
) -> pd.DataFrame:
    """Subset to `types`, rebuild cross-type lags + one-hot, drop incomplete."""
    df = df_base[df_base["election_type"].isin(types)].copy()
    df = add_cross_type_local_lags(df)
    add_election_type_onehot(df)
    raw_lags = [f"{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    return df.dropna(subset=demo_indicators + raw_lags + dev_lags)


def val_locations(df: pd.DataFrame) -> set[str]:
    val_mask = np.isclose(df["date_float"], VAL_DATE, atol=1e-3) & (
        df["election_type"] == VAL_TYPE
    )
    return set(df.loc[val_mask, "location"])


def restrict_val(df: pd.DataFrame, locs: set[str]) -> pd.DataFrame:
    """Keep all training rows; drop val rows whose location is not in `locs`."""
    val_mask = np.isclose(df["date_float"], VAL_DATE, atol=1e-3) & (
        df["election_type"] == VAL_TYPE
    )
    return df[~val_mask | df["location"].isin(locs)].copy()


def main() -> None:
    data_dir = Path("data")
    t0 = time.time()

    df_base, demo_indicators, national_means = build_augmented_base(data_dir)

    # National 2024 estimate: vote blocks from Legi 2024 polls, abstention from
    # the gap model on Legi+Pres-only national means (euro folds would pollute
    # the gap relationship). Same est for both conditions — only the deviation
    # model's training folds differ.
    polls = load_poll_tokens(data_dir)
    poll_feats = _build_national_poll_features(polls, [(VAL_TYPE, VAL_DATE)])
    est = {b: float(poll_feats[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    lp_nm = national_means[national_means["election_type"] != EURO]
    est["Abstention"], _ = estimate_national_abstention_from_gaps(lp_nm)
    print(f"\nNational 2024 estimates: {est}")

    df_legi_pres = prepare_condition(
        df_base, demo_indicators, ["Legislatives_T1", "Presidentielle_T1"]
    )
    df_euro = prepare_condition(df_base, demo_indicators, TYPES)

    common = val_locations(df_legi_pres) & val_locations(df_euro)
    df_legi_pres = restrict_val(df_legi_pres, common)
    df_euro = restrict_val(df_euro, common)
    print(
        f"\nCommon val rows: {len(common):,}  "
        f"(baseline train={len(df_legi_pres) - len(common):,}, "
        f"euro train={len(df_euro) - len(common):,})"
    )
    euro_train = df_euro[df_euro["election_type"] == EURO]
    print(
        f"Euro training folds added: "
        f"{sorted(euro_train['date_float'].round(2).unique().tolist())} "
        f"({len(euro_train):,} BV rows)"
    )

    results = {
        "Legi+Pres (baseline)": select_and_report(
            "LEGI+PRES (baseline)",
            df_legi_pres,
            demo_indicators,
            est,
            national_means,
            len(demo_indicators),
        ),
        "Legi+Pres+Euro": select_and_report(
            "LEGI+PRES+EURO",
            df_euro,
            demo_indicators,
            est,
            national_means,
            len(demo_indicators),
        ),
    }

    print(
        f"\n{'#' * 78}\nSUMMARY — val R² of LOO-selected model (common val rows)\n{'#' * 78}"
    )
    print(f"{'design':22s}" + "".join(f"{ABBR[tc]:>10s}" for tc in TARGET_COLS))
    for name, sel in results.items():
        print(f"{name:22s}" + "".join(f"{sel[tc][2]:10.4f}" for tc in TARGET_COLS))
    base = results["Legi+Pres (baseline)"]
    euro = results["Legi+Pres+Euro"]
    print(
        f"{'Δ (euro - base)':22s}"
        + "".join(f"{euro[tc][2] - base[tc][2]:+10.4f}" for tc in TARGET_COLS)
    )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
