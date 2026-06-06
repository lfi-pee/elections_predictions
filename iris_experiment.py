"""IRIS-resolution demographics: does sub-commune detail beat commune means?

Three designs, all evaluated through the existing pre-registered harness
(LOO OOF R² selection on training, single 2024 forward pass — no val tuning):

  commune   : 52 commune indicators (baseline, reproduces headline)
  iris      : same indicators at BV-weighted IRIS resolution (replaces commune)
  delta     : commune + within-commune deviation (iris - commune, NaN->0).
              Never drops a training election; only adds sub-commune signal
              where IRIS is available (elections >= 2017).

For `commune` vs `iris` we additionally restrict to IRIS-covered rows so the
comparison isolates resolution (not sample size).
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
    load_cross_type_data,
    add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.iris_features import build_bv_iris_demographics
from src.preregistered import run_loo_and_val, PREV_RAW

PCA_KS = [None, 5, 7, 10]


def attach_iris(df: pd.DataFrame, tokens: pd.DataFrame) -> list[str]:
    """merge_asof BV-keyed IRIS tokens onto df as `{ind}_iris` columns."""
    inds = sorted(tokens["candidate"].unique())
    df.sort_values("date_float", inplace=True)
    for ind in inds:
        d = (
            tokens[tokens["candidate"] == ind][
                ["location", "availability_date", "value"]
            ]
            .rename(columns={"value": f"{ind}_iris"})
            .sort_values("availability_date")
            .dropna(subset=["availability_date"])
        )
        merged = pd.merge_asof(
            df[["location", "date_float"]].sort_values("date_float"),
            d,
            left_on="date_float",
            right_on="availability_date",
            by="location",
            direction="backward",
        )
        df[f"{ind}_iris"] = merged[f"{ind}_iris"].values
        if "availability_date" in df.columns:
            df.drop(columns=["availability_date"], inplace=True)
    return [f"{i}_iris" for i in inds]


def select_and_report(label, df, demo_cols, est, national_means, n_demo_for_pca):
    """Run all PCA configs, LOO-select per block, print OOF + val vs PREV_RAW."""
    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    type_cols = [c for c in df.columns if c.startswith("type_")]
    non_demo = dev_lag1 + dev_lag2 + type_cols
    feat_cols = demo_cols + non_demo

    print(
        f"\n{'=' * 78}\n{label}  ({len(demo_cols)} demo cols, {len(df):,} rows)\n{'=' * 78}"
    )
    by_block = {tc: [] for tc in TARGET_COLS}
    for k in PCA_KS:
        cfg = (
            {"n_demo": n_demo_for_pca}
            if k is None
            else {"pca_k": k, "n_demo": n_demo_for_pca}
        )
        res = run_loo_and_val(label, df, feat_cols, est, national_means, cfg)
        tag = "raw" if k is None else f"PCA{k}"
        line = f"  {tag:6s}"
        for tc in TARGET_COLS:
            by_block[tc].append((res[tc]["oof_r2"], res[tc]["val_r2"], tag))
            line += (
                f"  {ABBR[tc]}:oof={res[tc]['oof_r2']:.3f} val={res[tc]['val_r2']:.3f}"
            )
        print(line, flush=True)

    print(f"  {'-' * 70}\n  LOO-SELECTED (best OOF) per block:")
    selected = {}
    for tc in TARGET_COLS:
        oof, val, tag = max(by_block[tc], key=lambda t: t[0])
        delta = val - PREV_RAW[tc]
        mark = (
            "BEAT" if delta > 0.0005 else ("~tie" if abs(delta) <= 0.0005 else "MISS")
        )
        selected[tc] = (tag, oof, val, delta, mark)
        print(
            f"    {tc:16s} {tag:6s}  oof={oof:.4f}  val={val:.4f}  "
            f"(prev={PREV_RAW[tc]:.4f}  Δ={delta:+.4f})  [{mark}]"
        )
    return selected


def main():
    data_dir = Path("data")
    t0 = time.time()

    print("Loading base (commune) data...", flush=True)
    df, demo_commune, national_means, poll_feats = load_cross_type_data(data_dir)
    add_election_type_onehot(df)

    est = {
        b: float(
            poll_feats[
                np.isclose(poll_feats["date_float"], 2024.5, atol=0.1)
                & (poll_feats["election_type"] == "Legislatives_T1")
            ][f"poll_{b}"].iloc[0]
        )
        for b in TARGET_BLOCKS
    }
    est["Abstention"], _ = estimate_national_abstention_from_gaps(national_means)

    print("Attaching IRIS demographics...", flush=True)
    iris_tokens = build_bv_iris_demographics(data_dir, use_cache=True)
    demo_iris = attach_iris(df, iris_tokens)

    # Per-row IRIS coverage is partial (no vintage has all 47 indicators), so we
    # never require full IRIS coverage. Two row-preserving encodings of the
    # sub-commune signal on the shared indicators:
    #   irisfill = IRIS value where present, else commune value (isolates level)
    #   wdev     = IRIS - commune (within-commune deviation), NaN->0 (anchored)
    shared = [i for i in demo_commune if f"{i}_iris" in demo_iris]
    new_cols = {}
    for i in shared:
        iris_v = df[f"{i}_iris"]
        new_cols[f"{i}_irisfill"] = iris_v.where(iris_v.notna(), df[i]).values
        new_cols[f"{i}_wdev"] = (iris_v - df[i]).fillna(0.0).values
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    demo_fill = [f"{i}_irisfill" for i in shared] + [
        i for i in demo_commune if i not in shared
    ]
    demo_wdev = [f"{i}_wdev" for i in shared]

    raw_lags = [f"{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    base = df.dropna(subset=demo_commune + raw_lags + dev_lags).copy()
    recent = base[base["date_float"] >= 2017.0].copy()  # IRIS-covered era
    iris_share = base[[f"{i}_iris" for i in shared]].notna().any(axis=1).mean()
    print(
        f"\nRows: full V1={len(base):,}  recent(>=2017)={len(recent):,}  "
        f"shared indicators={len(shared)}/{len(demo_commune)}  "
        f"rows with any IRIS={iris_share:.0%}"
    )

    results = {}
    # Full-sample designs (all 8 training dates; no rows dropped)
    results["commune_full"] = select_and_report(
        "COMMUNE (baseline, full V1)",
        base,
        demo_commune,
        est,
        national_means,
        len(demo_commune),
    )
    results["irisfill_full"] = select_and_report(
        "IRIS-fallback level (full V1)",
        base,
        demo_fill,
        est,
        national_means,
        len(demo_fill),
    )
    results["delta_full"] = select_and_report(
        "COMMUNE + within-commune Δ (full V1)",
        base,
        demo_commune + demo_wdev,
        est,
        national_means,
        len(demo_commune) + len(demo_wdev),
    )
    # Resolution-isolated: same recent rows, commune vs IRIS level
    results["commune_recent"] = select_and_report(
        "COMMUNE (recent >=2017 only)",
        recent,
        demo_commune,
        est,
        national_means,
        len(demo_commune),
    )
    results["irisfill_recent"] = select_and_report(
        "IRIS-fallback level (recent >=2017 only)",
        recent,
        demo_fill,
        est,
        national_means,
        len(demo_fill),
    )

    print(
        f"\n{'#' * 78}\nSUMMARY — val R² of LOO-selected model per design\n{'#' * 78}"
    )
    print(f"{'design':28s}" + "".join(f"{ABBR[tc]:>10s}" for tc in TARGET_COLS))
    for name, sel in results.items():
        print(f"{name:28s}" + "".join(f"{sel[tc][2]:10.4f}" for tc in TARGET_COLS))
    print(f"{'PREV_RAW':28s}" + "".join(f"{PREV_RAW[tc]:10.4f}" for tc in TARGET_COLS))
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
