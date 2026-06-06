"""Residential-mobility / churn features: do they help (esp. Abstention)?

Commune-level mobility (full temporal coverage, back to 2009.5) added on top
of the commune demographic baseline, evaluated through the pre-registered LOO
harness (selection on training OOF, single 2024 forward pass).
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
from src.mobility_features import build_commune_mobility, MOBILITY_INDICATORS
from src.preregistered import PREV_RAW
from iris_experiment import select_and_report


def attach_commune_tokens(df: pd.DataFrame, tokens: pd.DataFrame) -> list[str]:
    """merge_asof commune-keyed tokens onto df by commune code."""
    df["commune"] = df["location"].str.split("_").str[0].astype(str)
    df.sort_values("date_float", inplace=True)
    inds = sorted(tokens["candidate"].unique())
    for ind in inds:
        d = (
            tokens[tokens["candidate"] == ind][
                ["location", "availability_date", "value"]
            ]
            .rename(columns={"location": "commune", "value": ind})
            .sort_values("availability_date")
            .dropna(subset=["availability_date"])
        )
        d["commune"] = d["commune"].astype(str)
        merged = pd.merge_asof(
            df[["commune", "date_float"]].sort_values("date_float"),
            d,
            left_on="date_float",
            right_on="availability_date",
            by="commune",
            direction="backward",
        )
        df[ind] = merged[ind].values
        if "availability_date" in df.columns:
            df.drop(columns=["availability_date"], inplace=True)
    return inds


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

    print("Attaching commune mobility...", flush=True)
    mob = build_commune_mobility(data_dir, use_cache=True)
    mob_cols = attach_commune_tokens(df, mob)
    print(f"  mobility indicators attached: {mob_cols}")

    raw_lags = [f"{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    base = df.dropna(subset=demo_commune + raw_lags + dev_lags).copy()
    mob_ok = [
        m for m in MOBILITY_INDICATORS if m in df.columns and base[m].notna().any()
    ]
    for m in mob_ok:
        base[m] = base[m].fillna(base[m].median())
    print(f"\nRows: full V1={len(base):,}  mobility cols usable={mob_ok}")

    results = {}
    results["commune"] = select_and_report(
        "COMMUNE (baseline)", base, demo_commune, est, national_means, len(demo_commune)
    )
    results["commune+mobility"] = select_and_report(
        "COMMUNE + mobility",
        base,
        demo_commune + mob_ok,
        est,
        national_means,
        len(demo_commune) + len(mob_ok),
    )

    print(
        f"\n{'#' * 78}\nSUMMARY — val R² of LOO-selected model per design\n{'#' * 78}"
    )
    print(f"{'design':22s}" + "".join(f"{ABBR[tc]:>10s}" for tc in TARGET_COLS))
    for name, sel in results.items():
        print(f"{name:22s}" + "".join(f"{sel[tc][2]:10.4f}" for tc in TARGET_COLS))
    print(f"{'PREV_RAW':22s}" + "".join(f"{PREV_RAW[tc]:10.4f}" for tc in TARGET_COLS))
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
