"""Confirm channel A on the 2024 val forward pass (pre-registered closing step).

The lag-isolation decomposition (europeennes_lagiso_oof.py) found that adding
européennes as training ROWS with Legi+Pres-sourced lags (euro never a lag
source) lifts the fair OOF for Centre+Droite (+0.039 raw / +0.016 selected) and
is neutral elsewhere. This runs the single 2024 Legi T1 forward pass for that
condition vs baseline, on identical (common) val rows, per fair-OOF-selected k:
  G→PCA5, CD→raw, ED→PCA5, Ab→PCA5.

base : train Legi+Pres rows, LP lags → predict 2024 legi
rows : train Legi+Pres + euro rows, LP-sourced lags → predict 2024 legi
Val rows are identical (held 2024 legi rows have LP-sourced lags either way).
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from src.cross_type_dev import (
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
)
from src.cross_type_ridge import _build_national_poll_features, TARGET_BLOCKS
from src.load_polls import load_poll_tokens
from europeennes_experiment import build_augmented_base, prepare_condition, EURO, TYPES
from europeennes_oof import _train_only, KEY
from europeennes_typed_oof import _fit_predict
from europeennes_lagiso_oof import build_lp_sourced_lags, LP_TYPES

SELECTED_K = {"Gauche": 5, "Centre+Droite": None, "Extreme_Droite": 5, "Abstention": 5}


def _val_mask(df):
    return np.isclose(df["date_float"], VAL_DATE, atol=1e-3) & (
        df["election_type"] == VAL_TYPE
    )


def main() -> None:
    data_dir = Path("data")
    t0 = time.time()

    df_base, demo_indicators, national_means = build_augmented_base(data_dir)
    df_lp = prepare_condition(df_base, demo_indicators, LP_TYPES)
    df_rows = build_lp_sourced_lags(df_base, demo_indicators)

    polls = load_poll_tokens(data_dir)
    poll_feats = _build_national_poll_features(polls, [(VAL_TYPE, VAL_DATE)])
    est = {b: float(poll_feats[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    lp_nm = national_means[national_means["election_type"] != EURO]
    est["Abstention"], _ = estimate_national_abstention_from_gaps(lp_nm)

    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    type_cols = [c for c in df_rows.columns if c.startswith("type_")]
    feat_cols = demo_indicators + dev_lags + type_cols
    n_demo = len(demo_indicators)

    base_val = df_lp[_val_mask(df_lp)]
    rows_val_all = df_rows[_val_mask(df_rows) & df_rows["election_type"].isin(LP_TYPES)]
    common_val = set(base_val["location"]) & set(rows_val_all["location"])

    base_tr = df_lp[~_val_mask(df_lp)]
    base_v = base_val[base_val["location"].isin(common_val)]
    rows_tr = _train_only(df_rows)  # Legi+Pres + euro training rows
    rows_v = rows_val_all[rows_val_all["location"].isin(common_val)]

    # align val rows by location order for identical y
    base_v = base_v.sort_values("location")
    rows_v = rows_v.sort_values("location")

    print(f"Common 2024 val rows: {len(common_val):,}")
    print(f"Euro training rows added: {(rows_tr['election_type'] == EURO).sum():,}\n")

    print(f"{'block':16s} {'k':>5s} {'base_val':>10s} {'rows_val':>10s} {'Δ':>9s}")
    for tc in TARGET_COLS:
        k = SELECTED_K[tc]
        Xb_tr = base_tr[feat_cols].values.astype(np.float64)
        Xb_v = base_v[feat_cols].values.astype(np.float64)
        yb = base_tr[f"dev_{tc}"].values.astype(np.float64)
        base_pred = _fit_predict(Xb_tr, yb, Xb_v, n_demo, k) + est[tc]
        base_r2 = r2_score(base_v[tc].values, base_pred)

        Xr_tr = rows_tr[feat_cols].values.astype(np.float64)
        Xr_v = rows_v[feat_cols].values.astype(np.float64)
        yr = rows_tr[f"dev_{tc}"].values.astype(np.float64)
        rows_pred = _fit_predict(Xr_tr, yr, Xr_v, n_demo, k) + est[tc]
        rows_r2 = r2_score(rows_v[tc].values, rows_pred)

        tag = "raw" if k is None else f"PCA{k}"
        print(
            f"{tc:16s} {tag:>5s} {base_r2:10.4f} {rows_r2:10.4f} {rows_r2 - base_r2:+9.4f}"
        )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
