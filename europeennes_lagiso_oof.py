"""Decompose the européennes penalty: training-rows channel vs lag channel.

§8 showed adding euro folds degrades the fair OOF on held-out Legi+Pres folds,
and that it is not a slope-pooling artifact. Two channels remain entangled:
  A) euro as extra TRAINING ROWS
  B) euro as cross-type LAG inputs (a 2022 legislative's lag2 becomes the 2019
     européenne deviation)

This isolates them by rebuilding lags so européennes are used ONLY as training
rows, never as a lag source (LP-sourced lags via strict-backward merge_asof).
Three conditions on common held-out Legi+Pres folds, pooled slopes:
  base      : LP rows, LP-sourced lags
  rows_only : LP+euro rows, LP-sourced lags  → held LP rows keep base's exact
              lags, so the only change is the extra euro observations (channel A)
  full      : LP+euro rows, euro-informed lags (= §8 pooled)            (A+B)
  channel B = full - rows_only
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from src.cross_type_dev import (
    add_election_type_onehot,
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
)
from europeennes_experiment import build_augmented_base, prepare_condition, EURO, TYPES
from europeennes_oof import _train_only, KEY
from europeennes_typed_oof import _fit_predict

PCA_KS: list[int | None] = [None, 5]
LP_TYPES = ["Legislatives_T1", "Presidentielle_T1"]


def build_lp_sourced_lags(
    df_base: pd.DataFrame, demo_indicators: list[str]
) -> pd.DataFrame:
    """Dev lags for ALL rows, sourced only from Legi+Pres events (euro excluded
    as a lag source). lag1 = latest LP event strictly before; lag2 = the one
    before that."""
    df = df_base.sort_values("date_float").reset_index(drop=True).copy()
    lp = (
        df[df["election_type"].isin(LP_TYPES)]
        .sort_values(["location", "date_float"])
        .copy()
    )
    src = lp[["location", "date_float"]].copy()
    for b in BLOCKS_ABS:
        src[f"{b}_s1"] = lp[f"dev_{b}"].values
        src[f"{b}_s2"] = lp.groupby("location")[f"dev_{b}"].shift(1).values
    src = src.sort_values("date_float")

    merged = pd.merge_asof(
        df[["location", "date_float"]],
        src,
        by="location",
        on="date_float",
        direction="backward",
        allow_exact_matches=False,
    )
    for b in BLOCKS_ABS:
        df[f"dev_{b}_lag1"] = merged[f"{b}_s1"].values
        df[f"dev_{b}_lag2"] = merged[f"{b}_s2"].values
    add_election_type_onehot(df)
    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    return df.dropna(subset=demo_indicators + dev_lags)


def lagiso_fold_oof(df_lp, df_rows, df_full, feat_cols, n_demo, national_means, k):
    """Per-block fair OOF on common held LP folds: base / rows_only / full."""
    parts = {
        "base": _train_only(df_lp).set_index(KEY),
        "rows": _train_only(df_rows[df_rows["election_type"].isin(LP_TYPES)]).set_index(
            KEY
        ),
        "full": _train_only(df_full[df_full["election_type"].isin(LP_TYPES)]).set_index(
            KEY
        ),
    }
    common = parts["base"].index
    for p in parts.values():
        common = common.intersection(p.index)
    lp = {name: p.loc[common].reset_index() for name, p in parts.items()}

    euro_rows = _train_only(df_rows[df_rows["election_type"] == EURO])
    euro_full = _train_only(df_full[df_full["election_type"] == EURO])

    folds = (
        lp["base"][["election_type", "date_float"]].drop_duplicates().values.tolist()
    )
    nat = {
        (et, round(float(dt), 4)): national_means[
            (national_means["election_type"] == et)
            & np.isclose(national_means["date_float"], dt, atol=1e-3)
        ]
        for et, dt in folds
    }

    X = {name: lp[name][feat_cols].values.astype(np.float64) for name in lp}
    Xe_rows = euro_rows[feat_cols].values.astype(np.float64)
    Xe_full = euro_full[feat_cols].values.astype(np.float64)

    out = {}
    for tc in TARGET_COLS:
        y = {name: lp[name][f"dev_{tc}"].values.astype(np.float64) for name in lp}
        ye_rows = euro_rows[f"dev_{tc}"].values.astype(np.float64)
        ye_full = euro_full[f"dev_{tc}"].values.astype(np.float64)
        preds = {m: np.full(len(common), np.nan) for m in ("base", "rows", "full")}
        for et, dt in folds:
            held = np.isclose(lp["base"]["date_float"], dt, atol=1e-3) & (
                lp["base"]["election_type"] == et
            )
            tr = ~held
            nm = nat[(et, round(float(dt), 4))]
            nat_v = float(nm[tc].iloc[0]) if len(nm) else 0.0

            preds["base"][held] = (
                _fit_predict(X["base"][tr], y["base"][tr], X["base"][held], n_demo, k)
                + nat_v
            )
            preds["rows"][held] = (
                _fit_predict(
                    np.vstack([X["rows"][tr], Xe_rows]),
                    np.concatenate([y["rows"][tr], ye_rows]),
                    X["rows"][held],
                    n_demo,
                    k,
                )
                + nat_v
            )
            preds["full"][held] = (
                _fit_predict(
                    np.vstack([X["full"][tr], Xe_full]),
                    np.concatenate([y["full"][tr], ye_full]),
                    X["full"][held],
                    n_demo,
                    k,
                )
                + nat_v
            )

        y_true = lp["base"][tc].values.astype(np.float64)
        out[tc] = {m: r2_score(y_true, preds[m]) for m in preds}
    return out


def main() -> None:
    data_dir = Path("data")
    t0 = time.time()

    df_base, demo_indicators, national_means = build_augmented_base(data_dir)
    df_lp = prepare_condition(df_base, demo_indicators, LP_TYPES)
    df_full = prepare_condition(df_base, demo_indicators, TYPES)
    df_rows = build_lp_sourced_lags(df_base, demo_indicators)

    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    type_cols = [c for c in df_full.columns if c.startswith("type_")]
    feat_cols = demo_indicators + dev_lags + type_cols
    n_demo = len(demo_indicators)

    best = {tc: None for tc in TARGET_COLS}
    for k in PCA_KS:
        res = lagiso_fold_oof(
            df_lp, df_rows, df_full, feat_cols, n_demo, national_means, k
        )
        tag = "raw" if k is None else f"PCA{k}"
        print(f"\n[{tag}] fair OOF on held-out Legi+Pres folds:")
        for tc in TARGET_COLS:
            r = res[tc]
            print(
                f"  {ABBR[tc]:3s}  base={r['base']:.4f}  "
                f"rows_only={r['rows']:.4f} (A={r['rows'] - r['base']:+.4f})  "
                f"full={r['full']:.4f} (A+B={r['full'] - r['base']:+.4f})  "
                f"lag-channel B={r['full'] - r['rows']:+.4f}"
            )
            if best[tc] is None or r["rows"] > best[tc]["rows"]:
                best[tc] = {**r, "tag": tag}

    print(
        f"\n{'#' * 78}\nVERDICT — channel decomposition (best rows_only config)\n{'#' * 78}"
    )
    for tc in TARGET_COLS:
        r = best[tc]
        a = r["rows"] - r["base"]
        b = r["full"] - r["rows"]
        print(
            f"  {tc:16s} {r['tag']:5s}  base={r['base']:.4f}  "
            f"channel A (extra rows)={a:+.4f}  channel B (euro lags)={b:+.4f}"
        )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
