"""Confound-free re-selection: CT (Legi+Pres) vs Legi-only, identical rows.

The production per-block champion is *legi-only* for CD and G, but those were
selected by comparing OOF across candidates evaluated on DIFFERENT complete-case
samples ([[loo-harness-sample-size-confound]]). The europeennes head-to-head
hinted CT dominates legi-only — but used same-type lags for legi-only, which is
NOT the production candidate.

This reproduces the production candidates faithfully — both use the SAME
cross-type lags; they differ ONLY in the training rows (legi vs legi+pres) — and
scores them on IDENTICAL held-out legislative rows:

  Legi-only : train on legislative rows only
  CT        : train on legislative + presidential rows
  held-out  : each legislative training fold (cross-type lags give 2007 onward
              two lags via same-year pres), common rows, OOF R² pooled.
  + 2024 val forward pass (report only — not used for selection).
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
    load_cross_type_data,
    add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    BLOCKS_ABS,
    ABBR,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
)
from src.cross_type_ridge import TARGET_BLOCKS
from europeennes_typed_oof import _fit_predict

PCA_KS: list[int | None] = [None, 5, 7, 10]


def main() -> None:
    data_dir = Path("data")
    t0 = time.time()

    df, demo, national_means, poll_feats = load_cross_type_data(data_dir)
    add_election_type_onehot(df)

    raw_lags = [f"{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    type_cols = [c for c in df.columns if c.startswith("type_")]
    feat_cols = demo + dev_lags + type_cols
    n_demo = len(demo)
    d2 = df.dropna(subset=demo + raw_lags + dev_lags).reset_index(drop=True)

    val_m = (d2["election_type"] == VAL_TYPE) & np.isclose(
        d2["date_float"], VAL_DATE, atol=1e-3
    )
    train = d2[~val_m]
    is_legi = train["election_type"] == VAL_TYPE
    legi_dates = sorted(train.loc[is_legi, "date_float"].round(4).unique())
    print(f"Held-out legislative folds: {legi_dates}")
    n_held = {
        d: int(
            (
                (train["election_type"] == VAL_TYPE)
                & np.isclose(train["date_float"], d, atol=1e-3)
            ).sum()
        )
        for d in legi_dates
    }
    print(f"Held legi BVs/fold: {n_held}\n")

    nat = {
        d: national_means[
            (national_means["election_type"] == VAL_TYPE)
            & np.isclose(national_means["date_float"], d, atol=1e-3)
        ]
        for d in legi_dates
    }

    # OOF: held legi rows identical for both designs; CT just trains on more.
    oof = {tc: {"legi": {}, "ct": {}} for tc in TARGET_COLS}
    Xtr_all = train[feat_cols].values.astype(np.float64)
    legi_mask = is_legi.values
    for k in PCA_KS:
        tag = "raw" if k is None else f"PCA{k}"
        acc = {tc: {"legi": ([], []), "ct": ([], [])} for tc in TARGET_COLS}
        for d in legi_dates:
            held_mask = legi_mask & np.isclose(train["date_float"].values, d, atol=1e-3)
            X_h = Xtr_all[held_mask]
            ct_tr = ~held_mask  # all types except the held legi fold
            legi_tr = legi_mask & ~held_mask  # legi rows except held fold
            nat_v = {tc: float(nat[d][tc].iloc[0]) for tc in TARGET_COLS}
            for tc in TARGET_COLS:
                y = train[f"dev_{tc}"].values.astype(np.float64)
                yt = train[tc].values.astype(np.float64)[held_mask]
                for design, m in (("legi", legi_tr), ("ct", ct_tr)):
                    pred = _fit_predict(Xtr_all[m], y[m], X_h, n_demo, k) + nat_v[tc]
                    acc[tc][design][0].append(yt)
                    acc[tc][design][1].append(pred)
        for tc in TARGET_COLS:
            for design in ("legi", "ct"):
                yy = np.concatenate(acc[tc][design][0])
                pp = np.concatenate(acc[tc][design][1])
                oof[tc][design][tag] = r2_score(yy, pp)
        print(f"  [{tag}] done", flush=True)

    # 2024 val forward pass (report only)
    poll_2024 = poll_feats[
        np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
        & (poll_feats["election_type"] == VAL_TYPE)
    ]
    est = {b: float(poll_2024[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    est["Abstention"], _ = estimate_national_abstention_from_gaps(national_means)
    val = d2[val_m]
    Xv = val[feat_cols].values.astype(np.float64)
    legi_train_all = train[is_legi.values]
    val_r2 = {tc: {} for tc in TARGET_COLS}
    for k in PCA_KS:
        tag = "raw" if k is None else f"PCA{k}"
        for design, tr in (("legi", legi_train_all), ("ct", train)):
            Xt = tr[feat_cols].values.astype(np.float64)
            for tc in TARGET_COLS:
                pred = (
                    _fit_predict(
                        Xt, tr[f"dev_{tc}"].values.astype(np.float64), Xv, n_demo, k
                    )
                    + est[tc]
                )
                val_r2[tc].setdefault(design, {})[tag] = r2_score(val[tc].values, pred)

    print(
        f"\n{'#' * 78}\nHELD-OUT-LEGISLATIVE OOF: Legi-only vs CT (identical rows)\n{'#' * 78}"
    )
    for tc in TARGET_COLS:
        lb = max(oof[tc]["legi"].values())
        cb = max(oof[tc]["ct"].values())
        lk = max(oof[tc]["legi"], key=oof[tc]["legi"].get)
        ck = max(oof[tc]["ct"], key=oof[tc]["ct"].get)
        win = "CT" if cb > lb else "Legi-only"
        print(
            f"  {tc:16s} legi-only={lb:.4f} ({lk})  CT={cb:.4f} ({ck})  "
            f"Δ(CT−legi)={cb - lb:+.4f}  → {win}"
        )

    print(
        f"\n{'#' * 78}\n2024 VAL (report only): best-OOF-config forward pass\n{'#' * 78}"
    )
    for tc in TARGET_COLS:
        lk = max(oof[tc]["legi"], key=oof[tc]["legi"].get)
        ck = max(oof[tc]["ct"], key=oof[tc]["ct"].get)
        lv, cv = val_r2[tc]["legi"][lk], val_r2[tc]["ct"][ck]
        print(
            f"  {tc:16s} legi-only={lv:.4f} ({lk})  CT={cv:.4f} ({ck})  Δ={cv - lv:+.4f}"
        )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
