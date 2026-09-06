"""Go/no-go gate for the off-axis "Autre" bloc: its leave-one-election-out **share
R²** under the PRODUCTION deviation model (`cross_type_dev`).

Correct metric (cf. docs/autre_bloc_plan.md §9): is the off-axis residual vote
("Other" in the block routing — what falls outside G/CD/ED AFTER lineage repair and
the 2024 attribution overrides) predictable out-of-fold, on the axis the model is
validated on (per-BV vote-share R²), and comparably to the three modelled blocs?

Method — faithful to production, no reimplementation:
  * Promote "Other" to a block by patching the module globals `BLOCKS_ABS` /
    `TARGET_COLS` that `cross_type_dev`'s own build functions read. `_build_block_scores`
    already emits an "Other" column, so `build_per_type_national_means`,
    `add_deviation_targets` and `add_cross_type_local_lags` then build `natmean_Other`,
    `dev_Other` and its full lag machinery (precinct-reuse + commune fallback) exactly
    as for the other blocks.
  * `load_cross_type_data` rebuilds the base with Other (derived caches removed by the
    caller; raw election/demo caches reused).
  * LOO = leave-one-cross-type-election-out. Per held election: fit RidgeCV on `dev_<b>`
    of the other elections (features = demo + geo/time + dev lags + type one-hot, the
    production "CT-devlag" set), predict the held election's deviation, reconstruct the
    share as dev_pred + the held fold's national mean (oracle mode, as production
    validates intervals), clip to [0,100]. Pool per-BV `r2_score` over held LEGISLATIVE
    elections (the target regime); presidential folds still train every other fold.

    python3 -u -m src.autre_oof
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from src import cross_type_dev as D

OTHER = "Other"
BLOCKS5 = ["Gauche", "Centre+Droite", "Extreme_Droite", OTHER, "Abstention"]
ALPHA_GRID = np.logspace(-2, 6, 20)
EVAL_TYPES = ("Legislatives_T1",)


def main():
    # Promote Other to a block in the functions' view (they read these globals).
    D.BLOCKS_ABS = BLOCKS5
    D.TARGET_COLS = BLOCKS5
    import src.cross_type_ridge as R  # some helpers read TARGET_COLS from here too
    R.TARGET_COLS = BLOCKS5

    df, demo_cols, national_means, _polls = D.load_cross_type_data(Path("data"))
    type_cols = D.add_election_type_onehot(df)
    print(f"Dataset: {len(df):,} rows, targets={BLOCKS5}", flush=True)

    dev_lag1 = [f"dev_{b}_lag1" for b in BLOCKS5]
    dev_lag2 = [f"dev_{b}_lag2" for b in BLOCKS5]
    avail = lambda cols: [c for c in cols if c in df.columns and df[c].notna().any()]
    feats = demo_cols + ["latitude", "longitude", "date_float"] + avail(dev_lag1) + avail(dev_lag2) + type_cols

    # Require complete demo + the lags actually used (production V1-2lag drop).
    need = demo_cols + avail([f"{b}_lag1" for b in BLOCKS5]) + avail([f"{b}_lag2" for b in BLOCKS5]) \
        + avail(dev_lag1) + avail(dev_lag2)
    df = df.dropna(subset=need).copy()
    df = df[df[feats].notna().all(axis=1)].copy()

    # National mean per (type, date) for oracle reconstruction.
    nm = {(row["election_type"], int(round(row["date_float"] * 100))):
          {b: float(row[b]) for b in BLOCKS5}
          for _, row in national_means.iterrows()}
    et = df["election_type"].values
    dk = (df["date_float"].values.astype("float64") * 100).round().astype("int64")

    folds = sorted(set(zip(et, dk)))
    Xall = df[feats].values.astype(np.float64)

    oof = {b: {"t": [], "p": []} for b in BLOCKS5}
    for e_h, d_h in folds:
        if e_h not in EVAL_TYPES:
            continue
        held = (et == e_h) & (dk == d_h)
        if held.sum() == 0 or (~held).sum() == 0 or (e_h, d_h) not in nm:
            continue
        scaler = StandardScaler().fit(Xall[~held])
        Xtr, Xhe = scaler.transform(Xall[~held]), scaler.transform(Xall[held])
        natrow = nm[(e_h, d_h)]
        for b in BLOCKS5:
            devcol = df[f"dev_{b}"].values.astype(np.float64)
            m = RidgeCV(alphas=ALPHA_GRID).fit(Xtr, devcol[~held])
            share = np.clip(m.predict(Xhe) + natrow[b], 0.0, 100.0)
            oof[b]["t"].extend(df[b].values[held])
            oof[b]["p"].extend(share)

    print("\nLeave-one-legislative-election-out OOF share R² (production model):\n", flush=True)
    print(f"  {'block':16}{'OOF R²':>9}{'MAE':>9}{'share sd':>10}")
    for b in BLOCKS5:
        yt = np.array(oof[b]["t"]); yp = np.array(oof[b]["p"])
        print(f"  {b:16}{r2_score(yt, yp):9.3f}{np.mean(np.abs(yt-yp)):8.2f}p{yt.std():9.1f}p")


if __name__ == "__main__":
    main()
