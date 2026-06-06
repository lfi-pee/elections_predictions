"""Fold-fair OOF: does adding européennes folds improve out-of-sample
prediction of held-out LEGISLATIVE/PRESIDENTIAL elections?

The headline europeennes_experiment compares val R² on identical 2024 rows
(euro wins every block) but its LOO OOF is NOT cross-condition comparable: the
euro condition also holds out the européennes folds (a harder population), so
its OOF averages over different rows. This script fixes that.

For each Legi+Pres training fold f, on the SAME common held-out rows:
  - baseline trains on (common Legi+Pres folds minus f)
  - euro     trains on (common Legi+Pres folds minus f) + all européennes folds
Both predict f; OOF R² is pooled over identical held rows → a fair,
multi-election test of the user's "strictly better in OOF" rule.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from src.cross_type_dev import BLOCKS_ABS, ABBR, TARGET_COLS, VAL_DATE, VAL_TYPE
from europeennes_experiment import (
    build_augmented_base,
    prepare_condition,
    EURO,
    TYPES,
)

PCA_KS: list[int | None] = [None, 5, 7, 10]
ALPHA_GRID = np.logspace(-2, 6, 20)
KEY = ["location", "date_float", "election_type"]


def _train_only(df: pd.DataFrame) -> pd.DataFrame:
    v = np.isclose(df["date_float"], VAL_DATE, atol=1e-3) & (
        df["election_type"] == VAL_TYPE
    )
    return df[~v]


def _transform(X_tr, X_ev, n_demo, k):
    scaler = StandardScaler().fit(X_tr)
    X_tr, X_ev = scaler.transform(X_tr), scaler.transform(X_ev)
    if k is None:
        return X_tr, X_ev
    pca = PCA(n_components=k).fit(X_tr[:, :n_demo])
    return (
        np.hstack([pca.transform(X_tr[:, :n_demo]), X_tr[:, n_demo:]]),
        np.hstack([pca.transform(X_ev[:, :n_demo]), X_ev[:, n_demo:]]),
    )


def fold_fair_oof(df_lp, df_euro, feat_cols, n_demo, national_means, k):
    """OOF R² per block on common held-out Legi+Pres folds, base vs +euro."""
    lp = _train_only(df_lp).set_index(KEY)
    eu = _train_only(df_euro)
    eu_lp = eu[eu["election_type"] != EURO].set_index(KEY)
    euro_rows = eu[eu["election_type"] == EURO]

    common = lp.index.intersection(eu_lp.index)
    lp = lp.loc[common].reset_index()
    eu_lp = eu_lp.loc[common].reset_index()  # row-aligned with lp via common order

    folds = lp[["election_type", "date_float"]].drop_duplicates().values.tolist()
    nat = {
        (et, round(float(dt), 4)): national_means[
            (national_means["election_type"] == et)
            & np.isclose(national_means["date_float"], dt, atol=1e-3)
        ]
        for et, dt in folds
    }

    Xlp = lp[feat_cols].values.astype(np.float64)
    Xeu_lp = eu_lp[feat_cols].values.astype(np.float64)
    Xeuro = euro_rows[feat_cols].values.astype(np.float64)

    out = {}
    for tc in TARGET_COLS:
        y_lp = lp[f"dev_{tc}"].values.astype(np.float64)
        y_eu_lp = eu_lp[f"dev_{tc}"].values.astype(np.float64)
        y_euro = euro_rows[f"dev_{tc}"].values.astype(np.float64)
        oof_base = np.full(len(lp), np.nan)
        oof_euro = np.full(len(lp), np.nan)
        for et, dt in folds:
            held = np.isclose(lp["date_float"], dt, atol=1e-3) & (
                lp["election_type"] == et
            )
            tr = ~held
            nm = nat[(et, round(float(dt), 4))]
            nat_v = float(nm[tc].iloc[0]) if len(nm) else 0.0

            Xb_tr, Xb_ev = _transform(Xlp[tr], Xlp[held], n_demo, k)
            ab = RidgeCV(alphas=ALPHA_GRID).fit(Xb_tr, y_lp[tr]).alpha_
            oof_base[held] = (
                Ridge(alpha=ab, solver="cholesky").fit(Xb_tr, y_lp[tr]).predict(Xb_ev)
                + nat_v
            )

            Xe_tr_raw = np.vstack([Xeu_lp[tr], Xeuro])
            ye_tr = np.concatenate([y_eu_lp[tr], y_euro])
            Xe_tr, Xe_ev = _transform(Xe_tr_raw, Xeu_lp[held], n_demo, k)
            ae = RidgeCV(alphas=ALPHA_GRID).fit(Xe_tr, ye_tr).alpha_
            oof_euro[held] = (
                Ridge(alpha=ae, solver="cholesky").fit(Xe_tr, ye_tr).predict(Xe_ev)
                + nat_v
            )

        y_true = lp[tc].values.astype(np.float64)
        out[tc] = (
            r2_score(y_true, oof_base),
            r2_score(y_true, oof_euro),
        )
    return out


def main() -> None:
    data_dir = Path("data")
    t0 = time.time()

    df_base, demo_indicators, national_means = build_augmented_base(data_dir)
    df_lp = prepare_condition(
        df_base, demo_indicators, ["Legislatives_T1", "Presidentielle_T1"]
    )
    df_euro = prepare_condition(df_base, demo_indicators, TYPES)

    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    type_cols = [c for c in df_euro.columns if c.startswith("type_")]
    feat_cols = demo_indicators + dev_lags + type_cols
    n_demo = len(demo_indicators)

    print(
        "\nFold-fair OOF R² on common held-out Legi+Pres folds "
        "(base = LP-only training, euro = LP + européennes folds)\n"
    )
    print(f"{'config':8s}" + "".join(f"{ABBR[tc]:>22s}" for tc in TARGET_COLS))
    print(f"{'':8s}" + "".join(f"{'base→euro (Δ)':>22s}" for _ in TARGET_COLS))
    best = {tc: (-9.0, -9.0, "") for tc in TARGET_COLS}
    for k in PCA_KS:
        res = fold_fair_oof(df_lp, df_euro, feat_cols, n_demo, national_means, k)
        tag = "raw" if k is None else f"PCA{k}"
        line = f"{tag:8s}"
        for tc in TARGET_COLS:
            b, e = res[tc]
            line += f"  {b:.3f}→{e:.3f} ({e - b:+.3f})"
            if e > best[tc][1]:
                best[tc] = (b, e, tag)
        print(line, flush=True)

    print(
        f"\n{'#' * 78}\nVERDICT — fair-OOF best euro config vs its own base, per block\n{'#' * 78}"
    )
    for tc in TARGET_COLS:
        b, e, tag = best[tc]
        mark = (
            "BEAT" if e > b + 0.0005 else ("~tie" if abs(e - b) <= 0.0005 else "WORSE")
        )
        print(
            f"  {tc:16s} {tag:6s}  base_oof={b:.4f} → euro_oof={e:.4f}  Δ={e - b:+.4f}  [{mark}]"
        )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
