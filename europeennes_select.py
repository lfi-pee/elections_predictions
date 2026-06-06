"""Head-to-head selection: does euro-rows (LP-sourced lags) win CD's OOF?

Decision run for the channel-A result. Compares three training-set designs on
the TASK-CORRECT out-of-sample metric — predict held-out LEGISLATIVE folds — so
the euro-rows candidate is directly comparable to the production CD champion
(which is legi-only). Selection is on this OOF; the 2024 val is not consulted.

  Legi-only : legislative rows only, same-type lags   (production CD champion family)
  CT        : Legi+Pres rows, cross-type lags
  Euro-rows : Legi+Pres+euro rows, LP-SOURCED lags    (euro as observations only)

For each held-out legislative fold f, every design trains on its own rows minus
f and predicts the SAME common held-out legislative BVs (intersection across
designs, complete-case). OOF R² pooled over folds, per block, per PCA config.
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from src.cross_type_dev import BLOCKS_ABS, ABBR, TARGET_COLS, VAL_DATE, VAL_TYPE
from europeennes_experiment import build_augmented_base, prepare_condition
from europeennes_typed_oof import _fit_predict
from europeennes_lagiso_oof import build_lp_sourced_lags, LP_TYPES

PCA_KS: list[int | None] = [None, 5, 7, 10]


def _legi_train_dates(df):
    legi = df[(df["election_type"] == VAL_TYPE)]
    dates = sorted(
        d for d in legi["date_float"].round(4).unique() if abs(d - VAL_DATE) > 0.05
    )
    return dates


def main() -> None:
    data_dir = Path("data")
    t0 = time.time()

    df_base, demo, national_means = build_augmented_base(data_dir)
    designs = {
        "Legi-only": prepare_condition(df_base, demo, [VAL_TYPE]),
        "CT(Legi+Pres)": prepare_condition(df_base, demo, LP_TYPES),
        "Euro-rows": build_lp_sourced_lags(df_base, demo),
    }

    dev_lags = [f"dev_{b}_lag{j}" for b in BLOCKS_ABS for j in (1, 2)]
    type_cols = [c for c in designs["Euro-rows"].columns if c.startswith("type_")]
    feat_cols = demo + dev_lags + type_cols
    n_demo = len(demo)

    legi_dates = _legi_train_dates(designs["Legi-only"])
    # common held-out legislative BVs per fold (complete-case in all designs)
    held_locs = {}
    for d in legi_dates:
        sets = []
        for df in designs.values():
            m = (df["election_type"] == VAL_TYPE) & np.isclose(
                df["date_float"], d, atol=1e-3
            )
            sets.append(set(df.loc[m, "location"]))
        held_locs[d] = set.intersection(*sets)
    print(f"Held-out legislative folds: {legi_dates}")
    print(f"Common held BVs/fold: {[len(held_locs[d]) for d in legi_dates]}\n")

    nat = {
        d: national_means[
            (national_means["election_type"] == VAL_TYPE)
            & np.isclose(national_means["date_float"], d, atol=1e-3)
        ]
        for d in legi_dates
    }

    # results[block][design][cfg] = oof_r2
    results = {tc: {dn: {} for dn in designs} for tc in TARGET_COLS}
    for k in PCA_KS:
        tag = "raw" if k is None else f"PCA{k}"
        for dn, df in designs.items():
            val_m = (df["election_type"] == VAL_TYPE) & np.isclose(
                df["date_float"], VAL_DATE, atol=1e-3
            )
            train_all = df[~val_m].reset_index(drop=True)
            acc = {tc: ([], []) for tc in TARGET_COLS}  # (y_true, pred)
            for d in legi_dates:
                hl = held_locs[d]
                held_mask = (
                    (train_all["election_type"] == VAL_TYPE)
                    & np.isclose(train_all["date_float"], d, atol=1e-3)
                    & train_all["location"].isin(hl)
                )
                held = train_all[held_mask].sort_values("location")
                tr = train_all[~held_mask]
                X_tr = tr[feat_cols].values.astype(np.float64)
                X_h = held[feat_cols].values.astype(np.float64)
                nat_v = {tc: float(nat[d][tc].iloc[0]) for tc in TARGET_COLS}
                for tc in TARGET_COLS:
                    pred = (
                        _fit_predict(
                            X_tr,
                            tr[f"dev_{tc}"].values.astype(np.float64),
                            X_h,
                            n_demo,
                            k,
                        )
                        + nat_v[tc]
                    )
                    acc[tc][0].append(held[tc].values.astype(np.float64))
                    acc[tc][1].append(pred)
            for tc in TARGET_COLS:
                y = np.concatenate(acc[tc][0])
                p = np.concatenate(acc[tc][1])
                results[tc][dn][tag] = r2_score(y, p)
        print(f"  [{tag}] done", flush=True)

    print(f"\n{'#' * 78}\nHELD-OUT-LEGISLATIVE OOF R² per design × config\n{'#' * 78}")
    for tc in TARGET_COLS:
        print(f"\n{tc} ({ABBR[tc]}):")
        for dn in designs:
            row = "  ".join(
                f"{tag}={results[tc][dn][tag]:.4f}"
                for tag in ("raw", "PCA5", "PCA7", "PCA10")
            )
            print(f"  {dn:16s} {row}")

    print(
        f"\n{'#' * 78}\nSELECTION — best OOF per block (which design wins?)\n{'#' * 78}"
    )
    for tc in TARGET_COLS:
        best_dn, best_tag, best = "", "", -9.0
        for dn in designs:
            for tag, v in results[tc][dn].items():
                if v > best:
                    best, best_dn, best_tag = v, dn, tag
        euro_best = max(results[tc]["Euro-rows"].values())
        non_euro_best = max(
            max(results[tc][dn].values()) for dn in designs if dn != "Euro-rows"
        )
        delta = euro_best - non_euro_best
        winner = "EURO-ROWS" if best_dn == "Euro-rows" else best_dn
        print(
            f"  {tc:16s} → {winner:14s} ({best_tag}, OOF={best:.4f})   "
            f"euro_best−other_best={delta:+.4f}"
        )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
