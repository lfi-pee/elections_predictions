"""Is the europeennes fair-OOF degradation a slope-pooling artifact?

§8 showed that adding européennes folds degrades the fold-fair OOF on held-out
Legi+Pres elections. Hypothesis (user): the model pools ONE coefficient vector
across types — type one-hot only shifts the intercept, slopes are shared — so
euro rows drag the Legi+Pres slopes. If true, giving euro its OWN slopes should
remove the harm (and might even help via partial pooling).

Three conditions, same held-out common Legi+Pres folds:
  base        : train Legi+Pres only
  euro_pooled : train Legi+Pres + euro, ONE shared slope vector (= §8)
  euro_typed  : train Legi+Pres + euro, euro gets its own slopes
                (features interacted with a euro indicator; held Legi+Pres rows
                 have euro_ind=0 so they read the Legi+Pres-anchored slopes)

Verdict logic:
  typed ≈ base  → pooling artifact; euro neutral for the target regime
  typed > base  → euro transfers once typed; bankable
  typed < base  → not a pooling artifact; euro genuinely does not transfer
"""

from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from src.cross_type_dev import BLOCKS_ABS, ABBR, TARGET_COLS
from europeennes_experiment import build_augmented_base, prepare_condition, EURO, TYPES
from europeennes_oof import _train_only, _transform, KEY, ALPHA_GRID

PCA_KS: list[int | None] = [None, 5]


def _fit_predict(X_tr, y_tr, X_ev, n_demo, k, euro_ind_tr=None):
    """Transform, optionally add euro-interacted slopes, Ridge-fit, predict.

    euro_ind_tr: per-train-row euro indicator. When given, the transformed
    features are duplicated and masked by euro_ind so euro rows fit a separate
    slope; eval rows (held Legi+Pres, euro_ind=0) read only the main slopes.
    """
    Z_tr, Z_ev = _transform(X_tr, X_ev, n_demo, k)
    if euro_ind_tr is not None:
        Z_tr = np.hstack([Z_tr, Z_tr * euro_ind_tr[:, None]])
        Z_ev = np.hstack([Z_ev, np.zeros_like(Z_ev)])
    alpha = RidgeCV(alphas=ALPHA_GRID).fit(Z_tr, y_tr).alpha_
    return Ridge(alpha=alpha, solver="cholesky").fit(Z_tr, y_tr).predict(Z_ev)


def typed_fold_oof(df_lp, df_euro, feat_cols, n_demo, national_means, k):
    """Per-block fair OOF on common held Legi+Pres folds: base/pooled/typed."""
    lp = _train_only(df_lp).set_index(KEY)
    eu = _train_only(df_euro)
    eu_lp = eu[eu["election_type"] != EURO].set_index(KEY)
    euro_rows = eu[eu["election_type"] == EURO]

    common = lp.index.intersection(eu_lp.index)
    lp = lp.loc[common].reset_index()
    eu_lp = eu_lp.loc[common].reset_index()

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
        preds = {m: np.full(len(lp), np.nan) for m in ("base", "pooled", "typed")}
        for et, dt in folds:
            held = np.isclose(lp["date_float"], dt, atol=1e-3) & (
                lp["election_type"] == et
            )
            tr = ~held
            nm = nat[(et, round(float(dt), 4))]
            nat_v = float(nm[tc].iloc[0]) if len(nm) else 0.0

            X_eu_tr = np.vstack([Xeu_lp[tr], Xeuro])
            y_eu_tr = np.concatenate([y_eu_lp[tr], y_euro])
            ind = np.concatenate([np.zeros(int(tr.sum())), np.ones(len(Xeuro))])

            preds["base"][held] = (
                _fit_predict(Xlp[tr], y_lp[tr], Xlp[held], n_demo, k) + nat_v
            )
            preds["pooled"][held] = (
                _fit_predict(X_eu_tr, y_eu_tr, Xeu_lp[held], n_demo, k) + nat_v
            )
            preds["typed"][held] = (
                _fit_predict(X_eu_tr, y_eu_tr, Xeu_lp[held], n_demo, k, ind) + nat_v
            )

        y_true = lp[tc].values.astype(np.float64)
        out[tc] = {m: r2_score(y_true, preds[m]) for m in preds}
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

    best = {
        tc: {"base": -9.0, "pooled": -9.0, "typed": -9.0, "tag": ""}
        for tc in TARGET_COLS
    }
    for k in PCA_KS:
        res = typed_fold_oof(df_lp, df_euro, feat_cols, n_demo, national_means, k)
        tag = "raw" if k is None else f"PCA{k}"
        print(f"\n[{tag}] fair OOF on held-out Legi+Pres folds:")
        for tc in TARGET_COLS:
            r = res[tc]
            print(
                f"  {ABBR[tc]:3s}  base={r['base']:.4f}  "
                f"pooled={r['pooled']:.4f} ({r['pooled'] - r['base']:+.4f})  "
                f"typed={r['typed']:.4f} ({r['typed'] - r['base']:+.4f})"
            )
            if r["typed"] > best[tc]["typed"]:
                best[tc] = {**r, "tag": tag}

    print(
        f"\n{'#' * 78}\nVERDICT — best-typed-config vs its own base, per block\n{'#' * 78}"
    )
    for tc in TARGET_COLS:
        r = best[tc]
        d = r["typed"] - r["base"]
        mark = "BEAT" if d > 0.0005 else ("~neutral" if abs(d) <= 0.005 else "WORSE")
        recovered = r["typed"] - r["pooled"]
        print(
            f"  {tc:16s} {r['tag']:5s}  base={r['base']:.4f}  pooled={r['pooled']:.4f}  "
            f"typed={r['typed']:.4f}  (typed-base={d:+.4f}, recovered vs pooled={recovered:+.4f})  [{mark}]"
        )
    print(f"\nTotal time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
