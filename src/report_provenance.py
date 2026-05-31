"""Provenance de l'incertitude : part « national (sondages) » vs « lecture locale ».

Rejoue la calibration conforme dans deux modes (cf. `src/conformal.py`) :
  - oracle    : niveau national = vraie moyenne  -> erreur du modèle par bureau
  - réaliste  : niveau national = estimation sondagière -> modèle + sondages
L'erreur sondagière étant un décalage national commun à tous les bureaux d'une
élection, `var(réaliste) - var(oracle)` isole ce que les sondages pèsent dans
l'incertitude d'un bureau. Sorties : `provenance.json` + `fig_provenance.png`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.linewidth": 0.6})

from src.conformal import (
    BEST_RIDGE,
    BLOCKS_ABS,
    run_conformal_for_block,
    split_tv,
)
from src.cross_type_dev import (
    ABBR,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
    add_election_type_onehot,
    estimate_national_abstention_from_gaps,
    load_cross_type_data,
)
from src.cross_type_ridge import TARGET_BLOCKS

OUT = Path("report_app/data")
DATA = Path("data")
SHORT = {"G": "G", "CD": "CD", "ED": "ED", "Ab": "AB"}
COLOR = {"G": "#E4572E", "CD": "#4A90D9", "ED": "#6A4C93", "AB": "#9AA0A6"}
PAPER, INK, POLLS = "#FAFAF7", "#1A1A2E", "#C9CCD2"
NAME = {
    "G": "Gauche",
    "CD": "Centre+Droite",
    "ED": "Extrême Droite",
    "AB": "Abstention",
}


def _loo_nat_ests(poll_feats, national_means) -> dict:
    loo: dict = {}
    for _, r in poll_feats.iterrows():
        edate = round(float(r["date_float"]), 3)
        if edate > VAL_DATE - 0.1:
            continue
        loo[(r["election_type"], edate)] = {
            b: float(r[f"poll_{b}"]) for b in TARGET_BLOCKS
        }
    nm = national_means.sort_values("date_float").reset_index(drop=True)
    tr = nm[nm["date_float"] < VAL_DATE - 0.1].copy()
    g = tr.copy()
    g["gap_years"] = g["date_float"].diff()
    g = g.dropna(subset=["gap_years"]).reset_index(drop=True)
    X, y = g[["gap_years"]].values, g["Abstention"].values
    for i in range(len(g)):
        m = np.ones(len(g), dtype=bool)
        m[i] = False
        pred = float(LinearRegression().fit(X[m], y[m]).predict(X[[i]])[0])
        key = (g.iloc[i]["election_type"], round(float(g.iloc[i]["date_float"]), 3))
        if key in loo:
            loo[key]["Abstention"] = pred
    for _, row in tr.iterrows():
        key = (row["election_type"], round(float(row["date_float"]), 3))
        if key in loo and "Abstention" not in loo[key]:
            loo[key]["Abstention"] = row["Abstention"]
    return loo


def _datasets(df, demo_indicators, type_cols):
    lags = {
        s: [f"{p}{b}{s2}" for b in BLOCKS_ABS]
        for s, (p, s2) in {
            "r1": ("", "_lag1"),
            "r2": ("", "_lag2"),
            "d1": ("dev_", "_lag1"),
            "d2": ("dev_", "_lag2"),
        }.items()
    }
    need = lags["r1"] + lags["r2"] + lags["d1"] + lags["d2"]
    df_ct = df.dropna(subset=demo_indicators).dropna(subset=need)
    df_legi = df[df["election_type"] == VAL_TYPE].dropna(subset=demo_indicators)
    df_legi = df_legi.dropna(subset=need)
    nd_ct = lags["d1"] + lags["d2"] + type_cols
    nd_legi = lags["d1"] + lags["d2"]
    datasets = {"ct_v1_2": df_ct, "legi_v1_2": df_legi}
    feat_maps = {
        "ct": (demo_indicators, demo_indicators + nd_ct),
        "legi": (demo_indicators, demo_indicators + nd_legi),
    }
    return datasets, feat_maps


def compute() -> dict:
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(DATA)
    type_cols = add_election_type_onehot(df)
    abs_pred, _ = estimate_national_abstention_from_gaps(national_means)
    val_est = {
        b: float(
            poll_feats[
                np.isclose(poll_feats["date_float"], VAL_DATE, atol=0.1)
                & (poll_feats["election_type"] == VAL_TYPE)
            ][f"poll_{b}"].iloc[0]
        )
        for b in TARGET_BLOCKS
    }
    val_est["Abstention"] = abs_pred
    loo_nat = _loo_nat_ests(poll_feats, national_means)
    nm_val = national_means[
        (national_means["election_type"] == VAL_TYPE)
        & np.isclose(national_means["date_float"], VAL_DATE, atol=0.1)
    ]
    actual_nat = {tc: float(nm_val[tc].iloc[0]) for tc in TARGET_COLS}
    datasets, feat_maps = _datasets(df, demo_indicators, type_cols)

    def _r2(y: np.ndarray, p: np.ndarray) -> float:
        return round(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum(), 2)

    blocks: dict[str, dict[str, float]] = {}
    for tc in TARGET_COLS:
        _, data_key, feat_key, cfg = BEST_RIDGE[tc]
        demo_cols, all_cols = feat_maps[feat_key]
        cfg = dict(cfg) | {"n_demo": len(demo_cols)}
        train, val = split_tv(datasets[data_key])
        train = train[train[all_cols].notna().all(axis=1)].copy()
        val = val[val[all_cols].notna().all(axis=1)].copy()
        common = (tc, train, val, all_cols, demo_cols, val_est, national_means, cfg)
        oracle = run_conformal_for_block(*common, loo_national_ests=None)
        real = run_conformal_for_block(*common, loo_national_ests=loo_nat)
        sd_m = float(np.std(oracle["cal_residuals"]))
        sd_t = float(np.std(real["cal_residuals"]))
        share = max(sd_t**2 - sd_m**2, 0.0) / sd_t**2 if sd_t > 0 else 0.0
        y = oracle["y_true_val"]
        # r2_real: prediction anchored on the polls (what we actually ship).
        # r2_oracle: same local deviation, but anchored on the TRUE national level —
        # isolates the bureau-level skill, free of the national poll error.
        r2_real = _r2(y, oracle["val_pred"])
        r2_oracle = _r2(y, oracle["val_dev_pred"] + actual_nat[tc])
        blocks[SHORT[ABBR[tc]]] = {
            "national_share": round(100 * share, 1),
            "local_share": round(100 * (1 - share), 1),
            "sd_model": round(sd_m, 2),
            "sd_total": round(sd_t, 2),
            "q90_oracle": round(oracle["intervals"][90]["std_quantile"], 2),
            "q90_real": round(real["intervals"][90]["std_quantile"], 2),
            "r2_real": r2_real,
            "r2_oracle": r2_oracle,
        }
    return {"blocks": blocks}


def _figure(prov: dict) -> None:
    blocks = prov["blocks"]
    order = sorted(blocks, key=lambda b: blocks[b]["national_share"])
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    y = np.arange(len(order))
    for i, b in enumerate(order):
        loc = blocks[b]["local_share"]
        nat = blocks[b]["national_share"]
        ax.barh(i, loc, color=COLOR[b], edgecolor=PAPER)
        ax.barh(i, nat, left=loc, color=POLLS, edgecolor="#8a8f99", hatch="////")
        ax.text(
            loc / 2,
            i,
            f"{loc:.0f} %",
            va="center",
            ha="center",
            color=PAPER,
            fontsize=8,
            fontweight="bold",
        )
        ax.text(
            loc + nat / 2,
            i,
            f"{nat:.0f} %",
            va="center",
            ha="center",
            color=INK,
            fontsize=8,
        )
    ax.set_yticks(y)
    ax.set_yticklabels([NAME[b] for b in order], fontsize=9)
    ax.set_xlim(0, 100)
    ax.set_xticks([])
    for s in ("top", "right", "bottom", "left"):
        ax.spines[s].set_visible(False)
    ax.set_title(
        "D'où vient l'incertitude d'un bureau : notre lecture locale\n"
        "(couleur du bloc) contre le national des sondages (gris)",
        fontsize=9.5,
        loc="left",
    )
    fig.tight_layout()
    fig.savefig(OUT / "fig_provenance.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    prov = compute()
    (OUT / "provenance.json").write_text(json.dumps(prov, ensure_ascii=False, indent=1))
    _figure(prov)
    print(
        "provenance:",
        {b: prov["blocks"][b]["national_share"] for b in prov["blocks"]},
    )


if __name__ == "__main__":
    build()
