"""Test : γ (part de gauche du votant marginal) dépend-il de *tout* (démographie,
lags, momentum) au-delà du seul niveau de gauche ? — et est-ce que ça transfère ?

γ reste **identifié sur les vraies hausses de participation** : par paire de scrutins
consécutifs de même type, r = ΔLR/ΔT (ΔLR = Δ gauche % inscrits, ΔT = Δ participation),
pondéré par ΔT² (dans une feuille, l'optimum pondéré = la pente sans constante = γ). Un
HistGBM apprend γ(x) à partir de features **pré-swing** (connues avant le scrutin :
niveau de gauche passé, lags de dépôt, démographie INSEE, géo, type) — la démographie
n'est qu'un *prédicteur de la pente identifiée*, jamais un substitut de niveau (le piège
du résidu jumeaux). Backtest hors échantillon : on apprend ≤ fit_max, on prédit le
mouvement legi suivant, on compare γ riche vs courbe 1-D (niveau) vs γ plat (swing
uniforme). Métrique clé `het_corr` : la part hétérogène (γ̂ − γ_plat)·ΔT corrèle-t-elle
au vrai écart au swing uniforme ?
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from src import movability_turnout as mt

CACHE = Path("data/baseline_cache/cross_type_dev_base.parquet")
INDIC = Path("data/baseline_cache/cross_type_dev_indicators.txt")
LEGI = "Legislatives_T1"
BLOCKS = ["Gauche", "Centre+Droite", "Extreme_Droite", "Abstention"]


@dataclass
class Obs:
    df: pd.DataFrame
    feats: list[str]


def load() -> Obs:
    indic = INDIC.read_text().strip().split("\n")
    lags = [f"{b}_lag1" for b in BLOCKS] + [f"{b}_lag2" for b in BLOCKS]
    devs = [f"dev_{b}_lag1" for b in BLOCKS] + [f"dev_{b}_lag2" for b in BLOCKS]
    cols = [
        "location",
        "election_type",
        "date_float",
        "Gauche",
        "Abstention",
        "latitude",
        "longitude",
        *lags,
        *devs,
        *indic,
    ]
    df = pd.read_parquet(CACHE, columns=cols)
    df["T"] = 100.0 - df["Abstention"]
    df["LR"] = df["T"] * df["Gauche"] / 100.0
    df = df.sort_values(["location", "election_type", "date_float"])
    g = df.groupby(["location", "election_type"], sort=False)
    df["dT"] = df["T"] - g["T"].shift(1)
    df["dLR"] = df["LR"] - g["LR"].shift(1)
    df["prevG"] = g["Gauche"].shift(1)
    df["mom"] = df["dev_Gauche_lag1"] - df["dev_Gauche_lag2"]
    feats = ["prevG", "mom", "latitude", "longitude", *lags, *devs, *indic]
    df = df[np.abs(df["dT"]) > 0.5]
    df = df.dropna(subset=["dT", "dLR", "prevG"])
    df["r"] = df["dLR"] / df["dT"]
    df["w"] = df["dT"] ** 2
    return Obs(df.reset_index(drop=True), feats)


# Régularisation croissante : du modèle « moyen » (run précédent) au « très fort »
# (peu de capacité, gros l2, feuilles larges → tend vers le plat s'il n'y a rien).
GBM_GRID: list[tuple[str, dict[str, object]]] = [
    (
        "gbm-moyen",
        dict(
            max_depth=3,
            min_samples_leaf=300,
            l2_regularization=1.0,
            learning_rate=0.05,
            max_iter=400,
        ),
    ),
    (
        "gbm-fort",
        dict(
            max_depth=2,
            min_samples_leaf=3000,
            l2_regularization=10.0,
            learning_rate=0.03,
            max_iter=400,
        ),
    ),
    (
        "gbm-tres-fort",
        dict(
            max_depth=2,
            min_samples_leaf=12000,
            l2_regularization=30.0,
            learning_rate=0.02,
            max_iter=300,
            max_leaf_nodes=4,
        ),
    ),
]
RESID_PARAMS = dict(
    max_depth=2,
    min_samples_leaf=8000,
    l2_regularization=20.0,
    learning_rate=0.02,
    max_iter=300,
    max_leaf_nodes=4,
)


def fit_gbm(
    tr: pd.DataFrame, feats: list[str], y: np.ndarray, params: dict[str, object]
) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(
        loss="squared_error",
        early_stopping=True,
        validation_fraction=0.15,
        random_state=0,
        **params,
    )
    model.fit(tr[feats], y, sample_weight=tr["w"])
    return model


def curve_1d(tr: pd.DataFrame, level: np.ndarray, nbins: int = 20) -> np.ndarray:
    t = pd.DataFrame({"dT": tr.dT, "dLR": tr.dLR, "G": tr.prevG})
    res = mt.gamma_curve(t, nbins).sort_values("G_moyen")
    return np.clip(
        np.interp(level, res.G_moyen.to_numpy(), res.gamma_pct.to_numpy() / 100), 0, 1
    )


def flat(tr: pd.DataFrame) -> float:
    return float((tr.w * tr.r).sum() / tr.w.sum())


def backtest(o: Obs, fit_max: float, d_to: float) -> dict[str, tuple[float, float]]:
    """Apprend sur tout pair ≤ fit_max ; teste sur la transition legi → d_to.
    Rend {méthode: (RMSE de ΔLR, het_corr vs plat)}."""
    tr = o.df[o.df.date_float <= fit_max + 0.01]
    te = o.df[(o.df.election_type == LEGI) & np.isclose(o.df.date_float, d_to)]
    if not len(te):
        return {}
    dT, actual = te.dT.to_numpy(), te.dLR.to_numpy()
    gf = flat(tr)
    preds: dict[str, np.ndarray | float] = {
        "flat": gf,
        "courbe1d": curve_1d(tr, te.prevG.to_numpy()),
    }
    for name, p in GBM_GRID:
        preds[name] = np.clip(
            fit_gbm(tr, o.feats, tr.r.to_numpy(), p).predict(te[o.feats]), 0, 1
        )
    gc_tr = curve_1d(tr, tr.prevG.to_numpy())
    resid_model = fit_gbm(tr, o.feats, tr.r.to_numpy() - gc_tr, RESID_PARAMS)
    preds["courbe1d+gbm-resid"] = np.clip(
        curve_1d(tr, te.prevG.to_numpy()) + resid_model.predict(te[o.feats]), 0, 1
    )

    resid = actual - gf * dT
    out = {}
    for name, g in preds.items():
        rmse = float(np.sqrt(np.mean((actual - g * dT) ** 2)))
        het = 0.0 if np.isscalar(g) else float(np.corrcoef((g - gf) * dT, resid)[0, 1])
        out[name] = (round(rmse, 4), round(het, 4))
    return out


def perstation_slopes(tr: pd.DataFrame) -> pd.DataFrame:
    """Pente γ_b par station depuis SON propre historique de hausses de participation
    (sans constante : γ = ΣΔTΔLR/ΣΔT²), SE par résidus (dof = n−1), niveau moyen."""
    g = tr.groupby("location")
    a = g.agg(
        n=("dT", "size"),
        sxx=("dT", lambda v: float((v * v).sum())),
        syy=("dLR", lambda v: float((v * v).sum())),
        level=("prevG", "mean"),
    )
    a["sxy"] = g.apply(lambda d: float((d.dT * d.dLR).sum()), include_groups=False)
    a["gamma_raw"] = a.sxy / a.sxx
    rss = np.maximum(a.syy - a.sxy**2 / a.sxx, 1e-9)
    a["se2"] = np.where(a.n >= 2, rss / np.maximum(a.n - 1, 1) / a.sxx, np.inf)
    return a


def perstation_gamma(tr: pd.DataFrame, te: pd.DataFrame) -> np.ndarray:
    """γ par station test = courbe 1-D (socle) + écart propre rétréci (empirical Bayes
    vers la courbe). Stations sans historique informatif → courbe pure."""
    a = perstation_slopes(tr)
    prior = curve_1d(tr, a.level.to_numpy())  # μ_b = courbe au niveau de la station
    have = np.isfinite(a.gamma_raw.to_numpy()) & np.isfinite(a.se2.to_numpy())
    dev = a.gamma_raw.to_numpy() - prior
    w = np.where(have, 1.0 / np.maximum(a.se2.to_numpy(), 1e-9), 0.0)
    wv = w[have] / w[have].sum()
    tau2 = max(
        0.0, float(np.sum(wv * dev[have] ** 2)) - float(np.mean(a.se2.to_numpy()[have]))
    )
    k = np.where(have, tau2 / (tau2 + a.se2.to_numpy()), 0.0)
    shrunk_dev = pd.Series(np.where(have, k * dev, 0.0), index=a.index)
    base = curve_1d(tr, te.prevG.to_numpy())
    add = te.location.map(shrunk_dev).fillna(0.0).to_numpy()
    print(
        f"      [perstation] τ²={tau2:.5f} (variance inter-station de γ au-delà de la courbe) · "
        f"k moyen={float(np.nanmean(k[have])):.3f} · stations avec historique={int(have.sum()):,} · "
        f"part rétrécie vers la courbe (k<0,1)={float((k[have] < 0.1).mean()):.1%}"
    )
    return np.clip(base + add, 0, 1)


def run() -> None:
    o = load()
    print(
        f"{len(o.df):,} transitions station×scrutin | {len(o.feats)} features pré-swing"
    )
    print("\nBacktest hors échantillon (RMSE de ΔLR ; plus bas = mieux) :")
    for fit_max, d_to in [(2017.5, 2022.5), (2022.5, 2024.5)]:
        res = backtest(o, fit_max, d_to)
        if not res:
            continue
        best = min(res, key=lambda k: res[k][0])
        print(f"\n  ≤{fit_max:.0f} → legi {d_to:.0f} :")
        for name, (rmse, het) in res.items():
            mark = "  ← meilleur" if name == best else ""
            print(f"    {name:22s} RMSE {rmse:7.4f}   het {het:+.3f}{mark}")


def run_perstation() -> None:
    o = load()
    print(
        f"{len(o.df):,} transitions | per-station γ (historique propre) vs courbe 1-D"
    )
    for fit_max, d_to in [(2017.5, 2022.5), (2022.5, 2024.5)]:
        tr = o.df[o.df.date_float <= fit_max + 0.01]
        te = o.df[(o.df.election_type == LEGI) & np.isclose(o.df.date_float, d_to)]
        if not len(te):
            continue
        dT, actual = te.dT.to_numpy(), te.dLR.to_numpy()
        gf = flat(tr)
        rmse = lambda g: float(np.sqrt(np.mean((actual - g * dT) ** 2)))
        resid = actual - gf * dT
        het = lambda g: float(np.corrcoef((g - gf) * dT, resid)[0, 1])
        gc = curve_1d(tr, te.prevG.to_numpy())
        print(f"\n  ≤{fit_max:.0f} → legi {d_to:.0f} (n={len(te):,}) :")
        gp = perstation_gamma(tr, te)
        print(f"    flat                 RMSE {rmse(gf):7.4f}")
        print(f"    courbe1d             RMSE {rmse(gc):7.4f}   het {het(gc):+.3f}")
        print(f"    courbe1d+perstation  RMSE {rmse(gp):7.4f}   het {het(gp):+.3f}")


if __name__ == "__main__":
    import sys

    run() if "--gbm" in sys.argv else run_perstation()
