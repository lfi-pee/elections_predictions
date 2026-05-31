"""Défendabilité du plancher d'abstention : reproduit toute la chaîne de décision.

Le gisement mobilisation = `inscrits · conjoncturelle/100 · γ`, où `conjoncturelle =
max(0, abstention prédite − plancher)`. Tout le débat porte sur le **plancher** (γ est
quasi ponctuel — voir la bande bootstrap en fin de run). On compare ici, sur les mêmes
prédictions, les estimateurs de plancher envisagés et on chiffre les deux failles qui
ont écarté les premiers :

1. **Régime** : plancher poolé (legi+présid+euro) → le min est dominé par la présidentielle,
   il importe le décalage structurel legi↔présid dans le « conjoncturel ».
2. **Fuite de la cible** : la cible 2024 est le législatif le plus mobilisé ; un min qui
   l'inclut vaut sa valeur observée 2024 pour ~56 % des bureaux → `conjoncturelle` dégénère
   en résidu de prédiction (corrélé au résidu `pred − observé₂₀₂₄`).

Estimateur retenu (B) : **min des législatives strictement passées**, en niveau observé.
La renormalisation national+local (C) est calculée pour montrer pourquoi elle est écartée
(planchers hors bornes, poches « mobilisable » impossibles par bureau).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import movability_turnout, report_data, report_targets

LEGI = "Legislatives_T1"
CUTOFF = report_data.TARGET_FLOOR_CUTOFF


def _panel() -> pd.DataFrame:
    p = pd.read_parquet(
        report_data.TURNOUT_CACHE,
        columns=["location", "election_type", "date_float", "Abstention"],
    )
    return p[p.election_type == LEGI]


def _gisement(
    df: pd.DataFrame, floor: pd.Series, curve, ml: np.ndarray, ins: np.ndarray
) -> tuple[float, float, float, float]:
    fl = (
        df.merge(floor.rename("f"), left_on="location", right_index=True, how="left")
        .f.fillna(df.pred_AB)
        .clip(upper=df.pred_AB)
    )
    conj = np.clip(df.pred_AB.to_numpy() - fl.to_numpy(), 0.0, None)
    gamma = movability_turnout.apply_gamma(curve, df.pred_G.to_numpy()) / 100.0
    mob = ins * conj / 100.0 * gamma
    return (
        mob[ml].sum(),
        (ins * conj / 100.0)[ml].sum(),
        float(fl.min()),
        float(conj.max()),
    )


def gamma_band(draws: int = 200) -> tuple[float, float]:
    t = movability_turnout.panel_diffs(LEGI)
    t = t.assign(bin=pd.qcut(t["G"], 20, labels=False, duplicates="drop"))
    rng = np.random.default_rng(0)
    groups = [g for _, g in t.groupby("bin")]
    means = []
    for _ in range(draws):
        gv = []
        for g in groups:
            i = rng.integers(0, len(g), len(g))
            dT, dLR = g.dT.to_numpy()[i], g.dLR.to_numpy()[i]
            sxx = float((dT**2).sum())
            gv.append((dLR * dT).sum() / sxx * 100 if sxx else np.nan)
        means.append(np.nanmean(gv))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def run() -> None:
    df = report_data.load_wide().merge(
        report_data.load_context(), on="location", how="left"
    )
    legi = _panel()
    pool = pd.read_parquet(
        report_data.TURNOUT_CACHE, columns=["location", "election_type", "Abstention"]
    )
    curve = movability_turnout.fit_gamma(election_type=LEGI)
    ml = report_targets.mainland(df)
    ins = df.inscrits.to_numpy().astype(float)
    past = legi[legi.date_float < CUTOFF]

    nat = legi.groupby(legi.date_float.round(1)).Abstention.transform("mean")
    anchor = float(df.pred_AB.mean())
    renorm = (
        anchor
        + (legi[legi.date_float < CUTOFF].assign(d=legi.Abstention - nat))
        .groupby("location")
        .d.min()
    )

    estimators = {
        "poolé 3 types (régime mêlé)": pool.groupby("location").Abstention.min(),
        "legi, AVEC cible 2024 (fuite)": legi.groupby("location").Abstention.min(),
        "B — legi PASSÉES (retenu)": past.groupby("location").Abstention.min(),
        "C — renorm national+local": renorm,
    }
    print(
        f"{'estimateur':>32} {'gisement':>9} {'conj':>7} {'floor_min':>10} {'conj_max%':>10}"
    )
    for name, fl in estimators.items():
        mv, conj, fmin, cmax = _gisement(df, fl, curve, ml, ins)
        print(
            f"{name:>32} {mv / 1e6:>7.2f}M {conj / 1e6:>5.2f}M {fmin:>10.1f} {cmax:>10.1f}"
        )

    # Fuite chiffrée : part des bureaux dont le min legi == valeur observée 2024.
    obs24 = (
        legi[legi.date_float.round(1) == 2024.5]
        .set_index("location")
        .Abstention.astype(float)
    )
    d = df.merge(obs24.rename("o"), left_on="location", right_index=True, how="left")
    floor_leak = legi.groupby("location").Abstention.min()
    fl = d.merge(
        floor_leak.rename("f"), left_on="location", right_index=True, how="left"
    )
    is24 = np.isclose(fl.f, d.o, atol=0.05)
    conj_leak = np.clip(d.pred_AB - fl.f, 0, None)
    print(
        f"\nFuite : {is24.mean():.0%} des bureaux ont min legi == observé 2024 ; "
        f"corr(conj, max(0,pred−obs24)) = {np.corrcoef(conj_leak, np.clip(d.pred_AB - d.o, 0, None))[0, 1]:.2f}"
    )

    lo, hi = gamma_band()
    print(
        f"γ moyen légi : bande bootstrap 95 % [{lo:.1f} ; {hi:.1f}] % — "
        "quasi ponctuel : toute la sensibilité du gisement est dans le plancher."
    )


if __name__ == "__main__":
    run()
