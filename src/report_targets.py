"""Gisement « électeurs convaincables » de gauche (en individus, pas en bureaux).

Un seul gisement défendable : la **mobilisation**. Les abstentionnistes qui penchent
à gauche d'un bureau = abstentionnistes(b) × γ(b), où γ(b) est la **part de gauche du
votant marginal** lue sur la courbe participation (`movability_turnout`, MOVABILITY.md
§11) en fonction du niveau de gauche du bureau — quantité identifiée et stable, pas le
partage des exprimés locaux (circulaire, surestime jusqu'à 17 pts en bastion).
Abstention et niveau de gauche sont **prédits** (`pred_*`), pas observés : le livrable
est une prévision du prochain scrutin.

Le « potentiel latent » (attente démographique − réalisé) a été retiré : c'est un
résidu non identifié, de signe ambigu, qui ne repose sur aucun signal spatial
vérifiable (MOVABILITY.md §2/§10). Le total national d'un basculement reste de
l'arithmétique ; l'apport du modèle est la *répartition spatiale* du gisement. On
exclut l'outre-mer / l'étranger d'un ordre de déploiement de terrain (non démarchable,
erreur concentrée — voir le brief)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import movability_turnout

VOTE = ["G", "CD", "ED"]


def mainland(df: pd.DataFrame) -> np.ndarray:
    dep = df.code_departement.astype(str)
    return ~(dep.str.startswith("Z") | dep.str.match(r"9[789]")).to_numpy()


def mobilizable(df: pd.DataFrame, curve: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Par bureau : abstentionnistes prédits × γ(niveau de gauche prédit).

    Prédit, pas réalisé : le livrable est une prévision du prochain scrutin, donc
    l'abstention et le niveau de gauche viennent du modèle (`pred_*`), pas des
    résultats observés — qui ne seraient connus qu'après le vote."""
    ins = df.inscrits.to_numpy().astype(float)
    abstainers = ins * df.pred_AB.to_numpy() / 100.0
    gamma = movability_turnout.apply_gamma(curve, df.pred_G.to_numpy()) / 100.0
    return abstainers * gamma


def deployment(
    df: pd.DataFrame, mob: np.ndarray, ml: np.ndarray, top: int
) -> list[dict[str, object]]:
    sub = df.assign(_m=mob)[ml]
    g = (
        sub.groupby(["libelle_commune", "code_departement"])
        .agg(mob=("_m", "sum"), bv=("inscrits", "size"))
        .sort_values("mob", ascending=False)
        .head(top)
        .reset_index()
    )
    return [
        {
            "nom": r.libelle_commune,
            "dept": r.code_departement,
            "mob": int(r.mob),
            "bv": int(r.bv),
        }
        for r in g.itertuples()
    ]


def left_gain(
    df: pd.DataFrame, curve: tuple[np.ndarray, np.ndarray]
) -> tuple[dict[str, object], np.ndarray]:
    """Bloc résumé (mainland) + tableau par bureau pour la couche carte client."""
    mob = mobilizable(df, curve)
    ml = mainland(df)
    abstainers = df.inscrits.to_numpy() * df.pred_AB.to_numpy() / 100.0
    gamma = movability_turnout.apply_gamma(curve, df.pred_G.to_numpy())
    summary = {
        "method": "part de gauche du votant marginal (courbe participation)",
        "mobilization_voters": int(mob[ml].sum()),
        "total_abstainers": int(abstainers[ml].sum()),
        "gamma_mean": round(float(np.average(gamma[ml], weights=abstainers[ml])), 1),
        "deployment": deployment(df, mob, ml, top=12),
    }
    return summary, mob
