"""Gisement « électeurs convaincables » de gauche (en individus, pas en bureaux).

Un seul gisement défendable : la **mobilisation**. Les abstentionnistes qui penchent
à gauche d'un bureau = abstentionnistes **conjoncturels**(b) × γ(b), où γ(b) est la
**part de gauche du votant marginal** lue sur la courbe participation du **type de
scrutin projeté** (`movability_turnout`, MOVABILITY.md §11/§14) en fonction du niveau de
gauche du bureau — quantité identifiée et stable, pas le partage des exprimés locaux
(circulaire, surestime jusqu'à 17 pts en bastion).

**Abstention de fond vs conjoncturelle.** On ne mobilise pas l'abstentionniste chronique
(qui ne vote jamais, même quand l'enjeu monte). Le gisement de fin de campagne, c'est la
frange **conjoncturelle** : ceux qui votent quand la participation grimpe. On l'isole par
le plancher d'abstention historique du bureau (`abst_floor`, le minimum jamais atteint) :
`conjoncturelle = max(0, abstention prédite − plancher)`. Abstention et niveau de gauche
sont **prédits** (`pred_*`), pas observés : le livrable est une prévision.

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


def conjunctural_pct(df: pd.DataFrame) -> np.ndarray:
    """Part conjoncturelle de l'abstention (% inscrits) = abstention prédite − plancher
    historique du bureau, plancher = abstention de fond (chronique). Clip à 0."""
    floor = df.abst_floor.to_numpy() if "abst_floor" in df else df.pred_AB.to_numpy()
    return np.clip(df.pred_AB.to_numpy() - floor, 0.0, None)


def mobilizable(df: pd.DataFrame, curve: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Par bureau : abstentionnistes **conjoncturels** prédits × γ(niveau de gauche prédit).

    Prédit, pas réalisé : le livrable est une prévision du prochain scrutin, donc
    l'abstention et le niveau de gauche viennent du modèle (`pred_*`), pas des
    résultats observés — qui ne seraient connus qu'après le vote."""
    ins = df.inscrits.to_numpy().astype(float)
    conj = ins * conjunctural_pct(df) / 100.0
    gamma = movability_turnout.apply_gamma(curve, df.pred_G.to_numpy()) / 100.0
    return conj * gamma


def deployment(
    df: pd.DataFrame, mob: np.ndarray, ml: np.ndarray, top: int
) -> list[dict[str, object]]:
    """Communes par gisement mobilisable. Sert **volume** (mob, personnes) ET
    **rendement** (γ, % des conjoncturels qui penchent à gauche) : le volume suit la
    taille de la ville, le rendement est l'apport propre du modèle (le tri par γ dit où
    chaque porte frappée rapporte le plus)."""
    conj = df.inscrits.to_numpy().astype(float) * conjunctural_pct(df) / 100.0
    sub = df.assign(_m=mob, _c=conj)[ml]
    g = (
        sub.groupby(["libelle_commune", "code_departement"])
        .agg(mob=("_m", "sum"), conj=("_c", "sum"), bv=("inscrits", "size"))
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
            "gamma": round(100 * r.mob / r.conj, 1) if r.conj else 0.0,
        }
        for r in g.itertuples()
    ]


def left_gain(
    df: pd.DataFrame, curve: tuple[np.ndarray, np.ndarray]
) -> tuple[dict[str, object], np.ndarray]:
    """Bloc résumé (mainland) + tableau par bureau pour la couche carte client."""
    mob = mobilizable(df, curve)
    ml = mainland(df)
    ins = df.inscrits.to_numpy().astype(float)
    abstainers = ins * df.pred_AB.to_numpy() / 100.0
    conj = ins * conjunctural_pct(df) / 100.0
    gamma = movability_turnout.apply_gamma(curve, df.pred_G.to_numpy())
    summary = {
        "method": "abstentionnistes conjoncturels × part de gauche du votant marginal (γ législatif)",
        "mobilization_voters": int(mob[ml].sum()),
        "total_abstainers": int(abstainers[ml].sum()),
        "conjunctural_abstainers": int(conj[ml].sum()),
        "structural_abstainers": int((abstainers[ml] - conj[ml]).sum()),
        "gamma_mean": round(float(np.average(gamma[ml], weights=conj[ml])), 1),
        "deployment": deployment(df, mob, ml, top=12),
    }
    return summary, mob
