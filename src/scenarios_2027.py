"""Scénarios nationaux pour la prévision des Législatives 2027 — source unique.

2027 étant à venir, l'ancre nationale (Étape 1 du modèle : la moyenne nationale par
bloc) ne peut venir d'un résultat : elle est **posée par hypothèse**, réglable au
curseur sur le site. On fournit des **présélections** (`SCENARIOS`) ancrées sur les
intentions de vote 1er tour publiées (Ifop, OpinionWay, Elabe, Cluster17, Toluna-Harris,
juin–octobre 2025 ; agrégats Toute l'Europe / Wikipédia), voir
`data/polls/legislatives/legislatives_2027_hypotheses.csv`.

Les moyennes par bloc sont **renormalisées à 100** sur les trois blocs (Gauche,
Centre+Droite, Extrême Droite), comme les moyennes nationales historiques du modèle
(les « autres/divers » sont exclus). L'abstention est un axe à part (% des inscrits).

`left_config` / `radical_share` ne changent pas la prévision par bloc (le modèle prédit
le bloc Gauche entier) : ils pilotent la **jouabilité par circonscription** — une gauche
unie présente un seul candidat qui capte tout le bloc, une gauche divisée répartit le
bloc entre deux (ou trois) candidats, dont aucun ne pèse le total, ce qui change la
qualification au second tour.
"""

from __future__ import annotations


def _renorm3(g: float, cd: float, ed: float) -> dict[str, float]:
    s = g + cd + ed
    return {"G": round(100 * g / s, 1), "CD": round(100 * cd / s, 1), "ED": round(100 * ed / s, 1)}


# Abstention par défaut : une législative « à l'heure » (dans la foulée d'une
# présidentielle 2027) mobilise davantage qu'une législative de mi-mandat — on part de
# ~48 % d'abstention (contre ~52 % en 2022, ~57 % en 2017 hors effet présidentiel, et le
# creux à 33 % de la dissolution surprise de 2024). Réglable au curseur.
DEFAULT_ABSTENTION = 48.0

SCENARIOS = [
    {
        "key": "union",
        "label": "Union de la gauche large",
        "desc": "Un seul candidat de gauche par circonscription (type NFP/Front populaire). "
        "Tout le bloc Gauche derrière une candidature.",
        "means": {**_renorm3(24.3, 27.4, 32.3), "AB": DEFAULT_ABSTENTION},
        "left_config": "union",
        "radical_share": 1.0,
    },
    {
        "key": "split2",
        "label": "Gauche radicale vs néolibérale",
        "desc": "Scénario de référence : la gauche se scinde en un pôle radical (LFI) et un "
        "pôle social-démocrate/néolibéral (PS-Place publique-EELV-PCF) qui concourent "
        "séparément — le bloc est divisé entre deux candidatures.",
        "means": {**_renorm3(27.4, 26.0, 31.7), "AB": DEFAULT_ABSTENTION},
        "left_config": "split2",
        # Part du pôle radical dans le total de gauche : LFI 9,7 / (9,7+17,7) = 0,354.
        "radical_share": 0.354,
    },
    {
        "key": "frag",
        "label": "Fragmentation (statu quo)",
        "desc": "« Autre » : gauche dispersée sans pôle dominant (LFI / PS-PP / EELV-PCF "
        "chacun de son côté) et démobilisation partielle — aucune candidature ne fédère.",
        "means": {**_renorm3(20.0, 27.0, 33.0), "AB": DEFAULT_ABSTENTION + 4},
        "left_config": "split3",
        "radical_share": 0.35,
    },
    {
        "key": "droite_unie",
        "label": "Droites unies en face",
        "desc": "Une partie de LR fait alliance avec le RN (union des droites) : la barre à "
        "franchir pour la gauche (ici unie) monte fortement.",
        "means": {**_renorm3(24.3, 21.4, 38.3), "AB": DEFAULT_ABSTENTION},
        "left_config": "union",
        "radical_share": 1.0,
    },
]

# Présélection servie par défaut au chargement du site (le scénario de référence).
DEFAULT_SCENARIO = "split2"

# Bornes des curseurs nationaux (parts de bloc, % ; abstention % inscrits).
SLIDER_RANGES = {
    "G": [10.0, 50.0],
    "CD": [10.0, 55.0],
    "ED": [15.0, 55.0],
    "AB": [25.0, 60.0],
}
