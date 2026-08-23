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

# Base commune : le **soutien réel** de la gauche (somme des intentions 1er tour des
# composantes, gauche divisée = 27,4 % des exprimés bruts) et le rapport de force à droite
# (CD 26,0, ED/RN 31,7). Les trois premiers scénarios partent du **même total** — seule la
# **configuration** de la gauche change — pour isoler l'effet propre de l'union (à total
# égal, l'union convertit mieux en sièges ; la division en perd). Réglable au curseur.
_L, _CD, _ED = 27.4, 26.0, 31.7

SCENARIOS = [
    {
        "key": "union",
        "label": "Union de la gauche large",
        "desc": "À soutien de gauche égal, une seule candidature par circonscription "
        "(type NFP/Front populaire) capte tout le bloc. C'est la configuration qui "
        "convertit le mieux le soutien en sièges.",
        "means": {**_renorm3(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "union",
        "radical_share": 1.0,
    },
    {
        "key": "split2",
        "label": "Gauche radicale vs néolibérale",
        "desc": "Scénario de référence : même soutien de gauche, mais scindé en un pôle "
        "radical (LFI) et un pôle social-démocrate (PS-Place publique-EELV-PCF) qui "
        "concourent séparément — deux candidatures, qualification au 2nd tour plus dure.",
        "means": {**_renorm3(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "split2",
        # Part du pôle radical dans le total de gauche : LFI 9,7 / (9,7+17,7) = 0,354.
        "radical_share": 0.354,
    },
    {
        "key": "frag",
        "label": "Fragmentation (statu quo)",
        "desc": "« Autre » : même soutien, mais éclaté en trois (LFI / PS-PP / éco-PCF) "
        "sans pôle fédérateur — dispersion maximale, presque aucune qualification.",
        "means": {**_renorm3(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "split3",
        "radical_share": 0.354,
    },
    {
        "key": "droite_unie",
        "label": "Droites unies en face",
        "desc": "Même soutien de gauche (ici unie), mais une partie de LR bascule au RN "
        "(union des droites) : la barre à franchir au 2nd tour monte fortement.",
        # ~6 pts du centre-droit passent au RN ; le total de gauche est inchangé.
        "means": {**_renorm3(_L, _CD - 6.0, _ED + 6.0), "AB": DEFAULT_ABSTENTION},
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
