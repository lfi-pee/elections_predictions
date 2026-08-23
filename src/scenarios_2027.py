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
    """Parts renormalisées sur 3 blocs, sommant **exactement** à 100 (le 3e absorbe l'arrondi)."""
    s = g + cd + ed
    gg, cc = round(100 * g / s, 1), round(100 * cd / s, 1)
    return {"G": gg, "CD": cc, "ED": round(100 - gg - cc, 1)}


# Abstention par défaut : une législative « à l'heure » (dans la foulée d'une
# présidentielle 2027) mobilise davantage qu'une législative de mi-mandat — on part de
# ~48 % d'abstention (contre ~52 % en 2022, ~57 % en 2017 hors effet présidentiel, et le
# creux à 33 % de la dissolution surprise de 2024). Réglable au curseur.
DEFAULT_ABSTENTION = 48.0

# Base commune, rafraîchie au 2026-08-23 (cf. `legislatives_2027_hypotheses.csv`). La lecture
# de bloc la plus récente est la tendance agrégée PolitPro (août 2026, le baromètre législatif
# des instituts n'étant plus mis à jour depuis oct. 2025) : RN ~35, gauche unie ~24, Ensemble
# ~14, LR ~12. Traduits en blocs « propres » (autres/divers exclus, renormalisés à 100 par
# `_renorm3`) : soutien de gauche total _L ~27, centre+droite _CD = Ensemble+LR ~26, extrême
# droite _ED = RN ~35. Les trois premiers scénarios partent du **même total** — seule la
# **configuration** de la gauche change — pour isoler l'effet propre de l'union (à total égal,
# l'union convertit mieux en sièges ; la division en perd). Tout est réglable au curseur.
_L, _CD, _ED = 27.0, 26.0, 35.0

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
        # Part du pôle radical (LFI) dans le total de gauche = ancrage sondages : dernier test
        # législatif « gauche divisée » disponible (2025), LFI 9,7 / (9,7+17,7) = 0,354 (agrégat
        # Touteleurope) ; Elabe juin 2025 donnait 10/26 = 0,385. Aucun sondage législatif 2026
        # ne scinde la gauche (suivi reporté sur la présidentielle, où Mélenchon remonte : LFI à
        # la hausse). Valeur de départ, désormais **réglable au curseur** sur le site.
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
        "desc": "Gauche unie, mais union des droites : au 2nd tour, l'électorat LR se reporte "
        "sur le RN plutôt que de faire barrage — le « front républicain » s'effondre, la barre "
        "à franchir monte. (Le niveau national reste celui des curseurs.)",
        "means": {**_renorm3(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "union",
        "radical_share": 1.0,
        "right_union": True,
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
