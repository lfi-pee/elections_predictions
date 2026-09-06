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

import csv
import io
from pathlib import Path

_POLLS_CSV = (
    Path(__file__).resolve().parent.parent
    / "data/polls/legislatives/legislatives_2027_hypotheses.csv"
)


# Niveau national du bloc « Autre » (régionalistes/autonomistes hors axe G/CD/ED).
# ~1,8 % du vote exprimé national — stable d'un scrutin à l'autre (cf. src/autre_oof.py) ;
# ce n'est PAS un curseur : sa faible masse nationale est fixe, c'est sa **répartition
# spatiale** (concentrée en Corse et outre-mer) que le modèle prédit via `dev_Other`.
AUTRE_NATIONAL = 1.8


def _renorm3(g: float, cd: float, ed: float, total: float = 100.0) -> dict[str, float]:
    """Parts renormalisées sur 3 blocs, sommant **exactement** à `total` (le 3e absorbe
    l'arrondi). `total` < 100 laisse la place au bloc « Autre »."""
    s = g + cd + ed
    gg, cc = round(total * g / s, 1), round(total * cd / s, 1)
    return {"G": gg, "CD": cc, "ED": round(total - gg - cc, 1)}


def _means4(g: float, cd: float, ed: float) -> dict[str, float]:
    """Parts G/CD/ED/AU sommant à 100 : les trois blocs d'axe renormalisés sur
    (100 − Autre), plus le bloc « Autre » à son niveau national fixe."""
    return {**_renorm3(g, cd, ed, 100.0 - AUTRE_NATIONAL), "AU": AUTRE_NATIONAL}


def _read_polls(path: Path = _POLLS_CSV) -> list[dict]:
    """Lit le CSV d'hypothèses (lignes de commentaire `#` ignorées)."""
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    out = []
    for d in csv.DictReader(io.StringIO("\n".join(lines))):
        num = lambda k: (float(d[k]) if d.get(k, "").strip() else None)  # noqa: E731
        out.append(
            {"year": int(d["periode"][:4]), "rn": num("RN_allies"),
             "gu": num("gauche_unie_NFP"), "lfi": num("LFI"),
             "pspp": num("PS_PP_EELV_PCF"), "ens": num("Ensemble"), "lr": num("LR")}
        )
    return out


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def anchor_from_polls(rows: list[dict]) -> tuple[float, float, float, float]:
    """Ancre nationale (soutien de gauche _L, centre+droite _CD, extrême droite _ED, part
    radicale _RAD) par **moyenne simple** des sondages — la seule agrégation que la validation
    croisée du modèle avalise (estimateurs LOO = moyennes/dernière valeur ; cf. cross_type_dev.
    `estimate_national_abstention`), et la règle « si un sondage couvre le scrutin, on l'utilise
    directement ». Pas de moyenne exponentielle : elle n'est pas dans l'ensemble validé.

    - Blocs (_CD, _ED) : moyenne simple des lignes de l'**année la plus récente**.
    - Soutien de gauche _L : « soutien réel » = somme des composantes (gauche divisée) là où
      elle existe ; sinon la gauche unie de l'année récente, remise à l'échelle par le rapport
      historique divisé/unie (l'offre divisée somme un peu au-dessus d'une liste NFP unique).
    - Part radicale _RAD = moyenne simple de LFI/(LFI+PS·PP·EELV·PCF) sur les tests « divisés ».
    """
    y = max(r["year"] for r in rows)
    recent = [r for r in rows if r["year"] == y]
    ed = _mean([r["rn"] for r in recent])
    cd = _mean([r["ens"] for r in recent]) + _mean([r["lr"] for r in recent])
    div = [r for r in rows if r["lfi"] is not None]
    div_tot = _mean([r["lfi"] + r["pspp"] for r in div])
    uni_all = _mean([r["gu"] for r in rows if r["gu"] is not None])
    recent_div = _mean([r["lfi"] + r["pspp"] for r in recent if r["lfi"] is not None])
    recent_uni = _mean([r["gu"] for r in recent if r["gu"] is not None])
    if recent_div is not None:
        left = recent_div
    elif recent_uni is not None and uni_all:
        left = recent_uni * (div_tot / uni_all)
    else:
        left = div_tot
    rad = _mean([r["lfi"] / (r["lfi"] + r["pspp"]) for r in div])
    return round(left, 1), round(cd, 1), round(ed, 1), round(rad, 3)


# Abstention par défaut : une législative « à l'heure » (dans la foulée d'une
# présidentielle 2027) mobilise davantage qu'une législative de mi-mandat — on part de
# ~48 % d'abstention (contre ~52 % en 2022, ~57 % en 2017 hors effet présidentiel, et le
# creux à 33 % de la dissolution surprise de 2024). Réglable au curseur.
DEFAULT_ABSTENTION = 48.0

# Base commune, **calculée depuis le CSV** par moyenne simple (agrégation avalisée par la
# validation croisée — cf. `anchor_from_polls`). Sur les données au 2026-08-23 : soutien de
# gauche total _L ~27, centre+droite _CD = Ensemble+LR ~26, extrême droite _ED = RN ~35, part
# radicale _RAD ~0,37. Repli sur des constantes documentées si le CSV est absent. Les trois
# premiers scénarios partent du **même total** — seule la **configuration** de la gauche change
# — pour isoler l'effet propre de l'union (à total égal, l'union convertit mieux en sièges ; la
# division en perd). Tout reste réglable au curseur sur le site.
try:
    _L, _CD, _ED, _RAD = anchor_from_polls(_read_polls())
except Exception:  # CSV absent / illisible : ancre documentée de repli.
    _L, _CD, _ED, _RAD = 27.0, 26.0, 35.0, 0.354

SCENARIOS = [
    {
        "key": "union",
        "label": "Union de la gauche large",
        "desc": "À soutien de gauche égal, une seule candidature par circonscription "
        "(type NFP/Front populaire) capte tout le bloc. C'est la configuration qui "
        "convertit le mieux le soutien en sièges.",
        "means": {**_means4(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "union",
        "radical_share": 1.0,
    },
    {
        "key": "split2",
        "label": "Gauche radicale vs néolibérale",
        "desc": "Scénario de référence : même soutien de gauche, mais scindé en un pôle "
        "radical (LFI) et un pôle social-démocrate (PS-Place publique-EELV-PCF) qui "
        "concourent séparément — deux candidatures, qualification au 2nd tour plus dure.",
        "means": {**_means4(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "split2",
        # Part du pôle radical (LFI) dans le total de gauche = moyenne simple des tests
        # « gauche divisée » du CSV (cf. `anchor_from_polls`) : LFI 9,7/27,4 (Touteleurope) et
        # 10/26 (Elabe) ⇒ ~0,37. Aucun sondage législatif 2026 ne scinde la gauche (suivi
        # reporté sur la présidentielle, où Mélenchon remonte). Réglable au curseur sur le site.
        "radical_share": _RAD,
    },
    {
        "key": "frag",
        "label": "Fragmentation (statu quo)",
        "desc": "« Autre » : même soutien, mais éclaté en trois (LFI / PS-PP / éco-PCF) "
        "sans pôle fédérateur — dispersion maximale, presque aucune qualification.",
        "means": {**_means4(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
        "left_config": "split3",
        "radical_share": _RAD,
    },
    {
        "key": "droite_unie",
        "label": "Droites unies en face",
        "desc": "Gauche unie, mais union des droites : au 2nd tour, l'électorat LR se reporte "
        "sur le RN plutôt que de faire barrage — le « front républicain » s'effondre, la barre "
        "à franchir monte. (Le niveau national reste celui des curseurs.)",
        "means": {**_means4(_L, _CD, _ED), "AB": DEFAULT_ABSTENTION},
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
