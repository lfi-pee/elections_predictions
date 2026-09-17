"""Scénarios nationaux pour la prévision des Législatives 2027 — source unique.

2027 étant à venir, l'ancre nationale (Étape 1 du modèle : la moyenne nationale par
bloc) ne peut venir d'un résultat : elle est **posée par hypothèse**, réglable au
curseur sur le site. On fournit des **présélections** (`SCENARIOS`) ancrées sur les
intentions de vote 1er tour de la **présidentielle 2027** (Wikipédia, tous instituts, fenêtre
d'un an — `src/scrape_pres_2027`), agrégées exactement comme l'estimateur validé du modèle
(cf. `anchor_from_polls`). La part LFI dans la gauche en découle avec une décote « candidat →
parti » mesurée en LOO à horizon égal (`src/lfi_pres_discount`). Le baromètre législatif 2025
(gelé) ne sert plus qu'au rappel comparatif.

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
import json
from pathlib import Path

_PRES_CSV = (
    Path(__file__).resolve().parent.parent
    / "data/polls/presidentielle/2027/presidentielle_2027_t1_tidy.csv"
)
_DISCOUNT_JSON = Path(__file__).resolve().parent.parent / "data/polls/lfi_pres_discount.json"
# Ancienne source (baromètre législatif « hypothèse dissolution », gelé depuis oct. 2025) :
# conservée pour comparaison dans `ANCHOR["legislative_polls_2025"]`, plus utilisée pour l'ancre.
_LEGI_CSV = (
    Path(__file__).resolve().parent.parent
    / "data/polls/legislatives/legislatives_2027_hypotheses.csv"
)
# Fenêtre de l'estimateur validé : sondages des 12 mois précédant le scrutin (cross_type_ridge.
# _build_national_poll_features). En prévision, 12 mois avant le sondage le plus récent.
WINDOW_YEARS = 1.0


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


def _read_csv(path: Path) -> list[dict]:
    """Lit un CSV en ignorant les lignes de commentaire `#`."""
    lines = [ln for ln in path.read_text().splitlines() if ln and not ln.startswith("#")]
    return list(csv.DictReader(io.StringIO("\n".join(lines))))


def _read_pres_polls(path: Path = _PRES_CSV) -> list[dict]:
    """Sondages présidentiels 2027 « tidy » (src/scrape_pres_2027) : une ligne par candidat·e."""
    return [{"institut": d["institut"], "date": d["date_fin"], "hyp": int(d["hypothese"]),
             "candidat": d["candidat"], "bloc": d["bloc"], "v": float(d["valeur"])}
            for d in _read_csv(path)]


def _mean(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _date_float(iso: str) -> float:
    y, m, d = (int(x) for x in iso.split("-"))
    return y + (m - 1) / 12 + (d - 1) / 365


def anchor_from_polls(rows: list[dict], k_discount: float, window: float = WINDOW_YEARS) -> dict:
    """Ancre nationale depuis les sondages **présidentiels** 2027 — l'estimateur validé du modèle
    (Étape 1) : moyenne simple des sondages de la fenêtre d'un an, chaque sondage (institut × date)
    ramené à 100 sur les trois blocs, toutes hypothèses de candidats confondues (mêmes règles que
    `cross_type_ridge._build_national_poll_features`, dont la fenêtre ne contenait, pour toutes
    les législatives d'apprentissage 2007→2022, pratiquement que des sondages présidentiels :
    RMSE LOO par bloc G 6,3 / CD 6,3 / ED 7,5 pts, cf. `bayesian_polls`). Pas de moyenne
    exponentielle ni d'effet maison : hors de l'ensemble validé.

    Part radicale = **k × (Mélenchon / gauche)** moyenné par hypothèse ; k = décote « candidat →
    parti » mesurée en LOO **à l'horizon actuel de la prévision** (`lfi_pres_discount` : k(h), h =
    mois entre le dernier sondage et le 1er tour ; ≈0,9 à sept mois, ≈0,75 à la veille du vote,
    la montée tardive de Mélenchon étant absorbée par les sondages au fil de la campagne).

    Renvoie {"G", "CD", "ED" (parts sur 100), "rad", "rad_raw", "n_polls", "n_hyp", "from", "to"}.
    """
    latest = max(_date_float(r["date"]) for r in rows)
    w = [r for r in rows if _date_float(r["date"]) >= latest - window]
    by_poll: dict[tuple, dict[str, float]] = {}
    for r in w:
        acc = by_poll.setdefault((r["institut"], r["date"]), {"G": 0.0, "CD": 0.0, "ED": 0.0})
        if r["bloc"] in acc:
            acc[r["bloc"]] += r["v"]
    shares = {b: [] for b in ("G", "CD", "ED")}
    for acc in by_poll.values():
        tot = sum(acc.values())
        if tot > 0:
            for b in shares:
                shares[b].append(100.0 * acc[b] / tot)
    by_hyp: dict[tuple, dict[str, float]] = {}
    for r in w:
        if r["bloc"] != "G":
            continue
        acc = by_hyp.setdefault((r["institut"], r["date"], r["hyp"]), {"left": 0.0, "mel": 0.0})
        acc["left"] += r["v"]
        if r["candidat"] == "MÉLENCHON":
            acc["mel"] += r["v"]
    ratios = [a["mel"] / a["left"] for a in by_hyp.values() if a["left"] > 0 and a["mel"] > 0]
    rad_raw = _mean(ratios)
    return {"G": round(_mean(shares["G"]), 1), "CD": round(_mean(shares["CD"]), 1),
            "ED": round(_mean(shares["ED"]), 1), "rad_raw": round(rad_raw, 3),
            "rad": round(k_discount * rad_raw, 3), "n_polls": len(shares["G"]), "n_hyp": len(ratios),
            "from": min(r["date"] for r in w), "to": max(r["date"] for r in w)}


def legislative_polls_2025(path: Path = _LEGI_CSV) -> dict | None:
    """Rappel de l'ancienne ancre (baromètre législatif 2025, gelé) : part LFI dans la gauche
    des tests « gauche divisée », pour affichage comparatif seulement."""
    try:
        rows = _read_csv(path)
    except OSError:
        return None
    div = [r for r in rows if r.get("LFI", "").strip()]
    rad = _mean([float(r["LFI"]) / (float(r["LFI"]) + float(r["PS_PP_EELV_PCF"])) for r in div])
    return {"rad": round(rad, 3), "n": len(div)} if rad is not None else None


# Abstention par défaut : une législative « à l'heure » (dans la foulée d'une
# présidentielle 2027) mobilise davantage qu'une législative de mi-mandat — on part de
# ~48 % d'abstention (contre ~52 % en 2022, ~57 % en 2017 hors effet présidentiel, et le
# creux à 33 % de la dissolution surprise de 2024). Réglable au curseur.
DEFAULT_ABSTENTION = 48.0

# Base commune, **calculée depuis les sondages présidentiels** par moyenne simple (agrégation
# avalisée par la validation croisée — cf. `anchor_from_polls`). Les trois premiers scénarios
# partent du **même total** — seule la **configuration** de la gauche change — pour isoler l'effet
# propre de l'union (à total égal, l'union convertit mieux en sièges ; la division en perd). Tout
# reste réglable au curseur sur le site. Repli sur des constantes documentées si les données
# manquent (ancre du 2026-08-23, baromètre législatif).
try:
    _K = json.loads(_DISCOUNT_JSON.read_text())
    ANCHOR = anchor_from_polls(_read_pres_polls(), _K["k_poll"])
    ANCHOR.update({"source": "présidentielle 2027, 1er tour (Wikipédia)", "k_discount": _K["k_poll"],
                   "k_range": _K["k_poll_range"], "horizon_months": _K["horizon_months"],
                   "legislative_polls_2025": legislative_polls_2025()})
    _L, _CD, _ED, _RAD = ANCHOR["G"], ANCHOR["CD"], ANCHOR["ED"], ANCHOR["rad"]
except Exception:  # données absentes / illisibles : ancre documentée de repli.
    ANCHOR = {"source": "repli (constantes)", "G": 27.0, "CD": 26.0, "ED": 35.0, "rad": 0.354}
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
        # Part du pôle radical (LFI) dans le total de gauche = k × part de Mélenchon dans le vote
        # de gauche des sondages présidentiels (cf. `anchor_from_polls`, `lfi_pres_discount`).
        # Réglable au curseur sur le site.
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
