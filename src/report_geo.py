"""Découpe les contours bureau de vote en GeoJSON simplifiés par département.

`scan`   : repère les bureaux sans contour (territoires particuliers) →
           `data/report/contourless.json`. Indépendant de la table maître.
`export` : lit `bv_master.parquet`, simplifie chaque polygone, attache les
           propriétés légères nécessaires à la carte + au moteur de scénario
           client, et écrit `report_app/data/bv/<dept>.geojson` (un par dept).

Propriétés volontairement minimales (octets = fluidité) : prédictions par bloc
(le client recalcule bloc en tête + marge sous scénario), marge, disputé,
inscrits, point de bascule ED, nom de commune. Le détail lourd (réel, intervalles,
SHAP) est servi à la demande par `report_shap.py`.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import pandas as pd
from shapely.geometry import mapping, shape

CONTOURS = Path("data/geo/contours-bv.geojson")
MASTER = Path("data/report/bv_master.parquet")
CACHE = Path("data/report")
WHY_LEFT = Path("data/report/why_left.json")
OUT = Path("report_app/data/bv")
PRED_CSV = Path("data/predictions_with_intervals.csv")

SIMPLIFY_TOL = 0.00015
PRECISION = 5

# --------------------------------------------------------------------------------------
# Communes désynchronisées des contours
# --------------------------------------------------------------------------------------
# Les contours data.gouv sont figés sur le REU du 1er juin 2022 ; les prédictions portent
# la numérotation des scrutins récents. Une commune qui a RENUMÉROTÉ ses bureaux depuis
# (Bordeaux : 1101 → 1001, 1201 → 1021, 1301 → 1041…) n'a donc plus les mêmes clés des
# deux côtés du `join` par `codeBureauVote`. Deux dégâts, dont le second est le pire :
#
#   1. la quasi-totalité de ses bureaux tombe dans le `continue` silencieux ;
#   2. les rares codes qui coïncident PAR ACCIDENT sont servis — à Bordeaux, 18 sur 153,
#      dont les propriétés (prédictions, inscrits, plancher) sont celles d'un bureau et la
#      géométrie celle d'un AUTRE. Le contour 33063_1101 comptait 686 inscrit·es en 2022 ;
#      la carte y affichait les 1 349 du bureau numéroté 1101 en 2024.
#
# On ne devine pas la correspondance ici : elle demande un témoin indépendant des codes
# (l'écart d'inscrits entre un scrutin d'avant et un scrutin d'après), et l'export n'a pas
# à porter cette inférence. On refuse en revanche de servir un appariement massivement
# faux : sous COUV_MIN_COMMUNE d'électorat retrouvé, la commune entière sort de la carte
# des bureaux — elle reste servie à l'échelle communale, où aucun code n'intervient.
#
# Le seuil est DÉLIBÉRÉMENT BAS. Une couverture partielle est le cas NORMAL d'une commune
# qui a simplement créé ou fermé des bureaux : ses codes appariés désignent alors bien le
# même bureau des deux côtés, et les écarter détruirait des données justes. Mesuré sur le
# corpus : à 90 % le filtre retirerait 1 127 features dont la plupart sont correctes, à
# 50 % il en retire 38, toutes vérifiées fausses (Bordeaux, Saint-Victoret, Le Creusot).
#
# LIMITE CONNUE, à ne pas confondre avec ce que ce filtre attrape : une commune qui a
# REDÉCOUPÉ ses bureaux sans changer l'espace de codes (17 bureaux fusionnés en 11, par
# exemple) garde 100 % de couverture tout en n'ayant plus AUCUN appariement valide — son
# code 0001 de 2024 ne décrit plus le territoire du 0001 de 2022. La couverture est aveugle
# à ce cas ; seul l'écart d'inscrits entre deux millésimes le révèle (75 communes du corpus,
# cf. la méthode de prep_elections.construire_crosswalk_renumerotation dans devoirs_maison).
# Le détecter ici demanderait les inscrits d'un scrutin d'avant le millésime des contours.
COUV_MIN_COMMUNE = 0.50
# Sous ce nombre de bureaux, l'écart relève de la création ou fermeture d'un bureau, pas
# d'une renumérotation : on ne prive pas une petite commune de sa carte pour un bureau.
BV_MIN_COMMUNE = 5


def codes_contours() -> set[str]:
    """Codes de bureau présents dans le fichier de contours.

    Passe LÉGÈRE : on ne demande à ijson que la propriété utile, il saute les géométries —
    ce qui économise l'essentiel des 645 Mo du fichier."""
    with CONTOURS.open("rb") as f:
        return set(ijson.items(f, "features.item.properties.codeBureauVote"))


def communes_desynchronisees(inscrits: pd.Series) -> dict[str, float]:
    """{code commune → part d'électorat retrouvée} pour les communes dont la numérotation
    ne correspond plus à celle des contours (cf. le commentaire ci-dessus).

    `inscrits` est indexé par code de bureau (`location` de la table maître)."""
    avec = codes_contours()
    com = inscrits.index.to_series().str.slice(0, 5)
    tot = inscrits.groupby(com).sum()
    trouve = inscrits[inscrits.index.isin(avec)].groupby(com).sum().reindex(tot.index).fillna(0)
    nbv = inscrits.groupby(com).size()
    part = (trouve / tot.where(tot > 0)).fillna(0.0)
    vise = part[(part < COUV_MIN_COMMUNE) & (nbv >= BV_MIN_COMMUNE)]
    return vise.to_dict()


def _annoncer_desync(desync: dict[str, float], inscrits: pd.Series) -> None:
    if not desync:
        return
    com = inscrits.index.to_series().str.slice(0, 5)
    nbv = inscrits.groupby(com).size()
    pires = sorted(desync.items(), key=lambda kv: kv[1])[:5]
    detail = ", ".join(f"{c} ({nbv.get(c, 0)} BV, {p:.0%})" for c, p in pires)
    print(
        f"  ⚠ {len(desync)} commune(s) renumérotée(s) depuis le millésime des contours : "
        f"leurs bureaux ne sont PAS servis (appariement par code faux) — {detail}"
    )


def _pred_locations() -> set[str]:
    return set(pd.read_csv(PRED_CSV, usecols=["location"]).location.unique())


def scan() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    have = set()
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            have.add(feat["properties"]["codeBureauVote"])
    contourless = sorted(_pred_locations() - have)
    (CACHE / "contourless.json").write_text(json.dumps(contourless))
    print(f"scan: {len(have)} contours, {len(contourless)} bureaux sans contour")


def _round_geom(geom: dict) -> dict:
    g = shape(geom).simplify(SIMPLIFY_TOL, preserve_topology=True)

    def r(coords):
        return [[round(x, PRECISION), round(y, PRECISION)] for x, y in coords]

    m = mapping(g)
    if m["type"] == "Polygon":
        m = {"type": "Polygon", "coordinates": [r(ring) for ring in m["coordinates"]]}
    else:
        m = {
            "type": "MultiPolygon",
            "coordinates": [[r(ring) for ring in poly] for poly in m["coordinates"]],
        }
    return m


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    why = json.loads(WHY_LEFT.read_text()) if WHY_LEFT.exists() else {}
    m = pd.read_parquet(MASTER).set_index("location")
    props = m[
        [
            "pred_G",
            "pred_CD",
            "pred_ED",
            "pred_AB",
            "margin",
            "unc",
            "inscrits",
            "ed_tip",
            "mob",
            "abst_floor",
            "act_AB",
            "libelle_commune",
            "code_departement",
            "lag_fallback",
        ]
    ]
    desync = communes_desynchronisees(m["inscrits"])
    _annoncer_desync(desync, m["inscrits"])
    by_dept: dict[str, list[dict]] = defaultdict(list)
    kept = 0
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            loc = feat["properties"]["codeBureauVote"]
            if loc not in props.index or loc[:5] in desync:
                continue
            row = props.loc[loc]
            by_dept[row.code_departement].append(
                {
                    "type": "Feature",
                    "geometry": _round_geom(feat["geometry"]),
                    "properties": {
                        "l": loc,
                        "n": row.libelle_commune,
                        "pg": round(float(row.pred_G), 1),
                        "pc": round(float(row.pred_CD), 1),
                        "pe": round(float(row.pred_ED), 1),
                        "pa": round(float(row.pred_AB), 1),
                        "m": round(float(row.margin), 1),
                        "u": round(float(row.unc), 0),
                        "t": round(float(row.ed_tip), 1),
                        "mv": int(row.mob),
                        "i": int(row.inscrits),
                        "ab": int(round(row.inscrits * row.pred_AB / 100)),
                        # Conjunctural abstainers (predicted − historical floor) — the
                        # denominator γ is read against, so the hover shows the SAME
                        # left-share as the click panel (mob / conjunctural), never the
                        # mobilizable-over-all-abstainers ratio.
                        "cj": int(
                            round(
                                row.inscrits
                                * max(0.0, row.pred_AB - row.abst_floor)
                                / 100
                            )
                        ),
                        "w": why.get(loc, ""),
                        # Lower-confidence prediction: lag features fell back to
                        # the commune aggregate (own-BV history missing or from a
                        # reused precinct). Emitted only when true, to keep size down.
                        **({"fb": 1} if bool(row.lag_fallback) else {}),
                    },
                }
            )
            kept += 1
    for dept, feats in by_dept.items():
        fc = {"type": "FeatureCollection", "features": feats}
        (OUT / f"{dept}.geojson").write_text(
            json.dumps(fc, ensure_ascii=False, separators=(",", ":"))
        )
    size = sum(p.stat().st_size for p in OUT.glob("*.geojson")) / 1e6
    print(f"export: {kept} bureaux, {len(by_dept)} départements, {size:.1f} Mo")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    {"scan": scan, "export": export}[mode]()
