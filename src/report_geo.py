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
    by_dept: dict[str, list[dict]] = defaultdict(list)
    kept = 0
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            loc = feat["properties"]["codeBureauVote"]
            if loc not in props.index:
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
