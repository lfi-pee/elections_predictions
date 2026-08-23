"""Polygones de **circonscription** pour le site 2027 (carte choroplèthe).

Les contours officiels de circonscription ne sont pas dans le dépôt ; on les reconstruit
en **dissolvant** les contours bureau de vote par code de circonscription (carte
bureau→circo stable 2012–2024, déjà portée par `bv_master_2027`). Chaque polygone reçoit
les agrégats de déviation de la circo (repris de `circo.json`) : le client calcule
`pred_b = curseur_b + dev_b` puis le score de jouabilité, par circonscription.

    python3 -u -m src.report_geo_circo_2027
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import ijson
import pandas as pd
from shapely.geometry import mapping, shape
from shapely.ops import unary_union

from src.report_geo import CONTOURS

MASTER = Path("data/report/bv_master_2027.parquet")
CIRCO_JSON = Path("report_app/2027/data/circo.json")
OUT = Path("report_app/2027/data/circo.geojson")

SIMPLIFY_BV = 0.0025   # pré-simplification par bureau (la vue circo est dézoomée)
SIMPLIFY_OUT = 0.006  # simplification du polygone de circo dissous
PRECISION = 4


def _round_geom(geom):
    def r(coords):
        return [[round(x, PRECISION), round(y, PRECISION)] for x, y in coords]

    m = mapping(geom)
    if m["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [r(ring) for ring in m["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[r(ring) for ring in poly] for poly in m["coordinates"]]}


def export() -> None:
    loc2circo = pd.read_parquet(MASTER, columns=["location", "circo"]).dropna(subset=["circo"])
    loc2circo = dict(zip(loc2circo.location, loc2circo.circo))
    circo = json.loads(CIRCO_JSON.read_text())
    props_by_id = {
        cid: {
            "id": cid, "nm": circo["nm"][i], "dept": circo["dept"][i],
            "ins": circo["ins"][i], "nbv": circo["nbv"][i],
            "dG": circo["dG"][i], "dCD": circo["dCD"][i], "dED": circo["dED"][i],
            "dAB": circo["dAB"][i], "af": circo["af"][i],
        }
        for i, cid in enumerate(circo["id"])
    }

    geoms: dict[str, list] = defaultdict(list)
    n = 0
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            cid = loc2circo.get(feat["properties"]["codeBureauVote"])
            if cid is None:
                continue
            try:
                g = shape(feat["geometry"]).buffer(0).simplify(SIMPLIFY_BV, True)
                if not g.is_empty:
                    geoms[cid].append(g)
                    n += 1
            except Exception:
                pass
    print(f"  {n:,} contours bureau regroupés en {len(geoms)} circos ; dissolution…")

    feats = []
    for cid, gs in geoms.items():
        if cid not in props_by_id:
            continue
        try:
            merged = unary_union(gs).simplify(SIMPLIFY_OUT, True)
        except Exception:
            merged = unary_union([g.buffer(0) for g in gs]).simplify(SIMPLIFY_OUT, True)
        if merged.is_empty:
            continue
        feats.append({"type": "Feature", "geometry": _round_geom(merged),
                      "properties": props_by_id[cid]})
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              ensure_ascii=False, separators=(",", ":")))
    print(f"export circo : {len(feats)} circonscriptions, {OUT.stat().st_size / 1e6:.1f} Mo → {OUT}")


if __name__ == "__main__":
    export()
