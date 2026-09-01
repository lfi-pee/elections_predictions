"""Découpe les contours bureau de vote en GeoJSON par département pour le site **2027**.

Comme `report_geo.export`, mais les propriétés portent les **composantes de déviation**
(`dg/dc/de/da`, centrées) et non des prédictions figées : le client calcule
`pred_b = curseur_b + dev_b`, la marge, le bloc en tête, le gisement et les fourchettes
en direct au gré des curseurs. On sert aussi la demi-largeur conforme à 90 % par bloc
(`hg/hc/he/ha`), le plancher d'abstention (`af`) et la circo (`ri`).

    python3 -u -m src.report_geo_2027
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import ijson
import pandas as pd

from src.report_geo import (
    CONTOURS,
    _annoncer_desync,
    _round_geom,
    communes_desynchronisees,
)

MASTER = Path("data/report/bv_master_2027.parquet")
OUT = Path("report_app/2027/data/bv")


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    m = pd.read_parquet(MASTER).set_index("location")
    # Même refus que report_geo.export : une commune renumérotée depuis le millésime des
    # contours n'a plus les mêmes clés des deux côtés, et les rares codes qui coïncident
    # encore rattacheraient les déviations d'un bureau à la géométrie d'un autre.
    desync = communes_desynchronisees()
    _annoncer_desync(desync)
    by_dept: dict[str, list[dict]] = defaultdict(list)
    kept = 0
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            loc = feat["properties"]["codeBureauVote"]
            if loc not in m.index or loc[:5] in desync:
                continue
            row = m.loc[loc]
            props = {
                "l": loc,
                "n": row.libelle_commune,
                "dg": round(float(row.dev_G), 1),
                "dc": round(float(row.dev_CD), 1),
                "de": round(float(row.dev_ED), 1),
                "da": round(float(row.dev_AB), 1),
                "hg": round(float(row.hw90_G), 0),
                "hc": round(float(row.hw90_CD), 0),
                "he": round(float(row.hw90_ED), 0),
                "ha": round(float(row.hw90_AB), 0),
                "i": int(row.inscrits),
                "af": round(float(row.abst_floor), 1),
                "ri": row.circo if pd.notna(row.circo) else "",
            }
            if bool(row.lag_fallback):
                props["fb"] = 1
            by_dept[row.code_departement].append(
                {"type": "Feature", "geometry": _round_geom(feat["geometry"]), "properties": props}
            )
            kept += 1
    for dept, feats in by_dept.items():
        (OUT / f"{dept}.geojson").write_text(
            json.dumps({"type": "FeatureCollection", "features": feats},
                       ensure_ascii=False, separators=(",", ":"))
        )
    size = sum(p.stat().st_size for p in OUT.glob("*.geojson")) / 1e6
    print(f"export 2027 : {kept} bureaux, {len(by_dept)} départements, {size:.1f} Mo")


if __name__ == "__main__":
    export()
