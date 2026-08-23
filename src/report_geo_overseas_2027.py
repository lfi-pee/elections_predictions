"""Silhouettes des collectivités d'outre-mer **absentes des contours bureau** (Etalab ne
couvre que la métropole + les 5 DOM). Sans elles, Nouvelle-Calédonie, Polynésie,
Wallis-et-Futuna et Saint-Martin/St-Barthélemy tombent dans le repli « tuiles » de
`report_circo_display`. On récupère leur contour de territoire dans Natural Earth (1:10m,
`admin_0_map_units`), on simplifie, et on écrit un petit GeoJSON par **département** (code
ZN/ZP/ZW/ZX). `report_circo_display` place ensuite ces silhouettes dans les encarts (et,
pour un département à plusieurs circos, les découpe en bandes verticales d'aire égale, une
par circo — le découpage interne est arbitraire, seul le contour est réel).

    python3 -u -m src.report_geo_overseas_2027

Le téléchargement Natural Earth (~13 Mo) est mis en cache dans `data/geo/` (ignoré par git,
comme les contours bureau) ; l'artefact versionné reste `circo_display.geojson`.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path

from shapely.geometry import mapping, shape
from shapely.ops import unary_union

GEO_DIR = Path("data/geo")
NE_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
          "geojson/ne_10m_admin_0_map_units.geojson")
NE_CACHE = GEO_DIR / "ne_10m_admin_0_map_units.geojson"
OUT = GEO_DIR / "overseas_silhouettes.geojson"

# Département de circo → identifiants Natural Earth (ADM0_A3) à unir pour la silhouette.
# ZX-01 = Saint-Martin, mais la circo couvre aussi Saint-Barthélemy → on unit les deux.
TERR = {
    "ZN": ["NCL"],          # Nouvelle-Calédonie
    "ZP": ["PYF"],          # Polynésie française
    "ZW": ["WLF"],          # Wallis-et-Futuna
    "ZX": ["MAF", "BLM"],   # Saint-Martin + Saint-Barthélemy
}
SIMPLIFY = 0.004  # les silhouettes sont fortement réduites dans les encarts


def _fetch() -> dict:
    if not NE_CACHE.exists():
        GEO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"téléchargement Natural Earth → {NE_CACHE.name} …")
        urllib.request.urlretrieve(NE_URL, NE_CACHE)
    return json.loads(NE_CACHE.read_text())


def _round_geom(geom, p=4):
    def r(cs):
        return [[round(x, p), round(y, p)] for x, y in cs]
    m = mapping(geom)
    if m["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [r(ring) for ring in m["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[r(ring) for ring in poly] for poly in m["coordinates"]]}


def export() -> None:
    ne = _fetch()
    by_a3: dict[str, list] = {}
    for f in ne["features"]:
        a3 = f["properties"].get("ADM0_A3")
        by_a3.setdefault(a3, []).append(shape(f["geometry"]))

    feats = []
    for dept, codes in TERR.items():
        geoms = [g for c in codes for g in by_a3.get(c, [])]
        if not geoms:
            print(f"  !! {dept}: aucun contour Natural Earth pour {codes}")
            continue
        sil = unary_union(geoms).simplify(SIMPLIFY, True)
        feats.append({"type": "Feature", "properties": {"dept": dept},
                      "geometry": _round_geom(sil)})

    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                              ensure_ascii=False, separators=(",", ":")))
    print(f"silhouettes : {len(feats)} territoires → {OUT} ({OUT.stat().st_size / 1e3:.1f} Ko)")


if __name__ == "__main__":
    export()
