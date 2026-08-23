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
# Pays du monde (1:50m) pour les 11 circos de l'étranger, rendues en planisphère miniature.
NE_WORLD_URL = ("https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/"
                "geojson/ne_50m_admin_0_countries.geojson")
NE_WORLD_CACHE = GEO_DIR / "ne_50m_admin_0_countries.geojson"
OUT = GEO_DIR / "overseas_silhouettes.geojson"
OUT_ETR = GEO_DIR / "etranger_world.geojson"

# Département de circo → identifiants Natural Earth (ADM0_A3) à unir pour la silhouette.
# ZX-01 = Saint-Martin, mais la circo couvre aussi Saint-Barthélemy → on unit les deux.
TERR = {
    "ZN": ["NCL"],          # Nouvelle-Calédonie
    "ZP": ["PYF"],          # Polynésie française
    "ZW": ["WLF"],          # Wallis-et-Futuna
    "ZX": ["MAF", "BLM"],   # Saint-Martin + Saint-Barthélemy
}
SIMPLIFY = 0.004  # les silhouettes sont fortement réduites dans les encarts

# Territoires français : exclus du planisphère de l'étranger (ils ont leur propre circo, ou
# font partie de la métropole). ATA/HMD : pôle sud, ignorés (hors cadre).
FR_SKIP = {"FRA", "NCL", "PYF", "WLF", "MAF", "BLM", "ATF", "SPM", "GUF", "ATA", "HMD"}
# Moyen-Orient (→ 10e). Le reste de l'Asie et l'Océanie → 11e.
MID_EAST = {"SAU", "ARE", "BHR", "IRQ", "IRN", "JOR", "KWT", "LBN", "OMN", "QAT", "SYR", "YEM"}

# Composition officielle des 11 circonscriptions des Français de l'étranger (décret 2011-916),
# encodée par code ISO A3 + règles par continent/sous-région (Natural Earth). ZZ-0N = Ne circo.
CIRCO_ISO = {
    3: {"GBR", "IRL", "ISL", "NOR", "SWE", "DNK", "FIN", "EST", "LVA", "LTU",
        "FRO", "ALA", "GRL", "IMN", "GGY", "JEY"},                       # Europe du Nord
    4: {"BEL", "NLD", "LUX"},                                            # Benelux
    5: {"ESP", "PRT", "AND", "MCO"},                                     # Péninsule ibérique
    6: {"CHE", "LIE"},                                                   # Suisse, Liechtenstein
    7: {"DEU", "AUT", "POL", "CZE", "SVK", "HUN", "BGR", "ROU",          # Europe centrale
        "SVN", "HRV", "BIH", "SRB", "MNE", "MKD", "ALB"},                # + Balkans
    8: {"GRC", "ITA", "MLT", "CYP", "ISR", "TUR", "PSE", "SMR", "VAT"},  # Méditerranée orientale
    9: {"MAR", "DZA", "TUN", "ESH", "BFA", "CIV", "CPV", "GIN", "GMB",   # Maghreb + Afrique
        "GNB", "LBR", "MLI", "MRT", "NER", "SEN", "SLE"},                # de l'Ouest (partiel)
    11: {"BLR", "MDA", "UKR", "RUS", "GEO", "ARM", "AZE", "MDV"},        # Europe or. + Caucase + …
}


def _circo_of(iso, admin, cont, sub):
    """Numéro de circonscription (1..11) d'un pays du monde, ou None si exclu."""
    if iso in FR_SKIP:
        return None
    if iso in {"USA", "CAN", "BMU"}:
        return 1
    if cont in ("South America",) or sub in ("Caribbean", "Central America") or iso in {"FLK", "SGS"}:
        return 2
    for n, isos in CIRCO_ISO.items():
        if iso in isos:
            return n
    if admin == "Kosovo":
        return 7
    if admin == "Northern Cyprus":
        return 8
    if iso in {"MUS", "SYC", "SHN", "IOT"}:  # océan Indien africain / britannique
        return 10
    if cont == "Oceania":
        return 11
    if cont == "Asia":
        return 10 if iso in MID_EAST else 11
    if cont == "Africa" or admin == "Somaliland":  # reste de l'Afrique + Proche-Orient déjà pris
        return 10
    return None


def _fetch(url, cache) -> dict:
    if not cache.exists():
        GEO_DIR.mkdir(parents=True, exist_ok=True)
        print(f"téléchargement Natural Earth → {cache.name} …")
        urllib.request.urlretrieve(url, cache)
    return json.loads(cache.read_text())


def _round_geom(geom, p=4):
    def r(cs):
        return [[round(x, p), round(y, p)] for x, y in cs]
    m = mapping(geom)
    if m["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [r(ring) for ring in m["coordinates"]]}
    return {"type": "MultiPolygon",
            "coordinates": [[r(ring) for ring in poly] for poly in m["coordinates"]]}


def _export_com() -> None:
    ne = _fetch(NE_URL, NE_CACHE)
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
    print(f"silhouettes COM : {len(feats)} territoires → {OUT.name} ({OUT.stat().st_size / 1e3:.1f} Ko)")


def _export_etranger() -> None:
    """Planisphère : union des pays de chaque circonscription de l'étranger (ZZ-01..ZZ-11)."""
    world = _fetch(NE_WORLD_URL, NE_WORLD_CACHE)
    groups: dict[int, list] = {}
    unmapped = []
    for f in world["features"]:
        p = f["properties"]
        iso = p.get("ISO_A3_EH") or p.get("ISO_A3")
        n = _circo_of(iso, p.get("ADMIN"), p.get("CONTINENT"), p.get("SUBREGION"))
        if n is None:
            if iso not in FR_SKIP:
                unmapped.append(p.get("ADMIN"))
            continue
        groups.setdefault(n, []).append(shape(f["geometry"]).buffer(0))
    if unmapped:
        print(f"  (info) pays non affectés, ignorés : {', '.join(sorted(set(unmapped)))}")

    feats = []
    for n in range(1, 12):
        if n not in groups:
            print(f"  !! circo étranger {n} : aucun pays")
            continue
        merged = unary_union(groups[n]).simplify(0.08, preserve_topology=True)
        feats.append({"type": "Feature", "properties": {"id": f"ZZ-{n:02d}"},
                      "geometry": _round_geom(merged, p=3)})

    OUT_ETR.write_text(json.dumps({"type": "FeatureCollection", "features": feats},
                                  ensure_ascii=False, separators=(",", ":")))
    print(f"planisphère étranger : {len(feats)}/11 circos → {OUT_ETR.name} "
          f"({OUT_ETR.stat().st_size / 1e3:.1f} Ko)")


def export() -> None:
    _export_com()
    _export_etranger()


if __name__ == "__main__":
    export()
