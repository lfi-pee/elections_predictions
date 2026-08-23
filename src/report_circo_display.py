"""Disposition d'affichage « toutes circonscriptions » : la métropole reste géographique,
l'outre-mer et les Français de l'étranger sont ramenés en **encarts** à gauche de la carte
(comme les cartes électorales usuelles), pour que les **577** circos soient visibles et
cliquables — y compris les 11 circos de l'étranger (sans territoire) rendues en tuiles.

Entrées : `circo.geojson` (559 polygones dissous) + `circo.json` (577, agrégats/props).
Sorties : `circo_display.geojson` (577 features, géométrie d'affichage) + `circo_insets.json`
(cadres + libellés des encarts).

    python3 -u -m src.report_circo_display
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from shapely.geometry import box as shp_box, mapping, shape
from shapely.affinity import affine_transform

DIR = Path("report_app/2027/data")
GEO = DIR / "circo.geojson"
ARR = DIR / "circo.json"
OUT = DIR / "circo_display.geojson"
INSETS = DIR / "circo_insets.json"
# Silhouettes des COM absentes des contours bureau (Natural Earth), produites par
# `report_geo_overseas_2027` : ZN/ZP/ZW/ZX obtiennent ainsi un vrai contour au lieu de tuiles.
SILH = Path("data/geo/overseas_silhouettes.geojson")
# Planisphère de l'étranger : union des pays de chacune des 11 circos (ZZ-01..ZZ-11).
ETR = Path("data/geo/etranger_world.geojson")

# Libellés des territoires d'outre-mer / étranger (à défaut : le code).
TERR = {
    "ZA": "Guadeloupe", "ZB": "Martinique", "ZC": "Guyane", "ZD": "La Réunion",
    "ZM": "Mayotte", "ZS": "St-Pierre-et-Miquelon", "ZN": "Nouvelle-Calédonie",
    "ZP": "Polynésie fr.", "ZW": "Wallis-et-Futuna", "ZX": "St-Martin, St-Barth.",
    "ZZ": "Français de l'étranger",
}
# Ordre et emplacement des encarts. Colonne d'encarts à gauche (Atlantique), empilée en
# latitude ; l'étranger (11 tuiles) en bandeau sous la métropole.
LEFT_COL = ["ZS", "ZA", "ZB", "ZC", "ZD", "ZM", "ZN", "ZP", "ZW", "ZX"]


def _fit_transform(src_bounds, dst):
    """Matrice affine (shapely: [a,b,d,e,xoff,yoff]) qui place `src_bounds` dans la boîte
    `dst`=(x0,y0,x1,y1) en conservant le rapport d'aspect (85 % de remplissage, centré)."""
    sx0, sy0, sx1, sy1 = src_bounds
    sw, sh = max(sx1 - sx0, 1e-6), max(sy1 - sy0, 1e-6)
    dx0, dy0, dx1, dy1 = dst
    s = min((dx1 - dx0) / sw, (dy1 - dy0) / sh) * 0.96
    scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    dcx, dcy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
    return [s, 0, 0, s, dcx - s * scx, dcy - s * scy]


def _inflate(geom, dst, target=0.10):
    """Dilate une silhouette jusqu'à couvrir ~`target` de l'aire de l'encart `dst`, puis la
    reclippe à l'encart. Les territoires minuscules ou épars (Wallis-et-Futuna, Polynésie —
    quelques îlots noyés dans un vaste rectangle océanique) redeviennent visibles ; ceux qui
    remplissent déjà l'encart (Nouvelle-Calédonie) ne bougent quasiment pas (arrêt précoce)."""
    dx0, dy0, dx1, dy1 = dst
    cell = (dx1 - dx0) * (dy1 - dy0)
    if cell <= 0:
        return geom
    step, g = 0.02 * min(dx1 - dx0, dy1 - dy0), geom
    for _ in range(24):
        if g.area >= target * cell:
            break
        g = geom.buffer((_ + 1) * step)
    return g.intersection(shp_box(dx0, dy0, dx1, dy1))


def _split_vertical(geom, n):
    """Découpe `geom` en `n` bandes verticales d'aire égale (une par circo, ordre gauche→
    droite). Le contour reste réel ; seule la limite interbande est arbitraire — assumé pour
    les COM à plusieurs circos (Nouvelle-Calédonie 2, Polynésie 3)."""
    if n <= 1:
        return [geom]
    minx, miny, maxx, maxy = geom.bounds
    total = geom.area or 1e-9
    cuts = []
    for k in range(1, n):
        target, lo, hi = total * k / n, minx, maxx
        for _ in range(40):
            mid = (lo + hi) / 2
            left = geom.intersection(shp_box(minx - 1, miny - 1, mid, maxy + 1)).area
            lo, hi = (mid, hi) if left < target else (lo, mid)
        cuts.append((lo + hi) / 2)
    xs = [minx - 1] + cuts + [maxx + 1]
    return [geom.intersection(shp_box(xs[i], miny - 1, xs[i + 1], maxy + 1)) for i in range(n)]


def _square(cx, cy, r):
    return {"type": "Polygon", "coordinates": [[[cx - r, cy - r], [cx + r, cy - r],
            [cx + r, cy + r], [cx - r, cy + r], [cx - r, cy - r]]]}


def _round_geom(geom, p=4):
    def r(cs):
        return [[round(x, p), round(y, p)] for x, y in cs]
    m = mapping(geom)
    if m["type"] == "Polygon":
        return {"type": "Polygon", "coordinates": [r(ring) for ring in m["coordinates"]]}
    return {"type": "MultiPolygon", "coordinates": [[r(ring) for ring in poly] for poly in m["coordinates"]]}


def export() -> None:
    geo = json.loads(GEO.read_text())
    arr = json.loads(ARR.read_text())
    poly_by_id = {f["properties"]["id"]: f for f in geo["features"]}
    silhouettes = ({f["properties"]["dept"]: shape(f["geometry"])
                    for f in json.loads(SILH.read_text())["features"]} if SILH.exists() else {})
    ids_by_dept = defaultdict(list)
    for i, cid in enumerate(arr["id"]):
        ids_by_dept[arr["dept"][i]].append((cid, i))

    def props(i):
        return {k: arr[k][i] for k in ("id", "nm", "dept", "ins", "nbv", "dG", "dCD", "dED", "dAB", "af")}

    out = []
    insets = []

    # Métropole (dept numérique ou Corse) : géométrie inchangée.
    for f in geo["features"]:
        d = f["properties"]["dept"]
        if d[0].isdigit() or d in ("2A", "2B"):
            out.append(f)

    # Encarts outre-mer/étranger : tous ramenés à GAUCHE de la métropole (Atlantique,
    # lon < −6.5 → aucun chevauchement avec la métropole lon ≥ −5,1), en bloc compact.
    # 10 territoires en grille 3 colonnes × 4 lignes (cellules ~carrées → les petits
    # territoires comme Mayotte remplissent mieux et paraissent moins minuscules) ;
    # l'étranger en bandeau en dessous.
    reg_x0, reg_x1 = -14.5, -7.0
    reg_y0, reg_y1 = 42.0, 51.2
    n_cols, n_rows = 3, 4
    gap = 0.1
    cw = (reg_x1 - reg_x0) / n_cols
    ch = (reg_y1 - reg_y0) / n_rows
    for k, dept in enumerate(LEFT_COL):
        if dept not in ids_by_dept:
            continue
        col, row = k % n_cols, k // n_cols
        bx0 = reg_x0 + col * cw + gap
        bx1 = reg_x0 + (col + 1) * cw - gap
        by1 = reg_y1 - row * ch - gap
        by0 = reg_y1 - (row + 1) * ch + gap
        box = (bx0, by0, bx1, by1)
        insets.append({"dept": dept, "label": TERR.get(dept, dept), "box": [round(v, 3) for v in box]})
        entries = sorted(ids_by_dept[dept])  # par code circo → bandes ordonnées gauche→droite
        withpoly = [poly_by_id[cid] for cid, _ in entries if cid in poly_by_id]
        dst = (box[0] + 0.06, box[1] + 0.28, box[2] - 0.06, box[3] - 0.06)
        if withpoly:
            geoms = [shape(f["geometry"]) for f in withpoly]
            xs = [g.bounds[0] for g in geoms] + [g.bounds[2] for g in geoms]
            ys = [g.bounds[1] for g in geoms] + [g.bounds[3] for g in geoms]
            src = (min(xs), min(ys), max(xs), max(ys))
            m = _fit_transform(src, dst)
            for f in withpoly:
                out.append({"type": "Feature",
                            "geometry": _round_geom(affine_transform(shape(f["geometry"]), m)),
                            "properties": f["properties"]})
        elif dept in silhouettes:
            # Contour réel de la COM (Natural Earth) ajusté à l'encart, puis découpé en une
            # bande par circo → chaque circo reste cliquable et colorée par ses propres données.
            placed = affine_transform(silhouettes[dept], _fit_transform(silhouettes[dept].bounds, dst))
            placed = _inflate(placed, dst)  # rend visibles les COM minuscules / éparses
            for (cid, i), part in zip(entries, _split_vertical(placed, len(entries))):
                if part.is_empty:
                    continue
                out.append({"type": "Feature", "geometry": _round_geom(part), "properties": props(i)})
        else:
            # Tuiles carrées (territoire sans contour) alignées dans la boîte de l'encart.
            cnt = len(entries)
            tcols = min(cnt, 3)
            trows = (cnt + tcols - 1) // tcols
            tcw = (bx1 - bx0) / tcols
            tch = (by1 - by0) / max(trows, 1)
            r = min(tcw, tch) * 0.4
            for j, (cid, i) in enumerate(entries):
                cx = bx0 + tcw * (j % tcols + 0.5)
                cy = by1 - tch * (j // tcols + 0.5)
                out.append({"type": "Feature", "geometry": _square(cx, cy, r), "properties": props(i)})

    # Étranger : planisphère miniature SOUS le bloc outre-mer — chaque circo = l'union de ses
    # pays (report_geo_overseas_2027). Une SEULE transformation partagée → planisphère cohérent ;
    # bande unique et compacte (les 11 circos ne prennent qu'une rangée, gain de place vertical).
    zz = ids_by_dept.get("ZZ", [])
    if zz and ETR.exists():
        bx0, bx1, by0, by1 = -14.5, -7.0, 39.2, 41.6
        insets.append({"dept": "ZZ", "label": TERR["ZZ"], "box": [bx0, by0, bx1, by1]})
        world = {f["properties"]["id"]: shape(f["geometry"])
                 for f in json.loads(ETR.read_text())["features"]}
        allg = list(world.values())
        xs = [g.bounds[0] for g in allg] + [g.bounds[2] for g in allg]
        ys = [g.bounds[1] for g in allg] + [g.bounds[3] for g in allg]
        m = _fit_transform((min(xs), min(ys), max(xs), max(ys)),
                           (bx0 + 0.05, by0 + 0.05, bx1 - 0.05, by1 - 0.05))
        for cid, i in zz:
            g = world.get(cid)
            if g is not None:
                out.append({"type": "Feature",
                            "geometry": _round_geom(affine_transform(g, m)),
                            "properties": props(i)})
    elif zz:
        # Repli (planisphère non généré) : bandeau de 11 tuiles carrées, une rangée.
        bx0, bx1, by0, by1 = -14.5, -7.0, 39.2, 41.6
        insets.append({"dept": "ZZ", "label": TERR["ZZ"], "box": [bx0, by0, bx1, by1]})
        zcw = (bx1 - bx0) / len(zz)
        r = min(zcw, by1 - by0) * 0.4
        for j, (cid, i) in enumerate(zz):
            out.append({"type": "Feature",
                        "geometry": _square(bx0 + zcw * (j + 0.5), (by0 + by1) / 2, r),
                        "properties": props(i)})

    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": out},
                              ensure_ascii=False, separators=(",", ":")))
    INSETS.write_text(json.dumps(insets, ensure_ascii=False))
    print(f"display : {len(out)} circos (dont encarts), {len(insets)} encarts → {OUT.name} "
          f"({OUT.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    export()
