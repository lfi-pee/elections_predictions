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

from shapely.geometry import mapping, shape
from shapely.affinity import affine_transform

DIR = Path("report_app/2027/data")
GEO = DIR / "circo.geojson"
ARR = DIR / "circo.json"
OUT = DIR / "circo_display.geojson"
INSETS = DIR / "circo_insets.json"

# Libellés des territoires d'outre-mer / étranger (à défaut : le code).
TERR = {
    "ZA": "Guadeloupe", "ZB": "Martinique", "ZC": "Guyane", "ZD": "La Réunion",
    "ZM": "Mayotte", "ZS": "St-Pierre, St-Martin…", "ZN": "Nouvelle-Calédonie",
    "ZP": "Polynésie fr.", "ZW": "Wallis-et-Futuna", "ZX": "Outre-mer",
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
    s = min((dx1 - dx0) / sw, (dy1 - dy0) / sh) * 0.85
    scx, scy = (sx0 + sx1) / 2, (sy0 + sy1) / 2
    dcx, dcy = (dx0 + dx1) / 2, (dy0 + dy1) / 2
    return [s, 0, 0, s, dcx - s * scx, dcy - s * scy]


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
    # lon < −6.5 → aucun chevauchement avec la métropole lon ≥ −5,1), en un bloc compact.
    # 10 territoires en grille 2 colonnes × 5 lignes ; l'étranger en bandeau juste en dessous.
    reg_x0, reg_x1 = -19.5, -6.5
    reg_y0, reg_y1 = 43.2, 51.3
    n_cols, n_rows = 2, 5
    gap = 0.3
    cw = (reg_x1 - reg_x0) / n_cols
    ch = (reg_y1 - reg_y0) / n_rows
    for k, dept in enumerate(LEFT_COL):
        if dept not in ids_by_dept:
            continue
        col, row = k // n_rows, k % n_rows
        bx0 = reg_x0 + col * cw + gap
        bx1 = reg_x0 + (col + 1) * cw - gap
        by1 = reg_y1 - row * ch - gap
        by0 = reg_y1 - (row + 1) * ch + gap
        box = (bx0, by0, bx1, by1)
        insets.append({"dept": dept, "label": TERR.get(dept, dept), "box": [round(v, 3) for v in box]})
        entries = ids_by_dept[dept]
        withpoly = [poly_by_id[cid] for cid, _ in entries if cid in poly_by_id]
        if withpoly:
            geoms = [shape(f["geometry"]) for f in withpoly]
            b = geoms[0].bounds
            xs = [g.bounds[0] for g in geoms] + [g.bounds[2] for g in geoms]
            ys = [g.bounds[1] for g in geoms] + [g.bounds[3] for g in geoms]
            src = (min(xs), min(ys), max(xs), max(ys))
            m = _fit_transform(src, (box[0] + 0.2, box[1] + 0.15, box[2] - 0.2, box[3] - 0.15))
            for f in withpoly:
                out.append({"type": "Feature",
                            "geometry": _round_geom(affine_transform(shape(f["geometry"]), m)),
                            "properties": f["properties"]})
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

    # Étranger : bandeau de 11 tuiles SOUS le bloc outre-mer (toujours à gauche de la
    # métropole, pas en dessous d'elle) — 2 rangées de grosses tuiles.
    zz = ids_by_dept.get("ZZ", [])
    if zz:
        bx0, bx1, by0, by1 = -19.5, -6.5, 38.8, 42.7
        insets.append({"dept": "ZZ", "label": TERR["ZZ"], "box": [bx0, by0, bx1, by1]})
        zcols, zrows = 6, 2
        zcw = (bx1 - bx0) / zcols
        zch = (by1 - by0) / zrows
        r = min(zcw, zch) * 0.4
        for j, (cid, i) in enumerate(zz):
            cx = bx0 + zcw * (j % zcols + 0.5)
            cy = by1 - zch * (j // zcols + 0.5)
            out.append({"type": "Feature", "geometry": _square(cx, cy, r), "properties": props(i)})

    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": out},
                              ensure_ascii=False, separators=(",", ":")))
    INSETS.write_text(json.dumps(insets, ensure_ascii=False))
    print(f"display : {len(out)} circos (dont encarts), {len(insets)} encarts → {OUT.name} "
          f"({OUT.stat().st_size / 1e6:.1f} Mo)")


if __name__ == "__main__":
    export()
