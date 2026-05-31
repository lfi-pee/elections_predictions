"""Panneaux statiques de preuve + données dérivées pour le site.

- `coverage.json` : couverture conforme empirique (80/90/95) par bloc — servie en
  pastille numérique compacte (la promesse tenue, sans graphe de statisticien).
- `fig_method.svg` : schéma de méthode en trois temps (le modèle lu bureau par
  bureau · le gisement de gauche estimé · la validation croisée sur le passé) —
  cartes illustrées, chaque étape se montre elle-même. Aucune lettre grecque à
  l'écran (lisibilité néophyte) : « la part qui revient à gauche », jamais « γ ».

La précision-par-marge et l'histogramme de marges sont tracés côté client
(ils suivent les curseurs et restent en palette).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

MASTER = Path("data/report/bv_master.parquet")
OUT = Path("report_app/data")
LEVELS = (80, 90, 95)
BLOCKS = {
    "G": "Gauche",
    "CD": "Centre+Droite",
    "ED": "Extrême Droite",
    "AB": "Abstention",
}
COLOR = {"G": "#E4572E", "CD": "#4A90D9", "ED": "#6A4C93", "AB": "#9AA0A6"}
PALE = {"G": "#F5D5C9", "CD": "#CFE1F2", "ED": "#DBD2E8", "AB": "#E4E5E7"}
PAPER, INK = "#FAFAF7", "#1A1A2E"
ACCENT, CARD, LINE, MUT = "#cc2229", "#FFFFFF", "#E7E4DC", "#6B6B76"


def coverage(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    cov: dict[str, dict[int, float]] = {}
    for b in BLOCKS:
        act = df[f"act_{b}"].to_numpy()
        cov[b] = {}
        for lvl in LEVELS:
            lo, hi = df[f"lower{lvl}_{b}"].to_numpy(), df[f"upper{lvl}_{b}"].to_numpy()
            cov[b][lvl] = round(float(((act >= lo) & (act <= hi)).mean()) * 100, 1)
    return cov


def _gamma_by_type() -> dict[str, int]:
    """Part de gauche du votant marginal par type de scrutin (1 bin), même source
    identifiée que le site — pas de lettre grecque à l'écran, le concept en clair."""
    from src import movability_turnout as mt

    return {
        t: round(float(mt.gamma_curve(mt.panel_diffs(t), nbins=1).gamma_pct.iloc[0]))
        for t in ("Legislatives_T1", "Europeennes_T1", "Presidentielle_T1")
    }


def _card(
    x: float, y: float, w: float, h: float, fill: str = CARD, stroke: str = LINE
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1" filter="url(#sh)"/>'
    )


def _badge(x: float, y: float, n: str) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="11" fill="{ACCENT}"/>'
        f'<text x="{x}" y="{y + 4.2}" text-anchor="middle" font-size="12.5" '
        f'font-weight="700" fill="#fff">{n}</text>'
    )


def _seg_bar(
    x: float, y: float, w: float, h: float, segs: list[tuple[str, float]]
) -> str:
    out, cx = "", x
    for col, frac in segs:
        out += f'<rect x="{cx:.1f}" y="{y}" width="{w * frac:.1f}" height="{h}" rx="2" fill="{col}"/>'
        cx += w * frac
    return out


def _chevron(cx: float, cy: float) -> str:
    return (
        f'<path d="M{cx - 4},{cy - 6} L{cx + 4},{cy} L{cx - 4},{cy + 6}" fill="none" '
        f'stroke="{ACCENT}" stroke-width="2.4" stroke-linecap="round" '
        f'stroke-linejoin="round" opacity="0.9"/>'
    )


def _band_model() -> str:
    axs, w, cy = [16, 212, 408, 604], 180, 84
    nat = [("#E4572E", 0.30), ("#4A90D9", 0.34), ("#6A4C93", 0.36)]
    pred = [("#E4572E", 0.46), ("#4A90D9", 0.30), ("#6A4C93", 0.24)]
    titles = [
        "Le national",
        "+ lecture locale",
        "= prévision du bureau",
        "la fourchette",
    ]
    caps = [
        "la moyenne des sondages",
        "votes passés · INSEE · géo",
        "part des blocs + abstention",
        "fiable à 80 / 90 / 95 %",
    ]
    cards = "".join(_card(x, cy, w, 104) for x in axs)
    labels = "".join(
        f'<text x="{x + w / 2}" y="{cy + 70}" text-anchor="middle" font-size="12.5" '
        f'font-weight="700" fill="{INK}">{t}</text>'
        f'<text x="{x + w / 2}" y="{cy + 88}" text-anchor="middle" font-size="9" fill="{MUT}">{c}</text>'
        for x, t, c in zip(axs, titles, caps)
    )
    chev = "".join(_chevron(x - 8, cy + 50) for x in axs[1:])
    iy = cy + 30
    i0 = _seg_bar(axs[0] + 20, iy - 6, 140, 11, nat)
    base = f'<line x1="{axs[1] + 20}" y1="{iy}" x2="{axs[1] + 160}" y2="{iy}" stroke="#CBC9C2" stroke-width="2"/>'
    devs = "".join(
        f'<line x1="{axs[1] + 36 + k * 28}" y1="{iy}" x2="{axs[1] + 36 + k * 28}" '
        f'y2="{iy + d}" stroke="{ACCENT}" stroke-width="2.4" stroke-linecap="round"/>'
        for k, d in enumerate((-11, 8, -6, 12, -9))
    )
    i2 = _seg_bar(axs[2] + 20, iy - 7, 140, 13, pred)
    xa, xb = axs[3] + 24, axs[3] + 156
    i3 = (
        f'<line x1="{xa}" y1="{iy}" x2="{xb}" y2="{iy}" stroke="{INK}" stroke-width="1.6"/>'
        f'<line x1="{xa}" y1="{iy - 6}" x2="{xa}" y2="{iy + 6}" stroke="{INK}" stroke-width="1.6"/>'
        f'<line x1="{xb}" y1="{iy - 6}" x2="{xb}" y2="{iy + 6}" stroke="{INK}" stroke-width="1.6"/>'
        f'<circle cx="{(xa + xb) / 2}" cy="{iy}" r="4.5" fill="{ACCENT}"/>'
    )
    sub = (
        f'<text x="16" y="212" font-size="9.6" fill="{MUT}">Une régression Ridge entraînée sur '
        f"tous les scrutins : elle pèse les indicateurs INSEE, la géographie et les votes passés.</text>"
        f'<text x="16" y="226" font-size="9.6" fill="{MUT}">L\'abstention est prédite à part — faute '
        f"de sondage, calée sur la participation passée (d'où son incertitude surtout nationale).</text>"
    )
    return cards + i0 + base + devs + i2 + i3 + labels + chev + sub


def _sparkline(x: float, y: float, w: float, h: float, curves: dict) -> str:
    keys = (
        ("Legislatives_T1", COLOR["G"], 2.4),
        ("Europeennes_T1", COLOR["CD"], 1.4),
        ("Presidentielle_T1", COLOR["ED"], 1.4),
    )
    sx = lambda v: x + (v - 10.0) / 56.0 * w  # noqa: E731
    sy = lambda v: y + h - v / 60.0 * h  # noqa: E731
    out = f'<line x1="{x}" y1="{y + h}" x2="{x + w}" y2="{y + h}" stroke="#E0DDD5" stroke-width="1"/>'
    for key, col, sw in keys:
        pts = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in curves[key])
        out += (
            f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="{sw}" '
            f'stroke-linejoin="round" stroke-linecap="round" opacity="0.95"/>'
        )
    return out


def _band_gisement(gain: dict, gbt: dict[str, int], curves: dict) -> str:
    conj = f"{gain['conjunctural_abstainers'] / 1e6:.2f}".replace(".", ",")
    tot = f"{gain['total_abstainers'] / 1e6:.1f}".replace(".", ",")
    mob = f"{gain['mobilization_voters'] / 1e6:.2f}".replace(".", ",")
    frac = gain["conjunctural_abstainers"] / gain["total_abstainers"]
    cy, cw, lx, mx, rx = 268, 224, 16, 288, 560
    bx, by = lx + 18, cy + 42
    left = (
        _card(lx, cy, cw, 106)
        + f'<text x="{lx + cw / 2}" y="{cy + 24}" text-anchor="middle" font-size="11.5" '
        f'font-weight="700" fill="{INK}">Abstentionnistes conjoncturels</text>'
        + f'<rect x="{bx}" y="{by}" width="{cw - 36}" height="12" rx="3" fill="{PALE["AB"]}"/>'
        + f'<rect x="{bx}" y="{by}" width="{(cw - 36) * frac:.1f}" height="12" rx="3" fill="{COLOR["AB"]}"/>'
        + f'<text x="{lx + cw / 2}" y="{cy + 70}" text-anchor="middle" font-size="8.4" '
        f'fill="{MUT}">abstention prévue − plancher historique du bureau</text>'
        + f'<text x="{lx + cw / 2}" y="{cy + 90}" text-anchor="middle" font-size="11" fill="{INK}">'
        f'<tspan font-weight="700">{conj} M</tspan><tspan fill="{MUT}">&#160;sur&#160;{tot} M</tspan></text>'
    )
    mid = (
        _card(mx, cy, cw, 106)
        + f'<text x="{mx + cw / 2}" y="{cy + 22}" text-anchor="middle" font-size="11.5" '
        f'font-weight="700" fill="{INK}">la part qui revient à gauche</text>'
        + f'<text x="{mx + cw / 2}" y="{cy + 36}" text-anchor="middle" font-size="8.4" '
        f'fill="{MUT}">100 votants de plus → combien rejoignent la gauche ?</text>'
        + _sparkline(mx + 22, cy + 46, cw - 44, 44, curves)
    )
    right = (
        _card(rx, cy, cw, 106, fill=PALE["G"], stroke=COLOR["G"])
        + f'<text x="{rx + cw / 2}" y="{cy + 28}" text-anchor="middle" font-size="12" '
        f'font-weight="700" fill="{INK}">Mobilisables de gauche</text>'
        + f'<text x="{rx + cw / 2}" y="{cy + 66}" text-anchor="middle" font-size="26" '
        f'font-weight="800" fill="{COLOR["G"]}">{mob} M</text>'
        + f'<text x="{rx + cw / 2}" y="{cy + 88}" text-anchor="middle" font-size="9.5" '
        f'fill="{MUT}">en métropole</text>'
    )
    ops = (
        f'<text x="264" y="{cy + 60}" text-anchor="middle" font-size="26" fill="{MUT}">×</text>'
        f'<text x="536" y="{cy + 59}" text-anchor="middle" font-size="26" fill="{MUT}">=</text>'
    )
    legi, euro, pres = (
        gbt["Legislatives_T1"],
        gbt["Europeennes_T1"],
        gbt["Presidentielle_T1"],
    )
    cap = (
        f'<text x="16" y="402" font-size="9.6" fill="{MUT}">Cette part se mesure, jamais ne se '
        f"suppose : sur les scrutins passés du même type, quand la participation d'un bureau monte, "
        f"on lit combien des votants gagnés rejoignent la gauche.</text>"
        f'<text x="16" y="416" font-size="9.6" fill="{MUT}">Sur 100 revenants, combien votent à '
        f"gauche — stable dans le temps (+0,96), mais propre au scrutin : "
        f'<tspan font-weight="700" fill="{COLOR["G"]}">législatives ~{legi}</tspan> · '
        f'<tspan font-weight="700" fill="{COLOR["CD"]}">européennes ~{euro}</tspan> · '
        f'<tspan font-weight="700" fill="{COLOR["ED"]}">présidentielle ~{pres}</tspan> sur 100.</text>'
    )
    return left + ops + mid + right + cap


def _band_proof(acc: float, r2: dict[str, float]) -> str:
    x0, x1, cy = 70.0, 620.0, 494.0
    px = lambda yr: x0 + (yr - 2002) * (x1 - x0) / 22  # noqa: E731
    base = f'<line x1="{x0}" y1="{cy}" x2="{x1 + 8}" y2="{cy}" stroke="#E0DDD5" stroke-width="1.4"/>'
    past = "".join(
        f'<circle cx="{px(y)}" cy="{cy}" r="5.5" fill="{INK}" opacity="0.85"/>'
        for y in (2002, 2007, 2017, 2022)
    )
    drop = (
        f'<circle cx="{px(2012)}" cy="{cy}" r="5.5" fill="{PAPER}" stroke="{COLOR["ED"]}" stroke-width="2"/>'
        f'<text x="{px(2012)}" y="{cy - 16}" text-anchor="middle" font-size="9" '
        f'fill="{COLOR["ED"]}">retiré à tour de rôle</text>'
    )
    bracket = (
        f'<path d="M{px(2002) - 6} {cy + 16} V{cy + 22} H{px(2022) + 6} V{cy + 16}" '
        f'fill="none" stroke="#BFBCB4" stroke-width="1"/>'
        f'<text x="{(px(2002) + px(2022)) / 2}" y="{cy + 38}" text-anchor="middle" '
        f'font-size="10" fill="{MUT}">scrutins passés (2002–2022) · servent à choisir le modèle</text>'
    )
    sep = (px(2022) + px(2024)) / 2
    test = (
        f'<line x1="{sep}" y1="{cy - 26}" x2="{sep}" y2="{cy + 22}" stroke="#D8D5CD" '
        f'stroke-width="1" stroke-dasharray="3 3"/>'
        f'<circle cx="{px(2024)}" cy="{cy}" r="15" fill="{ACCENT}" opacity="0.14"/>'
        f'<circle cx="{px(2024)}" cy="{cy}" r="8" fill="{ACCENT}"/>'
        f'<text x="{px(2024)}" y="{cy - 16}" text-anchor="middle" font-size="11" '
        f'font-weight="700" fill="{INK}">2024</text>'
        f'<text x="{px(2024)}" y="{cy + 38}" text-anchor="middle" font-size="10" '
        f'fill="{MUT}">jamais vu · testé une fois</text>'
    )
    acc_fr = f"{acc:.1f}".replace(".", ",")
    legend = "   ·   ".join(
        f'<tspan fill="{COLOR[b]}">{name} {r2[b]:.2f}</tspan>'.replace(".", ",")
        for b, name in BLOCKS.items()
    )
    foot = (
        f'<text x="16" y="560" font-size="13" fill="{INK}">Bon bloc en tête désigné dans '
        f'<tspan font-weight="800" fill="{ACCENT}">{acc_fr} %</tspan> des bureaux.</text>'
        f'<text x="16" y="582" font-size="10.5" fill="{MUT}">R² par bloc (test 2024, hors '
        f"échantillon) :  {legend}</text>"
    )
    return base + past + drop + bracket + test + foot


def method_schema(
    acc: float, r2: dict[str, float], gain: dict[str, object], gbt: dict[str, int]
) -> None:
    curves = json.loads((OUT / "gamma_curve.json").read_text())
    defs = (
        '<defs><filter id="sh" x="-20%" y="-20%" width="140%" height="170%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#1A1A2E" '
        'flood-opacity="0.10"/></filter></defs>'
    )
    head = (
        f'<rect width="800" height="600" fill="{PAPER}"/>'
        f'<text x="16" y="30" font-size="17" font-weight="800" fill="{INK}">Comment ça marche</text>'
        f'<text x="784" y="30" text-anchor="end" font-size="11" fill="{MUT}">prévoir chaque bureau '
        f"· en déduire le gisement de gauche · le prouver sur le passé</text>"
        f'<line x1="16" y1="40" x2="784" y2="40" stroke="{LINE}" stroke-width="1"/>'
    )
    h1 = _badge(27, 66, "1") + (
        f'<text x="46" y="70" font-size="13" font-weight="700" fill="{INK}">'
        f"Le modèle, lu bureau par bureau</text>"
    )
    h2 = _badge(27, 250, "2") + (
        f'<text x="46" y="254" font-size="13" font-weight="700" fill="{INK}">'
        f"Le gisement — l'abstention de gauche, estimée et non supposée</text>"
    )
    h3 = _badge(27, 436, "3") + (
        f'<text x="46" y="440" font-size="13" font-weight="700" fill="{INK}">'
        f"La preuve — choisi sur le passé, testé une seule fois sur 2024</text>"
    )
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 800 600" '
        "font-family=\"Inter, 'Helvetica Neue', Arial, sans-serif\">"
        f"{defs}{head}"
        f"{h1}{_band_model()}"
        f"{h2}{_band_gisement(gain, gbt, curves)}"
        f"{h3}{_band_proof(acc, r2)}</svg>"
    )
    (OUT / "fig_method.svg").write_text(svg, encoding="utf-8")


def build() -> None:
    df = pd.read_parquet(MASTER)
    cov = coverage(df)
    (OUT / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=1))
    summary = json.loads((OUT / "summary.json").read_text())
    method_schema(
        summary["lead_accuracy"], summary["r2"], summary["left_gain"], _gamma_by_type()
    )
    print("figs: coverage + method |", {b: cov[b][90] for b in BLOCKS})


if __name__ == "__main__":
    build()
