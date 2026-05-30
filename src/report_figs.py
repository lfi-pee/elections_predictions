"""Panneaux statiques de preuve + données dérivées pour le site.

- `coverage.json` : couverture conforme empirique (80/90/95) par bloc — servie en
  pastille numérique compacte (la promesse tenue, sans graphe de statisticien).
- `fig_method.svg` : schéma de méthode (flux en deux temps + frise
  apprentissage/test) — la rigueur rendue lisible en cinq secondes.

La courbe de bascule, la précision-par-marge et l'histogramme de marges sont
tracés côté client (ils suivent les curseurs et restent en palette).
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
PAPER, INK = "#FAFAF7", "#1A1A2E"


def coverage(df: pd.DataFrame) -> dict[str, dict[int, float]]:
    cov: dict[str, dict[int, float]] = {}
    for b in BLOCKS:
        act = df[f"act_{b}"].to_numpy()
        cov[b] = {}
        for lvl in LEVELS:
            lo, hi = df[f"lower{lvl}_{b}"].to_numpy(), df[f"upper{lvl}_{b}"].to_numpy()
            cov[b][lvl] = round(float(((act >= lo) & (act <= hi)).mean()) * 100, 1)
    return cov


def _box(x: float, title: str, sub: str) -> str:
    return (
        f'<rect x="{x}" y="64" width="150" height="74" rx="9" fill="{PAPER}" '
        f'stroke="{INK}" stroke-width="1.4"/>'
        f'<text x="{x + 75}" y="96" text-anchor="middle" font-size="14" '
        f'font-weight="600" fill="{INK}">{title}</text>'
        f'<text x="{x + 75}" y="118" text-anchor="middle" font-size="10.5" '
        f'fill="#555">{sub}</text>'
    )


def method_schema(acc: float, r2: dict[str, float]) -> None:
    flux = [
        ("National", "sondages · participation"),
        ("+ Écart local", "démographie INSEE · vote passé"),
        ("Prédiction", "du bureau de vote"),
        ("Intervalle", "conforme 80/90/95"),
    ]
    xs = [20, 205, 390, 575]
    boxes = "".join(_box(x, t, s) for x, (t, s) in zip(xs, flux))
    arrows = "".join(
        f'<line x1="{x + 150}" y1="101" x2="{x + 185}" y2="101" stroke="{INK}" '
        f'stroke-width="1.4" marker-end="url(#a)"/>'
        for x in xs[:-1]
    )
    x0, x1 = 60.0, 700.0
    px = lambda yr: x0 + (yr - 2002) * (x1 - x0) / 22  # noqa: E731
    legend = "   ·   ".join(
        f'<tspan fill="{COLOR[b]}">{name} {r2[b]:.2f}</tspan>'.replace(".", ",")
        for b, name in BLOCKS.items()
    )
    acc_fr = f"{acc:.1f}".replace(".", ",")
    held = 2012
    ticks = "".join(
        f'<circle cx="{px(y)}" cy="232" r="6" fill="{INK}" opacity="0.85"/>'
        for y in (2002, 2007, 2017, 2022)
    )
    sep = (px(2022) + px(2024)) / 2
    cx = (px(2002) + px(2022)) / 2
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 760 360" font-family="DejaVu Sans, sans-serif">
<defs><marker id="a" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
<path d="M0,0 L7,3.5 L0,7 z" fill="{INK}"/></marker></defs>
<rect width="760" height="360" fill="{PAPER}"/>
<text x="20" y="34" font-size="17" font-weight="700" fill="{INK}">Comment ça marche : national + lecture locale, choisi par validation croisée</text>
{boxes}{arrows}
<text x="20" y="184" font-size="13" font-weight="600" fill="{INK}">Validation croisée par scrutin, puis un test unique</text>
<line x1="{x0}" y1="232" x2="{x1 + 12}" y2="232" stroke="#ddd" stroke-width="1"/>
{ticks}
<circle cx="{px(held)}" cy="232" r="6" fill="{PAPER}" stroke="{COLOR["ED"]}" stroke-width="2"/>
<text x="{px(held)}" y="213" text-anchor="middle" font-size="9.5" fill="{COLOR["ED"]}">retiré</text>
<path d="M{px(2002) - 6} 250 V256 H{px(2022) + 6} V250" fill="none" stroke="#999" stroke-width="1"/>
<text x="{cx}" y="272" text-anchor="middle" font-size="10.5" fill="#555">scrutins passés (2002–2022) · chacun retiré à tour de rôle pour choisir le modèle</text>
<line x1="{sep}" y1="206" x2="{sep}" y2="256" stroke="#ccc" stroke-width="1" stroke-dasharray="3 3"/>
<circle cx="{px(2024)}" cy="232" r="9" fill="{COLOR["ED"]}" stroke="{PAPER}" stroke-width="2"/>
<text x="{px(2024)}" y="213" text-anchor="middle" font-size="11" font-weight="600" fill="{INK}">2024</text>
<text x="745" y="272" text-anchor="end" font-size="10.5" fill="#555">tenu à l'écart · testé une fois</text>
<text x="20" y="312" font-size="13" fill="{INK}">Bon bloc en tête appelé dans <tspan font-weight="700">{acc_fr} %</tspan> des bureaux</text>
<text x="20" y="338" font-size="11">R² par bloc, test 2024 hors échantillon :  {legend}</text>
</svg>"""
    (OUT / "fig_method.svg").write_text(svg, encoding="utf-8")


def build() -> None:
    df = pd.read_parquet(MASTER)
    cov = coverage(df)
    (OUT / "coverage.json").write_text(json.dumps(cov, ensure_ascii=False, indent=1))
    summary = json.loads((OUT / "summary.json").read_text())
    method_schema(summary["lead_accuracy"], summary["r2"])
    print("figs: coverage + method |", {b: cov[b][90] for b in BLOCKS})


if __name__ == "__main__":
    build()
