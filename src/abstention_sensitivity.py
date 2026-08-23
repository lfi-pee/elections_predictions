"""Sensibilité du modèle de SIÈGES à l'hypothèse d'ABSTENTION.

Le seuil de qualification au 2nd tour est de 12,5 % des **inscrits**, soit
`12,5 / participation` en part d'**exprimés**. Quand l'abstention monte, ce seuil (en
exprimés) grimpe : il faut un score bien plus élevé pour qualifier une 3ᵉ candidature, donc
les **triangulaires se raréfient**, donc le **désistement « front républicain » se déclenche
moins** — or c'est le mécanisme qui fait barrage au RN. Conséquence redoutée : le nombre de
sièges RN dépend fortement de l'hypothèse d'abstention, et le scénario par DÉFAUT (48 %) se
situe loin de l'année de calage du désistement (2024 : ~33 %).

Ce script balaie l'abstention nationale (parts de vote FIXÉES à l'ancre du scénario, seul le
seuil bouge) et reporte, par circo : sièges G/CD/ED, nombre de triangulaires qualifiées face
au RN, et nombre de désistements déclenchés. Objectif : vérifier que la courbe du SEUIL est
LISSE (pas de falaise artificielle) et chiffrer ce canal.

⚠️ PORTÉE LIMITÉE. Ce script isole le SEUL canal du SEUIL (parts de vote gelées). Le curseur
d'abstention du SITE fait AUSSI jouer le couplage γ (turnoutAdjust) : moins d'abstention →
électeurs de retour plutôt à gauche → part de gauche EFFECTIVE plus haute. Dans l'appli réelle
c'est γ qui DOMINE, et il rend la gauche TRÈS sensible à l'abstention dans TOUS les cas (unie
comme divisée). Ne pas conclure de ce script seul sur la sensibilité vécue.

RÉSULTATS (canal du seuil seul ; ancre 2027 par défaut G/CD/ED = 30,4/29,6/40,0 ; 30 → 52 %) :

  - AUCUNE falaise : courbes lisses et monotones. Le seuil de qualif. (exprimés) passe de 17,9 %
    à 26,0 %, les triangulaires face au RN fondent (318→125 unie, 173→8 divisée), sans saut.

  - Le nombre de sièges RN est ROBUSTE au canal du seuil : ~212→219 (unie), ~225→226 (divisée).
    La raréfaction des désistements ne profite pas mécaniquement au RN.

  - À parts gelées, le canal du seuil déplace peu la gauche unie (184→177) et davantage la
    divisée (159→102). MAIS ce contraste est SECONDAIRE : une fois γ inclus (site réel), la
    gauche perd ~65 sièges de 30 % à 52 % d'abstention DANS LES DEUX configurations (unie
    232→168, divisée 201→130). L'ancienne conclusion « l'union amortit le risque d'abstention »
    était un artefact du gel des parts — RETIRÉE. Ce qui reste vrai : à abstention ÉGALE, l'union
    rapporte ~30 à 40 sièges de plus que la division.

    python3 -u -m src.abstention_sensitivity
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import winnability_2027 as W

CIRCO = Path("report_app/2027/data/circo.json")
SUMMARY = Path("report_app/2027/data/summary.json")


def _clamp(v):
    return min(100.0, max(0.0, v))


def _qual(g, cd, ed, ab, cfg, rad):
    """Rejoue la qualification pour instrumenter triangulaires/désistement."""
    turnout = max(0.05, 1 - ab / 100.0)
    thr = 12.5 / turnout
    left = W._left_candidates(g, cfg, rad)
    cands = sorted(left + [cd, ed], reverse=True)
    second = cands[1]
    _, qL = W._left_t2(left, second, thr)
    qC = cd >= second - 1e-9 or cd >= thr
    qE = ed >= second - 1e-9 or ed >= thr
    return qL, qC, qE, thr


def sweep(cfg: str, rad_share: float, right_union: bool, ab_grid, anchor, circo):
    dG = np.array(circo["dG"]); dCD = np.array(circo["dCD"])
    dED = np.array(circo["dED"]); dAB = np.array(circo["dAB"])
    ins = np.array(circo["ins"], float)
    rows = []
    for ab_nat in ab_grid:
        seats = {"G": 0, "CD": 0, "ED": 0}
        tri = desist = 0
        for i in range(len(dG)):
            g = _clamp(anchor["G"] + dG[i])
            cd = _clamp(anchor["CD"] + dCD[i])
            ed = _clamp(anchor["ED"] + dED[i])
            ab = _clamp(ab_nat + dAB[i])
            rad = 1.0 if cfg == "union" else min(0.68, max(0.12, rad_share + 0.006 * dG[i]))
            w = W.seat_winner(g, cd, ed, ab, cfg, rad, right_union)
            seats[w] += 1
            qL, qC, qE, thr = _qual(g, cd, ed, ab, cfg, rad)
            if qL and qC and qE and ed >= cd:      # triangulaire face au RN
                tri += 1
                if not right_union:
                    desist += 1
        rows.append((ab_nat, seats, tri, desist))
    return rows


def main():
    circo = json.loads(CIRCO.read_text())
    s = json.loads(SUMMARY.read_text())
    default = s["default_scenario"]
    sc = next(x for x in s["scenarios"] if x["key"] == default)
    anchor = sc["means"]
    rad_share = sc.get("radical_share", 0.369)
    ab_grid = list(range(30, 53, 2))

    for cfg, ru, lab in [("union", False, "gauche unie (config du backtest)"),
                         (sc["left_config"], sc.get("right_union", False),
                          f"scénario par défaut « {default} »")]:
        thr30 = 12.5 / (1 - 30 / 100)
        thr52 = 12.5 / (1 - 52 / 100)
        print(f"\n=== {lab} — ancre G/CD/ED = {anchor['G']}/{anchor['CD']}/{anchor['ED']} ===")
        print(f"    seuil de qualif. (exprimés) : {thr30:.1f}% à 30% abst. → {thr52:.1f}% à 52% abst.")
        print(f"    {'abst':>5} | {'G':>4} {'CD':>4} {'ED':>4} | {'triang.RN':>9} {'désist.':>7}")
        rows = sweep(cfg, rad_share, ru, ab_grid, anchor, circo)
        prev = None
        for ab, seats, tri, des in rows:
            jump = ""
            if prev is not None:
                dED = seats["ED"] - prev["ED"]
                if abs(dED) >= 15:
                    jump = f"  ← ΔED={dED:+d}"
            mark = "  ⟵ défaut" if abs(ab - anchor["AB"]) < 1 else ""
            print(f"    {ab:>4}% | {seats['G']:>4} {seats['CD']:>4} {seats['ED']:>4} | "
                  f"{tri:>9} {des:>7}{jump}{mark}")
            prev = seats


if __name__ == "__main__":
    main()
