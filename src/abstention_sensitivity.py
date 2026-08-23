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
au RN, et nombre de désistements déclenchés. Objectif : vérifier que la courbe est LISSE (pas
de falaise artificielle) et chiffrer l'ampleur de l'effet.

RÉSULTATS (ancre 2027 par défaut G/CD/ED = 30,4/29,6/40,0 ; abstention 30 → 52 %) :

  - AUCUNE falaise : les courbes sont lisses et monotones sur toute la plage. Le seuil de
    qualification (exprimés) passe de 17,9 % à 26,0 %, les triangulaires face au RN fondent
    (318→125 en gauche unie, 173→8 en gauche divisée), sans discontinuité de sièges.

  - Le nombre de sièges RN est ROBUSTE à l'hypothèse d'abstention : ~212→219 (gauche unie),
    ~225→226 (gauche divisée). La crainte « le RN gonfle quand l'abstention monte » est
    infondée — la raréfaction des désistements ne lui profite pas mécaniquement.

  - La sensibilité réelle est PROPRE À LA GAUCHE et dépend de la division :
        gauche UNIE     : 184 → 177 sièges  (amplitude 7)   → quasi insensible à l'abstention
        gauche DIVISÉE  : 159 → 102 sièges  (amplitude 57)  → très fragile à l'abstention
    Une gauche divisée dépend des triangulaires + désistements pour convertir ; ceux-ci
    s'effondrent quand le seuil grimpe. L'union AMORTIT le risque d'abstention. (Le couplage γ,
    tenu fixe ici, renforcerait encore l'écart : à basse abstention les revenants penchent à
    gauche.) C'est un argument rigoureux, chiffré, en faveur de l'union.

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
