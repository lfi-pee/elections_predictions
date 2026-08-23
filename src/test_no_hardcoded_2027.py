"""Audit « aucun chiffre figé » : les statistiques AFFICHÉES viennent des données servies, pas
de valeurs saisies à la main dans le HTML.

Depuis que les infobulles sont DYNAMIQUES (remplies en JS depuis summary.json), la garantie est
« par construction ». Ce test le vérifie : (1) l'infobulle des sièges dans index.html est un
gabarit VIDE (aucune statistique en dur) ; (2) le JS qui la remplit lit bien les champs servis ;
(3) la part LFI affichée vient du curseur, pas d'un littéral ; (4) les champs servis existent.

Il ÉCHOUE si quelqu'un re-fige un chiffre dans le HTML ou débranche le rendu des données.

    python3 -u -m src.test_no_hardcoded_2027
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SUMMARY = Path("report_app/2027/data/summary.json")
INDEX = Path("report_app/2027/index.html")
CONTROLS = Path("report_app/2027/js/controls.js")


def _tip_after(html: str, marker: str) -> str:
    i = html.find(marker)
    j = html.find('class="tip"', i) if i >= 0 else -1
    if j < 0:
        return ""
    return html[j:html.find("</span>", j)]


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    html = INDEX.read_text()
    js = CONTROLS.read_text()
    fails = []

    # (1) L'infobulle des sièges est un gabarit VIDE (dynamique) — aucune stat en dur.
    seat_tip = _tip_after(html, "Projection en sièges")
    if 'id="seat-info-tip"' not in seat_tip:
        fails.append("infobulle sièges : gabarit dynamique #seat-info-tip absent")
    stray = [n for n in re.findall(r"\d+", seat_tip) if n not in ("1", "2")]  # 1er/2nd tour ok
    if stray:
        fails.append(f"infobulle sièges : statistiques FIGÉES dans le HTML {stray} (doit être vide)")

    # (2) Le JS remplit l'infobulle depuis les champs servis (branché aux données).
    need = ["backtest_2024_e2e", "backtest_2024_allseats", "accuracy_seats",
            "first_round", ".all", "model.ED", "actual.ED"]
    miss = [k for k in need if k not in js]
    if miss:
        fails.append(f"renderSeatInfo ne lit pas les champs servis : {miss}")

    # (3) La part LFI affichée vient du curseur (variable), pas d'un pourcentage littéral.
    if "actuellement <b>${pct} %</b>" not in js:
        fails.append("infobulle LFI : la part de voix n'est pas dynamique (${pct} attendu)")

    # (4) Les champs servis existent et sont plausibles (sinon le rendu serait vide/faux).
    e, a = s.get("backtest_2024_e2e"), s.get("backtest_2024_allseats")
    if not e or not a:
        fails.append("summary.json : backtest_2024_e2e / _allseats manquant")
    else:
        for path, lo, hi in [(e["accuracy_seats"], 60, 100), (a["all"]["accuracy"], 60, 100),
                             (a["first_round"]["accuracy"], 80, 100), (e["actual"]["ED"], 90, 130)]:
            if not (lo <= path <= hi):
                fails.append(f"summary : valeur hors plage attendue ({path} ∉ [{lo},{hi}])")
    scn = next((x for x in s["scenarios"] if x["key"] == s["default_scenario"]), None)
    if scn and not (0.30 <= scn.get("radical_share", 0) <= 0.45):
        fails.append(f"radical_share servi {scn['radical_share']} hors plage sondages ~0,37")

    if fails:
        print("ÉCHEC — chiffres figés ou rendu débranché des données :")
        for f in fails:
            print("  ✗", f)
        sys.exit(1)
    print("✅ Infobulles dynamiques : chiffres tirés des données servies, rien n'est figé "
          "dans le HTML (sièges + part LFI).")


if __name__ == "__main__":
    main()
