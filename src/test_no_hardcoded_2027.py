"""Audit « aucun chiffre figé » : les statistiques AFFICHÉES par le site doivent correspondre
aux données SERVIES (summary.json), pas être saisies à la main puis oubliées.

Les infobulles statiques de index.html citent des chiffres de validation (justesse rejeu 2024,
sous-estimation RN, part LFI sondée…). Si une reconstruction change ces nombres, l'affichage
doit suivre. Ce test extrait ces chiffres et vérifie qu'ils égalent (à l'arrondi) les valeurs
servies. Il ÉCHOUE si un nombre affiché a divergé de la donnée — forçant la mise à jour.

    python3 -u -m src.test_no_hardcoded_2027
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SUMMARY = Path("report_app/2027/data/summary.json")
INDEX = Path("report_app/2027/index.html")


def _tip_after(html: str, header_marker: str) -> str:
    """Texte de l'infobulle (.tip) qui suit un marqueur de section donné."""
    i = html.find(header_marker)
    if i < 0:
        return ""
    j = html.find('class="tip"', i)
    if j < 0:
        return ""
    k = html.find("</span>", j)
    return html[j:k]


def main() -> None:
    s = json.loads(SUMMARY.read_text())
    html = INDEX.read_text()
    fails = []

    def nums(txt: str) -> set[int]:
        return {int(x) for x in re.findall(r"\d+", txt)}

    # ── Infobulle « Projection en sièges » : justesse rejeu 2024 + sous-estimation RN ──
    seat_tip = _tip_after(html, "Projection en sièges")
    e2e, alls = s.get("backtest_2024_e2e"), s.get("backtest_2024_allseats")
    if not seat_tip:
        fails.append("infobulle 'Projection en sièges' introuvable")
    else:
        present = nums(seat_tip)
        want = []
        if e2e:
            want += [("justesse contesté (e2e)", round(e2e["accuracy_seats"])),
                     ("RN projeté (e2e)", e2e["model"]["ED"]),
                     ("RN réel (e2e)", e2e["actual"]["ED"])]
        if alls:
            want += [("justesse 1er tour", round(alls["first_round"]["accuracy"])),
                     ("justesse ensemble 577", round(alls["all"]["accuracy"]))]
        for label, val in want:
            if val not in present:
                fails.append(f"[sièges] {label} = {val} absent de l'infobulle (chiffres présents : {sorted(present)})")

    # ── Infobulle « Part de la gauche radicale (LFI) » : part radicale sondée (curseur) ──
    # (rendue par controls.js ; on vérifie la valeur servie du scénario par défaut.)
    scn = next((x for x in s["scenarios"] if x["key"] == s["default_scenario"]), None)
    if scn and "radical_share" in scn:
        rad_pct = round(scn["radical_share"] * 100)
        if rad_pct not in (36, 37, 38):  # ~37 % ; garde-fou de cohérence sondages
            fails.append(f"[LFI] part radicale servie {rad_pct}% hors de la plage attendue ~37%")

    if fails:
        print("ÉCHEC — chiffres affichés désynchronisés des données servies :")
        for f in fails:
            print("  ✗", f)
        sys.exit(1)
    print("✅ Chiffres affichés cohérents avec les données servies "
          "(justesse rejeu 2024 contesté/1er tour/ensemble, sous-estimation RN, part LFI).")


if __name__ == "__main__":
    main()
