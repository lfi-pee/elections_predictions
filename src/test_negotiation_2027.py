"""Tests de la négociation 2027 (`negotiation_2027`) et de la mesure du report vers une
candidature LFI (`label_effect_2024`) : invariants du JSON servi + cohérence du modèle.

    python3 -u -m src.test_negotiation_2027
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from src import negotiation_2027 as N, winnability_2027 as W

NEG = Path("report_app/2027/data/negotiation.json")
EFF = Path("report_app/2027/data/label_effect_2024.json")
HTML = Path("report_app/2027/negotiation.html")


def main() -> None:
    fails: list[str] = []
    d = json.loads(NEG.read_text())
    e = json.loads(EFF.read_text())
    rows = d["rows"]

    # ── Mesure du report LFI : signée, cohérente avec le JSON servi ──
    k = e["duels_vs_rn"]
    if not (k["k_fi"] < k["k_union"]):
        fails.append("k_fi < k_union attendu (report LFI mesuré sous la moyenne de l'union)")
    if abs(e["model"]["cd2l_delta_lfi"] - (k["k_fi"] - k["k_union"])) > 1e-3:
        fails.append("delta_lfi ≠ k_fi − k_union")
    if d["params"]["cd2l_delta"]["lfi"] != e["model"]["cd2l_delta_lfi"]:
        fails.append("negotiation.json ne porte pas le delta servi par label_effect_2024.json")
    if e["sample"]["n_duels_vs_rn"] < 100:
        fails.append("échantillon de duels trop petit")

    # ── Le décalage joue dans le modèle de sièges, dans le bon sens, et 0 = modèle de la carte ──
    if W.seat_winner(30, 30, 40, 48, "union", 1.0) != W.seat_winner(30, 30, 40, 48, "union", 1.0, cd2l_delta=0.0):
        fails.append("cd2l_delta=0 doit être le modèle inchangé")
    flips = 0
    for g in range(26, 40):
        for ed in range(38, 50):
            cd = 100 - g - ed
            a = W.seat_winner(g, cd, ed, 48, "union", 1.0)
            b = W.seat_winner(g, cd, ed, 48, "union", 1.0, cd2l_delta=e["model"]["cd2l_delta_lfi"])
            if a == "G" and b != "G":
                flips += 1
            if b == "G" and a != "G":
                fails.append(f"candidature LFI gagne là où le candidat moyen perd : g={g} cd={cd} ed={ed}")
    if flips == 0:
        fails.append("aucune circo ne bascule : le décalage est inopérant")

    # ── JSON servi : 577 lignes, groupes partitionnés, probabilités bornées ──
    if len(rows) != 577:
        fails.append(f"{len(rows)} lignes (577 attendues)")
    groups = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
        if not r["pub"]:
            if r["group"] != "non_mesure" or r["p_lfi"] is not None:
                fails.append(f"{r['id']} non publiable mais chiffré")
            continue
        for key in ("p_lfi", "p_lfi_local", "p_lfi_ru", "q_lfi"):
            if not (0.0 <= r[key] <= 1.0):
                fails.append(f"{r['id']} {key} hors [0,1]")
        if r["group"] == "hors_union" and r["depute"]["groupe"] in N.LEFT_GROUPS:
            fails.append(f"{r['id']} hors union mais sortant·e dans un groupe de gauche")
        if (r["depute"]["groupe"] == N.LFI_GROUP) != (r["group"] == "acquis"):
            fails.append(f"{r['id']} acquis ⇔ sortant·e LFI violé")
        if r["group"] == "sans_enjeu" and r["p_lfi"] >= N.P_MIN:
            fails.append(f"{r['id']} sans enjeu mais chance ≥ P_MIN")
        if r["group"] == "en_jeu" and r["p_lfi"] < N.P_MIN:
            fails.append(f"{r['id']} en jeu mais chance < P_MIN")
        if r["posture"] != N.posture(r["group"], r["q_lfi"], r["rdev"]):
            fails.append(f"{r['id']} posture non reproductible")
    if groups != d["groups"]:
        fails.append(f"comptes de groupes incohérents {groups} ≠ {d['groups']}")
    if d["groups"].get("acquis") != d["totals"]["n_acquis"]:
        fails.append("n_acquis ≠ nombre d'acquis")
    if sum(d["postures"].values()) != sum(1 for r in rows if r["posture"]):
        fails.append("comptes de postures incohérents")
    for r in rows:
        for bad in ("p_other", "price", "rate", "q_other"):
            if bad in r:
                fails.append(f"champ {bad} encore servi : le partenaire ne doit entrer dans aucun calcul")
                break

    # ── Courbe : en jeu seulement, triée par p_lfi décroissant, cumul monotone ──
    c = d["curve"]
    by = {r["id"]: r for r in rows}
    neg = [by[i] for i in c["ids"]]
    if any(neg[i]["p_lfi"] < neg[i + 1]["p_lfi"] - 1e-9 for i in range(len(neg) - 1)):
        fails.append("courbe : ids non triés par p_lfi décroissant")
    if [r["rank"] for r in neg] != list(range(1, len(neg) + 1)):
        fails.append("rangs ≠ position dans la courbe")
    if any(r["group"] != "en_jeu" for r in neg):
        fails.append("courbe : contient des circos hors « en jeu »")
    if abs(c["cum_lfi"][-1] - sum(r["p_lfi"] for r in neg)) > 0.05:
        fails.append("cum_lfi final ≠ somme des p_lfi en jeu")

    # ── Ventilation 2024 : les nuances de gauche somment au total ; UG = candidature NFP ──
    for r in rows:
        if r.get("h2024_parts"):
            if abs(sum(r["h2024_parts"].values()) - r["h2024_G"]) > 0.11:
                fails.append(f"{r['id']} ventilation 2024 ≠ total gauche")
    if abs(float(d["split"]["near"]) - d["split"]["default_share"]) > 0.005:
        fails.append("la part sondages n'est pas le réglage par défaut")
    if "rad_gain" not in d["params"] or len(d["params"]["rad_clip"]) != 2:
        fails.append("règle de part locale LFI (rad_gain, rad_clip) absente des paramètres")
    # ── Force réelle : grille de parts LFI, q servi = grille à near, monotone en moyenne ──
    sp = d["split"]
    if sp["near"] not in sp["shares"] or sp["near"] not in sp["by_share"]:
        fails.append("split.near absent de la grille")
    for kk, b in sp["by_share"].items():
        for key in ("q_lfi", "w_lfi"):
            if len(b[key]) != 577 or not all(0.0 <= v <= 1.0 for v in b[key]):
                fails.append(f"split[{kk}].{key} : longueur ou bornes")
        if any(w > q + 1e-9 for w, q in zip(b["w_lfi"], b["q_lfi"])):
            fails.append(f"split[{kk}] : w_lfi > q_lfi")
    near = sp["by_share"][sp["near"]]
    for i, r in enumerate(rows):
        if r["pub"] and r["q_lfi"] != near["q_lfi"][i]:
            fails.append(f"{r['id']} q servi ≠ grille à near")
    means = [sum(sp["by_share"][kk]["q_lfi"]) for kk in sp["shares"]]
    if any(means[i] > means[i + 1] for i in range(len(means) - 1)):
        fails.append("q_lfi moyen non croissant avec la part LFI")

    # ── La page : un seul tableau, gabarits remplis en JS, aucun chiffre ni partenaire figé ──
    html = HTML.read_text()
    for anchor in ('id="m-label"', 'id="m-groups"', 'id="m-unc"', 'id="m-posture"', 'id="m-hors"', 'id="share"',
                   'id="neg-table"', 'data/negotiation.json', 'data/label_effect_2024.json'):
        if anchor not in html:
            fails.append(f"negotiation.html : {anchor} manquant")
    for bad in ('id="tiles"', 'class="chart"', "PS · écolo", "Prix", "notre prédiction"):
        if bad in html:
            fails.append(f"negotiation.html : {bad} ne doit plus apparaître")

    if fails:
        print("ÉCHEC :\n  - " + "\n  - ".join(fails[:30]))
        sys.exit(1)
    print(f"OK — négociation 2027 (LFI) : {len(rows)} circos, groupes {d['groups']}, postures {d['postures']}, "
          f"{flips} cas-grille basculés par le report LFI, mesuré sur {e['sample']['n_duels_vs_rn']} duels.")


if __name__ == "__main__":
    main()
