"""Tests de la négociation 2027 (`negotiation_2027`) et de la mesure d'étiquette
(`label_effect_2024`) : invariants du JSON servi + cohérence du modèle d'étiquette.

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

    # ── Mesure d'étiquette : centrée, signée, cohérente avec le JSON servi ──
    k = e["duels_vs_rn"]
    if not (k["k_fi"] < k["k_union"] < k["k_other"]):
        fails.append("k_fi < k_union < k_other attendu (pénalité LFI mesurée)")
    if abs(e["model"]["cd2l_delta_lfi"] - (k["k_fi"] - k["k_union"])) > 1e-3:
        fails.append("delta_lfi ≠ k_fi − k_union")
    if d["params"]["cd2l_delta"]["lfi"] != e["model"]["cd2l_delta_lfi"]:
        fails.append("negotiation.json ne porte pas le delta servi par label_effect_2024.json")
    if e["sample"]["n_duels_vs_rn"] < 100:
        fails.append("échantillon de duels trop petit")

    # ── Le décalage joue dans le modèle de sièges, dans le bon sens, et 0 = modèle de la carte ──
    if W.seat_winner(30, 30, 40, 48, "union", 1.0) != W.seat_winner(30, 30, 40, 48, "union", 1.0, cd2l_delta=0.0):
        fails.append("cd2l_delta=0 doit être le modèle inchangé")
    # duel G–RN serré : la gauche gagne avec le bonus « autre » mais perd avec la pénalité LFI
    flips = 0
    for g in range(26, 40):
        for ed in range(38, 50):
            cd = 100 - g - ed
            a = W.seat_winner(g, cd, ed, 48, "union", 1.0, cd2l_delta=e["model"]["cd2l_delta_other"])
            b = W.seat_winner(g, cd, ed, 48, "union", 1.0, cd2l_delta=e["model"]["cd2l_delta_lfi"])
            if a == "G" and b != "G":
                flips += 1
            if b == "G" and a != "G":
                fails.append(f"étiquette LFI gagne là où l'autre perd : g={g} cd={cd} ed={ed}")
    if flips == 0:
        fails.append("aucune circo ne bascule entre étiquettes : le décalage est inopérant")

    # ── JSON servi : 577 lignes, groupes partitionnés, probabilités bornées, prix cohérent ──
    if len(rows) != 577:
        fails.append(f"{len(rows)} lignes (577 attendues)")
    groups = {}
    for r in rows:
        groups[r["group"]] = groups.get(r["group"], 0) + 1
        if not r["pub"]:
            if r["group"] != "non_mesure" or r["p_lfi"] is not None:
                fails.append(f"{r['id']} non publiable mais chiffré")
            continue
        for key in ("p_lfi", "p_other", "p_avg", "p_lfi_local", "p_lfi_ru"):
            if not (0.0 <= r[key] <= 1.0):
                fails.append(f"{r['id']} {key} hors [0,1]")
        if abs(r["price"] - (r["p_other"] - r["p_lfi"])) > 2e-3:
            fails.append(f"{r['id']} prix ≠ p_other − p_lfi")
        if r["p_lfi"] > r["p_other"] + 1e-9:
            fails.append(f"{r['id']} p_lfi > p_other : la pénalité mesurée doit jouer dans un seul sens")
        if r["group"] == "hors_union" and r["depute"]["groupe"] in N.LEFT_GROUPS:
            fails.append(f"{r['id']} hors union mais sortant·e dans un groupe de gauche")
        if (r["depute"]["groupe"] == N.LFI_GROUP) != (r["group"] == "acquis"):
            fails.append(f"{r['id']} acquis ⇔ sortant·e LFI violé")
        if r["group"] == "sans_enjeu" and max(r["p_lfi"], r["p_other"]) >= N.P_MIN:
            fails.append(f"{r['id']} sans enjeu mais chance ≥ P_MIN")
        if r["group"] == "libre" and r["price"] > N.PRICE_FREE + 1e-9:
            fails.append(f"{r['id']} libre mais prix > PRICE_FREE")
        if r["group"] == "a_negocier" and r["price"] <= N.PRICE_FREE:
            fails.append(f"{r['id']} à négocier mais prix ≤ PRICE_FREE")
    if groups != d["groups"]:
        fails.append(f"comptes de groupes incohérents {groups} ≠ {d['groups']}")
    if d["groups"].get("acquis") != d["totals"]["n_acquis"]:
        fails.append("n_acquis ≠ nombre d'acquis")

    # ── Courbe : ordre par p_lfi décroissant, cumuls monotones, sommes exactes ──
    c = d["curve"]
    by = {r["id"]: r for r in rows}
    neg = [by[i] for i in c["ids"]]
    if any(neg[i]["p_lfi"] < neg[i + 1]["p_lfi"] - 1e-9 for i in range(len(neg) - 1)):
        fails.append("courbe : ids non triés par p_lfi décroissant")
    if [r["rank"] for r in neg] != list(range(1, len(neg) + 1)):
        fails.append("rangs ≠ position dans la courbe")
    if any(r["group"] not in ("libre", "a_negocier") for r in neg):
        fails.append("courbe : contient des acquis / sans enjeu / non mesurées")
    if abs(c["cum_lfi"][-1] - sum(r["p_lfi"] for r in neg)) > 0.05:
        fails.append("cum_lfi final ≠ somme des p_lfi négociables")
    if any(c["cum_lfi"][i] > c["cum_lfi"][i + 1] + 1e-9 or c["cum_price"][i] > c["cum_price"][i + 1] + 1e-9
           for i in range(len(neg) - 1)):
        fails.append("cumuls non monotones")

    # ── Rapport de force : grille de parts LFI, postures reproductibles depuis les q servis ──
    sp = d["split"]
    if sp["near"] not in sp["shares"] or sp["near"] not in sp["by_share"]:
        fails.append("split.near absent de la grille")
    for k, b in sp["by_share"].items():
        for kk in ("q_lfi", "q_other", "w_lfi", "w_other"):
            if len(b[kk]) != 577 or not all(0.0 <= v <= 1.0 for v in b[kk]):
                fails.append(f"split[{k}].{kk} : longueur ou bornes")
        # w ≤ q : gagner seul suppose de s'être qualifié seul.
        if any(w > q + 1e-9 for w, q in zip(b["w_lfi"], b["q_lfi"])):
            fails.append(f"split[{k}] : w_lfi > q_lfi")
    near = sp["by_share"][sp["near"]]
    for i, r in enumerate(rows):
        if r["pub"] and (r["q_lfi"] != near["q_lfi"][i] or r["q_other"] != near["q_other"][i]):
            fails.append(f"{r['id']} q servi ≠ grille à near")
        if r["posture"] != N.posture(r["group"], r["q_lfi"], r["q_other"]):
            fails.append(f"{r['id']} posture non reproductible")
    # Plus la part LFI monte, plus LFI seule se qualifie (monotone en moyenne).
    means = [sum(sp["by_share"][k]["q_lfi"]) for k in sp["shares"]]
    if any(means[i] > means[i + 1] for i in range(len(means) - 1)):
        fails.append("q_lfi moyen non croissant avec la part LFI")
    if sum(d["postures"].values()) != sum(1 for r in rows if r["posture"]):
        fails.append("comptes de postures incohérents")

    # ── La page ne fige aucun chiffre : ses tuiles et sa méthode sont des gabarits remplis en JS ──
    html = HTML.read_text()
    for anchor in ('id="tiles"', 'id="m-label"', 'id="m-groups"', 'id="m-unc"', 'id="m-posture"', 'id="m-hors"', 'id="share"',
                   'data/negotiation.json', 'data/label_effect_2024.json'):
        if anchor not in html:
            fails.append(f"negotiation.html : {anchor} manquant")

    if fails:
        print("ÉCHEC :\n  - " + "\n  - ".join(fails[:30]))
        sys.exit(1)
    print(f"OK — négociation 2027 : {len(rows)} circos, groupes {d['groups']}, postures {d['postures']}, "
          f"{flips} cas-grille basculés par l'étiquette, effet mesuré sur {e['sample']['n_duels_vs_rn']} duels.")


if __name__ == "__main__":
    main()
