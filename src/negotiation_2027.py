"""Négociation des circonscriptions 2027 pour LFI — l'avantage COMPARATIF par circo.

Question posée par l'outil : dans une gauche unie (une candidature par circonscription, les
partis se répartissant les 577), QUELLES circonscriptions LFI doit-elle demander pour finir
avec le plus de député·es ? Le score de jouabilité de la carte n'y répond pas : il dit où LA
GAUCHE peut gagner, sans savoir quelle étiquette porte la candidature. Or l'étiquette compte
(mesuré : `label_effect_2024`), et c'est l'argument que les partenaires opposent à LFI.

Une négociation est un ÉCHANGE : un partenaire cède une circo si elle lui coûte peu. Chaque
circo reçoit donc DEUX nombres, tous deux des probabilités de siège moyennées sur l'incertitude
nationale ET locale :

    p_lfi   = P(siège gagné par une candidature LFI)
    p_autre = P(siège gagné par une candidature PS / écologiste / PCF)
    prix    = p_autre − p_lfi   (sièges espérés que l'union perd en donnant la circo à LFI)

D'où quatre groupes lisibles par tout le monde autour de la table :
  • ACQUIS      — député·e LFI sortant·e : hors négociation, compté à part.
  • SANS ENJEU  — aucune étiquette de gauche n'a P_MIN de chance : la circo ne vaut rien à
                  personne, elle n'entre pas dans le troc (c'est la majorité des 577).
  • LIBRE       — prix ≤ PRICE_FREE : l'étiquette LFI ne coûte rien de mesurable à l'union.
                  Personne ne peut s'opposer à ce que LFI les réclame toutes.
  • À NÉGOCIER  — prix > PRICE_FREE : l'union perd quelque chose à donner la circo à LFI ;
                  LFI décide combien elle en dispute, le prix étant affiché.
Le classement est par p_lfi décroissant (ce que LFI maximise) ; le prix est la colonne d'à côté,
et le TAUX D'ÉCHANGE (prix / p_lfi = ce que l'union perd par siège que LFI espère) dit quelles
circos à négocier sont les moins chères à réclamer — et lesquelles offrir en échange (taux le
plus haut). Il n'y a pas de groupe « à céder » séparé : l'effet d'étiquette mesuré est un
décalage modéré et quasi uniforme, aucune circo ne voit l'autre étiquette doubler sa chance ;
le taux d'échange ordonne ce continuum au lieu d'y tracer une frontière arbitraire.
La courbe « sièges LFI espérés selon le nombre de circos prises dans cet ordre » répond à
l'autre question de la négociation : COMBIEN en demander (là où la courbe s'aplatit, la circo
marginale ne vaut plus rien).

Modèle : celui de la carte (`winnability_2027.seat_winner`, gauche UNIE, front républicain
standard), avec un seul ajout — le multiplicateur d'étiquette sur les reports centre-droit →
gauche mesuré sur 2024 (décalage additif du taux de report, `cd2l_delta`). Incertitude :
  • nationale : niveaux de bloc tirés autour de l'ancre sondages (`scenarios_2027`) avec les
    erreurs historiques de sondage législatif (RMSE par bloc, `bayesian_polls`, validation LOO
    2002→2022), renormalisés à 100 ; abstention fixée à la référence des scénarios (le couplage
    participation γ y est l'identité, invariant vérifié par `test_parity_2027`) ;
  • locale : bruit gaussien indépendant par circo et par bloc, σ = demi-largeur conforme circo
    à 90 % / 1,645 — le même que la fourchette Monte-Carlo du site.
Un classement à un seul réglage de curseur ne survivrait pas à une réunion : celui-ci moyenne
sur ce que les sondages peuvent se tromper. `p_lfi_local` (incertitude locale seule) est fourni
à côté pour lire ce qu'apporte l'incertitude nationale.

RAPPORT DE FORCE (l'« option extérieure »). Négocier, c'est aussi pouvoir menacer de ne pas
s'entendre. Pour chaque circo et pour une grille de parts nationales LFI-dans-la-gauche (le
curseur de la page), on rejoue le modèle en gauche DIVISÉE (LFI seule contre le reste de la
gauche, motif local de la présidentielle) et on mesure :
    q_lfi   = P(LFI seule se qualifie au 2nd tour)      q_autre = idem pour l'autre gauche
    w_lfi   = P(LFI seule emporte le siège)              w_autre = idem
Là où q_lfi est élevé, la menace d'y aller seule est crédible et la revendication LFI est
incontestable (« exiger ») ; là où la circo vaut cher mais LFI paraît faible, il faut l'obtenir
sur l'argument du prix (l'étiquette ne coûte presque rien), pas sur la force locale
(« obtenir ») ; là où LFI paraît forte mais le siège n'est pas gagnable, la céder ne coûte rien
et a l'air d'un sacrifice — et la menace d'y aller seule pèse sur le partenaire (« monnaie
d'échange »). C'est la partie « impression de force et de faiblesse » de la négociation.

    python3 -u -m src.negotiation_2027        # → report_app/2027/data/negotiation.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src import coverage_2027, deputes_an, label_effect_2024, scenarios_2027, winnability_2027 as W

SERVED = Path("report_app/2027/data")
OUT = SERVED / "negotiation.json"
REPARTITION = Path("data/nuance/nfp_repartition_2024.csv")

# Erreur historique de l'ancre nationale (sondages → résultat, législatives T1) : RMSE par bloc
# des erreurs LOO de `bayesian_polls` (λ=5, scrutins 2002/2007/2012/2017/2022 : G +11,6/−3,4/
# −5,1/+0,7/−4,9 ; CD −9,3/−1,7/+5,0/−9,2/+0,2 ; ED +2,7/+8,5/+2,4/+11,6/+8,0 pts). Larges parce
# que les législatives se sondent mal (2002, 2017) — c'est l'incertitude honnête d'une
# négociation menée des mois avant le scrutin.
NAT_SIGMA = {"G": 6.3, "CD": 6.3, "ED": 7.5}
Z90 = 1.645
DRAWS = 600
SEED = 2027
PRICE_FREE = 0.02      # prix ≤ 2 % d'un siège : « libre »
P_MIN = 0.05           # ni LFI ni l'autre étiquette n'atteint 5 % : « sans enjeu »
LFI_GROUP = "LFI-NFP"
SCENARIO = "union"
# Grille de parts nationales LFI-dans-la-gauche pour le rapport de force (curseur de la page).
SHARES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
LEVERAGE_Q = 0.5   # q_lfi ≥ 0,5 : LFI seule a plus d'une chance sur deux de se qualifier


def _load_served() -> tuple[dict, dict]:
    arr = json.loads((SERVED / "circo.json").read_text())
    summary = json.loads((SERVED / "summary.json").read_text())
    return arr, summary


def _scenario(key: str) -> dict:
    return next(s for s in scenarios_2027.SCENARIOS if s["key"] == key)


def _draw_national(rng: np.random.Generator, means: dict, n: int) -> np.ndarray:
    """n tirages (G, CD, ED) autour de l'ancre, renormalisés à 100 − Autre."""
    base = np.array([means["G"], means["CD"], means["ED"]])
    sig = np.array([NAT_SIGMA["G"], NAT_SIGMA["CD"], NAT_SIGMA["ED"]])
    x = np.maximum(1.0, base + rng.normal(size=(n, 3)) * sig)
    return x / x.sum(axis=1, keepdims=True) * (100.0 - means.get("AU", 0.0))


def simulate(arr: dict, summary: dict, deltas: dict[str, float], right_union: bool = False,
             national: bool = True, draws: int = DRAWS, seed: int = SEED) -> dict[str, np.ndarray]:
    """Probabilité de siège de gauche par circo pour chaque décalage d'étiquette de `deltas`
    ({nom: cd2l_delta}). Mêmes tirages (national + local) pour toutes les étiquettes, afin que
    p_lfi − p_autre ne porte que la différence d'étiquette."""
    scn = _scenario(SCENARIO)
    m = scn["means"]
    rng = np.random.default_rng(seed)
    n = len(arr["id"])
    hw = summary["circo_halfwidth_90"]
    sig = {b: hw[b] / Z90 for b in ("G", "CD", "ED")}
    nat = _draw_national(rng, m, draws) if national else np.tile(
        [m["G"], m["CD"], m["ED"]], (draws, 1))
    dG, dCD, dED = (np.array(arr[k]) for k in ("dG", "dCD", "dED"))
    dAU = np.array(arr.get("dAU", [0.0] * n))
    dAB = np.array(arr["dAB"])
    wins = {k: np.zeros(n) for k in deltas}
    for d in range(draws):
        eG, eCD, eED = (rng.normal(size=n) * sig[b] for b in ("G", "CD", "ED"))
        g = np.clip(nat[d, 0] + dG + eG, 0, 100)
        cd = np.clip(nat[d, 1] + dCD + eCD, 0, 100)
        ed = np.clip(nat[d, 2] + dED + eED, 0, 100)
        au = np.clip(m.get("AU", 0.0) + dAU, 0, 100)
        ab = np.clip(m["AB"] + dAB, 0, 100)
        for i in range(n):
            for k, dl in deltas.items():
                if W.seat_winner(g[i], cd[i], ed[i], ab[i], "union", 1.0, right_union,
                                 au=au[i], cd2l_delta=dl) == "G":
                    wins[k][i] += 1
    return {k: v / draws for k, v in wins.items()}


def simulate_split(arr: dict, summary: dict, shares: list[float], draws: int = DRAWS,
                   seed: int = SEED) -> dict[str, dict[str, np.ndarray]]:
    """Option extérieure par circo et par part nationale LFI : mêmes tirages national + local
    que `simulate` (même graine), gauche divisée en deux pôles avec le motif local `rdev`."""
    from src import radical_spatial
    scn = _scenario(SCENARIO)
    m = scn["means"]
    rng = np.random.default_rng(seed)
    n = len(arr["id"])
    hw = summary["circo_halfwidth_90"]
    sig = {b: hw[b] / Z90 for b in ("G", "CD", "ED")}
    nat = _draw_national(rng, m, draws)
    dG, dCD, dED = (np.array(arr[k]) for k in ("dG", "dCD", "dED"))
    dAU = np.array(arr.get("dAU", [0.0] * n))
    dAB = np.array(arr["dAB"])
    rdev = np.array(arr.get("rdev", [0.0] * n))
    out = {f"{s:.2f}": {k: np.zeros(n) for k in ("q_lfi", "q_other", "w_lfi", "w_other")} for s in shares}
    for d in range(draws):
        eG, eCD, eED = (rng.normal(size=n) * sig[b] for b in ("G", "CD", "ED"))
        g = np.clip(nat[d, 0] + dG + eG, 0, 100)
        cd = np.clip(nat[d, 1] + dCD + eCD, 0, 100)
        ed = np.clip(nat[d, 2] + dED + eED, 0, 100)
        au = np.clip(m.get("AU", 0.0) + dAU, 0, 100)
        ab = np.clip(m["AB"] + dAB, 0, 100)
        for s in shares:
            key = f"{s:.2f}"
            rad = np.clip(s + radical_spatial.RAD_GAIN * rdev, 0.05, 0.95)
            o = out[key]
            for i in range(n):
                qual, pole = W.split_outcome(g[i], cd[i], ed[i], ab[i], rad[i], au=au[i])
                o["q_lfi"][i] += qual[0]; o["q_other"][i] += qual[1]
                if pole == 0: o["w_lfi"][i] += 1
                elif pole == 1: o["w_other"][i] += 1
    return {k: {kk: vv / draws for kk, vv in v.items()} for k, v in out.items()}


def posture(group: str, q_lfi: float | None, q_other: float | None) -> str | None:
    """Posture de négociation = valeur de la circo (groupe) × rapport de force (qui, seul, se
    qualifierait au 2nd tour). Miroir exact de `negPosture` (js/negotiation.js).
      exiger    : circo précieuse, LFI seule se qualifie, pas l'autre gauche → terrain LFI.
      disputer  : précieuse, les deux se qualifieraient seuls → cœur de la négociation.
      obtenir   : précieuse, aucun des deux seul → l'union crée le siège ; argument = le prix.
      difficile : précieuse, seule l'autre gauche se qualifie → terrain du partenaire, à ne
                  demander qu'en échange.
      monnaie   : sans enjeu mais LFI y paraît au moins aussi forte que l'autre gauche → à
                  céder, ça a l'air d'un sacrifice.
      rien      : sans enjeu, LFI plus faible."""
    if group in ("acquis", "non_mesure") or q_lfi is None or q_other is None:
        return None
    lfi, oth = q_lfi >= LEVERAGE_Q, q_other >= LEVERAGE_Q
    if group in ("libre", "a_negocier"):
        return "exiger" if lfi and not oth else "disputer" if lfi and oth else "difficile" if oth else "obtenir"
    return "monnaie" if q_lfi >= q_other else "rien"


def _group(p_lfi: float, p_other: float, lfi_incumbent: bool) -> str:
    if lfi_incumbent:
        return "acquis"
    if max(p_lfi, p_other) < P_MIN:
        return "sans_enjeu"
    return "libre" if round(p_other - p_lfi, 3) <= PRICE_FREE else "a_negocier"


def build() -> dict:
    arr, summary = _load_served()
    eff = label_effect_2024.load()
    deltas = {"lfi": eff["model"]["cd2l_delta_lfi"], "other": eff["model"]["cd2l_delta_other"], "avg": 0.0}
    print(f"  décalages d'étiquette du taux de report (2024) : {deltas}")
    print(f"  Monte-Carlo {DRAWS} tirages × 577 circos × {len(deltas)} étiquettes …")
    p = simulate(arr, summary, deltas)
    p_ru = simulate(arr, summary, {"lfi": deltas["lfi"]}, right_union=True)["lfi"]
    p_loc = simulate(arr, summary, {"lfi": deltas["lfi"]}, national=False)["lfi"]
    print(f"  rapport de force : gauche divisée × {len(SHARES)} parts LFI …")
    split = simulate_split(arr, summary, SHARES)
    default_share = round(float(next(x for x in scenarios_2027.SCENARIOS if x["key"] == "split2")["radical_share"]), 3)
    near = f"{min(SHARES, key=lambda x: abs(x - default_share)):.2f}"

    deputes = deputes_an.load()
    with REPARTITION.open() as f:
        lab24 = {r["circo"]: r["parti"] for r in csv.DictReader(f)}
    try:
        win24 = label_effect_2024.union_winners_2024()
    except Exception as e:  # noqa: BLE001
        print(f"  (vainqueurs 2024 indisponibles : {e})")
        win24 = {}
    cov_val, cov_src = coverage_2027.coverage(arr, summary)
    thr = coverage_2027.threshold(summary)
    scn = _scenario(SCENARIO)
    m = scn["means"]

    rows = []
    for i, cid in enumerate(arr["id"]):
        pub = coverage_2027.flag(cov_val[i], thr) != "faible"
        dep = deputes.get(cid, {})
        lfi_inc = dep.get("groupe") == LFI_GROUP
        # Arrondi AVANT le groupage : le groupe servi doit être reproductible depuis les
        # probabilités servies (à 3 décimales), pas depuis des valeurs internes plus fines.
        pl, po, pa = (round(float(p[k][i]), 3) for k in ("lfi", "other", "avg"))
        g0 = min(100, max(0, m["G"] + arr["dG"][i]))
        cd0 = min(100, max(0, m["CD"] + arr["dCD"][i]))
        ed0 = min(100, max(0, m["ED"] + arr["dED"][i]))
        au0 = min(100, max(0, m.get("AU", 0) + arr.get("dAU", [0] * len(arr["id"]))[i]))
        rows.append({
            "id": cid, "nm": arr["nm"][i], "dept": arr["dept"][i], "ins": arr["ins"][i],
            "pub": pub,
            "p_lfi": pl if pub else None, "p_other": po if pub else None,
            "p_avg": pa if pub else None, "price": round(po - pl, 3) if pub else None,
            "rate": (round(max(0.0, po - pl) / pl, 3) if pl >= P_MIN else None) if pub else None,
            "p_lfi_local": round(float(p_loc[i]), 3) if pub else None,
            "p_lfi_ru": round(float(p_ru[i]), 3) if pub else None,
            "group": _group(pl, po, lfi_inc) if pub else "non_mesure",
            "q_lfi": round(float(split[near]["q_lfi"][i]), 3) if pub else None,
            "q_other": round(float(split[near]["q_other"][i]), 3) if pub else None,
            "rdev": arr.get("rdev", [0] * len(arr["id"]))[i],
            "pred": {"G": round(g0, 1), "CD": round(cd0, 1), "ED": round(ed0, 1), "AU": round(au0, 1)},
            "depute": {"nom": dep.get("nom", ""), "prenom": dep.get("prenom", ""),
                       "groupe": dep.get("groupe", ""), "bloc": dep.get("bloc", "")},
            "lab2024": lab24.get(cid), "union_won_2024": (win24.get(cid) == "UG") if win24 else None,
        })

    for r in rows:
        r["posture"] = posture(r["group"], r["q_lfi"], r["q_other"])
    postures = {k: sum(1 for r in rows if r["posture"] == k)
                for k in ("exiger", "disputer", "obtenir", "difficile", "monnaie", "rien")}
    print(f"  postures (part LFI {near}) : {postures}")
    # Classement : p_lfi décroissant parmi les circos négociables (hors acquis, hors non mesurées).
    neg = [r for r in rows if r["pub"] and r["group"] not in ("acquis", "sans_enjeu")]
    neg.sort(key=lambda r: (-r["p_lfi"], r["price"], r["id"]))
    for k, r in enumerate(neg, 1):
        r["rank"] = k
    acquis = [r for r in rows if r["group"] == "acquis"]
    base_lfi = sum(r["p_lfi"] for r in acquis if r["pub"])
    cum_lfi, cum_price, ids = [], [], []
    s_l = s_p = 0.0
    for r in neg:
        s_l += r["p_lfi"]; s_p += max(0.0, r["price"])
        cum_lfi.append(round(s_l, 2)); cum_price.append(round(s_p, 2)); ids.append(r["id"])
    # Repère : la carte 2024 (les 229 circos FI) rejouée avec le modèle d'étiquette.
    slate24 = [r for r in rows if r["lab2024"] == "FI" and r["pub"]]
    n24 = len([r for r in rows if r["lab2024"] == "FI"])
    exp24 = sum(r["p_lfi"] for r in slate24)
    eff_same_n = base_lfi + (cum_lfi[min(n24 - len(acquis), len(cum_lfi)) - 1] if n24 > len(acquis) else 0)
    groups = {g: sum(1 for r in rows if r["group"] == g) for g in
              ("acquis", "libre", "a_negocier", "sans_enjeu", "non_mesure")}
    print(f"  groupes : {groups}")
    print(f"  sièges LFI espérés — acquis : {base_lfi:.1f} ; carte 2024 ({n24} circos FI) : {exp24:.1f} ; "
          f"répartition efficace à {n24} circos : {eff_same_n:.1f}")
    return {
        "scenario": {"key": SCENARIO, "label": scn["label"], "means": m},
        "params": {"draws": DRAWS, "seed": SEED, "nat_sigma": NAT_SIGMA,
                   "local_sigma": {b: round(summary["circo_halfwidth_90"][b] / Z90, 2) for b in ("G", "CD", "ED")},
                   "price_free": PRICE_FREE, "p_min": P_MIN,
                   "cd2l_delta": deltas, "label_effect": eff["model"],
                   "label_effect_k": {"fi": eff["duels_vs_rn"]["k_fi"], "other": eff["duels_vs_rn"]["k_other"],
                                      "union": eff["duels_vs_rn"]["k_union"]},
                   "label_effect_ci95": eff["duels_vs_rn"]["diff_ci95"],
                   "label_effect_n": eff["sample"]["n_duels_vs_rn"], "lfi_group": LFI_GROUP},
        "groups": groups,
        "postures": postures,
        # Option extérieure par part nationale LFI (clé = part, tableaux alignés sur `rows`).
        "split": {"shares": [f"{s:.2f}" for s in SHARES], "default_share": default_share, "near": near,
                  "leverage_q": LEVERAGE_Q, "publishable": [r["pub"] for r in rows],
                  "by_share": {k: {kk: [round(float(x), 3) for x in vv] for kk, vv in v.items()}
                               for k, v in split.items()}},
        "totals": {"acquis_expected": round(base_lfi, 2), "n_acquis": len(acquis),
                   "slate2024_n": n24, "slate2024_expected": round(exp24, 2),
                   "efficient_same_n_expected": round(eff_same_n, 2),
                   "union_expected_avg": round(sum(r["p_avg"] for r in rows if r["pub"]), 1),
                   "union_expected_best_label": round(sum(max(r["p_lfi"], r["p_other"]) for r in rows if r["pub"]), 1)},
        "curve": {"ids": ids, "cum_lfi": cum_lfi, "cum_price": cum_price},
        "rows": rows,
    }


def main() -> None:
    out = build()
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"  → {OUT} ({OUT.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
