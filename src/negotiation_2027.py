"""Négociation des circonscriptions 2027 pour LFI — ce que chaque circo vaut pour LFI, et où
LFI est forte ou faible.

Question posée par l'outil : dans une gauche unie (une candidature par circonscription, les
partis se répartissant les 577), QUELLES circonscriptions LFI doit-elle demander pour finir
avec le plus de député·es ? Le score de jouabilité de la carte n'y répond pas : il dit où LA
GAUCHE peut gagner, sans savoir quelle étiquette porte la candidature. Ici tout est vu de LFI,
et de LFI seulement — les autres partis n'entrent dans aucun calcul.

Trois nombres par circo, tous moyennés sur l'incertitude nationale ET locale :

    p_lfi   = P(siège gagné par une candidature LFI dans une gauche unie)   → la VALEUR
    q_lfi   = P(LFI seule se qualifie au 2nd tour si la gauche se divise)   → la FORCE réelle
    rdev    = écart local du vote Mélenchon dans la gauche (présidentielle) → la force APPARENTE

D'où les groupes et les postures :
  • ACQUIS      — député·e LFI sortant·e : hors négociation, compté à part.
  • HORS UNION  — siège tenu par un·e élu·e de gauche HORS de l'union (dissident·e, LIOT, DEM :
                  Falorni, Habib, Serva…) : le bloc de gauche prédit inclut ses voix, qui ne se
                  reporteraient pas sur une candidature d'union → sorti du classement, signalé.
  • SANS ENJEU  — p_lfi < P_MIN : imprenable pour LFI.
  • EN JEU      — le reste : ce que LFI a intérêt à demander, classé par p_lfi décroissant.
  Postures :
    exiger   — en jeu ET LFI seule se qualifierait (q_lfi ≥ LEVERAGE_Q) : LFI n'a pas besoin de
               l'accord ici, la revendication est incontestable, la menace d'y aller seule crédible.
    obtenir  — en jeu mais LFI seule ne se qualifierait pas : la circo vaut cher, il faut l'obtenir
               par la négociation (l'argument : une candidature LFI y gagne le siège).
    monnaie  — sans enjeu mais LFI y PARAÎT forte (vote Mélenchon au-dessus du national) : la
               céder ne coûte rien et a l'air d'un sacrifice.
    rien     — sans enjeu, LFI faible : rien à jouer.
Le classement est par p_lfi décroissant : ce qui se négocie est un NOMBRE de circos, et à nombre
donné chaque circo vaut pour LFI exactement sa chance d'y élire un·e député·e. La courbe « sièges
LFI espérés selon le nombre de circos prises dans cet ordre » dit COMBIEN en demander.

Modèle : celui de la carte (`winnability_2027.seat_winner`, gauche UNIE, front républicain
standard), avec un ajout — le taux de report centre-droit → gauche propre à un·e candidat·e LFI,
mesuré sur 2024 (`label_effect_2024`, décalage additif `cd2l_delta`). Incertitude :
  • nationale : niveaux de bloc tirés autour de l'ancre sondages (`scenarios_2027`) avec les
    erreurs historiques de sondage législatif (RMSE par bloc, `bayesian_polls`, validation LOO
    2002→2022), renormalisés à 100 ; abstention fixée à la référence des scénarios ;
  • locale : bruit gaussien indépendant par circo et par bloc, σ = demi-largeur conforme circo
    à 90 % / 1,645 — le même que la fourchette Monte-Carlo du site.
La force réelle (q_lfi) est servie pour une grille de parts nationales LFI-dans-la-gauche (le
curseur de la page) : le rapport de force bascule avec le niveau national de LFI.

    python3 -u -m src.negotiation_2027        # → report_app/2027/data/negotiation.json
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from src import radical_spatial
from src import coverage_2027, deputes_an, label_effect_2024, scenarios_2027, winnability_2027 as W

SERVED = Path("report_app/2027/data")
OUT = SERVED / "negotiation.json"
HISTORY = SERVED / "comparison_history.json"   # résultats passés par circo (report_comparison_2027)
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
P_MIN = 0.05           # p_lfi < 5 % : « sans enjeu »
LFI_GROUP = "LFI-NFP"
LEFT_GROUPS = {"LFI-NFP", "SOC", "ECOS", "GDR"}
# Nuances 2024 codées à gauche par le modèle mais HORS de l'union (candidature non-UG).
LEFT_NON_UNION_NUANCES = {"DVG", "SOC", "ECO", "RDG", "REG", "DIV", "DSV", "COM", "FI", "VEC"}
SCENARIO = "union"
# Grille de parts nationales LFI-dans-la-gauche pour le rapport de force (curseur de la page).
SHARES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
LEVERAGE_Q = 0.5
# Bornes de la part locale de LFI dans la gauche (part nationale + écart Mélenchon).
RAD_CLIP = (0.05, 0.95)   # q_lfi ≥ 0,5 : LFI seule a plus d'une chance sur deux de se qualifier


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
    out = {f"{s:.2f}": {k: np.zeros(n) for k in ("q_lfi", "w_lfi")} for s in shares}
    for d in range(draws):
        eG, eCD, eED = (rng.normal(size=n) * sig[b] for b in ("G", "CD", "ED"))
        g = np.clip(nat[d, 0] + dG + eG, 0, 100)
        cd = np.clip(nat[d, 1] + dCD + eCD, 0, 100)
        ed = np.clip(nat[d, 2] + dED + eED, 0, 100)
        au = np.clip(m.get("AU", 0.0) + dAU, 0, 100)
        ab = np.clip(m["AB"] + dAB, 0, 100)
        for s in shares:
            key = f"{s:.2f}"
            rad = np.clip(s + radical_spatial.RAD_GAIN * rdev, *RAD_CLIP)
            o = out[key]
            for i in range(n):
                qual, pole = W.split_outcome(g[i], cd[i], ed[i], ab[i], rad[i], au=au[i])
                o["q_lfi"][i] += qual[0]
                if pole == 0: o["w_lfi"][i] += 1
    return {k: {kk: vv / draws for kk, vv in v.items()} for k, v in out.items()}


def posture(group: str, q_lfi: float | None, rdev: float | None) -> str | None:
    """Posture = valeur (groupe) × force. Miroir exact de `negPosture` (js/negotiation.js).
    exiger : en jeu, LFI seule se qualifierait · obtenir : en jeu, pas seule · monnaie : sans
    enjeu mais LFI y paraît forte (Mélenchon au-dessus du national) · rien."""
    if group in ("acquis", "hors_union", "non_mesure") or q_lfi is None:
        return None
    if group == "en_jeu":
        return "exiger" if q_lfi >= LEVERAGE_Q else "obtenir"
    return "monnaie" if (rdev or 0.0) > 0 else "rien"


def _group(p_lfi: float, lfi_incumbent: bool, outside_union: bool = False) -> str:
    if lfi_incumbent:
        return "acquis"
    if outside_union:
        return "hors_union"
    return "sans_enjeu" if p_lfi < P_MIN else "en_jeu"


def _extrapolations(h24: dict | None, nat24: dict, means: dict) -> tuple[float | None, float | None]:
    """Deux extrapolations simples de 2024 pour la GAUCHE, vérifiables par tous :
    « 2024 + évolution nationale » (chaque circo bouge du même nombre de points que la France) et
    « 2024 × évolution nationale » (du même pourcentage). Quatre blocs (G/CD/ED/Autre), plancher 0,
    renormalisation à 100."""
    if not h24:
        return None, None
    blocs = ("G", "CD", "ED", "AU")
    plus = [max(0.0, h24[b] + means.get(b, 0.0) - nat24[b]) for b in blocs]
    mult = [h24[b] * means.get(b, 0.0) / nat24[b] if nat24[b] > 0 else 0.0 for b in blocs]
    sp, sm = sum(plus), sum(mult)
    return (round(100 * plus[0] / sp, 1) if sp else None), (round(100 * mult[0] / sm, 1) if sm else None)


def _history() -> dict:
    """Résultats passés par circo (part de la gauche ; LFI seule en 2017) + nationaux 2024."""
    if not HISTORY.exists():
        return {}
    h = json.loads(HISTORY.read_text())
    return {e["key"]: e for e in h["elections"]}


def build() -> dict:
    arr, summary = _load_served()
    hist = _history()
    eff = label_effect_2024.load()
    deltas = {"lfi": eff["model"]["cd2l_delta_lfi"]}
    print(f"  décalage du taux de report pour une candidature LFI (2024) : {deltas}")
    print(f"  Monte-Carlo {DRAWS} tirages × 577 circos × {len(deltas)} étiquettes …")
    p = simulate(arr, summary, deltas)
    p_ru = simulate(arr, summary, {"lfi": deltas["lfi"]}, right_union=True)["lfi"]
    p_loc = simulate(arr, summary, {"lfi": deltas["lfi"]}, national=False)["lfi"]
    default_share = round(float(next(x for x in scenarios_2027.SCENARIOS if x["key"] == "split2")["radical_share"]), 3)
    # La part sondages elle-même est dans la grille (et sert de réglage par défaut) : la force
    # réelle affichée par défaut est calculée à la part mesurée, pas au cran de grille voisin.
    shares = sorted({f"{s:.2f}": s for s in list(SHARES) + [default_share]}.values())
    near = f"{default_share:.2f}"
    print(f"  rapport de force : gauche divisée × {len(shares)} parts LFI …")
    split = simulate_split(arr, summary, shares)

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
        # Siège pris en 2024 par une candidature codée à gauche mais hors union, et dont le
        # titulaire ne siège pas dans un groupe de gauche : la « gauche » prédite ici n'est pas
        # celle de l'union.
        outside = (win24.get(cid) in LEFT_NON_UNION_NUANCES and dep.get("groupe") not in LEFT_GROUPS) if win24 else False
        # Arrondi AVANT le groupage : le groupe servi doit être reproductible depuis les
        # probabilités servies (à 3 décimales), pas depuis des valeurs internes plus fines.
        pl = round(float(p["lfi"][i]), 3)
        g0 = min(100, max(0, m["G"] + arr["dG"][i]))
        cd0 = min(100, max(0, m["CD"] + arr["dCD"][i]))
        ed0 = min(100, max(0, m["ED"] + arr["dED"][i]))
        au0 = min(100, max(0, m.get("AU", 0) + arr.get("dAU", [0] * len(arr["id"]))[i]))
        h17 = hist.get("2017", {}).get("rows", {}).get(cid)
        h22 = hist.get("2022", {}).get("rows", {}).get(cid)
        h24 = hist.get("2024", {}).get("rows", {}).get(cid)
        ext_plus, ext_mult = _extrapolations(h24, hist["2024"]["national"], m) if h24 and "2024" in hist else (None, None)
        rows.append({
            "id": cid, "nm": arr["nm"][i], "dept": arr["dept"][i], "ins": arr["ins"][i],
            # Arguments vérifiables par tous : résultats passés (gauche ; LFI seule en 2017, seul
            # scrutin où LFI concourait sous sa nuance) et extrapolations simples de 2024.
            "h2017_G": round(h17["G"], 1) if h17 else None, "h2017_LFI": round(h17["LFI"], 1) if h17 and "LFI" in h17 else None,
            "h2022_G": round(h22["G"], 1) if h22 else None, "h2024_G": round(h24["G"], 1) if h24 else None,
            # Ventilation 2024 par nuance (UG = candidature NFP ; DVG/EXG/ECO… = gauche hors NFP).
            "h2024_parts": h24.get("parts") if h24 else None,
            "ext_plus_G": ext_plus if pub else None, "ext_mult_G": ext_mult if pub else None,
            "pub": pub,
            "p_lfi": pl if pub else None,
            "p_lfi_local": round(float(p_loc[i]), 3) if pub else None,
            "p_lfi_ru": round(float(p_ru[i]), 3) if pub else None,
            "group": _group(pl, lfi_inc, outside) if pub else "non_mesure",
            "q_lfi": round(float(split[near]["q_lfi"][i]), 3) if pub else None,
            "rdev": arr.get("rdev", [0] * len(arr["id"]))[i],
            "pred": {"G": round(g0, 1), "CD": round(cd0, 1), "ED": round(ed0, 1), "AU": round(au0, 1)},
            "depute": {"nom": dep.get("nom", ""), "prenom": dep.get("prenom", ""),
                       "groupe": dep.get("groupe", ""), "bloc": dep.get("bloc", "")},
            "lab2024": lab24.get(cid), "union_won_2024": (win24.get(cid) == "UG") if win24 else None,
        })

    for r in rows:
        r["posture"] = posture(r["group"], r["q_lfi"], r["rdev"])
    postures = {k: sum(1 for r in rows if r["posture"] == k) for k in ("exiger", "obtenir", "monnaie", "rien")}
    print(f"  postures (part LFI {near}) : {postures}")
    # Classement : p_lfi décroissant parmi les circos négociables (hors acquis, hors non mesurées).
    neg = [r for r in rows if r["group"] == "en_jeu"]
    neg.sort(key=lambda r: (-r["p_lfi"], r["id"]))
    for k, r in enumerate(neg, 1):
        r["rank"] = k
    acquis = [r for r in rows if r["group"] == "acquis"]
    base_lfi = sum(r["p_lfi"] for r in acquis if r["pub"])
    cum_lfi, ids = [], []
    s_l = 0.0
    for r in neg:
        s_l += r["p_lfi"]
        cum_lfi.append(round(s_l, 2)); ids.append(r["id"])
    # Repère : la carte 2024 (les 229 circos FI) rejouée avec le modèle d'étiquette.
    slate24 = [r for r in rows if r["lab2024"] == "FI" and r["pub"]]
    n24 = len([r for r in rows if r["lab2024"] == "FI"])
    exp24 = sum(r["p_lfi"] for r in slate24)
    eff_same_n = base_lfi + (cum_lfi[min(n24 - len(acquis), len(cum_lfi)) - 1] if n24 > len(acquis) else 0)
    groups = {g: sum(1 for r in rows if r["group"] == g) for g in
              ("acquis", "en_jeu", "sans_enjeu", "hors_union", "non_mesure")}
    print(f"  groupes : {groups}")
    print(f"  sièges LFI espérés — acquis : {base_lfi:.1f} ; carte 2024 ({n24} circos FI) : {exp24:.1f} ; "
          f"répartition efficace à {n24} circos : {eff_same_n:.1f}")
    return {
        "scenario": {"key": SCENARIO, "label": scn["label"], "means": m},
        "history": {k: {"label": e["label"], "source": e["source"], "national_G": round(e["national"]["G"], 1),
                        **({"national_LFI": round(e["national"]["LFI"], 1)} if "LFI" in e["national"] else {})}
                    for k, e in hist.items()},
        "params": {"draws": DRAWS, "seed": SEED, "nat_sigma": NAT_SIGMA,
                   "local_sigma": {b: round(summary["circo_halfwidth_90"][b] / Z90, 2) for b in ("G", "CD", "ED")},
                   "p_min": P_MIN,
                   "cd2l_delta": deltas, "label_effect": eff["model"],
                   "label_effect_k": {"fi": eff["duels_vs_rn"]["k_fi"], "union": eff["duels_vs_rn"]["k_union"]},
                   "label_effect_n_fi": eff["sample"]["n_duels_fi"],
                   "label_effect_ci95": eff["duels_vs_rn"]["diff_ci95"],
                   "label_effect_n": eff["sample"]["n_duels_vs_rn"], "lfi_group": LFI_GROUP,
                   # Part locale de LFI dans la gauche = part nationale + RAD_GAIN × écart Mélenchon,
                   # bornée : la même règle que la force réelle, réutilisée par la page pour
                   # ventiler le « calcul simple » 2027 entre LFI et le reste de la gauche.
                   "rad_gain": radical_spatial.RAD_GAIN, "rad_clip": list(RAD_CLIP)},
        "groups": groups,
        "postures": postures,
        # Option extérieure par part nationale LFI (clé = part, tableaux alignés sur `rows`).
        "split": {"shares": [f"{s:.2f}" for s in shares], "default_share": default_share, "near": near,
                  "leverage_q": LEVERAGE_Q, "publishable": [r["pub"] for r in rows],
                  "by_share": {k: {kk: [round(float(x), 3) for x in vv] for kk, vv in v.items()}
                               for k, v in split.items()}},
        "totals": {"acquis_expected": round(base_lfi, 2), "n_acquis": len(acquis),
                   "slate2024_n": n24, "slate2024_expected": round(exp24, 2),
                   "efficient_same_n_expected": round(eff_same_n, 2)},
        "curve": {"ids": ids, "cum_lfi": cum_lfi},
        "rows": rows,
    }


def main() -> None:
    out = build()
    OUT.write_text(json.dumps(out, ensure_ascii=False, separators=(",", ":")))
    print(f"  → {OUT} ({OUT.stat().st_size // 1024} Ko)")


if __name__ == "__main__":
    main()
