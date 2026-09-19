"""Tests de la négociation 2027 (`negotiation_2027`) et de la mesure du report vers une
candidature LFI (`label_effect_2024`) : invariants du JSON servi + cohérence du modèle.

    python3 -u -m src.test_negotiation_2027
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src import negotiation_2027 as N, winnability_2027 as W

NEG = Path("report_app/2027/data/negotiation.json")
EFF = Path("report_app/2027/data/label_effect_2024.json")
HTML = Path("report_app/2027/negotiation.html")
METHO = Path("report_app/2027/METHODOLOGY.md")


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
        for key in ("p_lfi", "p_lfi_local", "p_lfi_ru", "q_lfi", "q_oth", "p_left"):
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
        if r["posture"] != N.posture(r["group"], r["q_lfi"], r["q_oth"]):
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
    # ── Présidentielle : part brute de Mélenchon = moyenne nationale + écart servi ; parts du reste ──
    nat = d["presidential"]["national"]
    for r in rows:
        if r["mel"] is not None and abs(r["mel"] - (nat["LFI"] + r["rdev"])) > 0.002:
            fails.append(f"{r['id']} part Mélenchon ≠ nationale + écart")
        if (r["mel"] is None) != (r["rdev"] == 0):
            fails.append(f"{r['id']} disponibilité Mélenchon incohérente")
    # Valeur servie au niveau COMMUNAL (circo entièrement incluse dans une grande commune que le
    # scrutin présidentiel ne découpe pas) : elle doit être la MÊME pour toutes les circos de la
    # commune, et la commune doit être nommée — la page l'affiche telle quelle.
    by_com = {}
    for r in rows:
        if r.get("mel_com"):
            by_com.setdefault(r["mel_com"], set()).add(r["mel"])
    for com, vals in by_com.items():
        if len(vals) != 1:
            fails.append(f"{com} : valeur communale non unique ({sorted(vals)})")
    if any(r["mel"] is None and r.get("mel_com") for r in rows):
        fails.append("commune nommée sans valeur de Mélenchon")
    if abs(sum(d["parties_2027"]["shares_rest"].values()) - 1) > 0.002:
        fails.append("parts PS/EELV/PCF du reste ≠ 1")
    # ── Postures : règle 100 % probabilités simulées, et les deux pôles mesurés à l'identique ──
    post = [r for r in rows if r["posture"]]
    if not any(r["posture"] == "monnaie" for r in post):
        fails.append("aucune circo « monnaie d'échange »")
    for r in post:
        # « Rien à jouer » ⇔ le groupe « sans enjeu », exactement. La posture et le classement
        # lisent le MÊME seuil sur la MÊME quantité : ils ne peuvent plus se contredire.
        if (r["posture"] == "rien") != (r["group"] == "sans_enjeu"):
            fails.append(f"{r['id']} « rien à jouer » ⇎ groupe « sans enjeu »")
        # Le garde-fou : aucune posture de DEMANDE ni de CESSION PAYANTE sur un siège que le
        # classement dit imprenable pour LFI — sinon le tableau se contredit d'une colonne à
        # l'autre (« obtenir » sur une circo affichée à 4 % de chance).
        if r["posture"] != "rien" and r["group"] != "en_jeu":
            fails.append(f"{r['id']} posture « {r['posture']} » hors du groupe « en jeu »")
        if r["posture"] == "exiger" and r["q_lfi"] < N.LEVERAGE_Q:
            fails.append(f"{r['id']} exiger sans option extérieure LFI")
        # Monnaie = l'option extérieure est au PARTENAIRE, pas à LFI : c'est toute la définition.
        if r["posture"] == "monnaie" and not (r["q_oth"] >= N.LEVERAGE_Q > r["q_lfi"]):
            fails.append(f"{r['id']} monnaie sans q_oth ≥ seuil > q_lfi")
        if r["posture"] == "obtenir" and (r["q_lfi"] >= N.LEVERAGE_Q or r["q_oth"] >= N.LEVERAGE_Q):
            fails.append(f"{r['id']} obtenir alors qu'un pôle tient le siège seul")
    # La règle ne doit PAS être un franchissement de seuil par deux quantités quasi identiques :
    # les deux options extérieures doivent réellement se séparer (sinon monnaie = bruit d'arrondi).
    sep = sum(1 for r in post if abs(r["q_oth"] - r["q_lfi"]) >= 0.2)
    if sep < 50:
        fails.append(f"options extérieures trop proches pour trancher ({sep} circos séparées de ≥ 20 pts)")
    # p_lfi ≤ p_left PARTOUT, pas « en général » : `seat_winner` est croissante en cd2l_delta et
    # les deux probabilités sortent des mêmes tirages. C'est cet invariant qui rend « p_left <
    # P_MIN » inatteignable et autorise à l'avoir retiré de la règle — s'il tombe, la règle doit
    # être rouverte, pas rafistolée.
    viol = [r["id"] for r in rows if r["pub"] and r["p_lfi"] > r["p_left"] + 1e-9]
    if viol:
        fails.append(f"p_lfi > p_left sur {len(viol)} circos ({viol[:5]}) : la règle des postures "
                     f"supposait le contraire")
    if any(r["pub"] and r["p_left"] < N.P_MIN and r["group"] == "en_jeu" for r in rows):
        fails.append("une circo « en jeu » avec gauche unie < P_MIN : impossible si p_lfi ≤ p_left")

    # ── Intervalle de Wilson : bornes valides, encadrantes, et qui resserrent avec les tirages ──
    n = d["params"]["draws"]
    # Valeurs NUMÉRIQUES calculées à la main depuis la formule de Wilson
    # (c ± h)/(1 + z²/n), c = p̂ + z²/2n, h = z√(p̂(1−p̂)/n + z²/4n²) — et non une propriété que
    # `wilson` garantit par construction. Le test précédent n'affirmait que « lo ≤ p̂ ≤ hi », ce
    # que le `min(p, …)/max(p, …)` de `wilson` rend VRAI QUOI QU'IL ARRIVE : une faute de frappe
    # dans l'algèbre (z*z/(4*n*n) → z*z/(4*n)) passait le test, et le test de miroir avec le JS
    # ne l'aurait pas vue non plus puisqu'il compare deux fois la même formule fausse.
    for v, want in ((0.5, (0.487352, 0.512648)), (1.0, (0.999360, 1.0)),
                    (0.05, (0.044767, 0.055808)), (0.0, (0.0, 0.000640))):
        got = N.wilson(v, 6000)
        if max(abs(a - b) for a, b in zip(got, want)) > 1e-6:
            fails.append(f"Wilson({v}, 6000) = {got}, attendu {want} : l'algèbre a changé")
    if max(abs(a - b) for a, b in zip(N.wilson(1.0, 600), (0.993638, 1.0))) > 1e-6:
        fails.append(f"Wilson(1, 600) = {N.wilson(1.0, 600)}, attendu (0.993638, 1.0)")
    for v in (0.0, 0.001, 0.05, 0.5, 0.95, 0.999, 1.0):
        lo, hi = N.wilson(v, n)
        if not (0.0 <= lo <= v <= hi <= 1.0):
            fails.append(f"Wilson({v}, {n}) = [{lo}, {hi}] n'encadre pas la valeur ou sort de [0,1]")
    # Une borne exactement nulle est le cas le plus fréquent de la colonne : elle doit sortir
    # dans la convention « < 1 », pas dans la décimale (« 0,0–<1 » mêlait les deux, et « 0,0 »
    # est la certitude que la légende promet de ne jamais afficher).
    if N.ci_txt(0.0, n) != "<1":
        fails.append(f"ci_txt(0, {n}) = « {N.ci_txt(0.0, n)} » : une borne nulle doit s'écrire « <1 »")
    if N.wilson(0.5, n)[1] - N.wilson(0.5, n)[0] >= N.wilson(0.5, n // 4)[1] - N.wilson(0.5, n // 4)[0]:
        fails.append("l'intervalle de Wilson ne se resserre pas quand les tirages augmentent")
    if N.wilson(1.0, n)[1] - N.wilson(1.0, n)[0] <= 0:
        fails.append("Wilson dégénère à p = 1 (c'est tout l'intérêt de ne pas prendre l'approx. normale)")
    if "ci_z" not in d["params"]:
        fails.append("ci_z absent des paramètres : la page ne pourrait pas refaire l'intervalle")

    # ── METHODOLOGY.md : chaque chiffre DÉRIVÉ doit se retrouver dans le JSON servi ────────
    # Trois passes de revue adverse d'affilée ont trouvé des chiffres périmés dans ce fichier —
    # jamais dans le code, toujours dans la prose qui le décrit. Rien en CI ne pouvait les voir.
    # Ce garde-fou lit la méthodologie et recalcule ce qu'elle affirme. Il ne vérifie pas le
    # texte, seulement les nombres : ceux que personne ne recompte à la main.
    import re as _re
    md = METHO.read_text()
    pm = d["params"]
    nsh, nsg, ncr, lsg = pm["nat_shift"], pm["nat_sigma"], pm["nat_corr"], pm["local_sigma"]
    mono = [r for r in rows if r.get("posture") == "monnaie"]
    lab = {k: sum(1 for r in mono if r.get("lab2024") == k) for k in ("PS", "PE", "PCF", "FI")}
    strad = [r for r in rows if r["pub"] and r["p_left"] >= N.P_MIN > r["p_lfi"]]
    gaps = sorted(r["p_left"] - r["p_lfi"] for r in rows if r["pub"])
    half = (N.wilson(0.5, pm["draws"])[1] - 0.5) * 100

    def fr(x, n=1):
        # Arrondi au DEMI SUPÉRIEUR et moins typographique : il faut reproduire ce que la page
        # écrit (`toLocaleString` arrondit à l'écart de zéro, là où `%.1f` de Python suit la
        # règle du banquier — 3,65 donne « 3,7 » en JS et « 3,6 » en Python), sinon le garde-fou
        # signalerait un écart là où les deux fichiers sont d'accord.
        from decimal import Decimal, ROUND_HALF_UP
        q = Decimal(str(x)).quantize(Decimal("1." + "0" * n), rounding=ROUND_HALF_UP)
        return str(q).replace(".", ",").replace("-", "−")

    want = [
        (rf"{pm['draws'] // 1000}\s*000 tirages Monte-Carlo", "nombre de tirages"),
        (rf"sur-estimé l'extrême droite de\s*\n?\s*\+{fr(pm['nat_bias_raw']['ED'])} pts", "biais brut ED"),
        (rf"λ = {fr(pm['nat_shrink'], 2)}", "facteur de rétraction"),
        (rf"ED {fr(nsh['ED'])} · C\+D \+{fr(nsh['CD'])} ·\s*\n?\s*G \+{fr(nsh['G'])} pt", "décalage appliqué"),
        (rf"écart-type G {fr(nsg['G'])} · C\+D {fr(nsg['CD'])} · ED {fr(nsg['ED'])} pts", "écarts-types nationaux"),
        (rf"G/C\+D {fr(ncr['GCD'], 2)} · G/ED {fr(ncr['GED'], 2)} · C\+D/ED {fr(ncr['CDED'], 2)}", "corrélations"),
        (rf"G {fr(lsg['G'])} ·\s*\n?\s*C\+D {fr(lsg['CD'])} · ED {fr(lsg['ED'])} pts", "écarts-types locaux"),
        (rf"±{fr(half)} pt au plus large", "demi-largeur de Wilson à p = 0,5"),
        (rf"les {len(strad)} circonscriptions à cheval", "circos de part et d'autre du seuil"),
        (rf"{len(strad)} circonscriptions tombent de part et d'autre", "circos de part et d'autre (2e mention)"),
        (rf"les {len(mono)} circonscriptions « monnaie d'échange »", "compte monnaie"),
        (rf"{lab['PS']} PS, {lab['PE']} Écologistes, {lab['PCF']} PCF, {lab['FI']} LFI", "ventilation monnaie"),
        (rf"{fr(100 * gaps[len(gaps) // 2])} pt en médiane et {fr(100 * gaps[-1])} pts au maximum",
         "écart p_left − p_lfi"),
        (rf"gauche {round(min(r['pred']['G'] for r in strad))}-"
         rf"{round(max(r['pred']['G'] for r in strad))} % contre RN "
         rf"{round(min(r['pred']['ED'] for r in strad))}-"
         rf"{round(max(r['pred']['ED'] for r in strad))} %", "fourchettes des circos à cheval"),
    ]
    for pat, what in want:
        if not _re.search(pat, md):
            fails.append(f"METHODOLOGY.md : {what} ne correspond plus au JSON servi "
                         f"(motif attendu : {pat})")

    # ── Loi d'erreur de l'ancre : les invariants dont dépend TOUT le reste ───────────────
    from src import poll_error_model as PE
    pe = PE.load()
    err_mat = PE.errors(pe)
    # Exprimées en parts des trois blocs : chaque scrutin somme à zéro. Si cet invariant tombe,
    # le tirage sort du plan de somme nulle et le total national n'est plus tenu — ce que la
    # renormalisation masquerait en rabotant silencieusement la dispersion.
    if float(np.abs(err_mat.sum(axis=0)).max()) > 1e-9:
        fails.append(f"les erreurs d'ancre ne somment pas à zéro (max {np.abs(err_mat.sum(axis=0)).max():.2e})")
    if err_mat.shape[1] != len(pe["train"]) + 1:
        fails.append("2024 (hors échantillon) manque à la matrice d'erreur : l'exclure reviendrait "
                     "à retirer la seule observation qui contredit le biais mesuré")
    efit = PE.fit(pe)
    if not 0.0 <= efit["shrink"] <= 1.0:
        fails.append(f"facteur de rétraction hors [0,1] : {efit['shrink']}")
    if abs(float(efit["bias"].sum())) > 1e-9:
        fails.append("le biais rétracté ne somme pas à zéro : un facteur par bloc casserait le plan")
    # La covariance prédictive doit rester SINGULIÈRE dans la direction (1,1,1) : c'est elle qui
    # garantit qu'aucun tirage ne change le total, donc qu'aucune renormalisation n'est requise.
    if float(np.abs(efit["cov"] @ np.ones(3)).max()) > 1e-9:
        fails.append("la covariance prédictive n'est plus singulière dans la direction (1,1,1)")
    # Rétracter ne doit jamais amplifier ni retourner un biais.
    if np.any(np.abs(efit["bias"]) > np.abs(efit["bias_raw"]) + 1e-9) or np.any(efit["bias"] * efit["bias_raw"] < -1e-12):
        fails.append(f"rétraction incohérente : {efit['bias']} contre {efit['bias_raw']}")
    # Un biais nul doit donner une rétraction totale (λ = 0) : la rétraction ne peut pas
    # inventer un biais que les données ne portent pas.
    if PE.fit({"train": [{"pred": {b: 33.0 for b in PE.BLOCS}, "act": {b: 33.0 for b in PE.BLOCS}}] * 4,
               "holdout": {"pred": {"G": 34.0, "CD": 33.0, "ED": 33.0},
                           "act": {"G": 33.0, "CD": 33.5, "ED": 33.5}}}, True)["shrink"] > 0.5:
        fails.append("la rétraction laisse passer un biais qui tient dans son propre bruit")

    # ── Flou de rang : un test sur un ÉCART, donc sur la covariance, pas sur les marginales ──
    # Construit à la main deux circos PARFAITEMENT corrélées (mêmes tirages, même indicatrice) et
    # deux INDÉPENDANTES à la même distance : le critère correct déclare les premières
    # interchangeables et pas les secondes. Le critère marginal — « p̂_j tombe-t-il dans
    # l'intervalle de Wilson de p̂_i » — ne les distingue pas, puisqu'il ignore la covariance.
    nw = 4000
    x1 = np.zeros(nw, dtype=bool); x1[:nw // 2] = True                 # p̂ = 0,500
    x2 = x1.copy(); x2[nw // 2:nw // 2 + 60] = True                    # p̂ = 0,515, même motif
    pair = np.stack([x1, x2], axis=1)
    # Écart 0,015 ; la demi-largeur de Wilson vaut 0,0155 — le critère MARGINAL déclare donc ces
    # deux circos interchangeables. Mais leur bruit est corrélé à 0,97 (mêmes tirages nationaux),
    # donc la SE de l'écart vaut 0,0019 et l'écart fait 7,8 σ : elles ne le sont pas du tout.
    # C'est exactement le cas que la page manquait, et le seul qui sépare les deux critères.
    lo_m, hi_m = N.wilson(0.5, nw)
    if not lo_m <= 0.515 <= hi_m:
        fails.append("le cas témoin ne sépare plus les deux critères : revoir le test, pas le code")
    if N.rank_blur(pair, [0, 1])["max"] != 0:
        fails.append("rank_blur utilise la marginale au lieu de l'écart apparié : deux circos "
                     "corrélées à 0,97 et distantes de 7,8 σ sont déclarées interchangeables")
    same = np.stack([x1, x1], axis=1)
    if N.rank_blur(same, [0, 1])["med"] != 1:
        fails.append("rank_blur : deux circos aux tirages identiques doivent être interchangeables")
    rb = d["params"].get("rank_blur")
    if not rb or not (0 <= rb["med"] <= rb["max"] < d["groups"]["en_jeu"]):
        fails.append(f"rank_blur servi absent ou incohérent ({rb}) ; en jeu = {d['groups']['en_jeu']}")
    pg = d["params"].get("paired_gap")
    # Tout l'argument de la ligne d'entête du CSV : l'écart apparié est plus serré que la somme
    # des deux demi-largeurs. S'il cesse de l'être, la phrase ment et il faut la retirer.
    if not pg or not (0 < pg["paired_pt"] < pg["sum_pt"]):
        fails.append(f"paired_gap servi absent ou non resserré par l'appariement ({pg})")
    # L'écart-type national ANNONCÉ n'est pas celui que la renormalisation délivre : la page doit
    # servir le second, mesuré sur les tirages.
    # Le tirage national doit DÉLIVRER ce qu'il vise : depuis qu'il se fait dans le plan de
    # somme nulle, plus aucune renormalisation ne rabote la dispersion. Un écart ici veut dire
    # que la projection est revenue, et avec elle la sous-dispersion de 14 à 22 %.
    nsd, ns = d["params"].get("nat_sigma_delivered"), d["params"].get("nat_sigma")
    if not nsd or not ns or any(abs(nsd[b] - ns[b]) > 0.05 for b in ("G", "CD", "ED")):
        fails.append(f"le tirage national ne délivre pas sa cible : visé {ns}, délivré {nsd}")
    # Les CORRÉLATIONS aussi, pas seulement les écarts-types : c'est pour elles que le tirage a
    # été réécrit, et c'est le partage entre blocs — pas la dispersion de chacun — qui décide des
    # qualifications au 2nd tour. Un tirage indépendant passerait le test des écarts-types.
    xs = N._draw_national(np.random.default_rng(N.SEED), d["scenario"]["means"], 200_000)
    got = np.corrcoef(xs.T)
    want = d["params"].get("nat_corr") or {}
    for (i, a), (j, b) in (((0, "G"), (1, "CD")), ((0, "G"), (2, "ED")), ((1, "CD"), (2, "ED"))):
        if abs(got[i, j] - want.get(f"{a}{b}", 99)) > 0.01:
            fails.append(f"corrélation {a}/{b} délivrée {got[i, j]:.3f} ≠ servie {want.get(f'{a}{b}')}")
    # Le biais rétracté est tiré vers zéro, jamais amplifié, et garde le signe du biais brut.
    br, bs, lam = d["params"].get("nat_bias_raw"), d["params"].get("nat_bias"), d["params"].get("nat_shrink")
    if not (br and bs and lam is not None) or not 0.0 <= lam <= 1.0:
        fails.append(f"rétraction du biais absente ou hors [0,1] ({lam})")
    elif any(abs(bs[b]) > abs(br[b]) + 1e-9 or bs[b] * br[b] < 0 for b in ("G", "CD", "ED")):
        fails.append(f"le biais rétracté n'est pas entre zéro et le biais brut : {bs} vs {br}")
    # Les erreurs sont exprimées en parts des trois blocs : elles somment à zéro, donc le biais
    # appliqué aussi — sans quoi le tirage ne respecterait plus le total imposé.
    elif abs(sum(bs.values())) > 0.02:
        fails.append(f"le biais appliqué ne somme pas à zéro ({sum(bs.values()):+.3f}) : "
                     f"le tirage sortirait du plan et le total ne serait plus tenu")
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
    # Le curseur recalcule les postures dans la page : les invariants doivent tenir à CHAQUE cran,
    # pas seulement à celui servi. Sans quoi une circo imprenable pour LFI peut basculer en
    # « exiger » à une autre part nationale, sans qu'aucun test ne le voie.
    for kk in sp["shares"]:
        b = sp["by_share"][kk]
        for i, r in enumerate(rows):
            if not r["pub"]:
                continue
            po = N.posture(r["group"], b["q_lfi"][i], b["q_oth"][i])
            if po is not None and po != "rien" and r["group"] != "en_jeu":
                fails.append(f"{r['id']} posture « {po} » hors « en jeu » à la part {kk}")
            if po == "exiger" and b["q_lfi"][i] < N.LEVERAGE_Q:
                fails.append(f"{r['id']} exiger sans option extérieure à la part {kk}")

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
