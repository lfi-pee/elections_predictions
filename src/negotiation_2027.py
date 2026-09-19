"""Négociation des circonscriptions 2027 pour LFI — ce que chaque circo vaut pour LFI, et où
LFI est forte ou faible.

Question posée par l'outil : dans une gauche unie (une candidature par circonscription, les
partis se répartissant les 577), QUELLES circonscriptions LFI doit-elle demander pour finir
avec le plus de député·es ? Le score de jouabilité de la carte n'y répond pas : il dit où LA
GAUCHE peut gagner, sans savoir quelle étiquette porte la candidature. Ici tout est vu de LFI,
et de LFI seulement — les autres partis n'entrent dans aucun calcul.

Quatre probabilités par circo (plus un écart servi tel quel), toutes moyennées sur
l'incertitude nationale ET locale :

    p_lfi   = P(siège gagné par une candidature LFI dans une gauche unie)   → la VALEUR pour LFI
    p_left  = P(siège gagné par une candidature d'union MOYENNE)            → la VALEUR du siège
    q_lfi   = P(LFI seule se qualifie au 2nd tour si la gauche se divise)   → option extérieure LFI
    q_oth   = P(le reste de la gauche seul se qualifie, même division)      → option extérieure
                                                                              du PARTENAIRE
    rdev    = écart local du vote Mélenchon dans la gauche (présidentielle) → servi à DROITE
              (argument visible par tous), jamais utilisé dans la posture

Les quatre premières sont des probabilités du MÊME Monte-Carlo (mêmes tirages) : les colonnes
de gauche, postures comprises, ne reposent que là-dessus.

D'où les groupes et les postures :
  • ACQUIS      — député·e LFI sortant·e : hors négociation, compté à part.
  • HORS UNION  — siège tenu par un·e élu·e de gauche HORS de l'union (dissident·e, LIOT, DEM :
                  Falorni, Habib, Serva…) : le bloc de gauche prédit inclut ses voix, qui ne se
                  reporteraient pas sur une candidature d'union → sorti du classement, signalé.
  • SANS ENJEU  — p_lfi < P_MIN : imprenable pour LFI.
  • EN JEU      — le reste : ce que LFI a intérêt à demander, classé par p_lfi décroissant.
  Postures (règle complète dans `posture` ci-dessous) : la chance de LFI (p_lfi, lue via le
  groupe) dit s'il y a quelque chose à jouer POUR LFI ; les deux options extérieures (q_lfi,
  q_oth), mesurées à l'identique sur les deux pôles, disent qui peut se passer de l'accord.
  p_left n'entre pas dans la règle : elle est servie et affichée, rien de plus.
    exiger   — LFI peut gagner le siège ET le tient sans l'accord (q_lfi ≥ LEVERAGE_Q).
    obtenir  — LFI peut le gagner, aucun pôle ne le tient seul : il se gagne à la table.
    monnaie  — LFI peut le gagner mais l'option extérieure est au PARTENAIRE (q_oth ≥
               LEVERAGE_Q > q_lfi) : LFI devra le céder, autant l'échanger.
    rien     — LFI ne gagne pas le siège (groupe « sans enjeu »).
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
from src import coverage_2027, deputes_an, label_effect_2024, poll_error_model, scenarios_2027, winnability_2027 as W

SERVED = Path("report_app/2027/data")
OUT = SERVED / "negotiation.json"
HISTORY = SERVED / "comparison_history.json"   # résultats passés par circo (report_comparison_2027)
REPARTITION = Path("data/nuance/nfp_repartition_2024.csv")

# Erreur de l'ancre nationale : biais RÉTRACTÉ et covariance, mesurés sur les législatives
# T1 2002→2024 par `poll_error_model` (qui les extrait de `bayesian_polls` et les met en cache).
# Plus de constante écrite à la main ici : les six erreurs étaient recopiées dans ce fichier et
# avaient déjà dérivé de ce que le code produit (CD 2007 : −1,7 recopié contre −0,36 calculé).
#
# Deux choses que l'ancienne version ne faisait pas, et qui changent les chiffres publiés :
#   • le BIAIS est corrigé. Les sondages sur-prédisent l'extrême droite aux législatives — de
#     +4,1 pts en moyenne sur six scrutins. Tirer autour de l'ancre BRUTE plaçait l'extrême
#     droite trop haut à chaque tirage, et aucune largeur d'intervalle ne peut rattraper une
#     erreur de centre. La correction est rétractée vers zéro (λ ≈ 0,36) : six observations ne
#     suffisent pas à parier sur le biais brut, et 2024 est parti dans l'autre sens.
#   • la COVARIANCE est celle des données, et le tirage se fait dans le plan de somme nulle
#     (les trois blocs se partagent un total imposé, l'abstention et « Autre » étant fixes).
#     L'ancienne version tirait les trois blocs INDÉPENDAMMENT puis renormalisait : cette
#     projection rabotait 14 à 22 % de la dispersion visée et fabriquait des corrélations à peu
#     près égales entre toutes les paires, là où gauche et centre-droit sont de loin les plus
#     anticorrélés. Ici rien n'est raboté, parce qu'il n'y a plus rien à projeter.
Z90 = 1.645
# Tirages Monte-Carlo. La page AFFICHE l'erreur de simulation à côté de chaque probabilité
# (intervalle de Wilson à 95 %) : ce nombre est donc lu par le lecteur, pas seulement subi.
# À 600 tirages elle valait ±4,0 pts à p = 0,5 — plus large que l'écart entre deux postures ;
# à 6 000 elle vaut ±1,3 pt. Le build passe de 50 s à ~8 min, ce qui reste un coût de build.
DRAWS = 6000
SEED = 2027
CI_Z = 1.96            # bornes à 95 % sur le bruit de simulation (Wilson)
P_MIN = 0.05           # p_lfi < 5 % : « sans enjeu »
LFI_GROUP = "LFI-NFP"
LEFT_GROUPS = {"LFI-NFP", "SOC", "ECOS", "GDR"}
# Nuances 2024 codées à gauche par le modèle mais HORS de l'union (candidature non-UG).
LEFT_NON_UNION_NUANCES = {"DVG", "SOC", "ECO", "RDG", "REG", "DIV", "DSV", "COM", "FI", "VEC"}
SCENARIO = "union"
# Grille de parts nationales LFI-dans-la-gauche pour le rapport de force (curseur de la page).
SHARES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
# q ≥ 0,5 : le pôle a plus d'une chance sur deux de se qualifier seul — il tient le siège
# sans l'accord. Seuil des postures, appliqué à l'identique aux deux pôles.
LEVERAGE_Q = 0.5
# Bornes de la part locale de LFI dans la gauche (part nationale + écart Mélenchon).
RAD_CLIP = (0.05, 0.95)


def _load_served() -> tuple[dict, dict]:
    arr = json.loads((SERVED / "circo.json").read_text())
    summary = json.loads((SERVED / "summary.json").read_text())
    return arr, summary


def _scenario(key: str) -> dict:
    return next(s for s in scenarios_2027.SCENARIOS if s["key"] == key)


_FIT: dict | None = None


def error_fit() -> dict:
    """Biais rétracté + covariance de l'erreur d'ancre (mis en cache : lecture d'un JSON)."""
    global _FIT
    if _FIT is None:
        _FIT = poll_error_model.fit()
    return _FIT


def _sampler(f: dict) -> np.ndarray:
    """Racine carrée de la covariance. `eigh` et non Cholesky : la covariance est SINGULIÈRE
    par construction (direction (1,1,1), les erreurs somment à zéro), ce que Cholesky refuse.
    Le plancher à zéro n'absorbe que le −1e-16 numérique de la valeur propre nulle."""
    w, v = np.linalg.eigh(f["cov"])
    return v @ np.diag(np.sqrt(np.maximum(w, 0.0)))


def _draw_national(rng: np.random.Generator, means: dict, n: int) -> np.ndarray:
    """n tirages (G, CD, ED) : l'ancre MOINS une erreur tirée dans sa loi mesurée.

    `err = prédit − réel`, donc on soustrait. L'ancre somme déjà à 100 − Autre et l'erreur somme
    à zéro : chaque tirage respecte le total EXACTEMENT, sans renormalisation. Celle qui reste ne
    sert qu'au plancher (un bloc négatif n'a pas de sens). Le bloc le plus exposé est le
    centre-droit, à 4,1 écarts-types du plancher : aucun déclenchement sur les 6 000 tirages du
    build, 6 sur les 200 000 du contrôle `delivered_sigma`. Quand il se déclenche, la
    renormalisation cesse d'être neutre et redistribue sur les deux autres blocs — c'est
    précisément pourquoi le plancher est à 0,5 et non à 1,0 : assez bas pour ne jamais mordre là
    où on publie, assez haut pour qu'une part reste une part."""
    base = np.array([means["G"], means["CD"], means["ED"]])
    f = error_fit()
    e = f["bias"] + rng.standard_normal((n, 3)) @ _sampler(f).T
    x = np.maximum(0.5, base - e)
    return x / x.sum(axis=1, keepdims=True) * (100.0 - means.get("AU", 0.0))


def wilson(p: float, n: int, z: float = CI_Z) -> tuple[float, float]:
    """Intervalle de Wilson à 95 % sur une probabilité estimée par `n` tirages Monte-Carlo.

    C'est l'erreur de SIMULATION, et elle seule : de combien le chiffre bougerait si on relançait
    le Monte-Carlo avec une autre graine. L'incertitude de l'ÉLECTION (sondages + erreur locale)
    est déjà INTÉGRÉE dans le point estimé — la remettre autour serait la compter deux fois.

    Wilson plutôt que l'approximation normale parce que les valeurs affichées touchent les bords :
    à p̂ = 1, `p̂ ± z√(p̂(1−p̂)/n)` donne ±0, ce qui est faux — et 22 circos sont à p̂ = 1 dans les
    données servies. Wilson y donne [0,9936 ; 1] à 600 tirages, [0,99936 ; 1] à 6 000 : la borne
    HAUTE vaut exactement 1 (c + h = (1 + z²/n)/d = 1 quand p̂ = 1), c'est la convention
    d'affichage de `ci_txt` — pas Wilson — qui évite d'écrire « 100 ». Ce qui sauve la borne
    basse de l'absurdité, c'est Wilson. Miroir exact de `wilsonCI` (js/negotiation.js).
    """
    if n <= 0:
        return 0.0, 1.0
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    # Bornes serrées sur le point estimé : en arithmétique exacte l'intervalle de Wilson le
    # contient toujours, mais aux bords (p = 0, p = 1) la soustraction laisse un résidu flottant
    # de l'ordre de 1e-20 du mauvais côté. Un intervalle qui n'encadre pas le chiffre qu'il annote
    # est précisément le défaut qu'on corrige ailleurs sur cette page : on l'interdit ici par
    # construction plutôt que par tolérance.
    return min(p, max(0.0, c - h)), max(p, min(1.0, c + h))


def ci_txt(p: float, n: int, z: float = CI_Z) -> str:
    """Bornes telles que la page les écrit sous le chiffre. Même convention que le chiffre
    lui-même : jamais de certitude affichée (« >99 », « <1 »), une décimale sous 10 % pour que le
    seuil des 5 % reste lisible dans l'intervalle aussi, et bornes fondues en un seul jeton quand
    elles s'écrivent pareil (l'intervalle est alors plus étroit que ce que l'affichage distingue).
    Miroir exact de `ciTxt` (js/negotiation.js) ; le test de page compare les deux."""
    lo, hi = wilson(p, n, z)

    def f(v: float) -> str:
        if v >= 0.995:
            return ">99"
        # Pas de « v > 0 » ici : une borne exactement nulle est le cas le plus fréquent de la
        # colonne (363 des 571 q_lfi servis valent 0). L'exclure la renvoyait dans la branche
        # décimale, qui écrivait « 0,0–<1 » : deux conventions dans un même jeton, dont l'une
        # est la certitude que ce docstring promet de ne jamais afficher.
        if v < 0.005:
            return "<1"
        return f"{v * 100:.1f}".replace(".", ",") if p < 0.1 else str(round(v * 100))

    a, b = f(lo), f(hi)
    return a if a == b else f"{a}–{b}"


def rank_blur(w: np.ndarray, idx: list[int], z: float = CI_Z) -> dict[str, int]:
    """« Avec combien de voisines une circo est-elle interchangeable, au bruit de simulation près ? »

    Deux circos sont interchangeables si l'ÉCART de leurs chances n'est pas distinguable de zéro :
    |p̂_i − p̂_j| ≤ z·SE(p̂_i − p̂_j). C'est un test sur une DIFFÉRENCE, et la différence ne se lit pas
    dans les deux intervalles marginaux : les circos partagent les tirages nationaux, donc
    Var(p̂_i − p̂_j) = [Var_i + Var_j − 2·Cov_ij]/n, et seule la matrice par tirage donne Cov_ij.

    Compter « combien de p̂_j tombent dans l'intervalle de Wilson de p̂_i » — ce que faisait la page
    — répond à une autre question et se trompe de sens : la corrélation du bruit entre circos vaut
    ~0,34, en dessous du 0,5 qui rendrait les deux critères équivalents, donc SE(écart) est PLUS
    grand que la demi-largeur marginale et le flou de rang était SOUS-estimé.
    """
    x = w[:, idx].astype(np.float64)
    nd = x.shape[0]
    pi = x.mean(axis=0)
    # ddof=0 assumé et cohérent avec `paired_gap` à 6 000 tirages (écart < 0,01 %) :
    # c'est la covariance de l'ÉCHANTILLON de tirages, pas une estimation de population.
    cov = (x.T @ x) / nd - np.outer(pi, pi)
    var = np.diag(cov)
    se = np.sqrt(np.maximum(var[:, None] + var[None, :] - 2 * cov, 0.0) / nd)
    # `≤` et non `<` : deux circos toutes deux à p̂ = 1 ont un écart nul ET une SE nulle — elles
    # sont parfaitement interchangeables, un test strict les déclarerait distinctes.
    inside = np.abs(pi[:, None] - pi[None, :]) <= z * se + 1e-12
    np.fill_diagonal(inside, False)
    c = inside.sum(axis=1)
    # `round` et non `int` : avec un nombre PAIR de circos « en jeu » la médiane tombe sur
    # un demi, et `int` la tronquerait vers le bas — la page publie ce chiffre comme un fait.
    return {"med": int(round(float(np.median(c)))), "max": int(c.max())}


def paired_gap(wa: np.ndarray, wb: np.ndarray, idx: list[int], z: float = CI_Z) -> dict[str, float]:
    """Sur le plus grand écart d'étiquette du jeu : demi-largeur 95 % de l'écart p̂_b − p̂_a mesurée
    APPARIÉE (tirage par tirage) contre la somme des deux demi-largeurs marginales, que le lecteur
    du CSV combinerait faute de mieux. Sert le chiffre que la ligne d'entête du CSV annonce — il
    était écrit en dur et se serait tu si `DRAWS` changeait."""
    # `idx` : les circos PUBLIABLES seulement. Le chiffre part dans l'entête d'un CSV public ;
    # le maximum ne doit pas pouvoir être porté par une circo dont on a justement décidé de ne
    # rien publier, même s'il ne s'agit que de la largeur de son intervalle.
    a, b = wa[:, idx].astype(np.float64), wb[:, idx].astype(np.float64)
    nd = a.shape[0]
    pa, pb = a.mean(axis=0), b.mean(axis=0)
    i = int(np.argmax(np.abs(pb - pa)))
    paired = z * float((b[:, i] - a[:, i]).std(ddof=0)) / np.sqrt(nd)
    (la, ha), (lb, hb) = wilson(float(pa[i]), nd, z), wilson(float(pb[i]), nd, z)
    return {"paired_pt": round(paired * 100, 2),
            "sum_pt": round(((ha - la) + (hb - lb)) / 2 * 100, 2)}


def delivered_sigma(means: dict, draws: int = 200_000, seed: int = SEED) -> dict[str, float]:
    """Écart-type RÉELLEMENT délivré par `_draw_national`, bloc par bloc — mesuré, pas déduit.

    Ce n'est plus un correctif mais un CONTRÔLE : depuis que le tirage se fait dans le plan de
    somme nulle, plus rien ne rabote la dispersion et le délivré doit coïncider avec la cible
    (`error_fit()["sd"]`) au bruit d'échantillonnage près. C'est un test, et le test de données
    l'exige. Tant que le tirage passait par une renormalisation, l'écart était de 14 à 22 %.
    """
    x = _draw_national(np.random.default_rng(seed), means, draws)
    return {b: round(float(x[:, i].std(ddof=1)), 2) for i, b in enumerate(("G", "CD", "ED"))}


def simulate(arr: dict, summary: dict, deltas: dict[str, float], right_union: bool = False,
             national: bool = True, draws: int = DRAWS, seed: int = SEED,
             matrices: bool = False) -> dict[str, np.ndarray]:
    """Probabilité de siège de gauche par circo pour chaque décalage d'étiquette de `deltas`
    ({nom: cd2l_delta}). Mêmes tirages (national + local) pour toutes les étiquettes, afin que
    p_lfi − p_autre ne porte que la différence d'étiquette."""
    scn = _scenario(SCENARIO)
    m = scn["means"]
    rng = np.random.default_rng(seed)
    n = len(arr["id"])
    hw = summary["circo_halfwidth_90"]
    sig = {b: hw[b] / Z90 for b in ("G", "CD", "ED")}
    # Le tirage national est TOUJOURS consommé, même quand on ne s'en sert pas : sans cela le
    # flux du générateur se décale et `p_lfi_local` ne partagerait pas les erreurs locales de
    # `p_lfi`, alors que la page et le CSV les présentent comme une paire (l'écart porterait
    # ~1,3 pt de bruit de simulation qu'une comparaison appariée annule).
    drawn = _draw_national(rng, m, draws)
    nat = drawn if national else np.tile([m["G"], m["CD"], m["ED"]], (draws, 1))
    dG, dCD, dED = (np.array(arr[k]) for k in ("dG", "dCD", "dED"))
    dAU = np.array(arr.get("dAU", [0.0] * n))
    dAB = np.array(arr["dAB"])
    # `matrices` : garder l'indicatrice de victoire TIRAGE PAR TIRAGE, et pas seulement sa
    # moyenne. Indispensable à toute affirmation portant sur un ÉCART ou un RANG : deux circos
    # partagent les tirages nationaux, donc la variance de p̂_i − p̂_j n'est pas déductible des
    # deux marginales — il faut la covariance, qui n'existe qu'ici. 6 000 × 577 booléens = 3,5 Mo.
    wins = {k: (np.zeros((draws, n), dtype=bool) if matrices else np.zeros(n)) for k in deltas}
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
                    if matrices:
                        wins[k][d, i] = True
                    else:
                        wins[k][i] += 1
    return wins if matrices else {k: v / draws for k, v in wins.items()}


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
    out = {f"{s:.2f}": {k: np.zeros(n) for k in ("q_lfi", "q_oth", "w_lfi")} for s in shares}
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
                o["q_oth"][i] += qual[1]
                if pole == 0: o["w_lfi"][i] += 1
    return {k: {kk: vv / draws for kk, vv in v.items()} for k, v in out.items()}


def posture(group: str, q_lfi: float | None, q_oth: float | None) -> str | None:
    """Posture = TROIS PROBABILITÉS SIMULÉES par le même Monte-Carlo (mêmes tirages), rien
    d'autre — jamais un chiffre de la partie droite du tableau, et aucun seuil nouveau : les
    deux seuils sont ceux déjà posés, P_MIN et LEVERAGE_Q. Miroir exact de `negPosture`
    (js/negotiation.js).

      p_lfi  — LFI gagne-t-elle le siège en portant la candidature unique ? = y a-t-il un enjeu
               POUR LFI. Lu à travers `group` : « sans enjeu » ⇔ p_lfi < P_MIN.
      q_lfi  — si la gauche se divise, LFI seule atteint-elle le 2nd tour ?  = l'option extérieure DE LFI
      q_oth  — si la gauche se divise, le reste de la gauche seul l'atteint-il ? = celle DU PARTENAIRE

    `p_left` (la gauche unie gagne le siège, candidature d'union moyenne) n'entre PAS dans la
    règle et ne figure plus dans la signature : c'est une colonne affichée — la valeur du siège
    pour l'union, indépendamment de qui le porte — et rien de plus. Elle y entrait avant le
    garde-fou ; la tester serait aujourd'hui du code mort, `seat_winner` étant croissante en
    `cd2l_delta` et les deux probabilités sortant des MÊMES tirages, donc p_lfi ≤ p_left partout
    (invariant testé). p_left < P_MIN impliquerait p_lfi < P_MIN, c'est-à-dire « sans enjeu »,
    déjà traité. La garder en paramètre revenait à la LIRE dans le garde-fou (`p_left is None`)
    tout en affirmant en commentaire qu'on ne la lit pas — et cette lecture était de toute façon
    redondante : `p_left is None` ⟺ circo non publiable ⟺ `q_lfi is None`, déjà filtré.

    Les deux options extérieures sont mesurées à l'identique sur les deux pôles : la règle est
    symétrique, c'est elle qui dit qui peut se passer de l'accord.

      rien à jouer     LFI ne gagne pas ce siège (groupe « sans enjeu », p_lfi < P_MIN) :
                       rien à demander, rien à céder. On ne revendique pas un siège qu'on ne
                       gagne pas — et c'est le classement lui-même qui le dit.
      exiger           la gauche gagne le siège ET q_lfi ≥ LEVERAGE_Q — LFI tient le siège sans
                       l'accord : la revendication ne se refuse pas.
      monnaie d'échange  la gauche gagne le siège, q_lfi < LEVERAGE_Q ≤ q_oth — l'option
                       extérieure est du côté du partenaire, pas de LFI : LFI ne peut pas exiger
                       ce siège et devra le céder ; autant le céder contre autre chose. C'est un
                       vrai siège (la gauche unie le gagne), donc la concession a un prix.
      obtenir          la gauche gagne le siège et AUCUN pôle n'a d'option extérieure
                       (q_lfi < LEVERAGE_Q, q_oth < LEVERAGE_Q) — personne ne peut se passer de
                       l'accord : le siège se gagne à la table.
    """
    if group in ("acquis", "hors_union", "non_mesure") or q_lfi is None or q_oth is None:
        return None
    # Aucune posture de DEMANDE sur un siège que LFI ne gagne pas : `group` porte déjà ce verdict
    # (« sans enjeu » = p_lfi < P_MIN). p_lfi ne sépare jamais exiger/obtenir/monnaie — il ne fait
    # qu'interdire la revendication là où elle n'a pas d'objet.
    if group == "sans_enjeu":
        return "rien"
    if q_lfi >= LEVERAGE_Q:
        return "exiger"
    return "monnaie" if q_oth >= LEVERAGE_Q else "obtenir"


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


PARTIES_CSV = Path("data/polls/legislatives/legislatives_2027_partis_gauche.csv")


def _party_shares_rest() -> dict:
    """Part de chaque parti (PS, EELV, PCF) dans le RESTE de la gauche (hors LFI), moyenne simple
    des enquêtes qui les publient séparément. Sert à partager le « calcul simple » 2027 ; n'entre
    pas dans l'ancre du modèle."""
    import csv as _csv
    lines = [ln for ln in PARTIES_CSV.read_text().splitlines() if ln and not ln.startswith("#")]
    rows = list(_csv.DictReader(lines))
    acc = {"PS": [], "EELV": [], "PCF": []}
    for r in rows:
        tot = sum(float(r[k]) for k in acc)
        for k in acc:
            acc[k].append(float(r[k]) / tot)
    return {"shares_rest": {k: round(sum(v) / len(v), 3) for k, v in acc.items()}, "n_polls": len(rows),
            "polls": [f"{r['institut']} {r['periode']}" for r in rows],
            "levels": {k: [float(r[k]) for r in rows] for k in ("LFI", "PS", "EELV", "PCF")}, "clip": [0.02, 0.96]}


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
    print(f"  Monte-Carlo {DRAWS} tirages × 577 circos × 2 étiquettes (LFI + union moyenne) …")
    # Les deux étiquettes dans le MÊME appel : les tirages étaient déjà identiques (la graine est
    # la même et le flux ne dépend pas de `deltas`), mais un seul appel garde la matrice par
    # tirage des DEUX, seul moyen de chiffrer un écart ou un rang — cf. `rank_blur`/`paired_gap`.
    # Chance de la gauche unie avec la candidature d'union MOYENNE (report moyen mesuré en 2024,
    # décalage 0) : la valeur du siège pour l'union, étiquette quelconque — affichée, hors règle.
    wm = simulate(arr, summary, {"lfi": deltas["lfi"], "union": 0.0}, matrices=True)
    p_lfi_arr = wm["lfi"].mean(axis=0)
    p_left = wm["union"].mean(axis=0)
    p_ru = simulate(arr, summary, {"lfi": deltas["lfi"]}, right_union=True)["lfi"]
    p_loc = simulate(arr, summary, {"lfi": deltas["lfi"]}, national=False)["lfi"]
    default_share = round(float(next(x for x in scenarios_2027.SCENARIOS if x["key"] == "split2")["radical_share"]), 3)
    # La part sondages elle-même est dans la grille (et sert de réglage par défaut) : la force
    # réelle affichée par défaut est calculée à la part mesurée, pas au cran de grille voisin.
    shares = sorted({f"{s:.2f}": s for s in list(SHARES) + [default_share]}.values())
    near = f"{default_share:.2f}"
    print(f"  rapport de force : gauche divisée × {len(shares)} parts LFI …")
    split = simulate_split(arr, summary, shares)

    pres_src, pres, pres_nat, pres_com = radical_spatial.left_presidential_shares()
    parties = _party_shares_rest()
    print(f"  présidentielle {pres_src} : Mélenchon dans la gauche {pres_nat['LFI']:.3f} ; reste PS/EELV/PCF {({k: round(v, 3) for k, v in pres_nat.items() if k != 'LFI'})} ; sondages reste {parties['shares_rest']}")
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
        pl = round(float(p_lfi_arr[i]), 3)
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
            # Option extérieure du PARTENAIRE : même simulation, même tirages, autre pôle.
            "q_oth": round(float(split[near]["q_oth"][i]), 3) if pub else None,
            "rdev": arr.get("rdev", [0] * len(arr["id"]))[i],
            # Présidentielle : part BRUTE de Mélenchon dans le vote de gauche (colonne de droite,
            # la moyenne nationale est servie à part) ; écart local de chaque parti du reste.
            # `mel_com` non vide = circo entièrement incluse dans une commune que le scrutin ne
            # découpe pas (Paris, Marseille, Lyon…) : la valeur servie est celle de la commune,
            # identique pour toutes ses circos. La page le dit dans la cellule.
            "mel": round(pres[cid]["LFI"], 3) if cid in pres else None,
            "mel_com": pres_com.get(cid),
            "pres_rest": ({pt: round(pres[cid][pt] - pres_nat[pt], 3) for pt in ("PS", "EELV", "PCF") if pt in pres[cid]}
                          if cid in pres else None),
            "p_left": round(float(p_left[i]), 3) if pub else None,
            "pred": {"G": round(g0, 1), "CD": round(cd0, 1), "ED": round(ed0, 1), "AU": round(au0, 1)},
            "depute": {"nom": dep.get("nom", ""), "prenom": dep.get("prenom", ""),
                       "groupe": dep.get("groupe", ""), "bloc": dep.get("bloc", "")},
            "lab2024": lab24.get(cid), "union_won_2024": (win24.get(cid) == "UG") if win24 else None,
        })

    for r in rows:
        r["posture"] = posture(r["group"], r["q_lfi"], r["q_oth"])
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
    # Flou de rang et écart apparié : calculés ICI, sur les tirages, parce qu'ils portent sur des
    # DIFFÉRENCES — la page ne peut pas les redériver des intervalles marginaux qu'elle affiche.
    idx = {cid: i for i, cid in enumerate(arr["id"])}
    en_jeu_idx = [idx[r["id"]] for r in rows if r["group"] == "en_jeu"]
    ef = error_fit()
    rb = rank_blur(wm["lfi"], en_jeu_idx)
    pub_idx = [idx[r["id"]] for r in rows if r["pub"]]
    pg = paired_gap(wm["lfi"], wm["union"], pub_idx)
    print(f"  flou de rang (écart apparié) : médiane {rb['med']}, max {rb['max']} ; "
          f"plus grand écart d'étiquette ±{pg['paired_pt']} pt apparié vs ±{pg['sum_pt']} pt en "
          f"sommant les deux demi-largeurs")
    print(f"  groupes : {groups}")
    print(f"  sièges LFI espérés — acquis : {base_lfi:.1f} ; carte 2024 ({n24} circos FI) : {exp24:.1f} ; "
          f"répartition efficace à {n24} circos : {eff_same_n:.1f}")
    return {
        "scenario": {"key": SCENARIO, "label": scn["label"], "means": m},
        "history": {k: {"label": e["label"], "source": e["source"], "national_G": round(e["national"]["G"], 1),
                        **({"national_LFI": round(e["national"]["LFI"], 1)} if "LFI" in e["national"] else {})}
                    for k, e in hist.items()},
        "params": {"draws": DRAWS, "seed": SEED,
                   # Loi de l'erreur d'ancre, telle que `poll_error_model` la mesure : biais
                   # rétracté effectivement appliqué, écart-type et corrélations visés, et ce
                   # que le tirage délivre (contrôle : les deux doivent coïncider).
                   "nat_sigma": {b: round(float(ef["sd"][i]), 2) for i, b in enumerate(("G", "CD", "ED"))},
                   "nat_sigma_delivered": delivered_sigma(m),
                   "nat_bias": {b: round(float(ef["bias"][i]), 2) for i, b in enumerate(("G", "CD", "ED"))},
                   # Le DÉCALAGE APPLIQUÉ à l'ancre, servi à part et de signe opposé au biais :
                   # `_draw_national` renvoie ancre − erreur, donc un bloc SUR-prédit (biais > 0)
                   # est tiré vers le BAS. Servir les deux évite d'avoir à retourner le signe à
                   # l'affichage — ce qui avait été oublié, et la page annonçait une extrême
                   # droite rehaussée de 1,5 pt une phrase après avoir dit qu'on la corrigeait.
                   "nat_shift": {b: round(-float(ef["bias"][i]), 2) for i, b in enumerate(("G", "CD", "ED"))},
                   "nat_bias_raw": {b: round(float(ef["bias_raw"][i]), 2) for i, b in enumerate(("G", "CD", "ED"))},
                   "nat_shrink": round(float(ef["shrink"]), 3),
                   "nat_corr": {f"{a}{b}": round(float(ef["corr"][i][j]), 2)
                                for i, a in enumerate(("G", "CD", "ED"))
                                for j, b in enumerate(("G", "CD", "ED")) if i < j},
                   "nat_n": ef["n"],
                   "ci_z": CI_Z,
                   "rank_blur": rb,
                   "paired_gap": pg,
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
        "presidential": {"source": pres_src, "national": {k: round(v, 4) for k, v in pres_nat.items()},
                         "note": "Part de Mélenchon dans le vote de gauche ; PS/EELV/PCF = part de chaque candidat·e dans le vote des trois. Les résultats présidentiels s'arrêtent à la commune : une circonscription entièrement incluse dans une grande commune (Paris, Marseille, Lyon…) reçoit la valeur de CETTE commune, la même pour toutes ses circonscriptions."},
        "parties_2027": parties,
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
