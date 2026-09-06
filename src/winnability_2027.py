"""Score de jouabilité 1→5 de la **gauche** par circonscription (1 = victoire facile,
5 = quasi impossible), sous un état de curseurs nationaux + une configuration de gauche.

Le modèle prédit le **bloc** Gauche entier. Ce qui décide un siège, c'est la
configuration : une gauche **unie** présente un candidat qui capte tout le bloc ; une
gauche **divisée** répartit le bloc entre deux (radicale vs néolibérale) ou trois
candidatures, dont aucune ne pèse le total — d'où une qualification au 2nd tour bien plus
dure (seuil des 12,5 % des inscrits). C'est là que se joue le pari « union vs division ».

Proxy assumé, pas une simulation de siège : on part des parts de bloc au 1er tour
(seul ce que le modèle prédit), on applique la règle de qualification, puis une
estimation de report au 2nd tour (front républicain contre le RN). La logique est
répliquée à l'identique côté client (`js/winnability.js`) pour réagir aux curseurs.
"""

from __future__ import annotations

# Reports de 2nd tour (barrage) du RN. Contre le centre-droit, peu de reports RN→gauche.
BARRAGE_ED_TO_LEFT = 0.15  # duel gauche vs centre-droit : reports RN faibles
BARRAGE_ED_TO_CD = 0.45

# Reports du bloc Centre+Droite, DÉCOMPOSÉS Ensemble vs LR — un électeur macroniste et un
# électeur LR ne se reportent pas pareil, et « union des droites » ne concerne que LR. La part
# LR du bloc (`CD_LR_DEFAULT`) est réglable au curseur (défaut = LR 12/(Ensemble 14+LR 12),
# sondages 2026). Les taux ci-dessous reproduisent l'ancien barrage global (0,45 vers la gauche,
# 0,25 vers le RN) à cette composition ; « droites unies » = LR bascule vers le RN, Ensemble
# continue de faire barrage (le front s'affaiblit sans s'effondrer).
CD_LR_DEFAULT = 0.46
ENS_TO_LEFT, ENS_TO_RN = 0.55, 0.12        # Ensemble : barrage anti-RN net
LR_TO_LEFT, LR_TO_RN = 0.33, 0.40          # LR seul, hors union des droites : reports partagés
LR_TO_LEFT_RU, LR_TO_RN_RU = 0.10, 0.60    # LR sous « droites unies » : bascule vers le RN
# Réunification imparfaite au 2nd tour : quand un pôle de gauche est éliminé au 1er tour,
# ses voix ne se reportent qu'en partie sur le pôle de gauche qualifié (les électorats
# radical et social-démocrate ne fusionnent pas entièrement). C'est le coût propre de la
# division, distinct du niveau national : à niveau égal, la gauche unie fait mieux.
#
# MESURÉ (src/reunif_measure.py) sur les législatives à gauche réellement divisée. Référence
# 2012 (PS vs Front de Gauche vs EELV, découpage actuel) : régression écologique du gain T2 du
# pôle de gauche survivant sur les voix de gauche éliminées, en contrôlant les reports CD/ED et
# la mobilisation → β = 0,69–0,73 ; ratio direct (sur-estime, mobilisation incluse) ~0,86.
# Pool 2012+2017 : régression 0,66, ratio 0,86. 0,72 est donc en haut de la fourchette de
# régression et sous le ratio — valeur centrale-à-légèrement-haute, empiriquement soutenue.
REUNIF = 0.72

# Désistement (« front républicain ») — le mécanisme DOMINANT du 2nd tour, MESURÉ sur le réel
# 2024. En triangulaire face au RN, le pôle anti-RN le plus faible (gauche ou centre-droit) se
# retire au profit du plus fort : `DESIST_TO_STRONG` de ses voix vont au survivant anti-RN,
# `DESIST_TO_ED` fuient vers le RN, le reste s'abstient. Réglable au curseur ; le baisser
# modélise un front républicain qui se délite.
#
# Mesure directe (src/desist_2024_measure.py, comparaison T1→T2 des voix par bloc et par circo,
# sur les 271 triangulaires face au RN de 2024 où un pôle s'est effectivement désisté) : le
# désistant a transféré ~69 % de ses voix au survivant anti-RN, ~17 % ont fui vers le RN, ~14 %
# se sont abstenues. Report fortement ASYMÉTRIQUE : centre-droit→gauche 56 %, gauche→centre-droit
# 81 % (la gauche fait mieux barrage). DESIST_TO_ED = 0,17 reprend la fuite mesurée.
#
# DESIST_TO_STRONG (0,52) < 0,69 mesuré à dessein : le désistement mesuré (0,69) ne vaut que
# dans les triangulaires où un pôle s'est EFFECTIVEMENT retiré, or ~⅓ des triangulaires 2024 ont
# été MAINTENUES (pas de désistement). 0,52 est la force INCONDITIONNELLE (appliquée à toute
# triangulaire) qui reproduit le NOMBRE RÉEL de sièges RN de 2024 — c.-à-d. SANS BIAIS sur le RN.
# Backtest oracle (parts T1 réelles, gauche unie) : G/CD/ED = 169/223/109 contre 162/230/109
# réels (RN exact ; erreur totale de sièges minimale), justesse par circo ~82 %. On calibre sur
# le NOMBRE de sièges (l'objet de l'outil), pas sur la justesse par circo : à 0,60 la justesse par
# circo est marginalement meilleure (82,6 %) MAIS le RN est sous-estimé de ~18 sièges (91 vs 109)
# — biais inacceptable, surtout pour un public de gauche. Réglable au curseur (baisser = front qui
# se délite → RN plus fort).
DESIST_TO_STRONG = 0.52
DESIST_TO_ED = 0.17


def _left_candidates(g: float, cfg: str, rad: float) -> list[float]:
    """Parts (en % des exprimés) des candidatures de gauche selon la configuration."""
    if cfg == "union":
        return [g]
    if cfg == "split2":
        return [g * rad, g * (1 - rad)]
    # split3 : pôle radical + deux pôles (PS/PP, éco/PCF)
    other = g * (1 - rad)
    return [g * rad, other * 0.6, other * 0.4]


def _left_t2(left: list[float], second: float, thr: float) -> tuple[float, bool]:
    """Score de gauche réuni au 2nd tour + booléen « au moins un pôle qualifié ».
    Pôles qualifiés (top 2 ou ≥ seuil) pleins + pôles éliminés × REUNIF."""
    ql = [p for p in left if p >= second - 1e-9 or p >= thr]
    if not ql:
        return 0.0, False
    elim = sum(p for p in left if p not in ql)
    return sum(ql) + REUNIF * elim, True


def _cd_transfer(right_union: bool, cd_lr: float = CD_LR_DEFAULT) -> tuple[float, float]:
    """Reports du centre-droit au 2nd tour, composés Ensemble (barrage) + LR (ambivalent, part
    `cd_lr`). Sous union des droites, seul LR bascule vers le RN ; Ensemble fait toujours
    barrage — le front s'affaiblit sans s'effondrer. Renvoie (vers gauche, vers RN)."""
    ens = 1.0 - cd_lr
    lr_l, lr_e = (LR_TO_LEFT_RU, LR_TO_RN_RU) if right_union else (LR_TO_LEFT, LR_TO_RN)
    return ens * ENS_TO_LEFT + cd_lr * lr_l, ens * ENS_TO_RN + cd_lr * lr_e


def score_circo(g: float, cd: float, ed: float, ab: float, cfg: str, rad: float,
                right_union: bool = False, desist: float = DESIST_TO_STRONG,
                cd_lr: float = CD_LR_DEFAULT, au: float = 0.0) -> dict:
    """Renvoie {score 1..5, l_best, qualifies, margin_t2, opp}. `g/cd/ed` = parts exprimées
    (somme ~100), `ab` = abstention % inscrits. `au` = bloc « Autre » (régionaliste), pris en
    compte comme adversaire là où il domine (bastions)."""
    turnout = max(0.05, 1 - ab / 100.0)
    thr = 12.5 / turnout  # seuil de qualification en part d'exprimés (= 12,5 % des inscrits)

    left = _left_candidates(g, cfg, rad)
    l_best = max(left) if left else 0.0
    cands = left + [cd, ed] + ([au] if au > 0.0 else [])
    top2 = sorted(cands, reverse=True)[:2]
    leader, second = top2[0], top2[1]
    left_base, qualifies = _left_t2(left, second, thr)
    q_cd = cd >= second - 1e-9 or cd >= thr
    q_ed = ed >= second - 1e-9 or ed >= thr
    cd2l, cd2e = _cd_transfer(right_union, cd_lr)

    if not qualifies:
        opp = "AU" if (au > 0.0 and au >= cd and au >= ed) else ("ED" if ed >= cd else "CD")
        return {"score": 5, "l_best": round(l_best, 1), "qualifies": False,
                "margin_t2": None, "opp": opp}

    # 2nd tour : gauche réunie (réunification imparfaite si divisée) face à l'adversaire le plus
    # fort ; reports selon que cet adversaire est le RN, le centre-droit, ou le bloc « Autre ».
    if au > 0.0 and au >= cd and au >= ed:
        # Adversaire = pôle régionaliste dominant (bastion) : hors axe, pas de front
        # républicain ni de report de barrage — duel direct gauche vs Autre.
        opp = "AU"
        left_t2 = left_base
        opp_t2 = au
    elif ed >= cd:
        opp = "ED"
        if q_cd and q_ed and not right_union:
            # Triangulaire face au RN : le centre-droit se DÉSISTE pour la gauche (front
            # républicain). Renfort plus fort qu'un simple report de barrage.
            left_t2 = left_base + desist * cd
            opp_t2 = ed + DESIST_TO_ED * cd
        else:
            # Duel (CD éliminé) ou droites unies (pas de désistement) : barrage classique.
            left_t2 = left_base + cd2l * cd
            opp_t2 = ed + cd2e * cd
    else:
        opp = "CD"
        left_t2 = left_base + BARRAGE_ED_TO_LEFT * ed
        opp_t2 = cd + BARRAGE_ED_TO_CD * ed
    margin_t2 = left_t2 - opp_t2
    leads_first = l_best >= leader - 1e-9

    if leads_first and margin_t2 > 8:
        score = 1
    elif margin_t2 > 0:
        score = 2
    elif margin_t2 > -8:
        score = 3
    else:
        score = 4
    return {"score": score, "l_best": round(l_best, 1), "qualifies": True,
            "margin_t2": round(margin_t2, 1), "opp": opp}


def seat_winner(g: float, cd: float, ed: float, ab: float, cfg: str, rad: float,
                right_union: bool = False, desist: float = DESIST_TO_STRONG,
                cd_lr: float = CD_LR_DEFAULT, au: float = 0.0) -> str:
    """Bloc vainqueur du siège (G/CD/ED/AU), même modèle de 2nd tour que `score_circo`."""
    turnout = max(0.05, 1 - ab / 100.0)
    thr = 12.5 / turnout
    left = _left_candidates(g, cfg, rad)
    # Bloc « Autre » (régionaliste/autonomiste) = pôle sortant « collant » : là où il arrive en
    # tête au 1er tour (bastions corses/ultramarins), il conserve le siège. Hors de l'axe
    # G/CD/ED, il ne participe pas au front républicain ; là où il n'est PAS en tête, la logique à
    # trois pôles ci-dessous reprend inchangée — aucun effet sur les circos où l'Autre est
    # négligeable (pattern spatial ≈ 0 hors bastions). Miroir exact de winnability.js.
    if au > 0.0 and au >= max(left + [cd, ed]) - 1e-9:
        return "AU"
    cands = sorted(left + [cd, ed], reverse=True)
    second = cands[1]
    left_base, qL = _left_t2(left, second, thr)
    qC = cd >= second - 1e-9 or cd >= thr
    qE = ed >= second - 1e-9 or ed >= thr
    cd2l, cd2e = _cd_transfer(right_union, cd_lr)
    sL = left_base if qL else 0.0
    sC = cd if qC else 0.0
    sE = ed if qE else 0.0
    if not qL and left_base == 0.0:  # gauche éliminée : ses voix (g) se reportent
        if qC:
            sC += 0.55 * g
        if qE:
            sE += 0.10 * g
    if not qC:
        if qL:
            sL += cd2l * cd
        if qE:
            sE += cd2e * cd
    if not qE:
        if qL:
            sL += BARRAGE_ED_TO_LEFT * ed
        if qC:
            sC += BARRAGE_ED_TO_CD * ed
    # Désistement (front républicain) en triangulaire face au RN : le pôle anti-RN le plus
    # faible se retire au profit du plus fort. C'est le mécanisme qui recale les sièges RN sur
    # le réel 2024 (cf. en-tête). Sous « droites unies », le centre-droit refuse de se retirer.
    if qL and qC and qE:
        if sL >= sC:  # gauche = pôle anti-RN le plus fort → le centre-droit se désiste
            if not right_union:
                sL += desist * sC
                sE += DESIST_TO_ED * sC
                qC = False
        else:  # centre-droit le plus fort → la gauche se désiste (elle fait toujours barrage)
            sC += desist * sL
            sE += DESIST_TO_ED * sL
            qL = False
    opts = [("G", sL, qL), ("CD", sC, qC), ("ED", sE, qE)]
    opts = [o for o in opts if o[2]]
    opts.sort(key=lambda x: -x[1])
    if opts:
        return opts[0][0]
    return "G" if g >= cd and g >= ed else ("CD" if cd >= ed else "ED")


SCORE_LABELS = {
    1: "victoire facile",
    2: "jouable",
    3: "disputé",
    4: "difficile",
    5: "quasi impossible",
}
