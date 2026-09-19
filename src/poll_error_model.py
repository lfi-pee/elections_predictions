"""Modèle d'erreur de l'ancre nationale : ce que les sondages se sont trompés, et comment on
s'en sert pour tirer un niveau national.

La question à laquelle ce module répond : quand on pose une ancre G/CD/ED pour 2027, de combien
peut-elle être fausse, et dans quelle direction ? Trois choses, toutes mesurées, aucune posée :

  1. LE BIAIS. `bayesian_polls` garde l'erreur LOO de chaque scrutin d'apprentissage
     (`err = prédit − réel`). Sur les législatives, l'extrême droite est SUR-prédite à chaque
     fois : +2,2 / +6,9 / +2,3 / +11,2 / +7,3 pts de 2002 à 2022 — chiffres dans l'espace où ce
     module travaille (point 3), d'où le léger écart avec les erreurs brutes de `bayesian_polls`.
     Moyenne +6,0, t(4) = 3,5 : ce n'est pas du bruit. Un tirage centré sur l'ancre brute place
     donc l'extrême droite trop haut de plusieurs points à CHAQUE tirage, et le modèle ne peut
     pas s'en apercevoir — l'erreur est dans le centre, pas dans la largeur, et aucune largeur
     d'intervalle ne rattrape une erreur de centre.

  2. LA RÉTRACTION. Cinq scrutins, c'est peu, et le sixième — 2024, hors échantillon — est
     parti dans l'AUTRE sens (−4,6). Corriger du biais brut reviendrait à parier que cinq
     observations disent toute la vérité ; ne rien corriger, à parier qu'elles n'en disent
     rien. On prend donc la moyenne a posteriori sous un a priori centré sur ZÉRO dont
     l'échelle est estimée sur les données (empirical Bayes). Si les biais observés ne
     dépassent pas leur propre bruit d'échantillonnage, le facteur vaut zéro et rien n'est
     corrigé : la rétraction ne peut pas inventer un biais absent. C'est elle qui décide, pas
     nous.

  3. LA COVARIANCE. Les trois blocs ne se trompent pas indépendamment : une part surestimée
     est prise à une autre. Gauche et centre-droit sont anticorrélés à −0,8 dans les données.
     Le modèle précédent tirait les trois INDÉPENDAMMENT puis renormalisait à 100, ce qui
     fabrique une anticorrélation à peu près égale entre toutes les paires (−0,4 / −0,5 / −0,5)
     — l'ordre des corrélations réelles s'en trouvait presque inversé — et rabotait au passage
     14 à 22 % de la dispersion visée. On tire ici le vecteur d'erreur d'un seul coup dans la
     covariance mesurée, et on RÉ-ÉTALONNE l'échelle d'entrée pour que ce qui SORT après
     renormalisation ait bien l'écart-type visé.

2024 est inclus comme sixième observation. Il contredit la tendance, et c'est précisément
pourquoi l'exclure serait indéfendable : on ne retire pas la seule observation qui gêne. Son
inclusion rétracte DAVANTAGE (λ 0,64 → 0,36) et divise par deux et demi la correction appliquée
à l'extrême droite (−3,8 → −1,5 pt).

Deux réserves, qui vont dans le même sens et qu'il vaut mieux énoncer que laisser deviner. La
fenêtre 2024 contenait de vrais sondages législatifs, alors que celles de 2002→2022 n'avaient
pratiquement que des sondages présidentiels : 2024 estimait donc une tâche PLUS FACILE que celle
de 2027. Et les cinq erreurs d'apprentissage viennent de `per_election[best_lam]`, où best_lam a
été choisi pour minimiser la RMSE sur ces cinq scrutins mêmes. Les deux rendent les erreurs
observées un peu trop petites, donc Σ trop petite, donc λ trop GRAND : la rétraction devrait être
un peu plus forte que celle qu'on applique, et la correction un peu plus faible. C'est une borne
sur le sens de l'erreur, pas une correction de plus — on ne rétracte pas deux fois sur un
argument qualitatif.

    python3 -u -m src.poll_error_model        # → data/polls/national_errors.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE = Path("data/polls/national_errors.json")
BLOCS = ("G", "CD", "ED")
_BLOC_OF = {"Gauche": "G", "Centre+Droite": "CD", "Extreme_Droite": "ED"}


def compute() -> dict:
    """Extrait la matrice d'erreur depuis `bayesian_polls` (lent : recharge tous les sondages).

    Convention conservée telle quelle : `err = prédit − réel`, donc un err positif = le bloc a
    été SUR-prédit. Les scrutins d'apprentissage viennent de la sélection LOO de lambda ; 2024
    vient de l'estimation hors échantillon que `bayesian_polls` produit déjà."""
    from src.bayesian_polls import (loo_select_lambda, estimate_house_effects,
                                    estimate_national_bayesian)
    from src.cross_type_dev import load_cross_type_data, VAL_DATE
    from src.load_polls import load_poll_tokens

    data_dir = Path("data")
    _, _, national_means, _ = load_cross_type_data(data_dir)
    polls = load_poll_tokens(data_dir)
    best_lam, _, _, _, per_election = loo_select_lambda(polls, national_means, VAL_DATE)

    def _row(label, pred, act):
        # On garde prédit ET réel, pas seulement leur écart : l'écart seul ne permet pas de
        # repasser dans l'espace des PARTS DES TROIS BLOCS, qui est celui où le modèle tire.
        return {"election": label,
                "pred": {b: round(float(pred[k]), 2) for k, b in _BLOC_OF.items()},
                "act": {b: round(float(act[k]), 2) for k, b in _BLOC_OF.items()}}

    by_date = {float(r["date_float"]): r for _, r in national_means.iterrows()}
    train = []
    for e in per_election[best_lam]:
        if not str(e["election_type"]).startswith("Legislatives"):
            continue   # seules les législatives : c'est le scrutin qu'on prédit
        act = by_date[float(e["date_float"])]
        pred = {k: float(act[k]) + float(e[f"err_{k}"]) for k in _BLOC_OF}
        train.append(_row(f"{e['election_type']} {e['date_float']:.2f}", pred, act))

    house_fx = estimate_house_effects(polls, national_means, VAL_DATE)
    pred = estimate_national_bayesian(polls, VAL_DATE, house_fx, best_lam)
    holdout = _row(f"Legislatives_T1 {VAL_DATE:.2f}", pred, by_date[float(VAL_DATE)])

    return {"convention": "err = predit - reel, en PARTS DES TROIS BLOCS renormalisees a 100",
            "lambda": best_lam, "source": "src.bayesian_polls (LOO)",
            "train": train, "holdout": holdout}


def load() -> dict:
    if not CACHE.exists():
        raise FileNotFoundError(f"{CACHE} absent : lancer `python3 -m src.poll_error_model`")
    return json.loads(CACHE.read_text())


def errors(d: dict | None = None, with_holdout: bool = True) -> np.ndarray:
    """Matrice (3 blocs × n scrutins) des erreurs, 2024 compris par défaut.

    Erreurs exprimées en PARTS DES TROIS BLOCS, prédit et réel étant chacun ramené à 100 avant
    la soustraction. Ce n'est pas un détail de présentation : c'est exactement l'espace où
    `_draw_national` travaille (l'abstention et le bloc « Autre » sont tenus fixes, donc les
    trois blocs se partagent un total imposé). Chaque colonne somme alors à ZÉRO, la covariance
    est singulière dans la direction (1,1,1), et un tirage dans cette covariance respecte le
    total SANS renormalisation — donc sans la distorsion que la renormalisation infligeait à la
    dispersion (−14 à −22 %) et à l'ordre des corrélations.

    La raison de fond n'est pas seulement cet alignement d'espaces : les DÉNOMINATEURS diffèrent.
    `pred` somme déjà à 100, tandis que `act` porte un bloc « Autre » sans contrepartie prédite
    (les sommes réelles vont de 95,0 à 98,8). Soustraire les deux tels quels donnerait à chaque
    bloc un biais parasite d'environ +1 pt qui ne serait qu'un écart de normalisation.

    Ramener les deux à 100 (projection PROPORTIONNELLE) n'est pas la seule façon de rejoindre le
    plan de somme nulle : on pourrait retrancher la moyenne de l'écart (projection par CENTRAGE).
    Le choix n'est pas neutre — le centrage donnerait λ = 0,21 et une correction d'extrême droite
    de −0,8 pt au lieu de −1,5. La proportionnelle est la bonne ici parce que l'écart vient
    justement d'un rapport de normalisation et non d'un décalage additif commun ; la sensibilité
    mérite d'être au dossier.
    """
    d = d or load()
    rows = d["train"] + ([d["holdout"]] if with_holdout else [])

    def norm(v: dict) -> np.ndarray:
        x = np.array([v[b] for b in BLOCS], dtype=float)
        return x / x.sum() * 100.0

    return np.array([norm(r["pred"]) - norm(r["act"]) for r in rows]).T


def fit(d: dict | None = None, with_holdout: bool = True) -> dict:
    """Biais rétracté et covariance prédictive de l'erreur d'ancre.

    Rétraction par un facteur SCALAIRE, a priori centré sur zéro — moyenne a posteriori,
    facteur estimé par la méthode des moments (empirical Bayes). Ce n'est PAS la forme
    James–Stein « 1 − (p−2)σ²/‖b̂‖² », qui dégénérerait ici : la contrainte de somme nulle
    ramène la dimension effective à 2, et le facteur (p−2) s'y annulerait.

        λ = max(0, 1 − tr(Σ)/n / ‖b̂‖²)

    ‖b̂‖² surestime systématiquement le vrai ‖β‖², parce qu'il contient en plus le bruit
    d'échantillonnage de la moyenne, dont l'espérance vaut tr(Σ)/n. λ est la part de ‖b̂‖² qui
    reste une fois ce bruit retiré : si les biais observés ne dépassent pas leur propre bruit,
    λ = 0 et tout est ramené à zéro. La rétraction ne peut pas inventer un biais absent.

    Pourquoi UN scalaire et non un λ par bloc : les erreurs somment à zéro (parts des trois
    blocs), et trois facteurs différents casseraient cette somme — le biais rétracté ne
    tiendrait plus dans le plan où le modèle tire, et il faudrait le reprojeter, c'est-à-dire
    défaire la rétraction qu'on vient de faire. Un scalaire préserve la contrainte exactement.

    La covariance prédictive vaut Σ·(1 + λ/n). Le terme ajouté est la variance A POSTERIORI du
    biais, λ·Σ/n, et non la variance d'échantillonnage de λb̂, qui vaudrait λ²·Σ/n : la
    correction appliquée est une moyenne a posteriori, c'est donc l'incertitude a posteriori
    qu'il faut reporter. La distinction n'est pas cosmétique — elle vaut ici 6 % d'écart-type en
    plus plutôt que 2 %, dans le sens de la prudence. Le facteur porte sur Σ tout entier, donc
    reste lui aussi dans le plan de somme nulle.

    CE QUE CETTE COVARIANCE NE DIT PAS. Σ est estimée sur n scrutins et ensuite traitée
    comme CONNUE : seule l'incertitude sur la moyenne est reportée. Avec 5 degrés de liberté,
    une prédictive de Student élargirait les écarts-types marginaux d'environ 29 %, et les
    corrélations (−0,61 / −0,41 / −0,47) ont une erreur-type de l'ordre de 0,3 — elles ne sont
    pas distinguables les unes des autres. La page les affiche à deux décimales : c'est la
    seule quantité du tableau servie sans ses bornes, et c'est une limite assumée, pas un
    oubli — la corriger demanderait de tirer la covariance elle-même (Wishart inverse), ce qui
    déplacerait l'incertitude sans qu'on sache la valider sur six observations.
    """
    e = errors(d, with_holdout)
    n = e.shape[1]
    b_hat = e.mean(axis=1)
    cov = np.cov(e, ddof=1)
    # ‖b̂‖² au dénominateur : un biais EXACTEMENT nul (données synthétiques, ou un jour des
    # sondages sans biais) doit donner λ = 0, pas une division par zéro — c'est précisément le
    # cas que le docstring promet de traiter.
    nb2 = float(b_hat @ b_hat)
    lam = 0.0 if nb2 <= 0.0 else max(0.0, 1.0 - float(np.trace(cov)) / n / nb2)
    bias = lam * b_hat
    pred_cov = cov * (1.0 + lam / n)
    sd = np.sqrt(np.diag(pred_cov))
    return {"n": n, "bias_raw": b_hat, "shrink": lam, "bias": bias, "cov": pred_cov,
            "sd": sd, "corr": np.divide(pred_cov, np.outer(sd, sd),
                                        out=np.zeros_like(pred_cov), where=np.outer(sd, sd) > 0)}


def main() -> None:
    d = compute()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    f = fit(d)
    print(f"→ {CACHE} ({len(d['train'])} scrutins d'apprentissage + {d['holdout']['election']})")
    print(f"  rétraction λ = {f['shrink']:.3f} (commune aux trois blocs)")
    for i, b in enumerate(BLOCS):
        print(f"  {b:>2} : biais brut {f['bias_raw'][i]:+5.2f} → rétracté {f['bias'][i]:+5.2f}"
              f" ; écart-type prédictif {f['sd'][i]:.2f}")
    print("  corrélations :", {f"{BLOCS[i]}/{BLOCS[j]}": round(float(f["corr"][i, j]), 2)
                               for i in range(3) for j in range(i + 1, 3)})


if __name__ == "__main__":
    main()
