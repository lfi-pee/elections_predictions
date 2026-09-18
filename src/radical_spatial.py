"""Motif spatial de la part RADICALE (LFI/Mélenchon) DANS le vote de gauche, par circonscription.

Le modèle de sièges scinde le bloc de gauche en pôles (radical vs sociaux-démocrates) selon une
part `rad`. Cette part se **conjugue** à partir de deux sources qui mesurent des choses
différentes — exactement comme le modèle de blocs (niveau national posé, motif spatial appris) :

  • NIVEAU national  ← SONDAGES (`scenarios_2027.anchor_from_polls`, ~0,37 = LFI/(LFI+PS·PP+
    EELV+PCF)). C'est la seule chose que les sondages mesurent : il n'existe pas de sondage
    LFI-dans-la-gauche par circonscription.
  • FORME par circo   ← ce module : la DÉVIATION au niveau national (quelles circos votent plus
    ou moins LFI que la moyenne), recentrée à moyenne (pondérée) nulle pour ne PAS déplacer le
    niveau posé par les sondages.

  rad_circo = clamp( part_sondages  +  RAD_GAIN · déviation_élection , 0,05 , 0,95 )

SOURCE de la forme — RÈGLE SÉLECTIONNÉE PAR VALIDATION CROISÉE, pas choisie à la main
(`src/lfi_geo_loo.py`, LOO complet sur toutes les élections à gauche divisée à pôle radical
étiqueté) : **la présidentielle (1er tour) la plus RÉCENTE disponible avant le scrutin visé**,
part de Mélenchon dans le vote de gauche. C'est la seule famille de source qui GÉNÉRALISE hors
échantillon (R² OOF poolé +0,298 vs part plate ; les cartes de listes législatives/européennes
font PIRE que la moyenne plate : −0,5 à −0,6). Le vote présidentiel personnel de Mélenchon,
dense dans chaque commune, est un instrument stable ; l'offre de listes législatives, non.

GRANULARITÉ des résultats présidentiels : la COMMUNE. 537 circos sont donc mesurées sur les
communes qu'elles contiennent entièrement ; les 40 autres sont elles-mêmes incluses dans une
grande commune que le scrutin ne découpe pas (les 18 de Paris, les 7 de Marseille, Lyon, Nice,
Toulouse…) et reçoivent la valeur de CETTE commune, la même pour toutes ses circos. C'est la
seule chose que la donnée sait dire là : exacte au niveau de la ville, muette sur les écarts
internes — et bien plus proche du vrai que l'écart nul (= part nationale) servi auparavant, ces
villes votant Mélenchon nettement plus que la moyenne DANS la gauche. La validation LOO de la
règle de source (`lfi_geo_loo`) porte, elle, sur les seules circos directement mesurées.

Rien n'est figé en dur : les scrutins présidentiels sont DÉCOUVERTS dans les données et le plus
récent d'avant la cible est retenu automatiquement. Pour 2027 cela donne la présidentielle 2022 ;
pour une cible ultérieure, la présidentielle suivante, sans toucher au code. Seule la LIGNÉE
politique (qui est « radical » / qui est « gauche ») est une définition du trait, pas une
sélection d'élections — les bulletins présidentiels ne portent pas d'étiquette de nuance.

    python3 -u -m src.radical_spatial          # motif servi pour 2027 (→ présidentielle 2022)
    python3 -u -m src.lfi_geo_loo              # la validation LOO de la règle de source
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.reunif_measure import _commune2circo, CAND, MASTER

# Cible par défaut : législatives 2027 (année + 6/12).
TARGET_DATE = 2027.5

# Dispersion appliquée telle que mesurée, sans amplification ni calage sur une cible de sièges.
RAD_GAIN = 1.0

# --- LIGNÉE politique (définition du trait, appliquée à N'IMPORTE quelle présidentielle) ---
# Le pôle radical = mouvance Mélenchon / Insoumis / Front de Gauche. Les présidentielles récentes
# ne portent pas de nuance (seulement le nom) ; 2012 est codée en nuance de candidat → on couvre
# les deux. « Gauche » = ensemble des candidats de gauche du scrutin (numérateur ⊂ dénominateur).
RAD_NAME = {"MÉLENCHON"}
RAD_NUANCE = {"MELE", "FI", "FG", "LFI"}
LEFT_NAME = {"MÉLENCHON", "HOLLANDE", "HAMON", "JADOT", "HIDALGO", "ROUSSEL",
             "POUTOU", "ARTHAUD", "JOLY", "LAGUILLER", "BESANCENOT", "BUFFET", "BOVÉ"}
LEFT_NUANCE = {"MELE", "HOLL", "JOLY", "POUT", "ARTH"}  # candidats de gauche, présidentielle 2012


def _year(eid: str) -> float:
    """Année flottante d'un id de scrutin ('2022_pres_t1' → 2022.3, la présidentielle = printemps)."""
    return int(eid[:4]) + 0.3


def select_source(target_date: float = TARGET_DATE, cand: pd.DataFrame | None = None) -> str:
    """RÈGLE LOO : la présidentielle (1er tour) la plus récente STRICTEMENT avant la cible.
    Les élections sont découvertes dans les données — aucune liste figée."""
    if cand is None:
        cand = pd.read_parquet(CAND, columns=["id_election"])
    pres = {e for e in cand.id_election.unique() if e.endswith("_pres_t1")}
    prior = sorted((e for e in pres if _year(e) < target_date), key=_year)
    if not prior:
        raise RuntimeError(f"Aucune présidentielle disponible avant {target_date}")
    return prior[-1]


# Lignée par parti, pour ventiler le RESTE de la gauche (hors pôle radical) : le·la candidat·e
# présidentiel·le de chaque parti, à n'importe quelle présidentielle.
PARTY_NAME = {"PS": {"HIDALGO", "HAMON", "HOLLANDE"}, "EELV": {"JADOT", "JOLY"}, "PCF": {"ROUSSEL", "BUFFET"}}


def _split_containers() -> pd.DataFrame:
    """Communes à cheval sur plusieurs circonscriptions (Paris, Marseille, Lyon…), ventilées par
    circo : `commune`, `circo`, `a` = part des inscrits de la commune tombant dans la circo, `nom`
    = libellé de la commune. Les résultats présidentiels s'arrêtent à la COMMUNE : une circo
    entièrement contenue dans une telle commune n'a aucune donnée propre — voir `_circo_left`."""
    bm = pd.read_parquet(MASTER, columns=["location", "circo", "inscrits", "libelle_commune"]).dropna(subset=["circo"])
    bm["commune"] = bm.location.str.split("_").str[0]
    g = bm.groupby(["commune", "circo"]).agg(inscrits=("inscrits", "sum"),
                                             nom=("libelle_commune", "first")).reset_index()
    nc = g.groupby("commune").circo.nunique()
    g = g[g.commune.isin(nc[nc > 1].index)].copy()
    g["a"] = g.inscrits / g.groupby("commune").inscrits.transform("sum")
    return g


def _circo_left(target_date: float = TARGET_DATE) -> tuple[str, pd.DataFrame]:
    """Par circo, à la présidentielle sélectionnée par la règle LOO : voix de gauche, voix du pôle
    radical et voix de chaque parti du reste. Deux origines, jamais mélangées :

      • cas général — agrégation des communes ENTIÈREMENT contenues dans la circo ;
      • circo SANS aucune commune entière (les 18 de Paris, les 7 de Marseille, Lyon, Nice,
        Toulouse, Montpellier…) — elle est incluse dans une seule grande commune que le scrutin
        présidentiel ne découpe pas : on lui sert la valeur de CETTE commune, répartie au prorata
        des inscrits. Les 18 circos de Paris reçoivent donc toutes la part parisienne : exact pour
        la commune, muet sur les écarts d'une circo à l'autre — c'est dit à l'affichage. L'ancien
        comportement (aucune valeur → déviation nulle → part nationale) était pire : ces villes
        votent Mélenchon nettement plus que la moyenne DANS la gauche.

    Colonnes : left, rad, PS, EELV, PCF, `com` (code de la commune servie, vide sinon), `nom`."""
    c2c = _commune2circo(drop_split=True)
    cand = pd.read_parquet(CAND, columns=["id_election", "code_commune", "nuance", "nom", "voix"])
    src = select_source(target_date, cand)
    d = cand[cand.id_election == src].copy()
    d["code_commune"] = d.code_commune.astype(str)
    d = d[d.nuance.isin(LEFT_NUANCE) | d.nom.isin(LEFT_NAME)]
    is_rad = d.nuance.isin(RAD_NUANCE) | d.nom.isin(RAD_NAME)
    com = pd.DataFrame({"left": d.groupby("code_commune").voix.sum(),
                        "rad": d[is_rad].groupby("code_commune").voix.sum(),
                        **{pt: d[d.nom.isin(names)].groupby("code_commune").voix.sum()
                           for pt, names in PARTY_NAME.items()}}).fillna(0.0)
    cols = ["left", "rad", *PARTY_NAME]
    whole = com[com.index.isin(c2c)].copy()
    whole["circo"] = whole.index.map(c2c)
    tab = whole.groupby("circo")[cols].sum()
    tab["com"], tab["nom"] = "", ""
    fb = _split_containers()
    fb = fb[fb.commune.isin(com.index) & ~fb.circo.isin(tab.index)]
    if not fb.empty:
        fb = fb.join(com[cols], on="commune")
        fb[cols] = fb[cols].mul(fb.a, axis=0)
        f = fb.groupby("circo")[cols].sum()
        held = fb.sort_values("a").groupby("circo").tail(1).set_index("circo")
        f["com"], f["nom"] = held.commune, held.nom
        tab = pd.concat([tab, f])
    return src, tab[tab.left > 0]


def radical_deviation(target_date: float = TARGET_DATE) -> dict[str, float]:
    """{circo → déviation de la part radicale-dans-la-gauche} (recentrée, moyenne pondérée ≈ 0),
    depuis la présidentielle sélectionnée par la règle LOO, au découpage circo actuel."""
    _, tab = _circo_left(target_date)
    rs = tab.rad / tab.left
    mean = float(np.average(rs, weights=tab.left))
    return {ci: float(v - mean) for ci, v in rs.items()}


def left_presidential_shares(target_date: float = TARGET_DATE) -> tuple[str, dict[str, dict[str, float]], dict[str, float], dict[str, str]]:
    """Par circo, à la présidentielle sélectionnée par la règle LOO : part de Mélenchon dans le
    vote de gauche (« LFI », identique à `radical_deviation` + moyenne) et part de chaque parti
    (PS, EELV, PCF) dans le vote des trois. Retourne (scrutin, {circo: {parti: part}},
    {parti: part nationale pondérée}, {circo: nom de la commune} pour les seules circos servies
    à la valeur de leur commune)."""
    src, tab = _circo_left(target_date)
    out = {ci: {"LFI": float(r.rad / r.left)} for ci, r in tab.iterrows()}
    nat = {"LFI": float(tab.rad.sum() / tab.left.sum())}
    tot3 = sum(tab[pt] for pt in PARTY_NAME)
    for pt in PARTY_NAME:
        if tab[pt].sum() == 0:
            continue
        nat[pt] = float(tab[pt].sum() / tot3.sum())
        for ci in out:
            if tot3[ci] > 0:
                out[ci][pt] = float(tab.at[ci, pt] / tot3[ci])
    return src, out, nat, {ci: n for ci, n in tab.nom.items() if n}


if __name__ == "__main__":
    src, tab = _circo_left()
    d = radical_deviation()
    v = np.array(list(d.values()))
    print(f"cible {TARGET_DATE} → source sélectionnée (règle LOO, présidentielle la + récente "
          f"avant la cible) : {src}")
    nc = int((tab.nom != "").sum())
    print(f"circos couvertes : {len(d)} — dont {nc} servies à la valeur de LEUR commune "
          f"(circo entièrement incluse dans une commune que le scrutin ne découpe pas) : "
          f"{', '.join(sorted(set(tab.nom[tab.nom != ''])))}")
    print(f"déviation part radicale-dans-la-gauche — écart-type {v.std():.3f}, "
          f"min {v.min():+.3f}, max {v.max():+.3f}")
    print(f"RAD_GAIN = {RAD_GAIN} · niveau national = sondages (scenarios_2027) ; "
          f"validation de la règle : src/lfi_geo_loo.py")
