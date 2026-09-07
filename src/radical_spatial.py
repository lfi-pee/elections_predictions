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

from src.reunif_measure import _commune2circo, CAND

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


def radical_deviation(target_date: float = TARGET_DATE) -> dict[str, float]:
    """{circo → déviation de la part radicale-dans-la-gauche} (recentrée, moyenne pondérée ≈ 0),
    depuis la présidentielle sélectionnée par la règle LOO, au découpage circo actuel."""
    c2c = _commune2circo(drop_split=True)
    cand = pd.read_parquet(CAND, columns=["id_election", "code_commune", "nuance", "nom", "voix"])
    src = select_source(target_date, cand)
    d = cand[cand.id_election == src].copy()
    d["circo"] = d.code_commune.astype(str).map(c2c)
    d = d.dropna(subset=["circo"])
    is_left = d.nuance.isin(LEFT_NUANCE) | d.nom.isin(LEFT_NAME)
    d = d[is_left]
    d["p"] = np.where(d.nuance.isin(RAD_NUANCE) | d.nom.isin(RAD_NAME), "R", "S")
    tab = d.groupby(["circo", "p"]).voix.sum().unstack("p").fillna(0.0)
    tab["left"] = tab.get("R", 0.0) + tab.get("S", 0.0)
    tab = tab[tab.left > 0]
    tab["rs"] = tab.get("R", 0.0) / tab["left"]
    mean = float(np.average(tab.rs, weights=tab.left))
    return {ci: float(rs - mean) for ci, rs in tab.rs.items()}


if __name__ == "__main__":
    src = select_source()
    d = radical_deviation()
    v = np.array(list(d.values()))
    print(f"cible {TARGET_DATE} → source sélectionnée (règle LOO, présidentielle la + récente "
          f"avant la cible) : {src}")
    print(f"circos couvertes : {len(d)}")
    print(f"déviation part radicale-dans-la-gauche — écart-type {v.std():.3f}, "
          f"min {v.min():+.3f}, max {v.max():+.3f}")
    print(f"RAD_GAIN = {RAD_GAIN} · niveau national = sondages (scenarios_2027) ; "
          f"validation de la règle : src/lfi_geo_loo.py")
