"""Motif spatial de la part RADICALE (LFI) DANS le vote de gauche, par circonscription.

Le modèle de sièges divise le bloc de gauche en pôles (radical/LFI vs sociaux-démocrates) selon
une part `rad`. Le curseur en fixe la MOYENNE nationale (sondages législatifs 2025 : LFI ~37 %
du vote de gauche) ; le MOTIF par circo — où LFI sur/sous-performe au sein de la gauche — vient
des données.

SOURCE = 2024 EUROPÉENNES (1er tour, juin 2024) : le scrutin à gauche divisée le plus RÉCENT où
chaque liste a son étiquette (LFI-Aubry vs PS·PP-Glucksmann vs EELV vs PCF), deux mois avant les
législatives 2024. Bien plus récent que 2017 (dernière législative divisée), et sa géographie
intra-gauche est quasi contemporaine du scrutin visé. On mesure la part LFI du vote de gauche
par circo, on la recentre (déviation de moyenne pondérée ≈ 0) : le curseur fournit le niveau, ce
motif fournit la forme. `RAD_GAIN` = 1,0 : on applique la dispersion MESURÉE telle quelle, sans
amplification (aucun calage sur une cible de sièges). À la part LFI sondée (~37 %), cela donne
~27 % des sièges de gauche à LFI en gauche divisée — cohérent : un pôle minoritaire convertit
moins que sa part de voix (le pôle le plus fort d'une circo rafle le siège).

    python3 -u -m src.radical_spatial   # imprime moyenne/écart-type + couverture
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.reunif_measure import _commune2circo, CAND

SOURCE = "2024_euro_t1"
# Listes de gauche aux européennes 2024. Pôle radical = liste LFI (Aubry) uniquement — c'est la
# définition du pôle « radical » du modèle (le reste : PS·Place publique, EELV, PCF, div. gauche).
_RAD = {"LFI"}
_LEFT = {"LFI", "LUG", "LVEC", "LCOM", "LECO", "LDVG", "LRDG", "LEXG"}

# Dispersion appliquée telle que mesurée (pas d'amplification) : rad_circo =
# clamp(curseur + RAD_GAIN · déviation_2024E, 0.05, 0.95).
RAD_GAIN = 1.0


def radical_deviation() -> dict[str, float]:
    """{circo → déviation de la part LFI-dans-la-gauche} (recentrée, moyenne pondérée ≈ 0),
    mesurée sur les européennes 2024, circo au découpage actuel (mapping commune majoritaire)."""
    c2c = _commune2circo(drop_split=True)
    c = pd.read_parquet(CAND, columns=["id_election", "code_commune", "nuance", "voix"])
    c = c[c.id_election == SOURCE].copy()
    c["circo"] = c.code_commune.astype(str).map(c2c)
    c = c.dropna(subset=["circo"])
    c = c[c.nuance.isin(_LEFT)]
    c["p"] = np.where(c.nuance.isin(_RAD), "R", "S")
    tab = c.groupby(["circo", "p"]).voix.sum().unstack("p").fillna(0.0)
    tab["left"] = tab.get("R", 0.0) + tab.get("S", 0.0)
    tab = tab[tab.left > 0]
    tab["rs"] = tab.get("R", 0.0) / tab["left"]
    mean = float(np.average(tab.rs, weights=tab.left))
    return {ci: float(rs - mean) for ci, rs in tab.rs.items()}


if __name__ == "__main__":
    d = radical_deviation()
    v = np.array(list(d.values()))
    print(f"source : {SOURCE} | circos couvertes : {len(d)}")
    print(f"déviation part LFI-dans-la-gauche — écart-type {v.std():.3f}, "
          f"min {v.min():+.3f}, max {v.max():+.3f}")
    print(f"RAD_GAIN = {RAD_GAIN} (dispersion appliquée telle quelle)")
