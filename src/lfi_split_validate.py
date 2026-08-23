"""Validation du PARTAGE des sièges de gauche entre pôles (radical/LFI vs sociaux-démocrates).

En configuration divisée, le modèle scinde le bloc de gauche en pôles (`_left_candidates`) :
split2 = [g·rad, g·(1−rad)], où `rad` = part radicale (LFI), défaut 0,369 (sondages), modulée
localement par `+0,006·dG` (le pôle radical pèse davantage là où la gauche est forte). Le siège
de gauche va au pôle qualifié le plus fort. Comme rad<0,5, le pôle social-démocrate l'emporte
partout SAUF là où la modulation fait passer rad>0,5 — c.-à-d. les bastions de gauche, où LFI
est effectivement le plus implanté. On vérifie ici que ce mécanisme produit une part de sièges
LFI RÉALISTE.

CONCLUSION (voir aussi la mesure de variance spatiale ci-dessous) :

  DÉFAUT ACTUEL cassé. À `radical_share = 0,369`, le modèle n'attribue que ~17 % des sièges de
  gauche au pôle radical, et la courbe est une FALAISE : 0,40→33 %, 0,45→70 %, 0,50→100 %. Cause :
  la modulation `+0,006·dG` donne une part radicale quasi CONSTANTE entre circos (écart-type
  ~0,02), alors que la part réelle de LFI DANS la gauche varie fortement d'une circo à l'autre
  (écart-type mesuré ~0,17 en 2017 : bastions de banlieue/Marseille vs reste). Le pôle radical ne
  gagne donc un siège que lorsque le curseur pousse rad>0,5 PARTOUT d'un coup → bascule brutale.

  REPÈRE réel. 2024 (LFI 74/175 ≈ 42 %) est un artefact d'UNION NÉGOCIÉE (candidatures réparties
  parti par parti), pas une compétition de 1er tour — mauvais étalon pour un scénario DIVISÉ. Le
  seul vrai « gauche divisée » avec étiquettes distinctes est 2017 : LFI ~17 sièges ≈ 29 % des
  sièges de gauche pour une part radicale du vote de gauche de ~0,52. C'est l'étalon d'un scénario
  divisé : à part de VOIX donnée, la part de SIÈGES radicale en est proche (winner-take-all par
  bastion), NON largement supérieure.

  FIX RETENU (cf. src/radical_spatial.py). Remplacer la modulation linéaire par la variance
  spatiale RÉELLE et RÉCENTE : part LFI-dans-la-gauche par circo mesurée aux EUROPÉENNES 2024 (le
  scrutin divisé le plus récent), recentrée sur la moyenne du curseur, dispersion appliquée telle
  quelle (RAD_GAIN 1,0, sans calage sur une cible). Supprime la falaise, fonde le motif sur des
  données contemporaines → ~27 % des sièges de gauche à LFI en divisé (≠ 42 % de 2024, artefact
  d'union négociée). Les chiffres ci-dessous illustrent le défaut de l'ANCIEN modèle (0,006·dG).

    python3 -u -m src.lfi_split_validate
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src import winnability_2027 as W

CIRCO = Path("report_app/2027/data/circo.json")


def _left_pole(g, cd, ed, ab, rad):
    """Rejoue seat_winner en identifiant le pôle de gauche vainqueur (0 = radical/LFI, 1 = soc-dém).
    Renvoie None si la gauche ne gagne pas le siège."""
    turnout = max(0.05, 1 - ab / 100.0)
    thr = 12.5 / turnout
    if W.seat_winner(g, cd, ed, ab, "split2", rad, False) != "G":
        return None
    left = W._left_candidates(g, "split2", rad)
    cands = sorted(left + [cd, ed], reverse=True)
    second = cands[1]
    ql = [(p, i) for i, p in enumerate(left) if p >= second - 1e-9 or p >= thr]
    if ql:
        return max(ql, key=lambda x: x[0])[1]
    return int(np.argmax(left))


def main():
    c = json.loads(CIRCO.read_text())
    keys = ("r24G", "r24CD", "r24ED", "r24AB")
    mask = [all(c[k][i] is not None for k in keys) for i in range(len(c["id"]))]
    G = np.array([c["r24G"][i] for i in range(len(mask)) if mask[i]], float)
    CD = np.array([c["r24CD"][i] for i in range(len(mask)) if mask[i]], float)
    ED = np.array([c["r24ED"][i] for i in range(len(mask)) if mask[i]], float)
    AB = np.array([c["r24AB"][i] for i in range(len(mask)) if mask[i]], float)
    ins = np.array([c["ins"][i] for i in range(len(mask)) if mask[i]], float)
    natG = float(np.average(G, weights=ins))
    dG = G - natG  # déviation de gauche par circo (repère de la modulation)
    print(f"Rejeu gauche divisée sur {len(G)} circos (parts 1er tour réelles 2024) ; "
          f"gauche nationale = {natG:.1f} %")
    print(f"Repère réel 2024 : ~42 % des sièges de gauche à LFI (74/175), pour ~37 % du vote de gauche.\n")
    print(f"  {'rad_share':>9} | {'sièges G':>8} {'dont LFI':>8} {'dont s-dém':>10} | {'part LFI sièges':>15}")
    for rs in [0.30, 0.35, 0.369, 0.40, 0.45, 0.50]:
        radical = socdem = 0
        for i in range(len(G)):
            rad = min(0.68, max(0.12, rs + 0.006 * dG[i]))
            p = _left_pole(G[i], CD[i], ED[i], AB[i], rad)
            if p == 0:
                radical += 1
            elif p == 1:
                socdem += 1
        tot = radical + socdem
        share = radical / tot if tot else float("nan")
        mark = "  ⟵ défaut" if abs(rs - 0.369) < 1e-6 else ""
        print(f"  {rs:>9.3f} | {tot:>8} {radical:>8} {socdem:>10} | {share*100:>14.1f}%{mark}")


if __name__ == "__main__":
    main()
