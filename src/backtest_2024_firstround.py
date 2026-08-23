"""Complète le backtest de sièges 2024 par les circos gagnées au **1er tour**.

`backtest_2024_seats` n'évalue que les circos passées au **2nd tour** (501) : le modèle de
sièges y modélise un 2nd tour, or 76 circos ont été emportées dès le 1er tour (majorité absolue,
pas de 2nd tour) — surtout des sièges SÛRS (RN dans le Nord, NFP à Paris/en banlieue). Les
exclure fait porter la justesse affichée sur le sous-ensemble le plus DUR (les circos disputées).

Ici on couvre ces 76 : vainqueur réel = bloc du candidat élu au 1er tour (bloc majoritaire du
1er tour), prédiction = `seat_winner` sur les parts de 1er tour réelles. On reporte la justesse
sur les circos de 1er tour, et la justesse COMBINÉE sur l'ensemble des 577 — pour ne pas laisser
croire que la validation ignore un septième des sièges.

    python3 -u -m src.backtest_2024_firstround
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import winnability_2027 as W
from src.backtest_2024_seats import CAND, MASTER24, _bloc, _first_round_and_winner


def _t1_majority_bloc(loc2circo: dict) -> pd.Series:
    """Bloc arrivé en tête au 1er tour par circo (proxy du bloc du candidat élu en cas de
    victoire au 1er tour : le candidat majoritaire appartient au bloc dominant)."""
    c = pq.read_table(CAND, columns=["id_election", "id_brut_miom", "nuance", "voix"]).to_pandas()
    c = c[c.id_election == "2024_legi_t1"].copy()
    c["circo"] = c.id_brut_miom.map(loc2circo)
    c["bloc"] = c.nuance.map(_bloc)
    c = c.dropna(subset=["circo", "bloc"])
    agg = c.groupby(["circo", "bloc"]).voix.sum().reset_index()
    return agg.loc[agg.groupby("circo").voix.idxmax()].set_index("circo").bloc


def summary() -> dict:
    t1, winner, idx = _first_round_and_winner()
    bm = pd.read_parquet(MASTER24, columns=["location", "circo"]).dropna(subset=["circo"])
    loc2circo = dict(zip(bm.location, bm.circo))
    t1maj = _t1_majority_bloc(loc2circo)

    # Contesté (2nd tour) : justesse brute (nb de circos) sur idx.
    con_ok = 0
    for c in idx:
        t = t1[c]
        p = W.seat_winner(t["G"], t["CD"], t["ED"], t["AB"], "union", 1.0, False)
        con_ok += int(p == winner[c])

    # 1er tour (pas de 2nd tour) : circos de t1 hors idx, vainqueur = bloc majoritaire T1.
    fr = [c for c in t1 if c not in winner.index and c in t1maj.index]
    fr_ok = 0
    by = {"G": [0, 0], "CD": [0, 0], "ED": [0, 0]}
    for c in fr:
        t, a = t1[c], t1maj[c]
        p = W.seat_winner(t["G"], t["CD"], t["ED"], t["AB"], "union", 1.0, False)
        by[a][1] += 1
        by[a][0] += int(p == a)
        fr_ok += int(p == a)

    n_con, n_fr = len(idx), len(fr)
    return {
        "contested": {"n": n_con, "correct": con_ok, "accuracy": round(100 * con_ok / n_con, 1)},
        "first_round": {"n": n_fr, "correct": fr_ok,
                        "accuracy": round(100 * fr_ok / n_fr, 1) if n_fr else None,
                        "by_bloc": by},
        "all": {"n": n_con + n_fr, "correct": con_ok + fr_ok,
                "accuracy": round(100 * (con_ok + fr_ok) / (n_con + n_fr), 1)},
    }


if __name__ == "__main__":
    r = summary()
    print(f"Contesté (2nd tour) : {r['contested']['correct']}/{r['contested']['n']} "
          f"= {r['contested']['accuracy']} %")
    print(f"1er tour (sièges sûrs) : {r['first_round']['correct']}/{r['first_round']['n']} "
          f"= {r['first_round']['accuracy']} %  (par bloc {r['first_round']['by_bloc']})")
    print(f"ENSEMBLE 577 : {r['all']['correct']}/{r['all']['n']} = {r['all']['accuracy']} %")
