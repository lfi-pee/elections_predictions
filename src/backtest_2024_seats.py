"""Backtest du modèle de SIÈGES sur le résultat réel des législatives 2024 — la validation
affichée par le bouton « Rejouer 2024 » du site.

On alimente le modèle de 2nd tour (`winnability_2027`) avec les parts de 1er tour **réelles**
2024 par circonscription (mode oracle : on isole l'erreur du modèle de sièges, pas celle de la
prévision de 1er tour), la gauche en configuration **unie** (comme le NFP en 2024), et on
compare le vainqueur projeté par circo au **vrai** vainqueur du 2nd tour (nuance du candidat
arrivé en tête, agrégée par circo depuis `candidats_results`).

    python3 -u -m src.backtest_2024_seats
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import winnability_2027 as W

MASTER24 = Path("data/report/bv_master.parquet")
CAND = Path("data/elections/agregees/candidats_results.parquet")
VOTE = ["G", "CD", "ED"]

# Nuances 2024 → bloc à 3 (les régionalistes/divers non mappés sont ignorés).
_ED = {"RN", "UDR", "UXD", "REC", "EXD"}
_G = {"UG", "DVG", "PS", "ECO", "SOC", "FI", "COM", "RDG", "LFI", "ECOLO"}
_CD = {"ENS", "RE", "LR", "DVD", "HOR", "DVC", "UDI", "MODEM", "NC", "LC"}


def _bloc(n: str) -> str | None:
    return "ED" if n in _ED else "G" if n in _G else "CD" if n in _CD else None


def backtest() -> dict:
    bm = pd.read_parquet(
        MASTER24,
        columns=["location", "circo", "inscrits", "act_G", "act_CD", "act_ED", "act_AB"],
    )
    bm = bm[bm.circo.notna()].copy()
    loc2circo = dict(zip(bm.location, bm.circo))

    def wa(g: pd.DataFrame, col: str) -> float:
        w = g.inscrits.to_numpy(float)
        return float(np.average(g[col], weights=w)) if w.sum() else float(g[col].mean())

    t1 = {
        c: dict(ins=float(g.inscrits.sum()), G=wa(g, "act_G"), CD=wa(g, "act_CD"),
                ED=wa(g, "act_ED"), AB=wa(g, "act_AB"))
        for c, g in bm.groupby("circo")
    }
    # Niveau national 2024 (parts d'exprimés, renormalisées 3 blocs) — sert le préréglage bouton.
    ins = bm.inscrits.to_numpy(float)
    nat = {b: float(np.average(bm[f"act_{b}"], weights=ins)) for b in VOTE}
    s = sum(nat.values())
    levels = {b: round(100 * nat[b] / s, 1) for b in VOTE}
    levels["ED"] = round(100 - levels["G"] - levels["CD"], 1)
    levels["AB"] = round(float(np.average(bm.act_AB, weights=ins)), 1)

    c2 = pq.read_table(
        CAND, columns=["id_election", "id_brut_miom", "nuance", "voix"]
    ).to_pandas()
    c2 = c2[c2.id_election == "2024_legi_t2"].copy()
    c2["circo"] = c2.id_brut_miom.map(loc2circo)
    c2["bloc"] = c2.nuance.map(_bloc)
    c2 = c2.dropna(subset=["circo", "bloc"])
    gt = c2.groupby(["circo", "bloc"]).voix.sum().reset_index()
    winner = gt.loc[gt.groupby("circo").voix.idxmax()].set_index("circo").bloc

    idx = [c for c in t1 if c in winner.index]
    actual = {b: 0 for b in VOTE}
    model = {b: 0 for b in VOTE}
    ok = wsum = 0.0
    for c in idx:
        t, a = t1[c], winner[c]
        actual[a] += 1
        p = W.seat_winner(t["G"], t["CD"], t["ED"], t["AB"], "union", 1.0, False)
        model[p] += 1
        wsum += t["ins"]
        ok += t["ins"] if p == a else 0.0
    return {
        "n_circo": len(idx),
        "actual": actual,
        "model": model,
        "accuracy": round(100 * ok / wsum, 1),
        "levels": levels,
    }


if __name__ == "__main__":
    r = backtest()
    print(f"circos: {r['n_circo']}  justesse (inscrits) : {r['accuracy']}%")
    print(f"  réel   G/CD/ED = {r['actual']['G']}/{r['actual']['CD']}/{r['actual']['ED']}")
    print(f"  modèle G/CD/ED = {r['model']['G']}/{r['model']['CD']}/{r['model']['ED']}")
    print(f"  niveau national 2024 (3 blocs) : {r['levels']}")
