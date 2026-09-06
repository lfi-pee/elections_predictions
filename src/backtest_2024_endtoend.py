"""Backtest **de bout en bout** (« à l'aveugle ») du modèle sur les législatives 2024.

Le backtest oracle (`backtest_2024_seats`) alimente le modèle de sièges avec les parts de
1er tour **réelles** 2024 : il n'isole que l'erreur du modèle de sièges. Ici on ferme la
chaîne complète :

  1. On **retire entièrement 2024 de l'entraînement** du modèle de déviation (il n'apprend
     que sur 2002→2022). Il **prédit** alors le motif spatial (déviation au national) de
     chaque bureau en 2024, à l'aveugle — exactement comme il prédira 2027.
  2. On pose le niveau **national** 2024 réel (l'ancre que le site délègue aux curseurs /
     sondages ; on ne teste pas ici la prévision nationale, seulement le motif local + les
     sièges). `pred_b = national_b + déviation_prédite_b`, borné à [0, 100].
  3. On agrège au niveau circonscription, on renormalise à 3 blocs, et on fait tourner le
     **modèle de sièges** (`winnability_2027`, gauche unie comme le NFP 2024).
  4. On compare le vainqueur projeté au **vrai** vainqueur du 2nd tour.

C'est LA validation qu'un analyste exigera : « tournez-le à l'aveugle avant 2024 — combien
de sièges a-t-il appelés correctement ? ». Le chiffre affiché par le site restait jusqu'ici
un backtest oracle ; celui-ci mesure la prévision complète.

    python3 -u -m src.backtest_2024_endtoend
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src import winnability_2027 as W
from src.cross_type_dev import BLOCKS_ABS, TARGET_COLS, load_cross_type_data
from src.forecast_2027 import PCA_K, _feat_cols, _transform, fit_block
from src.backtest_2024_seats import MASTER24, CAND, _bloc

VAL_DATE = 2024.5
VAL_TYPE = "Legislatives_T1"
ABBR = {"Gauche": "G", "Centre+Droite": "CD", "Extreme_Droite": "ED", "Abstention": "AB"}


def _ground_truth():
    """Vrai vainqueur du 2nd tour par circo + parts de 1er tour réelles (act_) par circo,
    strictement comme `backtest_2024_seats` (même source, même mapping)."""
    bm = pd.read_parquet(
        MASTER24,
        columns=["location", "circo", "inscrits", "act_G", "act_CD", "act_ED", "act_AB"],
    )
    bm = bm[bm.circo.notna()].copy()

    c2 = pq.read_table(
        CAND, columns=["id_election", "id_brut_miom", "nuance", "voix"]
    ).to_pandas()
    c2 = c2[c2.id_election == "2024_legi_t2"].copy()
    loc2circo = dict(zip(bm.location, bm.circo))
    c2["circo"] = c2.id_brut_miom.map(loc2circo)
    c2["bloc"] = c2.nuance.map(_bloc)
    c2 = c2.dropna(subset=["circo", "bloc"])
    gt = c2.groupby(["circo", "bloc"]).voix.sum().reset_index()
    winner = gt.loc[gt.groupby("circo").voix.idxmax()].set_index("circo").bloc
    return bm, winner


def backtest() -> dict:
    df, demo_indicators, national_means, poll_feats = load_cross_type_data(Path("data"))
    demo_cols, feat_all = _feat_cols(demo_indicators)

    legi = df[df["election_type"] == VAL_TYPE].copy()
    raw_lag = [f"dev_{b}_lag{k}" for b in BLOCKS_ABS for k in (1, 2)]
    legi = legi.dropna(subset=demo_indicators + raw_lag + [f"dev_{b}" for b in BLOCKS_ABS])

    # ── Hold-out : entraînement 2002→2022 STRICT ; cible = les vraies lignes 2024 ──
    train = legi[legi["date_float"] < VAL_DATE - 0.1].copy()
    target = legi[np.isclose(legi["date_float"], VAL_DATE, atol=1e-2)].copy()
    target = target.dropna(subset=feat_all)
    print(
        f"  Entraînement (2024 exclu) : {len(train):,} lignes, "
        f"plis {sorted(train.date_float.round(2).unique())}"
    )
    print(f"  Cible 2024 (aveugle) : {len(target):,} bureaux")

    ins = target["inscrits"].to_numpy(np.float64)
    ins = np.where(np.isfinite(ins) & (ins > 0), ins, 1.0)

    pred = {"location": target["location"].to_numpy()}
    # Le bloc « Autre » n'a pas de colonne réelle 2024 dans MASTER24 : le backtest de bout en
    # bout valide la chaîne G/CD/ED/Ab (les blocs d'axe). On l'exclut donc ici.
    e2e_cols = [c for c in TARGET_COLS if c != "Other"]
    for tc in e2e_cols:
        k = PCA_K[tc]
        scaler, pca, ridge = fit_block(tc, train, feat_all, demo_cols, k)
        dev = ridge.predict(_transform(scaler, pca, len(demo_cols), target, feat_all))
        dev = dev - np.average(dev, weights=ins)  # motif spatial seul (moyenne pondérée = 0)
        pred[f"dev_{tc}"] = dev

    pbv = pd.DataFrame(pred)

    # ── Niveau national réel 2024 (échelle act_, ancre « oracle national ») + circo ──
    bm, winner = _ground_truth()
    m = pbv.merge(
        bm[["location", "circo", "inscrits", "act_G", "act_CD", "act_ED", "act_AB"]],
        on="location", how="inner",
    )
    w = m.inscrits.to_numpy(float)
    nat = {tc: float(np.average(m[f"act_{ABBR[tc]}"], weights=w)) for tc in e2e_cols}

    # pred_b = national_b + déviation prédite (motif spatial), borné.
    for tc in e2e_cols:
        m[f"p_{ABBR[tc]}"] = np.clip(nat[tc] + m[f"dev_{tc}"], 0.0, 100.0)

    # ── Agrégation circo (pondérée inscrits) ──
    def wavg(g, col):
        wg = g.inscrits.to_numpy(float)
        return float(np.average(g[col], weights=wg)) if wg.sum() else float(g[col].mean())

    rows = []
    for c, g in m.groupby("circo"):
        rows.append(dict(
            circo=c, ins=float(g.inscrits.sum()),
            pG=wavg(g, "p_G"), pCD=wavg(g, "p_CD"), pED=wavg(g, "p_ED"), pAB=wavg(g, "p_AB"),
            aG=wavg(g, "act_G"), aCD=wavg(g, "act_CD"), aED=wavg(g, "act_ED"), aAB=wavg(g, "act_AB"),
        ))
    circo = pd.DataFrame(rows)

    # ── Diagnostic 1er tour : MAE des parts circo prédites vs réelles ──
    mae = {b: float(np.average(np.abs(circo[f"p{b}"] - circo[f"a{b}"]),
                               weights=circo.ins)) for b in ("G", "CD", "ED")}

    # ── Modèle de sièges sur les parts PRÉDITES (aveugle), gauche unie ──
    idx = circo[circo.circo.isin(winner.index)]
    actual = {b: 0 for b in ("G", "CD", "ED")}
    model = {b: 0 for b in ("G", "CD", "ED")}
    ok = wsum = ok_n = 0.0
    for _, r in idx.iterrows():
        a = winner[r.circo]
        actual[a] += 1
        p = W.seat_winner(r.pG, r.pCD, r.pED, r.pAB, "union", 1.0, False)
        model[p] += 1
        wsum += r.ins
        ok += r.ins if p == a else 0.0
        ok_n += 1.0 if p == a else 0.0

    return {
        "n_circo": int(len(idx)),
        "actual": actual,
        "model": model,
        "accuracy": round(100 * ok / wsum, 1),
        "accuracy_seats": round(100 * ok_n / len(idx), 1),
        "n_correct": int(ok_n),
        "mae_t1": {k: round(v, 2) for k, v in mae.items()},
        "national_2024": {ABBR[tc]: round(nat[tc], 1) for tc in e2e_cols},
    }


if __name__ == "__main__":
    from src import backtest_2024_seats as O

    print("=== Backtest de bout en bout (2024 retiré de l'entraînement) ===")
    r = backtest()
    print(f"\ncircos : {r['n_circo']}   justesse (pondérée inscrits) : {r['accuracy']}%"
          f"   |   sièges bien appelés : {r['n_correct']}/{r['n_circo']} ({r['accuracy_seats']}%)")
    print(f"  réel   G/CD/ED = {r['actual']['G']}/{r['actual']['CD']}/{r['actual']['ED']}")
    print(f"  modèle G/CD/ED = {r['model']['G']}/{r['model']['CD']}/{r['model']['ED']}")
    print(f"  MAE parts 1er tour circo (pts) : {r['mae_t1']}")
    print(f"  national 2024 posé : {r['national_2024']}")

    print("\n=== Rappel : backtest ORACLE (parts 1er tour réelles) ===")
    o = O.backtest()
    print(f"circos : {o['n_circo']}   justesse : {o['accuracy']}%")
    print(f"  modèle G/CD/ED = {o['model']['G']}/{o['model']['CD']}/{o['model']['ED']}")
