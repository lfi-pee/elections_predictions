"""Décote « candidat → parti » : de la part de Mélenchon dans le vote de gauche (sondages
présidentiels) à la part de LFI dans le vote de gauche au scrutin de liste/législatif suivant.

POURQUOI. La part radicale nationale (`radical_share`) posée par `scenarios_2027` vient des
sondages présidentiels 2027 (seule série encore vivante). Or le vote présidentiel personnel de
Mélenchon surestime largement ce que la liste LFI obtient ensuite dans la gauche : en résultats,
2012 0,25 → 0,14 (legi), 2017 0,71 → 0,39 (legi), 2022 0,69 → 0,29 (euro 2024). Utiliser le
rapport brut des sondages présidentiels ferait AUSSI mal que la moyenne plate (validation ci-
dessous) ; il faut une décote k.

MESURE (LOO, 3 plis). Pour chaque présidentielle P suivie d'un scrutin à gauche divisée S
(2012→legi 2012, 2017→legi 2017, 2022→euro 2024) : x_P = Mélenchon/gauche dans les sondages des
3 derniers mois avant P (candidats finaux), y_S = pôle radical/gauche au résultat de S.
k_pli = y_S / x_P ; la prédiction de chaque pli avec le k moyen des deux AUTRES plis donne le RMSE
LOO, comparé au null (moyenne des y des autres plis) et au rapport brut (k = 1).

Résultat (2026-09) : null 0,15 ; brut 0,15 (échec) ; k × sondages 0,07 ; k ≈ 0,63 (0,53–0,78).
Trois observations : validé dans la direction, mince en précision — servi comme décote, avec sa
fourchette, jamais comme certitude.

Sortie : `data/polls/lfi_pres_discount.json` (lu par `scenarios_2027`).

    python3 -u -m src.lfi_pres_discount
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.lfi_geo_loo import OBS
from src.load_polls import load_poll_tokens
from src.reunif_measure import CAND

OUT = Path("data/polls/lfi_pres_discount.json")

# Présidentielle → scrutin suivant à gauche divisée avec pôle radical étiqueté (ids de lfi_geo_loo).
PAIRS = [("2012_pres_t1", "2012_legi_t1"), ("2017_pres_t1", "2017_legi_t1"), ("2022_pres_t1", "2024_euro_t1")]
# Candidats de gauche FINAUX de chaque présidentielle (les fenêtres longues mêlent des candidats
# hypothétiques qui gonflent le dénominateur) ; fenêtre = 3 derniers mois avant le 1er tour.
FINAL_LEFT = {2012: ["HOLLANDE", "MÉLENCHON", "JOLY", "POUTOU", "ARTHAUD"],
              2017: ["HAMON", "MÉLENCHON", "JADOT", "POUTOU", "ARTHAUD"],
              2022: ["HIDALGO", "MÉLENCHON", "JADOT", "ROUSSEL", "POUTOU", "ARTHAUD"]}
WINDOW_YEARS = 0.25
PRES_DATE = 0.3  # la présidentielle = printemps (cf. radical_spatial._year)


def _result_ratio(cand: pd.DataFrame, eid: str) -> float:
    _, _, _, rad, left, col = next(o for o in OBS if o[0] == eid)
    d = cand[cand.id_election == eid]
    return float(d[d[col].isin(rad)].voix.sum() / d[d[col].isin(left)].voix.sum())


def _poll_ratio(polls: pd.DataFrame, year: int) -> tuple[float, int]:
    n = polls[(polls.location == "National") & (polls.election_type.astype(str) == "Presidentielle_T1")]
    hi = year + PRES_DATE
    w = n[(n.date_float >= hi - WINDOW_YEARS) & (n.date_float <= hi)].copy()
    w["cand"] = w.candidate.astype(str).str.upper()
    w = w[w.cand.str.contains("|".join(FINAL_LEFT[year]))]
    ratios = []
    for _, d in w.groupby(["date_float", "metric_type"]):
        left = d.value.sum()
        mel = d[d.cand.str.contains("MÉLENCHON")].value.sum()
        if left > 0 and mel > 0:
            ratios.append(mel / left)
    return float(np.mean(ratios)), len(ratios)


def compute() -> dict:
    cand = pd.read_parquet(CAND, columns=["id_election", "nuance", "nom", "voix"])
    polls = load_poll_tokens(Path("data"))
    folds = []
    for pres, nxt in PAIRS:
        year = int(pres[:4])
        x_poll, n_polls = _poll_ratio(polls, year)
        folds.append({"pres": pres, "next": nxt, "n_polls": n_polls,
                      "x_poll": round(x_poll, 4), "x_result": round(_result_ratio(cand, pres), 4),
                      "y": round(_result_ratio(cand, nxt), 4)})
    for f in folds:
        f["k_poll"] = round(f["y"] / f["x_poll"], 4)
        f["k_result"] = round(f["y"] / f["x_result"], 4)

    def rmse(pred):
        return float(np.sqrt(np.mean([(p - f["y"]) ** 2 for p, f in zip(pred, folds)])))

    others = lambda i: [f for j, f in enumerate(folds) if j != i]  # noqa: E731
    loo = {
        "null": rmse([np.mean([o["y"] for o in others(i)]) for i in range(len(folds))]),
        "raw_poll_ratio": rmse([f["x_poll"] for f in folds]),
        "k_x_poll": rmse([np.mean([o["k_poll"] for o in others(i)]) * f["x_poll"] for i, f in enumerate(folds)]),
        "k_x_result": rmse([np.mean([o["k_result"] for o in others(i)]) * f["x_result"] for i, f in enumerate(folds)]),
    }
    k = float(np.mean([f["k_poll"] for f in folds]))
    return {
        "k_poll": round(k, 4),
        "k_poll_range": [round(min(f["k_poll"] for f in folds), 4), round(max(f["k_poll"] for f in folds), 4)],
        "k_result": round(float(np.mean([f["k_result"] for f in folds])), 4),
        "loo_rmse": {a: round(b, 4) for a, b in loo.items()},
        "folds": folds,
        "window_years": WINDOW_YEARS,
        "note": "k = moyenne des y/x_poll sur 3 plis (présidentielle → scrutin divisé suivant) ; "
                "LOO : chaque pli prédit avec le k des deux autres.",
    }


def load() -> dict:
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    res = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print("Décote candidat → parti (Mélenchon/gauche aux sondages présidentiels → LFI/gauche au scrutin suivant)")
    for f in res["folds"]:
        print(f"  {f['pres']} → {f['next']}: sondages {f['x_poll']:.3f} ({f['n_polls']} sondages) | résultat prés. "
              f"{f['x_result']:.3f} | suivant {f['y']:.3f} | k_poll {f['k_poll']:.2f}")
    print(f"  RMSE LOO : {res['loo_rmse']}")
    print(f"  k servi = {res['k_poll']:.3f}  (plis {res['k_poll_range']}) → {OUT}")
