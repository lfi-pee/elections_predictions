"""Décote « candidat → parti » À HORIZON ÉGAL : de la part de Mélenchon dans le vote de gauche
(sondages présidentiels, h mois avant le 1er tour) à la part de LFI dans le vote de gauche au
scrutin de liste/législatif suivant.

POURQUOI. La part radicale nationale (`radical_share`, `scenarios_2027`) vient des sondages
présidentiels 2027 (seule série encore vivante). Or le vote présidentiel personnel de Mélenchon
surestime largement ce que la liste LFI obtient ensuite dans la gauche : résultats 2012 0,25 →
0,14 (legi), 2017 0,71 → 0,39 (legi), 2022 0,69 → 0,29 (euro 2024). Utiliser le rapport brut
des sondages ferait aussi mal que la moyenne plate ; il faut une décote k.

POURQUOI À HORIZON ÉGAL. Mélenchon monte tard (2012 : 6 % à l'automne 2011, 11 % le jour du
vote ; 2022 : 10 % → 22 %). Une décote mesurée sur les sondages des derniers mois avant le
scrutin ne s'applique donc pas aux sondages de septembre de l'année précédente, où se trouve la
prévision aujourd'hui. On mesure k(h) : pour chaque présidentielle P suivie d'un scrutin à
gauche divisée S (2012→legi 2012, 2017→legi 2017, 2022→euro 2024), x_P(h) = Mélenchon/gauche
moyenné par hypothèse sur les 12 mois finissant h mois avant P (même fenêtre que l'ancre) ;
y_S = pôle radical/gauche au résultat de S ; k_pli(h) = y_S / x_P(h). La décote servie est
k(h_now), h_now = distance entre le sondage 2027 le plus récent et le 1er tour 2027 ; elle se
rapproche de k(0) à mesure que la campagne avance (les sondages absorbent la montée tardive).

VALIDATION. Pour chaque h, LOO 3 plis : chaque pli prédit avec le k moyen des deux autres ;
comparé au null (moyenne des y des autres plis) et au rapport brut (k = 1). Trois observations :
une direction validée, une précision mince — servi comme décote, avec sa fourchette.

Sources : data/polls/presidentielle/{2012,2017}/presidentielle_YYYY_t1_tidy.csv
(`scrape_pres_history`), nsppolls 2022 (hypothèses identifiées), résultats (parquet candidats).
Sortie : data/polls/lfi_pres_discount.json (lu par `scenarios_2027`).

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
from src.reunif_measure import CAND

OUT = Path("data/polls/lfi_pres_discount.json")
TIDY = "data/polls/presidentielle/{year}/presidentielle_{year}_t1_tidy.csv"
NSP_2022 = Path("data/polls/presidentielle/2022/nsppolls_presidentielle_2022.csv")

# Présidentielle → scrutin suivant à gauche divisée avec pôle radical étiqueté (ids de lfi_geo_loo).
PAIRS = [("2012_pres_t1", "2012_legi_t1"), ("2017_pres_t1", "2017_legi_t1"), ("2022_pres_t1", "2024_euro_t1")]
# 1er tour. 2027 : date présumée (avril 2027) — une semaine de décalage ne change rien de
# matériel à l'horizon (exprimé en mois).
PRES_T1 = {2012: "2012-04-22", 2017: "2017-04-23", 2022: "2022-04-10", 2027: "2027-04-11"}
WINDOW_MONTHS = 12
CURVE_H = list(range(0, 13))
NSP_LEFT = "insoumise|socialiste|EE-LV|communiste|Lutte ouvrière|NPA|gauche"


def _result_ratio(cand: pd.DataFrame, eid: str) -> float:
    _, _, _, rad, left, col = next(o for o in OBS if o[0] == eid)
    d = cand[cand.id_election == eid]
    return float(d[d[col].isin(rad)].voix.sum() / d[d[col].isin(left)].voix.sum())


def hypothesis_rows(year: int) -> pd.DataFrame:
    """Une ligne par hypothèse sondée : date de fin de terrain, voix Mélenchon, total gauche."""
    if year == 2022:
        d = pd.read_csv(NSP_2022)
        d = d[(d.tour == "Premier tour") & d.parti.fillna("").str.contains(NSP_LEFT, regex=True)].copy()
        d["date"] = pd.to_datetime(d.fin_enquete)
        d["mel"] = np.where(d.candidat.str.contains("Mélenchon"), d.intentions, 0.0)
        g = d.groupby(["id", d.hypothese.fillna("").astype(str)]).agg(date=("date", "max"), mel=("mel", "sum"), left=("intentions", "sum"))
    else:
        d = pd.read_csv(TIDY.format(year=year), comment="#")
        d = d[(d.bloc == "G") & ~d.candidat.fillna("").str.contains("MACRON")].copy()
        d["date"] = pd.to_datetime(d.date_fin)
        d["mel"] = np.where(d.candidat.fillna("").str.contains("MÉLENCHON"), d.valeur, 0.0)
        g = d.groupby(["institut", "date_fin", "hypothese"]).agg(date=("date", "max"), mel=("mel", "sum"), left=("valeur", "sum"))
    g = g[(g.mel > 0) & (g.left > 0)].reset_index(drop=True)
    g["ratio"] = g.mel / g.left
    return g


def poll_ratio(rows: pd.DataFrame, t1: str, h_months: float, window: int = WINDOW_MONTHS) -> tuple[float, int]:
    hi = pd.Timestamp(t1) - pd.Timedelta(days=round(h_months * 30.44))
    lo = hi - pd.DateOffset(months=window)
    w = rows[(rows.date > lo) & (rows.date <= hi)]
    return (float(w.ratio.mean()) if len(w) else float("nan")), int(len(w))


def _rmse(errs) -> float:
    return float(np.sqrt(np.mean(np.square(errs))))


def _loo(folds: list[dict], xkey: str) -> dict:
    ys = [f["y"] for f in folds]
    null = [np.mean([o["y"] for j, o in enumerate(folds) if j != i]) - f["y"] for i, f in enumerate(folds)]
    kx = [np.mean([o["y"] / o[xkey] for j, o in enumerate(folds) if j != i]) * f[xkey] - f["y"] for i, f in enumerate(folds)]
    raw = [f[xkey] - f["y"] for f in folds]
    return {"null": round(_rmse(null), 4), "raw_ratio": round(_rmse(raw), 4), "k_x_ratio": round(_rmse(kx), 4),
            "errors_k": [round(float(e), 4) for e in kx]}


def compute() -> dict:
    cand = pd.read_parquet(CAND, columns=["id_election", "nuance", "nom", "voix"])
    rows = {int(p[:4]): hypothesis_rows(int(p[:4])) for p, _ in PAIRS}
    rows27 = hypothesis_rows(2027)
    latest = rows27.date.max()
    h_now = (pd.Timestamp(PRES_T1[2027]) - latest).days / 30.44

    def folds_at(h: float) -> list[dict]:
        out = []
        for pres, nxt in PAIRS:
            y = int(pres[:4])
            x, n = poll_ratio(rows[y], PRES_T1[y], h)
            out.append({"pres": pres, "next": nxt, "h": round(h, 2), "x": round(x, 4), "n_hyp": n,
                        "y": round(_result_ratio(cand, nxt), 4), "x_result": round(_result_ratio(cand, pres), 4)})
        for f in out:
            f["k"] = round(f["y"] / f["x"], 4)
        return out

    now = folds_at(h_now)
    curve = []
    for h in CURVE_H:
        fs = folds_at(h)
        curve.append({"h": h, "k": round(float(np.mean([f["k"] for f in fs])), 4),
                      "k_range": [round(min(f["k"] for f in fs), 4), round(max(f["k"] for f in fs), 4)],
                      "loo_rmse": _loo(fs, "x")["k_x_ratio"], "n_hyp": [f["n_hyp"] for f in fs]})
    k = float(np.mean([f["k"] for f in now]))
    x27, n27 = poll_ratio(rows27, latest.strftime("%Y-%m-%d"), 0)
    return {
        "k_poll": round(k, 4),
        "k_poll_range": [round(min(f["k"] for f in now), 4), round(max(f["k"] for f in now), 4)],
        "horizon_months": round(h_now, 2),
        "latest_poll_2027": latest.strftime("%Y-%m-%d"),
        "pres_t1_2027_assumed": PRES_T1[2027],
        "x_2027": round(x27, 4),
        "loo_rmse": _loo(now, "x"),
        "k_result": round(float(np.mean([f["y"] / f["x_result"] for f in now])), 4),
        "folds": now,
        "curve": curve,
        "window_months": WINDOW_MONTHS,
        "note": "k = moyenne des y/x(h) sur 3 plis (présidentielle → scrutin divisé suivant), x(h) = Mélenchon/gauche "
                "par hypothèse sur 12 mois finissant h mois avant le 1er tour, h = horizon actuel de la prévision 2027. "
                "LOO : chaque pli prédit avec le k des deux autres.",
    }


def load() -> dict:
    return json.loads(OUT.read_text())


if __name__ == "__main__":
    res = compute()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    print(f"Décote candidat → parti à horizon égal : h = {res['horizon_months']} mois "
          f"(dernier sondage 2027 {res['latest_poll_2027']}, 1er tour présumé {res['pres_t1_2027_assumed']})")
    for f in res["folds"]:
        print(f"  {f['pres']} → {f['next']}: x(h) {f['x']:.3f} ({f['n_hyp']} hyp.) | résultat prés. {f['x_result']:.3f} "
              f"| suivant {f['y']:.3f} | k {f['k']:.2f}")
    print(f"  RMSE LOO à h : {res['loo_rmse']}")
    print(f"  k servi = {res['k_poll']:.3f} (plis {res['k_poll_range']}) ; Mélenchon/gauche 2027 (12 mois) {res['x_2027']:.3f} "
          f"→ part LFI {res['k_poll'] * res['x_2027']:.3f}")
    print("  courbe k(h) :", " ".join(f"h{c['h']}={c['k']:.2f}/rmse{c['loo_rmse']:.3f}" for c in res["curve"]))
