"""Socle de données du site **2027** (prévision, curseurs nationaux, jouabilité circo).

Diffère de `report_data.py` (démonstration 2024) :

- On sert des **composantes de déviation** par bureau (`dev_*` centrées), pas des
  prédictions figées : le client calcule `pred_b = curseur_b + dev_b` en direct.
- Pas de « réel » ni de justesse mesurée (2027 n'a pas eu lieu) ; la preuve de compétence
  vient de la validation croisée passée (2024 en pli hold-out : justesse 83,3 %).
- On agrège la déviation par **commune** (couche dézoomée) et par **circonscription**
  (score de jouabilité 1→5, recalculé au curseur).

Entrée : `data/predictions_2027.csv` (cf. `forecast_2027.py`) + contexte repris de
`data/report/bv_master.parquet` (mêmes 70 083 bureaux). Sorties : `report_app/2027/data/`.

    python3 -u -m src.report_data_2027
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from src import scenarios_2027, winnability_2027

PRED = Path("data/predictions_2027.csv")
MASTER24 = Path("data/report/bv_master.parquet")
SUMMARY24 = Path("report_app/data/summary.json")
GAMMA24 = Path("report_app/data/gamma_curve.json")
CACHE = Path("data/report")
SERVED = Path("report_app/2027/data")

BLOCKS = {"Gauche": "G", "Centre+Droite": "CD", "Extreme_Droite": "ED", "Abstention": "AB"}
VOTE = ["G", "CD", "ED"]
CTX = [
    "inscrits", "code_departement", "code_commune", "libelle_commune",
    "circo", "abst_floor", "lat", "lon", "has_contour",
]


def load_master() -> pd.DataFrame:
    df = pd.read_csv(PRED)
    df["b"] = df["block"].map(BLOCKS)
    wide = {}
    for src, dst in (("dev_pred", "dev"), ("hw_90", "hw90")):
        w = df.pivot(index="location", columns="b", values=src)
        wide |= {f"{dst}_{b}": w[b] for b in w.columns}
    m = pd.DataFrame(wide).reset_index()
    fb = df.groupby("location")["lag_fallback"].first().rename("lag_fallback")
    m = m.merge(fb, on="location", how="left")
    m["lag_fallback"] = m["lag_fallback"].fillna(False).astype(bool)
    ctx = pd.read_parquet(MASTER24, columns=["location", *CTX])
    return m.merge(ctx, on="location", how="left")


def _wmean(col: pd.Series, w: pd.Series) -> float:
    ww = w.to_numpy(float)
    return float(np.average(col, weights=ww)) if ww.sum() else float(col.mean())


def aggregate(df: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Déviation moyenne pondérée par inscrits + plancher d'abstention agrégé, par clé."""
    recs = []
    for kv, g in df.groupby(keys):
        w = g.inscrits
        rec = dict(zip(keys, kv if isinstance(kv, tuple) else (kv,)))
        rec |= {f"d{b}": round(_wmean(g[f"dev_{b}"], w), 3) for b in ("G", "CD", "ED", "AB")}
        rec["ins"] = int(g.inscrits.sum())
        rec["af"] = round(_wmean(g.abst_floor, w), 2)
        rec["nbv"] = int(len(g))
        recs.append(rec)
    return pd.DataFrame(recs)


def build_communes(df: pd.DataFrame) -> pd.DataFrame:
    com = aggregate(df, ["code_commune"])
    first = df.groupby("code_commune").agg(
        nom=("libelle_commune", "first"),
        dept=("code_departement", "first"),
        lat=("lat", "mean"),
        lon=("lon", "mean"),
    ).reset_index()
    com = com.merge(first, on="code_commune", how="left").dropna(subset=["lat", "lon"])
    return com.reset_index(drop=True)


def build_circo(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df.circo.notna()].copy()
    cir = aggregate(sub, ["circo"])
    # Commune-ancre = plus gros vivier d'inscrits de la circo (pour nommer le point).
    anchor = (
        sub.groupby(["circo", "libelle_commune"]).inscrits.sum().reset_index()
        .sort_values("inscrits").groupby("circo").tail(1)
        .set_index("circo").libelle_commune
    )
    dept = sub.groupby("circo").code_departement.first()
    cir["nm"] = cir.circo.map(anchor)
    cir["dept"] = cir.circo.map(dept)
    return cir.reset_index(drop=True)


def circo_halfwidth(cv90: dict[str, float]) -> dict[str, float]:
    """Demi-largeur 90 % de l'erreur au niveau CIRCO, par bloc, pour la fourchette de sièges
    Monte-Carlo du site. Les intervalles conformes `cv90` sont calibrés par BUREAU ; on les
    ramène au grain circo par le rapport observé σ_circo/σ_bv des résidus 2024 (pred−réel). Ce
    rapport (~0,7) est bien supérieur à 1/√N : au sein d'une circo les erreurs du modèle sont
    fortement corrélées (même sociologie, mêmes candidats), elles ne se moyennent donc presque
    pas — appliquer 1/√N sous-estimerait grossièrement l'incertitude de sièges."""
    cols = ["circo", "inscrits"] + [f"{p}_{b}" for p in ("pred", "act") for b in VOTE]
    m = pd.read_parquet(MASTER24, columns=cols)
    m = m[m.circo.notna()].copy()
    out = {}
    for b in VOTE:
        resid_bv = (m[f"pred_{b}"] - m[f"act_{b}"]).to_numpy()
        circo_resid = m.groupby("circo").apply(
            lambda g: float(np.average(g[f"pred_{b}"] - g[f"act_{b}"],
                                       weights=g.inscrits.to_numpy(float)))
        )
        ratio = float(circo_resid.std() / (resid_bv.std() or 1.0))
        out[b] = round(cv90[b] * ratio, 1)
    return out


def scenario_means(scn: dict) -> dict[str, float]:
    return scn["means"]


def winnability_distribution(cir: pd.DataFrame, scn: dict) -> dict:
    """Répartition des circos par score 1→5 (pondérée en nombre et en inscrits) pour un
    scénario donné, au niveau national des présélections."""
    m = scenario_means(scn)
    counts = {s: 0 for s in range(1, 6)}
    ins = {s: 0 for s in range(1, 6)}
    ru = scn.get("right_union", False)
    cfg = scn["left_config"]
    seats = {"G": 0, "CD": 0, "ED": 0}
    for r in cir.itertuples():
        G, CD, ED, AB = m["G"] + r.dG, m["CD"] + r.dCD, m["ED"] + r.dED, m["AB"] + r.dAB
        G, CD, ED, AB = (min(100, max(0, v)) for v in (G, CD, ED, AB))
        # Part radicale modulée localement (miroir de compute.js).
        rad = 1.0 if cfg == "union" else min(0.68, max(0.12, scn["radical_share"] + 0.006 * r.dG))
        res = winnability_2027.score_circo(G, CD, ED, AB, cfg, rad, ru)
        counts[res["score"]] += 1
        ins[res["score"]] += int(r.ins)
        seats[winnability_2027.seat_winner(G, CD, ED, AB, cfg, rad, ru)] += 1
    return {"counts": counts, "inscrits": ins, "seats": seats,
            "playable": sum(counts[s] for s in (1, 2, 3))}


def build() -> None:
    SERVED.mkdir(parents=True, exist_ok=True)
    df = load_master()
    print(f"  bureaux : {len(df):,} | circos : {df.circo.nunique()}")

    # Table maître 2027 pour la découpe géo (report_geo_2027).
    df.to_parquet(CACHE / "bv_master_2027.parquet", index=False)

    com = build_communes(df)
    com_out = com[["code_commune", "nom", "dept", "ins", "nbv", "lat", "lon",
                   "dG", "dCD", "dED", "dAB", "af"]]
    (SERVED / "communes.json").write_text(com_out.to_json(orient="records"))
    code2idx = {c: i for i, c in enumerate(com.code_commune)}

    cir = build_circo(df)
    circo_arrays = {
        "id": cir.circo.tolist(),
        "nm": cir.nm.fillna("").tolist(),
        "dept": cir.dept.fillna("").tolist(),
        "ins": [int(v) for v in cir.ins],
        "nbv": [int(v) for v in cir.nbv],
        **{k: [round(float(v), 3) for v in cir[k]] for k in ("dG", "dCD", "dED", "dAB", "af")},
    }
    (SERVED / "circo.json").write_text(json.dumps(circo_arrays, separators=(",", ":")))

    # Preuve : chiffres hors-échantillon de 2024 (désormais un pli de la validation croisée).
    s24 = json.loads(SUMMARY24.read_text())
    scen_out = [
        {**{k: s[k] for k in ("key", "label", "desc", "means", "left_config", "radical_share")},
         "right_union": s.get("right_union", False)}
        for s in scenarios_2027.SCENARIOS
    ]
    per_scn = {
        s["key"]: winnability_distribution(cir, s) for s in scenarios_2027.SCENARIOS
    }
    cv90 = {b: round(float(df[f"hw90_{b}"].median()), 1) for b in ("G", "CD", "ED", "AB")}
    summary = {
        "n_bv": int(len(df)),
        "n_circo": int(len(cir)),
        "total_inscrits": int(df.inscrits.sum()),
        "scenarios": scen_out,
        "default_scenario": scenarios_2027.DEFAULT_SCENARIO,
        "slider_ranges": scenarios_2027.SLIDER_RANGES,
        "winnability": per_scn,
        "proof_2024": {
            "lead_accuracy": s24.get("lead_accuracy"),
            "r2": s24.get("r2"),
            "n_bv": s24.get("n_bv"),
        },
        "cv_halfwidth_90": cv90,
        # Incertitude au niveau CIRCO (pour la fourchette de sièges Monte-Carlo du site) : la
        # demi-largeur conforme est calibrée par BUREAU ; au niveau circo, les erreurs sont
        # fortement corrélées (mêmes réalités locales), donc on met à l'échelle par le rapport
        # observé σ_circo/σ_bv des résidus 2024 — et non par 1/√N (qui supposerait l'indépendance).
        "circo_halfwidth_90": circo_halfwidth(cv90),
        "lag_fallback_bv": int(df.lag_fallback.sum()),
    }
    (SERVED / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))

    if GAMMA24.exists():
        shutil.copy(GAMMA24, SERVED / "gamma_curve.json")

    ref = per_scn[scenarios_2027.DEFAULT_SCENARIO]
    print(f"  communes : {len(com):,} | circos servies : {len(cir)}")
    print(f"  scénario défaut '{scenarios_2027.DEFAULT_SCENARIO}' — jouables (1-3) : "
          f"{ref['playable']}/{len(cir)}  répartition {ref['counts']}")
    for s in scenarios_2027.SCENARIOS:
        d = per_scn[s["key"]]
        print(f"    {s['label']:32s} jouables 1-3 : {d['playable']:3d}  {d['counts']}")


if __name__ == "__main__":
    build()
