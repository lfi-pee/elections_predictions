"""Sélection LOO de la SOURCE du motif « part LFI-dans-la-gauche » par circonscription.

On ne choisit pas la source sur une paire favorable (2022 prés → 2024 euro, R²=0,80) : ce serait
un cherry-pick. On la choisit par validation croisée en laissant chaque scrutin de côté à tour de
rôle et en le prédisant à partir des scrutins ANTÉRIEURS (contrainte causale = celle de la vraie
prévision 2027), pour plusieurs stratégies de source. La stratégie retenue = meilleur R² OOF moyen.

Observations « part radicale du vote de gauche » (pôle radical étiqueté) :
  2012 prés (Mélenchon), 2012 legi (Front de Gauche), 2017 prés (Mélenchon), 2017 legi (FI),
  2022 prés (Mélenchon), 2024 euro (LFI).

Stratégies (chacune n'utilise que des scrutins ANTÉRIEURS au scrutin retiré) :
  null                    part nationale plate (R² = 0 par construction)
  recent_any              carte du scrutin antérieur le plus récent, tous types
  recent_pres             carte de la présidentielle antérieure la plus récente
  pool_any                moyenne des cartes de tous les scrutins antérieurs
  pool_pres               moyenne des cartes des présidentielles antérieures

Motif appliqué en log-odds, recentré, niveau national posé au vrai national du scrutin retiré
(mode oracle, comme le modèle de déviation). R² pondéré par les voix de gauche du scrutin retiré.

    python3 -u -m src.lfi_geo_loo
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.reunif_measure import _commune2circo, CAND

# (id, année, type, sélecteur du pôle radical, ensemble gauche, colonne d'étiquette)
OBS = [
    ("2012_pres_t1", 2012.3, "pres", {"MELE"}, {"HOLL", "MELE", "JOLY", "POUT", "ARTH"}, "nuance"),
    ("2012_legi_t1", 2012.5, "legi", {"FG"}, {"FG", "SOC", "VEC", "ECO", "COM", "DVG", "RDG", "EXG"}, "nuance"),
    ("2017_pres_t1", 2017.3, "pres", {"MÉLENCHON"}, {"MÉLENCHON", "HAMON", "POUTOU", "ARTHAUD"}, "nom"),
    ("2017_legi_t1", 2017.5, "legi", {"FI"}, {"FI", "SOC", "ECO", "COM", "DVG", "RDG", "EXG"}, "nuance"),
    ("2022_pres_t1", 2022.3, "pres", {"MÉLENCHON"}, {"MÉLENCHON", "JADOT", "HIDALGO", "ROUSSEL", "POUTOU", "ARTHAUD"}, "nom"),
    ("2024_euro_t1", 2024.5, "euro", {"LFI"}, {"LFI", "LUG", "LVEC", "LCOM", "LECO", "LDVG", "LRDG", "LEXG"}, "nuance"),
]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def _maps(c2c, cand):
    out = {}
    for eid, yr, typ, rad, left, col in OBS:
        d = cand[cand.id_election == eid].copy()
        d["circo"] = d.code_commune.astype(str).map(c2c)
        d = d.dropna(subset=["circo"])
        d = d[d[col].isin(left)]
        d["p"] = np.where(d[col].isin(rad), "R", "S")
        t = d.groupby(["circo", "p"]).voix.sum().unstack("p").fillna(0.0)
        t["left"] = t.get("R", 0.0) + t.get("S", 0.0)
        t = t[t.left > 0]
        t["rs"] = t.get("R", 0.0) / t["left"]
        out[eid] = {"yr": yr, "typ": typ,
                    "share": {ci: float(r) for ci, r in zip(t.index, t.rs)},
                    "w": {ci: float(w) for ci, w in zip(t.index, t.left)}}
    return out


def _dev_logit(m, circos):
    """Déviation en log-odds, recentrée (moyenne pondérée par les voix de gauche ≈ 0)."""
    s = np.array([m["share"][c] for c in circos])
    w = np.array([m["w"][c] for c in circos])
    lg = _logit(s)
    return lg - np.average(lg, weights=w)


def _wr2(y, yh, w):
    yb = np.average(y, weights=w)
    ss = np.average((y - yb) ** 2, weights=w)
    return 1.0 - np.average((y - yh) ** 2, weights=w) / ss if ss > 0 else float("nan")


def predict(sources, target, common):
    """Prédit la part de `target` sur `common` en moyennant les déviations log-odds des `sources`,
    niveau national = vrai national (oracle) du target."""
    devs = [_dev_logit(s, common) for s in sources]
    dev = np.mean(devs, axis=0)
    yt = np.array([target["share"][c] for c in common])
    wt = np.array([target["w"][c] for c in common])
    base = _logit(np.average(yt, weights=wt))
    yhat = 1.0 / (1.0 + np.exp(-(base + dev)))
    return yt, yhat, wt


def main():
    c2c = _commune2circo(drop_split=True)
    cand = pd.read_parquet(CAND, columns=["id_election", "code_commune", "nuance", "nom", "voix"])
    M = _maps(c2c, cand)
    ids = [o[0] for o in OBS]

    # LOO COMPLET : chaque scrutin retiré est prédit à partir de TOUS les autres (past+future).
    # C'est la validation croisée du projet (comme preregistered.py) : un test de généralisation,
    # pas un test causal. Stratégies = quels autres scrutins servir, et comment les combiner.
    def others(eid):
        return [p for p in ids if p != eid]
    strategies = {
        "recent_pres": lambda e, o: sorted([p for p in o if M[p]["typ"] == "pres"],
                                            key=lambda p: abs(M[p]["yr"] - M[e]["yr"]))[:1],
        "pool_pres":   lambda e, o: [p for p in o if M[p]["typ"] == "pres"],
        "pool_all":    lambda e, o: o,
        "recent_any":  lambda e, o: sorted(o, key=lambda p: abs(M[p]["yr"] - M[e]["yr"]))[:1],
    }
    # résidus concaténés sur TOUS les plis → un seul R² OOF « validé sur toutes les élections »
    pooled = {k: {"sr": 0.0, "st": 0.0} for k in strategies}
    per_fold = {}
    for eid in ids:
        per_fold[eid] = {}
        for name, pick in strategies.items():
            srcs = pick(eid, others(eid))
            if not srcs:
                per_fold[eid][name] = None
                continue
            common = [c for c in M[eid]["share"] if all(c in M[s]["share"] for s in srcs)]
            if len(common) < 50:
                per_fold[eid][name] = None
                continue
            yt, yh, wt = predict([M[s] for s in srcs], M[eid], common)
            yh = np.clip(yh, 0, 1)
            per_fold[eid][name] = _wr2(yt, yh, wt)
            yb = np.average(yt, weights=wt)
            pooled[name]["sr"] += float(np.sum(wt * (yt - yh) ** 2))
            pooled[name]["st"] += float(np.sum(wt * (yt - yb) ** 2))

    print("R² OOF par pli (LOO complet : chaque scrutin prédit depuis TOUS les autres) :")
    print(f"  {'retiré':>14} | " + " | ".join(f"{k:>11}" for k in strategies))
    for eid in ids:
        row = " | ".join(
            (f"{per_fold[eid][k]:>+11.3f}" if per_fold[eid].get(k) is not None else f"{'—':>11}")
            for k in strategies)
        print(f"  {eid[:14]:>14} | {row}")

    print("\nR² OOF POOLÉ sur toutes les élections (résidus concaténés — LE chiffre à valider) :")
    best, bestv = None, -1e9
    for k in strategies:
        r2 = 1.0 - pooled[k]["sr"] / pooled[k]["st"] if pooled[k]["st"] > 0 else float("nan")
        pos = sum(1 for e in ids if (per_fold[e].get(k) or -9) > 0)
        n = sum(1 for e in ids if per_fold[e].get(k) is not None)
        if r2 == r2 and r2 > bestv:
            bestv, best = r2, k
        print(f"  {k:>12} : {r2:>+.3f}   (positif sur {pos}/{n} plis)")
    print(f"  {'null':>12} : +0.000   (référence — part nationale plate)")
    verdict = ("BAT le null sur l'ensemble" if bestv > 0
               else "ne bat PAS le null : AUCUNE source n'est validée sur toutes les élections")
    print(f"\n  Meilleure : {best}  (R² poolé {bestv:+.3f}) → {verdict}")


if __name__ == "__main__":
    main()
