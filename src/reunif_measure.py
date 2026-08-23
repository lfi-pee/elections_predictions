"""Mesure empirique de la RÉUNIFICATION d'une gauche divisée au 2nd tour (coefficient `REUNIF`).

Le modèle de sièges (`winnability_2027`) applique, quand la gauche est divisée, un coefficient
`REUNIF` (défaut 0,72) : les voix d'un pôle de gauche ÉLIMINÉ au 1er tour ne se reportent qu'à
hauteur de `REUNIF` sur le pôle de gauche QUALIFIÉ au 2nd tour. C'était jusqu'ici le seul
coefficient de report SANS base empirique — 2024 (NFP uni) et 2022 (NUPES) ne permettent pas de
le mesurer. On le mesure ici sur les législatives à gauche RÉELLEMENT divisée : 2007, 2012, 2017.

Méthode (au niveau circonscription, cartographie 2024 — stable depuis le redécoupage de 2010,
donc valable pour 2012/2017 ; 2007 sur l'ancien découpage → indicatif seulement) :
  - On mappe chaque bureau à sa circo par sa commune (mapping majoritaire ; 0,4 % des communes
    sont à cheval sur 2 circos → léger bruit, robustesse vérifiée en les excluant).
  - Par circo, on agrège les voix T1 et T2 par nuance, regroupées en pôles gauche / CD / ED.
  - On retient les circos où UN SEUL candidat de gauche subsiste au T2 alors que ≥1 pôle de
    gauche a été éliminé au T1 (le cas où `REUNIF` s'applique).
  - Deux estimateurs du report gauche→gauche :
      (a) ratio direct  (surv_gauche_T2 − surv_gauche_T1) / (gauche_éliminée_T1), sur le
          sous-ensemble où l'élimination est DOMINÉE par la gauche (peu de CD/ED éliminé,
          donc peu de contamination) ;
      (b) régression écologique pondérée  Δsurv = β_L·élim_gauche + β_CD·élim_cd + β_ED·élim_ed
          + intercept (mobilisation) ; β_L = report gauche→gauche, en contrôlant les autres blocs.

RÉSULTATS (hors communes à cheval, l'estimation la plus propre) :
  - 2012 (référence : PS vs Front de Gauche vs EELV, gauche divisée SUBSTANTIELLE sur le
    découpage actuel) → ratio (a) 0,89 ; régression (b) β_gauche = 0,73.
  - Pool 2012+2017 → ratio 0,86 ; régression β_gauche = 0,66.
  - 2017 = cas atypique (effondrement du PS sous la vague En Marche, peu de cas propres →
    estimation instable, β même négatif) ; 2007 = ancien découpage. Les deux, indicatifs.

  La régression (qui neutralise les reports CD/ED éliminés et la mobilisation du T2) est
  l'estimateur honnête ; le ratio SURESTIME (il attribue TOUT le gain du survivant au report
  de gauche). Le défaut posé à la main `REUNIF = 0,72` tombe en HAUT de la fourchette de
  régression 2012 (0,69–0,73) et sous le ratio (~0,86) : valeur centrale-à-légèrement-haute,
  empiriquement SOUTENUE — ce n'était pas une supposition. Aucun ajustement nécessaire.

    python3 -u -m src.reunif_measure
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

CAND = "data/elections/agregees/candidats_results.parquet"
MASTER = "data/report/bv_master_2027.parquet"

# Nuances → pôle. LEFT couvre tous les sous-pôles (radical/soc-dém/éco/communiste) : REUNIF est
# la réunification INTER-pôles de gauche, on ne distingue donc pas à l'intérieur.
LEFT = {"SOC", "COM", "VEC", "ECO", "ECOLO", "EXG", "DVG", "FI", "NUP", "FG", "RDG",
        "DXG", "UG", "GEN", "PRG", "LFI", "RDGX"}
CD = {"UMP", "LR", "DVD", "REM", "ENS", "UDI", "MDM", "UDF", "UDFD", "CEN", "DVC", "NCE",
      "NC", "HOR", "MAJ", "ALLI", "PRV", "CPNT"}
ED = {"FN", "RN", "REC", "EXD", "MNR", "UXD", "DLF", "MPF", "DSV", "DXD"}


def _pole(n: str) -> str | None:
    if n in LEFT:
        return "L"
    if n in CD:
        return "CD"
    if n in ED:
        return "ED"
    return None  # DIV / REG / autres : ignorés (ambigus, marginaux)


def _commune2circo(drop_split: bool = False):
    bm = pd.read_parquet(MASTER, columns=["location", "circo", "inscrits"]).dropna(subset=["circo"])
    bm["commune"] = bm.location.str.split("_").str[0]
    g = bm.groupby(["commune", "circo"]).inscrits.sum().reset_index()
    if drop_split:
        nc = g.groupby("commune").circo.nunique()
        g = g[g.commune.isin(nc[nc == 1].index)]
    maj = g.sort_values("inscrits").groupby("commune").tail(1)
    return dict(zip(maj.commune, maj.circo))


def _load(elec: str, c2c: dict) -> pd.DataFrame:
    cols = ["id_election", "code_commune", "nuance", "voix"]
    c = pd.read_parquet(CAND, columns=cols)
    c = c[c.id_election.isin([f"{elec}_t1", f"{elec}_t2"])].copy()
    c["circo"] = c.code_commune.astype(str).map(c2c)
    c = c.dropna(subset=["circo"])
    c["pole"] = c.nuance.map(_pole)
    c["rnd"] = c.id_election.str[-2:]
    return c


def _circo_table(c: pd.DataFrame) -> pd.DataFrame:
    """Par circo : voix gauche T1/T2 (survivante vs éliminée) et éliminés CD/ED T1."""
    recs = []
    for circo, g in c.groupby("circo"):
        t1 = g[g.rnd == "t1"]
        t2 = g[g.rnd == "t2"]
        if t2.empty:
            continue
        # nuances de gauche présentes au T2 = pôle(s) survivant(s)
        surv_nu = set(t2[t2.pole == "L"].nuance.unique())
        if len(surv_nu) != 1:
            continue  # on veut exactement UN survivant de gauche (REUNIF = report vers lui)
        l1 = t1[t1.pole == "L"]
        surv1 = l1[l1.nuance.isin(surv_nu)].voix.sum()
        elim_L = l1[~l1.nuance.isin(surv_nu)].voix.sum()
        if elim_L <= 0 or surv1 <= 0:
            continue
        surv2 = t2[t2.nuance.isin(surv_nu)].voix.sum()
        elim_cd = t1[(t1.pole == "CD") & (~t1.nuance.isin(t2.nuance))].voix.sum()
        elim_ed = t1[(t1.pole == "ED") & (~t1.nuance.isin(t2.nuance))].voix.sum()
        recs.append(dict(circo=circo, surv1=surv1, surv2=surv2, elim_L=elim_L,
                         elim_cd=elim_cd, elim_ed=elim_ed, dsurv=surv2 - surv1))
    return pd.DataFrame(recs)


def _estimate(df: pd.DataFrame, lab: str):
    if df.empty:
        print(f"[{lab}] aucun cas exploitable")
        return
    # (a) ratio direct sur le sous-ensemble « élimination dominée par la gauche »
    dom = df[df.elim_L >= 2.0 * (df.elim_cd + df.elim_ed)]
    ratio = (dom.dsurv / dom.elim_L).clip(-0.5, 1.5)
    w = dom.elim_L
    ra = float(np.average(ratio, weights=w)) if len(dom) else float("nan")
    # (b) régression écologique pondérée (sans intercept forcé → intercept = mobilisation)
    X = df[["elim_L", "elim_cd", "elim_ed"]].to_numpy(float)
    X = np.column_stack([X, np.ones(len(df))])
    y = df.dsurv.to_numpy(float)
    sw = np.sqrt(df.elim_L.to_numpy(float))  # pondère les gros transferts
    beta, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    print(f"[{lab}]  n={len(df)}  (dont {len(dom)} à élimination dominée par la gauche)")
    print(f"   (a) ratio direct gauche→gauche (pondéré)      : {ra:+.3f}")
    print(f"   (b) régression β_gauche={beta[0]:+.3f}  β_cd={beta[1]:+.3f}  "
          f"β_ed={beta[2]:+.3f}  (intercept={beta[3]:,.0f} voix)")
    return ra, float(beta[0])


def main():
    for drop in (False, True):
        c2c = _commune2circo(drop_split=drop)
        tag = "hors communes à cheval" if drop else "mapping majoritaire (toutes communes)"
        print(f"\n================ {tag} ================")
        pooled = []
        for elec in ["2007_legi", "2012_legi", "2017_legi"]:
            c = _load(elec, c2c)
            tab = _circo_table(c)
            _estimate(tab, elec)
            if elec != "2007_legi":  # 2007 = ancien découpage → indicatif, hors pool
                pooled.append(tab)
        if pooled:
            _estimate(pd.concat(pooled, ignore_index=True), "POOL 2012+2017 (découpage actuel)")


if __name__ == "__main__":
    main()
