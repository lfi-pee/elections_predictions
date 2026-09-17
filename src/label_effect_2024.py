"""Effet de l'ÉTIQUETTE du candidat de gauche sur le 2nd tour — mesuré sur le réel 2024.

Le modèle de sièges traite la gauche unie comme un bloc homogène : les reports du centre-droit
vers « la gauche » au 2nd tour ne dépendent pas du parti qui porte la candidature. Or c'est
précisément l'argument central d'une négociation de circonscriptions : « ton étiquette fait
perdre des voix de report ». Ce module MESURE cet effet, plutôt que de le supposer.

Données : 2024 est le seul scrutin où (a) une candidature de gauche UNIQUE par circo a affronté
le RN au 2nd tour à grande échelle, et (b) le parti de chaque candidat d'union est connu — pas
par la nuance du ministère (toutes `UG`), mais par la RÉPARTITION des circonscriptions entre
partis du Nouveau Front populaire (`data/nuance/nfp_repartition_2024.csv`, 546 circos : FI 229,
PS 175, Pôle écologiste 92, PCF 50 ; source data.gouv.fr, David Libeau, CC0 ; Corse et outre-mer
absents — candidatures locales hors répartition nationale).

Échantillon PRINCIPAL : les DUELS candidat UG contre candidat RN/UXD (centre-droit éliminé au
1er tour). Pour chacun, on observe combien le candidat de gauche a récupéré des voix
« libérées » (centre-droit + autres + petits candidats de gauche éliminés) :

    k = (voix UG au T2 − voix UG au T1) / voix libérées au T1        (tout en % des inscrits)

et on compare k selon l'étiquette du candidat UG. Pourquoi le duel : la configuration est
identique pour tous (un adversaire RN, un réservoir de reports), donc la différence de k entre
étiquettes ne peut venir que du candidat — pas d'une différence de configuration. Le biais de
sélection (LFI a reçu des circos plus dures) est neutralisé par construction : k est un TAUX
de report, et on contrôle en plus par le niveau de gauche au 1er tour (terciles) et par une
régression de la marge de 2nd tour sur la marge de 1er tour.

Ce qu'on en tire pour le modèle : un DÉCALAGE ADDITIF par étiquette du taux de report
centre-droit → gauche (`cd2l` en duel, `desist` en désistement), centré sur la moyenne pondérée
de l'union 2024 (pour que le modèle moyen — calibré sur le nombre de sièges RN 2024 — reste
inchangé) :

    delta_LFI = k_FI − k_union        delta_autre = k_non-FI − k_union

Additif et non multiplicatif : k est mesuré par unité de voix libérées, exactement l'unité des
coefficients du modèle ; un multiplicateur sur `cd2l` (≈0,45) aurait rendu un tiers d'effet en
moins que la différence mesurée (≈ −1,4 pt d'inscrits de marge, retrouvée ainsi).

Résultat (voir JSON servi) : k_FI ≈ 0,53 contre ≈ 0,62 pour PS/écologistes/PCF ; écart −0,09
(IC 95 % bootstrap ≈ [−0,11 ; −0,06]), présent dans les trois terciles de force de la gauche
(donc PAS compensé dans les bastions), soit ≈ −1,4 pt d'inscrits sur la marge de 2nd tour à
marge de 1er tour égale. L'effet est SIGNÉ et testé : s'il avait été nul ou positif, le JSON
l'aurait dit et la négociation se serait réduite à la jouabilité du bloc.

    python3 -u -m src.label_effect_2024        # → report_app/2027/data/label_effect_2024.json
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

CAND = Path("data/elections/agregees/candidats_results.parquet")
GEN = Path("data/elections/agregees/general_results.parquet")
MASTER = Path("data/report/bv_master_2027.parquet")
REPARTITION = Path("data/nuance/nfp_repartition_2024.csv")
OUT = Path("report_app/2027/data/label_effect_2024.json")

LEFT = {"UG", "DVG", "EXG", "ECO", "SOC", "RDG", "FI", "COM", "VEC"}
CD = {"ENS", "LR", "DVD", "DVC", "HOR", "UDI"}
ED = {"RN", "UXD", "REC", "EXD", "DSV"}
LABELS = ["FI", "PS", "PE", "PCF"]
LABEL_NAMES = {"FI": "La France insoumise", "PS": "Parti socialiste",
               "PE": "Pôle écologiste", "PCF": "Parti communiste"}
BOOT = 4000
SEED = 0


def _bloc(n: str) -> str:
    return "G" if n in LEFT else "CD" if n in CD else "ED" if n in ED else "AU"


def load_repartition() -> dict[str, str]:
    r = pd.read_csv(REPARTITION)
    return dict(zip(r.circo, r.parti))


def second_rounds() -> pd.DataFrame:
    """Une ligne par circo où un candidat UG était au 2nd tour 2024 : parts de 1er tour par bloc
    (% inscrits), voix UG aux deux tours, adversaires présents, étiquette du candidat UG."""
    m = pd.read_parquet(MASTER, columns=["location", "circo"]).dropna(subset=["circo"])
    loc2c = dict(zip(m.location, m.circo))
    c = pd.read_parquet(CAND, columns=["id_election", "code_commune", "code_bv", "nuance", "voix"])
    c = c[c.id_election.isin(["2024_legi_t1", "2024_legi_t2"])].copy()
    c["circo"] = (c.code_commune.astype(str) + "_" + c.code_bv.astype(str)).map(loc2c)
    c = c.dropna(subset=["circo"])
    c["rnd"] = c.id_election.str[-2:]
    c["bloc"] = c.nuance.map(_bloc)
    g = pd.read_parquet(GEN, columns=["id_election", "code_commune", "code_bv", "inscrits"])
    g = g[g.id_election == "2024_legi_t1"].copy()
    g["circo"] = (g.code_commune.astype(str) + "_" + g.code_bv.astype(str)).map(loc2c)
    ins = g.dropna(subset=["circo"]).groupby("circo").inscrits.sum()

    t1 = c[c.rnd == "t1"].groupby(["circo", "bloc"]).voix.sum().unstack().fillna(0.0)
    ug1 = c[(c.rnd == "t1") & (c.nuance == "UG")].groupby("circo").voix.sum()
    t2 = c[c.rnd == "t2"].groupby(["circo", "nuance"]).voix.sum().reset_index()
    lab = load_repartition()
    rows = []
    for circo, grp in t2.groupby("circo"):
        ug = grp[grp.nuance == "UG"]
        if len(ug) != 1 or circo not in ins.index:
            continue
        n_ins = float(ins[circo])
        pct = lambda v: 100.0 * float(v) / n_ins  # noqa: E731
        nu = set(grp.nuance)
        blocs2 = {_bloc(n) for n in nu}
        r = t1.loc[circo]
        rows.append(dict(
            circo=circo, label=lab.get(circo), inscrits=n_ins,
            G1=pct(r.get("G", 0)), CD1=pct(r.get("CD", 0)), ED1=pct(r.get("ED", 0)),
            AU1=pct(r.get("AU", 0)), UG1=pct(ug1.get(circo, 0)), UG2=pct(ug.voix.iloc[0]),
            ED2=pct(grp[grp.nuance.map(_bloc) == "ED"].voix.sum()),
            CD2=pct(grp[grp.nuance.map(_bloc) == "CD"].voix.sum()),
            n_t2=len(grp), vs_ed="ED" in blocs2, vs_cd="CD" in blocs2,
            ug_won=bool(grp.loc[grp.voix.idxmax(), "nuance"] == "UG"),
        ))
    d = pd.DataFrame(rows)
    # Réservoir de voix libérées au T1 = tout ce qui n'est ni le candidat UG ni le bloc ED.
    d["pool"] = d.CD1 + d.AU1 + (d.G1 - d.UG1)
    d["k"] = (d.UG2 - d.UG1) / d.pool
    d["k_ed"] = (d.ED2 - d.ED1) / d.pool
    d["m1"] = d.UG1 - d.ED1
    d["m2"] = d.UG2 - d.ED2
    return d


def _wmean(x: pd.Series, w: pd.Series) -> float:
    return float(np.average(x, weights=w))


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    res = y - X @ b
    s2 = res @ res / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.inv(X.T @ X)))
    return b, se


def measure(d: pd.DataFrame) -> dict:
    duels = d[d.vs_ed & ~d.vs_cd & (d.n_t2 == 2) & d.label.isin(LABELS)].copy()
    per = {}
    for lab, sub in duels.groupby("label"):
        per[lab] = dict(n=int(len(sub)), k_weighted=round(_wmean(sub.k, sub.pool), 4),
                        k_mean=round(float(sub.k.mean()), 4), k_median=round(float(sub.k.median()), 4),
                        k_to_rn=round(_wmean(sub.k_ed, sub.pool), 4),
                        win_rate=round(float(sub.ug_won.mean()), 3),
                        t1_margin_mean=round(float(sub.m1.mean()), 2),
                        left_t1_mean=round(float(sub.UG1.mean()), 2))
    fi, oth = duels[duels.label == "FI"], duels[duels.label != "FI"]
    k_fi, k_oth, k_all = _wmean(fi.k, fi.pool), _wmean(oth.k, oth.pool), _wmean(duels.k, duels.pool)
    rng = np.random.default_rng(SEED)
    diffs = np.empty(BOOT)
    for b in range(BOOT):
        i = rng.integers(0, len(fi), len(fi))
        j = rng.integers(0, len(oth), len(oth))
        diffs[b] = _wmean(fi.k.values[i], fi.pool.values[i]) - _wmean(oth.k.values[j], oth.pool.values[j])
    lo, hi = np.percentile(diffs, [2.5, 97.5])

    # Régression de la marge de 2nd tour (UG − RN, % inscrits) sur la marge de 1er tour, le
    # réservoir centre-droit et les étiquettes (PS = référence).
    X = np.column_stack([np.ones(len(duels)), duels.m1, duels.CD1, duels.AU1,
                         duels.label == "FI", duels.label == "PE", duels.label == "PCF"]).astype(float)
    b, se = _ols(X, duels.m2.values)
    names = ["const", "t1_margin", "CD1", "AU1", "FI", "PE", "PCF"]
    reg = {n: {"coef": round(float(bb), 3), "se": round(float(ss), 3)} for n, bb, ss in zip(names, b, se)}

    # Hétérogénéité : l'écart FI / non-FI par tercile de force de la gauche au 1er tour.
    duels["tercile"] = pd.qcut(duels.UG1, 3, labels=["faible", "moyen", "fort"])
    het = {}
    for t, sub in duels.groupby("tercile", observed=True):
        f, o = sub[sub.label == "FI"], sub[sub.label != "FI"]
        het[str(t)] = dict(n_fi=int(len(f)), n_other=int(len(o)),
                           k_fi=round(_wmean(f.k, f.pool), 4), k_other=round(_wmean(o.k, o.pool), 4))

    # Configurations secondaires (rapportées, non utilisées) : triangulaires maintenues face au
    # RN (taux de victoire) et duels face au centre-droit (reports RN → gauche).
    tri = d[d.vs_ed & d.vs_cd & (d.n_t2 >= 3) & d.label.isin(LABELS)]
    tri_out = {lab: dict(n=int(len(s)), win_rate=round(float(s.ug_won.mean()), 3),
                         left_t1_mean=round(float(s.UG1.mean()), 2)) for lab, s in tri.groupby("label")}
    dcd = d[~d.vs_ed & d.vs_cd & (d.n_t2 == 2) & d.label.isin(LABELS)].copy()
    dcd["pool_cd"] = dcd.ED1 + dcd.AU1 + (dcd.G1 - dcd.UG1)
    dcd["k_cd"] = (dcd.UG2 - dcd.UG1) / dcd.pool_cd
    dcd_out = {lab: dict(n=int(len(s)), k_weighted=round(_wmean(s.k_cd, s.pool_cd), 4),
                         win_rate=round(float(s.ug_won.mean()), 3)) for lab, s in dcd.groupby("label")}

    return {
        "source": {
            "repartition": str(REPARTITION), "n_repartition": int(len(load_repartition())),
            "note": "Répartition des circonscriptions entre partis du NFP (data.gouv.fr, D. Libeau, "
                    "CC0) ; résultats T1/T2 2024 du ministère agrégés par circonscription.",
        },
        "sample": {"n_ug_second_round": int(len(d)), "n_duels_vs_rn": int(len(duels)),
                   "n_duels_fi": int(len(fi)), "n_duels_other": int(len(oth))},
        "duels_vs_rn": {"by_label": per, "label_names": LABEL_NAMES,
                        "k_fi": round(k_fi, 4), "k_other": round(k_oth, 4), "k_union": round(k_all, 4),
                        "diff_fi_minus_other": round(k_fi - k_oth, 4),
                        "diff_ci95": [round(float(lo), 4), round(float(hi), 4)],
                        "bootstrap": BOOT, "seed": SEED,
                        "margin_regression": reg, "by_left_strength": het},
        "triangulaires_vs_rn": tri_out,
        "duels_vs_cd": dcd_out,
        # Paramètres consommés par negotiation_2027 : multiplicateurs sur les reports
        # centre-droit → gauche selon l'étiquette, normalisés à l'union 2024.
        "model": {"cd2l_delta_lfi": round(k_fi - k_all, 4), "cd2l_delta_other": round(k_oth - k_all, 4),
                  "margin_effect_lfi_pts_inscrits": reg["FI"]["coef"],
                  "significant": bool(hi < 0 or lo > 0),
                  "sign": "penalty" if k_fi < k_oth else "bonus"},
    }


def load() -> dict:
    return json.loads(OUT.read_text())


def main() -> None:
    d = second_rounds()
    res = measure(d)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1))
    du = res["duels_vs_rn"]
    print(f"  {res['sample']['n_duels_vs_rn']} duels UG–RN 2024 étiquetés "
          f"({res['sample']['n_duels_fi']} FI, {res['sample']['n_duels_other']} autres)")
    for lab in LABELS:
        p = du["by_label"].get(lab)
        if p:
            print(f"   {lab:4s} n={p['n']:3d}  report k={p['k_weighted']:.3f}  victoire={p['win_rate']:.0%}")
    print(f"  k(FI) − k(autres) = {du['diff_fi_minus_other']:+.3f}  IC95 {du['diff_ci95']}")
    print(f"  marge T2 à marge T1 égale : FI {du['margin_regression']['FI']['coef']:+.2f} "
          f"± {du['margin_regression']['FI']['se']:.2f} pt d'inscrits")
    print("  par force de gauche :", {t: (v['k_fi'], v['k_other']) for t, v in du['by_left_strength'].items()})
    print(f"  → décalage du taux de report : LFI {res['model']['cd2l_delta_lfi']:+.3f}, "
          f"autres {res['model']['cd2l_delta_other']:+.3f} ({res['model']['sign']}, "
          f"{'significatif' if res['model']['significant'] else 'NON significatif'})")
    print(f"  → {OUT}")


if __name__ == "__main__":
    main()


def union_winners_2024() -> dict[str, str]:
    """Nuance du vainqueur 2024 par circo (élu au 1er tour si pas de 2nd tour) : `UG` là où
    l'union de gauche a pris le siège. Sert de repère « 2024 » à la négociation 2027."""
    m = pd.read_parquet(MASTER, columns=["location", "circo"]).dropna(subset=["circo"])
    loc2c = dict(zip(m.location, m.circo))
    c = pd.read_parquet(CAND, columns=["id_election", "code_commune", "code_bv", "nuance", "voix",
                                       "nom", "prenom"])
    c = c[c.id_election.isin(["2024_legi_t1", "2024_legi_t2"])].copy()
    c["circo"] = (c.code_commune.astype(str) + "_" + c.code_bv.astype(str)).map(loc2c)
    c = c.dropna(subset=["circo"])
    c["rnd"] = c.id_election.str[-2:]
    cand = c.groupby(["circo", "rnd", "nuance", "nom", "prenom"]).voix.sum().reset_index()
    out = {}
    for rnd in ("t1", "t2"):  # le 2nd tour écrase le 1er là où il a eu lieu
        sub = cand[cand.rnd == rnd]
        win = sub.loc[sub.groupby("circo").voix.idxmax()]
        out.update(dict(zip(win.circo, win.nuance)))
    return out
