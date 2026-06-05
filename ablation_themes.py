"""Push the audit: ablate demographics by THEME, per block, LOO vs val.

Goal: ensure no demographic sub-group is carried for its test benefit while
being LOO-neutral/negative. Red flag = removing it raises/keeps LOO but drops
val (ΔVal>0.003 & ΔLOO<=0.0005 → the group only helps the 2024 test).
"""

from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*SettingWithCopy.*")

import numpy as np
from pathlib import Path

from src.cross_type_dev import (
    load_cross_type_data,
    add_election_type_onehot,
    BLOCKS_ABS,
    TARGET_COLS,
    VAL_DATE,
    VAL_TYPE,
    estimate_national_abstention_from_gaps,
)
from src.cross_type_ridge import TARGET_BLOCKS
from src.preregistered import run_loo_and_val

THEMES = {
    "housing": [
        "Pct_Proprietaires",
        "Pct_HLM",
        "Pct_Locataires",
        "Pct_Logements_Vacants",
        "Pct_Maisons",
        "Pct_Petits_Logements",
        "Pct_Grands_Logements",
        "Pct_Logement_Gratuit",
        "Pct_Chauff_Elec",
        "Pct_Chauff_Fioul",
        "Pct_Chauff_Gaz",
        "Pct_Logements_Anciens",
        "Pct_Suroccupation",
    ],
    "family": [
        "Pct_Menages_Seuls",
        "Pct_Familles_Monoparentales",
        "Pct_Couples_Avec_Enfants",
        "Pct_Couples_Sans_Enfants",
        "Pct_Familles_Nombreuses",
        "Pct_Maries",
        "Pct_Celibataires",
        "Pct_Divorces",
        "Pct_Veufs",
        "Pct_Pacses",
        "Pct_Union_Libre",
    ],
    "csp": [
        "Taux_Chomage",
        "Pct_Ouvriers",
        "Pct_Cadres",
        "Pct_Employes",
        "Pct_Prof_Intermediaires",
        "Pct_Agriculteurs",
        "Pct_Artisans",
        "Pct_Emploi_Agriculture",
        "Pct_Emploi_Industrie",
        "Pct_Emploi_Construction",
        "Pct_Emploi_Tertiaire",
        "Pct_Retraites",
        "Pct_Etudiants",
        "Pct_Autres_Inactifs",
    ],
    "education": [
        "Pct_Sans_Diplome",
        "Pct_Bac_Plus_5",
        "Pct_Bac",
        "Pct_CAP_BEP",
        "Pct_BEPC",
        "Pct_Bac_Plus_2",
        "Pct_Bac_Plus_3_4",
    ],
    "age": [
        "Pct_Age_18_24",
        "Pct_Age_60_Plus",
        "Pct_Age_30_44",
        "Pct_Age_45_59",
        "Pct_Age_0_14",
        "Pct_Age_75_Plus",
    ],
    "immigration": ["Pct_Immigres"],
}


def main():
    dd = Path("data")
    df, demo, nm, pf = load_cross_type_data(dd)
    tcs = add_election_type_onehot(df)
    p24 = pf[
        np.isclose(pf.date_float, VAL_DATE, atol=0.1) & (pf.election_type == VAL_TYPE)
    ]
    est = {b: float(p24[f"poll_{b}"].iloc[0]) for b in TARGET_BLOCKS}
    est["Abstention"] = estimate_national_abstention_from_gaps(nm)[0]

    dl1 = [f"dev_{b}_lag1" for b in BLOCKS_ABS]
    dl2 = [f"dev_{b}_lag2" for b in BLOCKS_ABS]
    rl1 = [f"{b}_lag1" for b in BLOCKS_ABS]
    rl2 = [f"{b}_lag2" for b in BLOCKS_ABS]
    ct = df.dropna(subset=demo + rl1 + rl2 + dl1 + dl2)
    legi = df[df.election_type == VAL_TYPE].dropna(subset=demo + rl1 + rl2 + dl1 + dl2)

    def run_set(data, extra, tag):
        out = {}
        nd_base = dl1 + dl2 + (tcs if tag == "CT" else [])
        out["full"] = run_loo_and_val(
            "full", data, demo + nd_base, est, nm, {"pca_k": 5, "n_demo": len(demo)}
        )
        for th, cols in THEMES.items():
            keep = [c for c in demo if c not in cols]
            out[th] = run_loo_and_val(
                th,
                data,
                keep + nd_base,
                est,
                nm,
                {"pca_k": 5, "n_demo": len(keep)},
            )
        return out

    for tag, data in [("CT", ct), ("Legi", legi)]:
        print(f"\n  running {tag}...", flush=True)
        res = run_set(data, None, tag)
        print(
            f"\n{'=' * 70}\n{tag}-PCA5 — remove each demo THEME (contrib = full − ablated)"
        )
        print("=" * 70)
        print(f"{'theme':12s} {'block':14s} {'ΔLOO':>8s} {'ΔVal':>8s}   flag")
        for th in THEMES:
            for tc in TARGET_COLS:
                dloo = res["full"][tc]["oof_r2"] - res[th][tc]["oof_r2"]
                dval = res["full"][tc]["val_r2"] - res[th][tc]["val_r2"]
                flag = "⚠ TEST-ONLY" if (dval > 0.003 and dloo <= 0.0005) else ""
                if flag or abs(dloo) > 0.002 or abs(dval) > 0.005:
                    print(f"{th:12s} {tc:14s} {dloo:+8.4f} {dval:+8.4f}   {flag}")


if __name__ == "__main__":
    main()
