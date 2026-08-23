"""Measure the REAL 2024 second-round transfer rate in triangulaires face au RN.

The 2027 seat model has a hand-set désistement coefficient (`DESIST_TO_STRONG`,
default 0.50) justified only by a seat-count backtest. We actually have the real
2024 T1 and T2 results, so this script measures the transfer directly.

Method (all at circonscription level, 577 circos):
  - Join BV-level candidate results (candidats_results.parquet) to circo via
    bv_master_2027 (location = code_commune_code_bv → circo).
  - Aggregate T1 and T2 votes per circo per bloc (gauche / centre-droit / RN-ED).
  - Find triangulaires face au RN: at T1 the RN bloc + BOTH anti-RN blocs each
    qualified (top-2 OR >= 12.5% of inscrits); at T2 one anti-RN bloc is absent
    (it withdrew — "désistement / front républicain").
  - For each such désistement, measure where the withdrawn bloc's T1 votes went:
        to_strong = (survivor_T2 - survivor_T1) / withdrawn_T1
        to_rn     = (rn_T2       - rn_T1)       / withdrawn_T1
        abst/rest = 1 - to_strong - to_rn
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CAND = "data/elections/agregees/candidats_results.parquet"
GEN = "data/elections/agregees/general_results.parquet"
MASTER = "data/report/bv_master_2027.parquet"

# Nuance → bloc, matching the 3 blocs of the seat model.
LEFT = {"UG", "DVG", "EXG", "ECO", "SOC", "RDG", "FI", "COM", "VEC"}
CD = {"ENS", "LR", "DVD", "DVC", "HOR", "UDI"}
ED = {"RN", "UXD", "REC", "EXD", "DSV"}


def bloc(n: str) -> str | None:
    if n in LEFT:
        return "G"
    if n in CD:
        return "CD"
    if n in ED:
        return "ED"
    return None  # DIV, REG, DIV-other → ignored (marginal, ambiguous)


def load() -> pd.DataFrame:
    loc2circo = pd.read_parquet(MASTER, columns=["location", "circo"]).dropna(subset=["circo"])
    loc2circo = dict(zip(loc2circo.location, loc2circo.circo))

    c = pd.read_parquet(CAND, columns=["id_election", "code_commune", "code_bv", "nuance", "voix"])
    c = c[c.id_election.isin(["2024_legi_t1", "2024_legi_t2"])].copy()
    c["location"] = c.code_commune.astype(str) + "_" + c.code_bv.astype(str)
    c["circo"] = c.location.map(loc2circo)
    c = c.dropna(subset=["circo"])
    c["bloc"] = c.nuance.map(bloc)

    # inscrits per circo (from T1 general results)
    g = pd.read_parquet(GEN, columns=["id_election", "code_commune", "code_bv", "inscrits"])
    g = g[g.id_election == "2024_legi_t1"].copy()
    g["location"] = g.code_commune.astype(str) + "_" + g.code_bv.astype(str)
    g["circo"] = g.location.map(loc2circo)
    insc = g.dropna(subset=["circo"]).groupby("circo").inscrits.sum()

    # votes per circo × round × bloc
    tab = (c.dropna(subset=["bloc"])
           .assign(rnd=c.id_election.str[-2:])
           .groupby(["circo", "rnd", "bloc"]).voix.sum().unstack("bloc").fillna(0.0))
    return tab, insc


def main() -> None:
    tab, insc = load()
    circos = tab.index.get_level_values("circo").unique()

    recs = []
    for circo in circos:
        try:
            t1 = tab.loc[(circo, "t1")]
            t2 = tab.loc[(circo, "t2")]
        except KeyError:
            continue
        thr = 0.125 * insc.get(circo, np.nan)
        if not np.isfinite(thr):
            continue
        # RN must reach T2
        if t2.get("ED", 0) <= 0:
            continue
        g1, cd1, ed1 = t1.get("G", 0), t1.get("CD", 0), t1.get("ED", 0)
        g2, cd2 = t2.get("G", 0), t2.get("CD", 0)
        # both anti-RN blocs qualified at T1 (>=12.5% inscrits) → a triangulaire was possible
        if g1 < thr or cd1 < thr or ed1 < thr:
            continue
        # exactly one anti-RN bloc present at T2 → the other withdrew (désistement)
        g_in, cd_in = g2 > 0, cd2 > 0
        if g_in == cd_in:
            continue  # both present (real triangulaire) or both gone → not a clean désistement
        if g_in:  # CD withdrew for G
            withdrawn, w1, surv1, surv2 = "CD", cd1, g1, g2
        else:  # G withdrew for CD
            withdrawn, w1, surv1, surv2 = "G", g1, cd1, cd2
        rn1, rn2 = ed1, t2.get("ED", 0)
        if w1 <= 0:
            continue
        to_strong = (surv2 - surv1) / w1
        to_rn = (rn2 - rn1) / w1
        recs.append(dict(circo=circo, withdrew=withdrawn, w1=w1,
                         to_strong=to_strong, to_rn=to_rn,
                         rest=1 - to_strong - to_rn))

    df = pd.DataFrame(recs)
    print(f"Triangulaires face au RN with a clean désistement: {len(df)}")
    print(f"  CD withdrew for gauche : {(df.withdrew=='CD').sum()}")
    print(f"  gauche withdrew for CD : {(df.withdrew=='G').sum()}")
    print()
    # weight by size of the withdrawn bloc (large circos count more)
    for lab, sub in [("ALL", df), ("CD→G", df[df.withdrew == "CD"]), ("G→CD", df[df.withdrew == "G"])]:
        if not len(sub):
            continue
        w = sub.w1
        print(f"[{lab}]  n={len(sub)}")
        for col in ["to_strong", "to_rn", "rest"]:
            unw = sub[col].mean()
            wt = np.average(sub[col], weights=w)
            med = sub[col].median()
            print(f"   {col:9s}  mean={unw:+.3f}  vote-weighted={wt:+.3f}  median={med:+.3f}")
        print()
    # clip to [0,1] view of the headline transfer
    ts = df.to_strong.clip(0, 1)
    print(f"to_strong clipped[0,1]: vote-weighted={np.average(ts, weights=df.w1):.3f}  median={ts.median():.3f}")


if __name__ == "__main__":
    main()


def backtest() -> None:
    """Replay the 2024 seat backtest at several desist values, using REAL T1 shares."""
    import winnability_2027 as W

    loc2circo = pd.read_parquet(MASTER, columns=["location", "circo"]).dropna(subset=["circo"])
    loc2circo = dict(zip(loc2circo.location, loc2circo.circo))

    c = pd.read_parquet(CAND, columns=["id_election", "code_commune", "code_bv", "nuance", "voix"])
    c = c[c.id_election.isin(["2024_legi_t1", "2024_legi_t2"])].copy()
    c["location"] = c.code_commune.astype(str) + "_" + c.code_bv.astype(str)
    c["circo"] = c.location.map(loc2circo)
    c = c.dropna(subset=["circo"])
    c["bloc"] = c.nuance.map(bloc)
    c["rnd"] = c.id_election.str[-2:]

    g = pd.read_parquet(GEN, columns=["id_election", "code_commune", "code_bv", "inscrits", "exprimes", "abstentions"])
    g["location"] = g.code_commune.astype(str) + "_" + g.code_bv.astype(str)
    g["circo"] = g.location.map(loc2circo)
    g = g.dropna(subset=["circo"])
    t1g = g[g.id_election == "2024_legi_t1"].groupby("circo")[["inscrits", "exprimes", "abstentions"]].sum()

    # T1 bloc shares (% exprimés), real winner (bloc with most T2 votes)
    t1 = c[c.rnd == "t1"].dropna(subset=["bloc"]).groupby(["circo", "bloc"]).voix.sum().unstack("bloc").fillna(0.0)
    t2 = c[c.rnd == "t2"].dropna(subset=["bloc"]).groupby(["circo", "bloc"]).voix.sum().unstack("bloc").fillna(0.0)
    winner = t2.idxmax(axis=1)

    circos = [x for x in t1.index if x in t1g.index and x in winner.index]
    exprimes = t1g.exprimes
    ab_pct = (t1g.abstentions / t1g.inscrits * 100.0)

    real = winner.reindex(circos)
    real_counts = real.value_counts().to_dict()
    print(f"\nReal 2024 seat winners (by max T2 votes, n={len(circos)}): {real_counts}")

    for d in [0.0, 0.50, 0.60, 0.69, 0.80]:
        pred = {}
        for circo in circos:
            row = t1.loc[circo]
            ex = exprimes[circo]
            if ex <= 0:
                continue
            gg = row.get("G", 0) / ex * 100
            cd = row.get("CD", 0) / ex * 100
            ed = row.get("ED", 0) / ex * 100
            pred[circo] = W.seat_winner(gg, cd, ed, ab_pct[circo], "union", 0.5, False, d)
        p = pd.Series(pred)
        acc = (p == real.reindex(p.index)).mean()
        pc = p.value_counts().to_dict()
        print(f"desist={d:.2f}  acc={acc:.3f}  seats={ {k: pc.get(k,0) for k in ['G','CD','ED']} }")


if __name__ == "__main__" and False:
    pass
