"""Export CSV de la **liste des 577 circonscriptions avec leur prévision 2027**.

Lit les données SERVIES au site (`report_app/2027/data/circo.json` + `summary.json`) et rejoue
le même modèle que la carte (`winnability_2027`, miroir de `js/compute.js`) : la ligne exportée
est exactement ce qu'affiche le panneau d'une circo à l'état initial du scénario choisi.

    python3 -u -m src.export_circos_2027                    # scénario de référence → circos_2027.csv
    python3 -u -m src.export_circos_2027 --scenario union
    python3 -u -m src.export_circos_2027 --scenario all     # 4 scénarios empilés (colonne `scenario`)

Garde-fou : chaque ligne porte `couverture_2024` / `fiabilite` (cf. `coverage_2027`). Les circos
où les forces régionalistes échappent aux trois blocs du modèle — Guyane, Corse, Pacifique… —
sortent marquées `faible` : leurs colonnes sont remplies, mais leur score ne veut rien dire et ne
doit pas être publié. Le script les liste en fin d'exécution. Elles sont 19 tant que la chaîne
n'a pas été reconstruite avec la table d'attribution (`src/attribution_2027`), 11 après.

Portée : l'état INITIAL des présélections (abstention = AB_REF, coefficients calibrés). Les
scénarios sont posés à l'abstention de référence, où le couplage participation γ du site est
l'identité (invariant vérifié par `test_parity_2027`) : l'export coïncide donc au chiffre près
avec la carte tant qu'aucun curseur n'a bougé. Un état de curseurs quelconque n'est pas
exportable ici — il se lit sur le site.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src import coverage_2027, radical_spatial, winnability_2027 as W

SERVED = Path("report_app/2027/data")
OUT = Path("circos_2027.csv")

COLS = [
    "scenario", "circo", "dept", "commune_ancre", "inscrits", "bureaux",
    "pred_G", "pred_CD", "pred_ED", "pred_AB",
    "ic90_G", "ic90_CD", "ic90_ED",
    "part_radicale", "gauche_meilleur_pole", "gauche_qualifiee",
    "marge_t2", "adversaire", "score", "score_label", "vainqueur",
    "couverture_2024", "couverture_source", "fiabilite",
]


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return min(hi, max(lo, v))


def rows(scn: dict, arr: dict, hw: dict[str, float], cov: tuple, thr: float) -> list[dict]:
    """Une ligne par circo pour un scénario, dans l'ordre servi (= ordre de la carte)."""
    m, cfg, ru = scn["means"], scn["left_config"], scn.get("right_union", False)
    cov_val, cov_src = cov
    out = []
    for i, cid in enumerate(arr["id"]):
        g = _clamp(m["G"] + arr["dG"][i])
        cd = _clamp(m["CD"] + arr["dCD"][i])
        ed = _clamp(m["ED"] + arr["dED"][i])
        ab = _clamp(m["AB"] + arr["dAB"][i])
        # Part radicale (LFI) DANS la gauche : moyenne = scénario, motif spatial = 2017 amplifié.
        rad = 1.0 if cfg == "union" else min(
            0.95, max(0.05, scn["radical_share"] + radical_spatial.RAD_GAIN * arr["rdev"][i]))
        r = W.score_circo(g, cd, ed, ab, cfg, rad, ru)
        out.append({
            "scenario": scn["key"], "circo": cid, "dept": arr["dept"][i],
            "commune_ancre": arr["nm"][i], "inscrits": arr["ins"][i], "bureaux": arr["nbv"][i],
            "pred_G": round(g, 1), "pred_CD": round(cd, 1), "pred_ED": round(ed, 1),
            "pred_AB": round(ab, 1),
            **{f"ic90_{b}": hw[b] for b in ("G", "CD", "ED")},
            "part_radicale": round(rad, 3),
            "gauche_meilleur_pole": r["l_best"],
            "gauche_qualifiee": int(r["qualifies"]),
            "marge_t2": r["margin_t2"], "adversaire": r["opp"],
            "score": r["score"], "score_label": W.SCORE_LABELS[r["score"]],
            "vainqueur": W.seat_winner(g, cd, ed, ab, cfg, rad, ru),
            # Garde-fou de publication : part du vote 2024 que la nomenclature de blocs
            # rattache réellement (cf. coverage_2027). Sous le seuil, la ligne reste complète
            # mais NE DOIT PAS être publiée telle quelle — le modèle n'y voit qu'une fraction
            # de l'électorat (forces régionalistes hors blocs).
            "couverture_2024": cov_val[i], "couverture_source": cov_src[i] or "",
            "fiabilite": coverage_2027.flag(cov_val[i], thr),
        })
    return out


def check(recs: list[dict], key: str, summary: dict) -> str:
    """Les totaux de l'export doivent reproduire la répartition servie (summary.winnability)."""
    ref = summary["winnability"][key]
    seats = {b: sum(1 for r in recs if r["vainqueur"] == b) for b in ("G", "CD", "ED")}
    counts = {s: sum(1 for r in recs if r["score"] == s) for s in range(1, 6)}
    ok = seats == {k: int(v) for k, v in ref["seats"].items()} and \
        counts == {int(k): int(v) for k, v in ref["counts"].items()}
    return (f"sièges G/CD/ED {seats['G']}/{seats['CD']}/{seats['ED']} | "
            f"jouables (1-3) {sum(counts[s] for s in (1, 2, 3))} | "
            f"{'✅ conforme aux chiffres servis' if ok else '❌ DIVERGE de summary.json'}")


def main() -> None:
    summary = json.loads((SERVED / "summary.json").read_text())
    keys = [s["key"] for s in summary["scenarios"]]
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--scenario", default=summary["default_scenario"], choices=[*keys, "all"])
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()

    arr = json.loads((SERVED / "circo.json").read_text())
    hw = summary["circo_halfwidth_90"]
    cov = coverage_2027.coverage(arr, summary)
    thr = coverage_2027.threshold(summary)
    wanted = keys if a.scenario == "all" else [a.scenario]

    recs: list[dict] = []
    for scn in summary["scenarios"]:
        if scn["key"] not in wanted:
            continue
        rs = rows(scn, arr, hw, cov, thr)
        print(f"  {scn['key']:<12} {scn['label']:<32} {check(rs, scn['key'], summary)}")
        recs += rs

    with a.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(recs)
    print(f"{len(recs)} lignes ({len(arr['id'])} circos × {len(wanted)} scénario(s)) → {a.out}")

    flagged = sorted({r["circo"] for r in recs if r["fiabilite"] != coverage_2027.OK})
    if flagged:
        print(f"\n⚠ {len(flagged)} circos NON PUBLIABLES (couverture de blocs < {thr} % — forces "
              f"régionalistes hors nomenclature). Leurs lignes sont complètes mais leur score ne "
              f"vaut rien :\n   {' '.join(flagged)}\n   Détail : python3 -m src.coverage_2027")


if __name__ == "__main__":
    main()
