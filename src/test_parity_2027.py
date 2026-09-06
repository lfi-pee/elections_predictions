"""Test de PARITÉ Python ↔ JavaScript du modèle de sièges 2027.

Le cœur de calcul existe en double : `src/winnability_2027.py` (chiffres servis + backtests) et
`report_app/2027/js/compute.js` (recalcul en direct au curseur). Rien ne garantissait qu'ils
donnent la même chose — le bug d'abstention (audit ≠ appli) était de cette famille. Ce test
exécute le VRAI compute.js (via Node, `src/parity_harness.js`) et compare, sur une grille de
cas, `seatWinner` / `scoreCirco` / les CONSTANTES aux valeurs Python. Toute dérive future casse
le test.

On vérifie aussi l'INVARIANT γ qui rend le chiffre servi cohérent avec l'appli : à l'abstention
de référence (AB_REF), `turnoutAdjust` doit être l'identité — c'est pourquoi la distribution
servie (calculée SANS γ) égale l'état initial du site (calculé AVEC γ) tant que les scénarios
sont posés à AB_REF.

    python3 -u -m src.test_parity_2027
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src import coverage_2027, radical_spatial, winnability_2027 as W

JS_DIR = Path("report_app/2027/js")
HARNESS = Path("src/parity_harness.js")
GAMMA = Path("report_app/2027/data/gamma_curve.json")
TOL = 1e-6


def _node() -> str:
    n = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    if not Path(n).exists():
        sys.exit("Node introuvable (installez-le ; cf. $HOME/.local/node/bin).")
    return n


def _grid() -> list[dict]:
    """Grille de cas couvrant configs, union des droites, part radicale, niveaux et abstention,
    y compris des bords (gauche faible/forte, seuils de qualif.)."""
    cases = []
    share_sets = [
        (28, 27, 35), (35, 25, 30), (20, 30, 40), (45, 25, 22), (12, 30, 45),
        (31, 34, 35), (26, 24, 40), (18, 22, 48), (40, 30, 20), (15, 15, 60),
    ]
    for (g, cd, ed) in share_sets:
        for ab in (30, 40, 48, 55):
            for cfg in ("union", "split2", "split3"):
                for ru in (False, True):
                    for rad in (0.20, 0.369, 0.50, 0.70):
                        if cfg == "union" and rad != 0.369:
                            continue  # rad sans effet en union
                        cases.append(dict(g=float(g), cd=float(cd), ed=float(ed),
                                          ab=float(ab), cfg=cfg, ru=ru, rad=rad, dAB=0.0, au=0.0))
    # Cas dédiés au bloc « Autre » (bastions régionalistes) : Autre en tête, second, ou marginal,
    # en union comme en division — exerce le pôle « collant » et l'adversaire Autre de scoreCirco.
    for (g, cd, ed, au) in [(20, 20, 18, 40), (18, 24, 32, 26), (10, 37, 23, 34),
                            (26, 44, 12, 18), (30, 25, 25, 15), (12, 12, 12, 55)]:
        for ab in (40, 48):
            for cfg in ("union", "split2"):
                cases.append(dict(g=float(g), cd=float(cd), ed=float(ed), ab=float(ab),
                                  cfg=cfg, ru=False, rad=0.369, dAB=0.0, au=float(au)))
    # Cas dédiés à l'invariant γ : abstention = AB_REF + dAB → turnoutAdjust doit être identité.
    for dAB in (-8.0, -3.0, 0.0, 4.0, 9.0):
        cases.append(dict(g=32.0, cd=30.0, ed=38.0, ab=48.0 + dAB, cfg="union",
                          ru=False, rad=0.369, dAB=dAB, _gamma_invariant=True))
    return cases


def _py(c: dict) -> dict:
    au = c.get("au", 0.0)
    sc = W.score_circo(c["g"], c["cd"], c["ed"], c["ab"], c["cfg"], c["rad"], c["ru"], au=au)
    win = W.seat_winner(c["g"], c["cd"], c["ed"], c["ab"], c["cfg"], c["rad"], c["ru"], au=au)
    return dict(win=win, score=sc["score"], qualifies=sc["qualifies"],
                margin_t2=sc["margin_t2"], opp=sc["opp"])


def _eq(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b) <= TOL
    return a == b


def main() -> None:
    cases = _grid()
    gamma = json.loads(GAMMA.read_text())
    circo_arr, summary = coverage_2027.load()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump({"cases": cases, "gamma": gamma, "circoArr": circo_arr,
                   "summary": summary,
                   "coverage": json.loads(coverage_2027.COVERAGE.read_text())}, f)
        vec_path = f.name
    try:
        raw = subprocess.run([_node(), str(HARNESS), str(JS_DIR), vec_path],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(vec_path)
    res = json.loads(raw)
    js_out, js_consts = res["out"], res["consts"]

    # ── 1. Constantes ──
    py_consts = {
        "desist": W.DESIST_TO_STRONG, "cdLR": W.CD_LR_DEFAULT, "ed2left": W.BARRAGE_ED_TO_LEFT,
        "reunif": W.REUNIF, "DESIST_ED": W.DESIST_TO_ED, "RAD_GAIN": radical_spatial.RAD_GAIN,
        "CDT": {"ensL": W.ENS_TO_LEFT, "ensE": W.ENS_TO_RN, "lrL": W.LR_TO_LEFT, "lrE": W.LR_TO_RN,
                "lrLru": W.LR_TO_LEFT_RU, "lrEru": W.LR_TO_RN_RU},
    }
    const_fail = []
    for k, v in py_consts.items():
        if k == "CDT":
            for kk, vv in v.items():
                if not _eq(vv, js_consts["CDT"].get(kk)):
                    const_fail.append(f"CDT.{kk}: py={vv} js={js_consts['CDT'].get(kk)}")
        elif not _eq(v, js_consts.get(k)):
            const_fail.append(f"{k}: py={v} js={js_consts.get(k)}")

    # ── 2. Sorties du modèle + invariant γ ──
    out_fail, gamma_fail = [], []
    for i, c in enumerate(cases):
        j = js_out[i]
        if c.get("_gamma_invariant"):
            if not (_eq(j["ta_g"], c["g"]) and _eq(j["ta_cd"], c["cd"]) and _eq(j["ta_ed"], c["ed"])):
                gamma_fail.append(f"dAB={c['dAB']}: γ≠identité → ({j['ta_g']:.3f},{j['ta_cd']:.3f},{j['ta_ed']:.3f})")
            continue
        p = _py(c)
        for key in ("win", "score", "qualifies", "margin_t2", "opp"):
            # margin_t2 : Python le renvoie ARRONDI à 0,1 (affichage) ; la DÉCISION de score
            # utilise la valeur non arrondie des deux côtés (d'où score/win/opp identiques). On
            # compare donc le JS arrondi à 0,1 → tolérance de l'arrondi.
            if key == "margin_t2" and p[key] is not None and j[key] is not None:
                if abs(round(j[key], 1) - p[key]) <= TOL:
                    continue
            if not _eq(p[key], j[key]):
                out_fail.append(f"[{c['cfg']} ru={c['ru']} rad={c['rad']} "
                                f"g/cd/ed/ab={c['g']}/{c['cd']}/{c['ed']}/{c['ab']}] "
                                f"{key}: py={p[key]} js={j[key]}")

    # ── 3. Couverture de la nomenclature de blocs (garde-fou de publication) ──
    # Le marquage « non publiable » décide ce que la carte grise et ce que l'export refuse de
    # présenter comme un chiffre : s'il diverge entre Python et JS, l'export et le site ne
    # censurent pas les mêmes circos. On compare sur les 577 circos RÉELLEMENT servies.
    cov_fail, n_low = [], -1
    js_cov = res.get("cov")
    py_thr = coverage_2027.threshold(summary)
    py_val, py_src = coverage_2027.coverage(circo_arr, summary)
    py_flag = [coverage_2027.flag(v, py_thr) for v in py_val]
    n_low = sum(1 for f in py_flag if f != coverage_2027.OK)
    if not js_cov:
        cov_fail.append("coverage.js n'a rien renvoyé (harnais ou données servies absents)")
    else:
        if not _eq(py_thr, js_cov["thr"]):
            cov_fail.append(f"seuil : py={py_thr} js={js_cov['thr']}")
        for i, cid in enumerate(circo_arr["id"]):
            for name, pv, jv in (("couverture", py_val[i], js_cov["val"][i]),
                                 ("origine", py_src[i], js_cov["src"][i]),
                                 ("fiabilite", py_flag[i], js_cov["flag"][i])):
                if not _eq(pv, jv):
                    cov_fail.append(f"{cid} {name}: py={pv} js={jv}")

    n_model = sum(1 for c in cases if not c.get("_gamma_invariant"))
    n_gamma = sum(1 for c in cases if c.get("_gamma_invariant"))
    print(f"Parité constantes  : {'OK' if not const_fail else 'ÉCHEC'} "
          f"({len(py_consts) + len(py_consts['CDT']) - 1} constantes)")
    print(f"Parité modèle      : {'OK' if not out_fail else 'ÉCHEC'} "
          f"({n_model} cas × 5 champs)")
    print(f"Invariant γ@AB_REF : {'OK' if not gamma_fail else 'ÉCHEC'} ({n_gamma} cas)")
    print(f"Parité couverture  : {'OK' if not cov_fail else 'ÉCHEC'} "
          f"({len(circo_arr['id'])} circos"
          f", {n_low} marquées non publiables)")
    for f in (const_fail + out_fail[:20] + gamma_fail + cov_fail[:20]):
        print("  ✗", f)
    if const_fail or out_fail or gamma_fail or cov_fail:
        sys.exit(1)
    print("\n✅ Python et JavaScript calculent le MÊME modèle de sièges (constantes + sorties + γ) "
          "et marquent les MÊMES circos non publiables.")


if __name__ == "__main__":
    main()
