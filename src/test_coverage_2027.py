"""Test du GARDE-FOU de publication : le site montre-t-il vraiment qu'une circo n'est pas
couverte par la nomenclature de blocs ?

`src/coverage_2027.py` marque les circos où les forces régionalistes/autonomistes échappent aux
trois blocs du modèle (Guyane, Martinique, Nouvelle-Calédonie, Polynésie, Corse…). Marquer ne
sert à rien si le rendu l'oublie : ce test exécute le VRAI JavaScript du site sur les VRAIES
données servies et vérifie que, sur ces circos,

  1. la carte les sort de la choroplèthe (les deux expressions de couleur lisent l'état `pub`
     et virent au gris) et les entoure d'un liseré tireté (liste d'ids non vide, cohérente) ;
  2. le panneau AFFICHE un bandeau d'avertissement et le tag « non mesurée » ;
  3. le panneau N'AFFICHE PAS de siège probable ni de score — les deux chiffres indéfendables ;
  4. l'avertissement cite la couverture réelle mesurée ;
  5. la légende porte la pastille, avec le bon compte ;
  6. une circo normale, elle, garde son score et son siège probable (pas de sur-marquage).

Il ÉCHOUE si quelqu'un débranche le garde-fou en retouchant panel.js, map.js ou config.js.

    python3 -u -m src.test_coverage_2027
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from src import coverage_2027 as C

JS_DIR = Path("report_app/2027/js")
DATA_DIR = Path("report_app/2027/data")
HARNESS = Path("src/coverage_harness.js")

# Témoins : la pire circo encore marquée, une circo corse (non-attribution assumée : groupe
# LIOT, hors axe), une circo RÉPARÉE par la table d'attribution qui doit avoir retrouvé son
# score, et une circo normale qui ne doit RIEN perdre.
PROBES = ["2B-01", "2A-02", "ZC-02", "75-01"]


def _node() -> str:
    n = shutil.which("node") or str(Path.home() / ".local/node/bin/node")
    if not Path(n).exists():
        sys.exit("Node introuvable (installez-le ; cf. $HOME/.local/node/bin).")
    return n


def main() -> None:
    arr, summary = C.load()
    thr = C.threshold(summary)
    vals, srcs = C.coverage(arr, summary)
    flags = [C.flag(v, thr) for v in vals]
    py_low = [cid for cid, f in zip(arr["id"], flags) if f != C.OK]
    idx = {cid: i for i, cid in enumerate(arr["id"])}

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(PROBES, f)
        ids_path = f.name
    try:
        raw = subprocess.run([_node(), str(HARNESS), str(JS_DIR), str(DATA_DIR), ids_path],
                             capture_output=True, text=True, check=True).stdout
    finally:
        os.unlink(ids_path)
    r = json.loads(raw)
    fails = []

    # (1) Carte : les deux expressions de couleur neutralisent les circos marquées.
    for name in ("win_color_expr", "seat_color_expr"):
        if '"pub"' not in r[name] or r["grey"] not in r[name]:
            fails.append(f"carte : {name} ne neutralise pas les circos marquées (état `pub`)")
    if sorted(r["unpublishable"]) != sorted(py_low):
        fails.append("carte : la liste du liseré tireté diffère du marquage Python")

    # (2-4) Panneau, circo par circo.
    for cid in PROBES:
        p, expect_pub = r["panels"][cid], flags[idx[cid]] == C.OK
        if p["publishable"] != expect_pub:
            fails.append(f"{cid} : le site le juge publiable={p['publishable']}, Python {expect_pub}")
        if expect_pub:
            # (6) Pas de sur-marquage : une circo normale garde tout.
            for k, want in (("warning_box", False), ("unmeasured_tag", False),
                            ("seat_line", True), ("no_score_heading", False)):
                if p[k] is not want:
                    fails.append(f"{cid} (normale) : {k}={p[k]}, attendu {want}")
            continue
        for k, want in (("warning_box", True), ("unmeasured_tag", True),
                        ("seat_line", False), ("no_score_heading", True)):
            if p[k] is not want:
                fails.append(f"{cid} (marquée) : {k}={p[k]}, attendu {want}")
        cov, src = vals[idx[cid]], srcs[idx[cid]]
        if cov is not None:
            shown = f"{cov:.1f}".replace(".", ",")
            if shown not in p["warning_text"]:
                fails.append(f"{cid} : l'avertissement ne cite pas la couverture {shown} %")
            if "de cette circonscription" not in p["warning_text"]:
                fails.append(f"{cid} : l'avertissement ne situe pas la mesure")

    # (5) Légende. La pastille « non mesurée » n'est requise QUE s'il reste des circos marquées.
    # Bloc « Autre » modélisé (4e bloc) ⇒ plus aucune circo hors couverture ⇒ garde-fou retiré.
    if py_low and not r["legend_has_chip"]:
        fails.append("légende : pastille « non mesurée » absente alors que des circos sont marquées")
    if r["n_low"] != len(py_low):
        fails.append(f"légende : compte {r['n_low']} ≠ {len(py_low)} circos marquées")

    print(f"Seuil de couverture : {thr} % | circos marquées : {len(py_low)}")
    print(f"Carte (gris + liseré) : {'OK' if not any('carte' in f for f in fails) else 'ÉCHEC'}")
    print(f"Panneau ({len(PROBES)} témoins) : "
          f"{'OK' if not any(p in f for f in fails for p in PROBES) else 'ÉCHEC'}")
    print(f"Légende : {'OK' if not any('légende' in f for f in fails) else 'ÉCHEC'}")
    for f in fails:
        print("  ✗", f)
    if fails:
        sys.exit(1)
    if py_low:
        print("\n✅ Le site refuse de publier un score sur les circos que la nomenclature ne "
              "couvre pas — carte grisée, bandeau d'avertissement, ni score ni siège annoncés.")
    else:
        print("\n✅ Bloc « Autre » modélisé : les 577 circos sont couvertes et publiables "
              "(garde-fou de grisage retiré) ; Python et JavaScript sont d'accord.")


if __name__ == "__main__":
    main()
