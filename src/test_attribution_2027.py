"""Test de la table d'attribution des voix régionalistes
(`data/nuance/attribution_regionalistes_2024.csv`).

Une ligne mal orthographiée ne provoque aucune erreur : elle ne correspond simplement à
personne, la voix reste hors blocs et la circonscription reste fausse — en silence. C'est le
mode de panne à couvrir. On vérifie donc, contre les résultats OFFICIELS du 1er tour 2024 :

  1. chaque ligne désigne un candidat qui EXISTE, dans la circonscription indiquée ;
  2. la nuance déclarée correspond à celle du ministère ;
  3. cette nuance est bien HORS des trois blocs — sinon l'attribution est inutile, voire nuisible
     (elle écraserait un classement correct) ;
  4. toute attribution (bloc ≠ NA) porte une preuve non vide, et tout bloc est valide ;
  5. la clé d'injection dans la chaîne (`pipeline_overrides`) est bien formée et sans collision ;
  6. l'attribution ne fait que RÉCUPÉRER du vote : couverture croissante, jamais au-delà de 100 % ;
  7. `report_app/2027/data/coverage.json` est à jour de la table — sinon le site marque les
     mauvaises circonscriptions.

    python3 -u -m src.test_attribution_2027
"""

from __future__ import annotations

import json
import sys

from src import attribution_2027 as A

VALID = {"G", "CD", "ED", "NA"}


def main() -> None:
    table = A.load_table()
    res = A.load_results()
    sets = A.block_sets()
    fails = []

    by_circo = {cid: {A.normalize_name(k["nom"]): k for k in cands}
                for cid, cands in res.items()}

    for (cid, nom), row in table.items():
        key = A.normalize_name(nom)
        # (1) le candidat existe, dans cette circo
        if cid not in by_circo:
            fails.append(f"{cid} : circonscription inconnue du fichier officiel")
            continue
        cand = by_circo[cid].get(key)
        if cand is None:
            near = [n for n in by_circo[cid] if key.split()[-1] in n]
            fails.append(f"{cid} « {nom} » : aucun candidat de ce nom"
                         + (f" — voulu dire {near} ?" if near else ""))
            continue
        # (2) la nuance déclarée est la bonne
        if row["nuance"] and cand["nuance"] != row["nuance"]:
            fails.append(f"{cid} « {nom} » : nuance {row['nuance']} déclarée, "
                         f"{cand['nuance']} au ministère")
        # (3) la nuance est bien hors blocs
        blk = next((b for b, ns in sets.items() if cand["nuance"] in ns), None)
        if blk:
            fails.append(f"{cid} « {nom} » : nuance {cand['nuance']} DÉJÀ classée {blk} — "
                         f"l'attribution écraserait un classement correct")
        # (4) bloc valide + preuve
        if row["bloc"] not in VALID:
            fails.append(f"{cid} « {nom} » : bloc « {row['bloc']} » invalide (attendu {VALID})")
        if row["bloc"] != "NA" and not row["preuve"].strip():
            fails.append(f"{cid} « {nom} » : attribution sans preuve")

    # (5) clés d'injection dans la chaîne d'entraînement
    ov = A.pipeline_overrides()
    n_attrib = sum(1 for r in table.values() if r["bloc"] in ("G", "CD", "ED"))
    if len(ov) != n_attrib:
        fails.append(f"collision de clés : {n_attrib} attributions → {len(ov)} clés uniques")
    for (dept, nom) in ov:
        if not dept or not nom:
            fails.append(f"clé mal formée : ({dept!r}, {nom!r})")

    # (6) l'attribution récupère du vote, sans jamais dépasser 100 %
    rows = A.build()
    for r in rows:
        if r["couverture_apres"] < r["couverture_avant"] - 1e-9:
            fails.append(f"{r['circo']} : couverture en BAISSE "
                         f"({r['couverture_avant']} → {r['couverture_apres']})")
        if r["couverture_apres"] > 100.0 + 1e-9:
            fails.append(f"{r['circo']} : couverture > 100 % ({r['couverture_apres']})")

    # (7) le fichier servi reflète la table
    if not A.COVERAGE_OUT.exists():
        fails.append(f"{A.COVERAGE_OUT} absent — lancez `python3 -m src.attribution_2027`")
    else:
        served = json.loads(A.COVERAGE_OUT.read_text())
        stale = []
        for key, col in (("cov_avant", "couverture_avant"), ("cov_apres", "couverture_apres")):
            got = served.get(key, {})
            stale += [f"{r['circo']}/{key}" for r in rows
                      if abs(got.get(r["circo"], -1) - r[col]) > 1e-6]
        if stale:
            fails.append(f"coverage.json périmé sur {len(stale)} circos ({stale[:5]}…) — "
                         f"relancez `python3 -m src.attribution_2027`")

    moved = [r for r in rows if r["recupere"] > 0.05]
    print(f"Table : {len(table)} décisions ({n_attrib} attributions, "
          f"{len(table) - n_attrib} non-attributions assumées)")
    print(f"Candidats retrouvés au ministère : "
          f"{'OK' if not any('aucun candidat' in f for f in fails) else 'ÉCHEC'}")
    print(f"Circos corrigées : {len(moved)} | vote récupéré : "
          f"{sum(r['recupere'] for r in moved):.1f} pts cumulés")
    for f in fails:
        print("  ✗", f)
    if fails:
        sys.exit(1)
    print("\n✅ Table d'attribution cohérente avec les résultats officiels 2024, "
          "et couverture servie à jour.")


if __name__ == "__main__":
    main()
