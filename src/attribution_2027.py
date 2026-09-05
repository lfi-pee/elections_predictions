"""Attribution des voix régionalistes/diverses de 2024 aux trois blocs du modèle.

Le modèle range chaque candidat dans Gauche / Centre+Droite / Extrême Droite. Les nuances
REG, DIV et DSV du ministère n'appartiennent à aucun des trois : leurs voix tombent dans
« Autre ». 1,77 % du vote national — négligeable partout, sauf là où la force DOMINANTE porte
une de ces étiquettes. À Cayenne, 74 % du vote exprimé de 2024 échappait ainsi au modèle, qui
n'y voyait la gauche qu'à 1,2 % alors qu'elle détient le siège.

Ce module lit les résultats OFFICIELS 2024 par circonscription, applique
`data/nuance/attribution_regionalistes_2024.csv` (une décision par candidat, avec sa preuve —
groupe parlementaire rejoint, investiture de coalition, mandat antérieur, parti) et recalcule
les parts de bloc et la couverture par circo, AVANT et APRÈS attribution.

Ce que ce module corrige tout de suite : la couche 2024 (parts réelles servies au bouton
« Rejouer 2024 » et mesure de couverture de `coverage_2027`). Ce qu'il NE corrige pas : les
déviations 2027 (`dev_*`), sorties du modèle entraîné avec l'ancienne nomenclature. Elles
exigent `./rebuild_2027.sh` avec cette table branchée dans la chaîne d'entraînement.

    python3 -u -m src.attribution_2027            # rapport avant/après
    python3 -u -m src.attribution_2027 --csv OUT  # + export par circo
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
import urllib.request
from pathlib import Path

TABLE = Path("data/nuance/attribution_regionalistes_2024.csv")
RESULTS = Path("data/elections/legislatives_2024_t1_circo.csv")
RESULTS_URL = (
    "https://static.data.gouv.fr/resources/elections-legislatives-des-30-juin-et-7-juillet-"
    "2024-resultats-definitifs-du-1er-tour/20240710-171413/"
    "resultats-definitifs-par-circonscriptions-legislatives.csv"
)
RIDGE = Path("src/cross_type_ridge.py")
SERVED_CIRCO = Path("report_app/2027/data/circo.json")
COVERAGE_OUT = Path("report_app/2027/data/coverage.json")

# Bloc de la table (G/CD/ED) → vocabulaire de la chaîne d'entraînement (cross_type_ridge).
PIPELINE_BLOCK = {"G": "Gauche", "CD": "Centre+Droite", "ED": "Extreme_Droite"}

# Codes département du ministère → codes internes du projet (report_geo_2027).
DEPT = {"971": "ZA", "972": "ZB", "973": "ZC", "974": "ZD", "975": "ZS", "976": "ZM",
        "986": "ZW", "987": "ZP", "988": "ZN", "ZX": "ZX", "ZZ": "ZZ"}


def block_sets() -> dict[str, set[str]]:
    """Les trois ensembles de nuances, lus dans `cross_type_ridge.py` SANS l'importer.

    Ce sont trois listes de chaînes ; les importer entraînerait scikit-learn et toute la
    chaîne d'entraînement pour un outil qui ne fait que lire un CSV. On les extrait donc du
    source — même source de vérité, aucune duplication à maintenir.
    """
    src = RIDGE.read_text()
    out = {}
    for name, key in (("LEFT", "G"), ("CENTER_RIGHT", "CD"), ("EXTREME_RIGHT", "ED")):
        m = re.search(rf"^{name}\s*=\s*\{{\n(.*?)^\}}", src, re.S | re.M)
        if not m:
            raise RuntimeError(f"{name} introuvable dans {RIDGE} — la structure a changé.")
        out[key] = {ln.strip().rstrip(",").strip('"') for ln in m.group(1).splitlines()
                    if ln.strip() and not ln.strip().startswith("#")}
    for k, sentinel in (("G", "FI"), ("CD", "LR"), ("ED", "RN")):
        if sentinel not in out[k]:
            raise RuntimeError(f"ensemble {k} suspect : « {sentinel} » absent.")
    return out


def normalize_name(name: str) -> str:
    """Nom de candidat en forme canonique : minuscules, sans accent, tirets et apostrophes
    ramenés à des espaces. Les deux côtés d'une correspondance DOIVENT passer par ici — le
    ministère écrit « DUPONT-AIGNAN », la table d'agrégation « Dupont Aignan »."""
    s = unicodedata.normalize("NFD", (name or "").strip().lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[-'’]", " ", s).split())


def dept_key(circo: str) -> str:
    """Département au format de `cross_type_ridge._dept_of` : 3 caractères outre-mer
    (97x/98x, où « 97 » seul confondrait des territoires distincts), 2 sinon."""
    d = circo.split("-")[0]
    inv = {v: k for k, v in DEPT.items()}
    return inv.get(d, d)


def pipeline_overrides() -> dict[tuple[str, str], str]:
    """Table d'attribution au format attendu par `CANDIDATE_BLOCK_OVERRIDES` :
    {(département, nom canonique) : bloc}. Les lignes `NA` (non-attribution assumée) sont
    volontairement absentes : leurs voix restent dans « Autre »."""
    out = {}
    for (cid, nom), row in load_table().items():
        if row["bloc"] in PIPELINE_BLOCK:
            out[(dept_key(cid), normalize_name(nom))] = PIPELINE_BLOCK[row["bloc"]]
    return out


def circo_id(dept: str, code: str) -> str:
    d = DEPT.get(dept) or (dept if dept in ("2A", "2B") else dept.zfill(2))
    n = re.sub(r"^" + re.escape(dept), "", code) or code[-2:]
    return f"{d}-{int(n):02d}"


def fetch_results() -> Path:
    if not RESULTS.exists():
        RESULTS.parent.mkdir(parents=True, exist_ok=True)
        print(f"  téléchargement des résultats officiels 2024 → {RESULTS}")
        urllib.request.urlretrieve(RESULTS_URL, RESULTS)
    return RESULTS


def load_results() -> dict[str, list[dict]]:
    """Candidats par circo depuis le fichier ministère (bloc de 9 colonnes répété)."""
    csv.field_size_limit(10 ** 9)
    with fetch_results().open(encoding="utf-8") as f:
        r = csv.reader(f, delimiter=";")
        hdr = next(r)
        i0 = hdr.index("Numéro de panneau 1")
        out = {}
        for x in r:
            expr = float(x[9].replace(",", ".") or 0)
            cands, j = [], i0
            while j + 8 < len(x) and x[j].strip():
                voix = float(x[j + 5].replace(",", ".") or 0)
                cands.append({"nuance": x[j + 1].strip(), "nom": f"{x[j+3]} {x[j+2]}".strip(),
                              "voix": voix, "pct": 100 * voix / expr if expr else 0.0})
                j += 9
            out[circo_id(x[0], x[2])] = cands
    return out


def load_table() -> dict[tuple[str, str], dict]:
    rows = csv.DictReader(ln for ln in TABLE.read_text().splitlines() if not ln.startswith("#"))
    return {(r["circo"], r["nom"]): r for r in rows}


def table_by_key() -> dict[tuple[str, str], dict]:
    """Table indexée (circo, nom canonique) — tolère les graphies du fichier ministère."""
    return {(cid, normalize_name(nom)): r for (cid, nom), r in load_table().items()}


def classify(cands: list[dict], sets: dict[str, set[str]],
             table: dict, cid: str) -> list[dict]:
    """Ajoute à chaque candidat son bloc `avant` (nuance seule) et `apres` (table appliquée)."""
    for k in cands:
        nu = k["nuance"]
        b = next((blk for blk, s in sets.items() if nu in s), None)
        k["avant"] = b
        dec = table.get((cid, normalize_name(k["nom"])))
        if b is None and dec and dec["bloc"] in ("G", "CD", "ED"):
            k["apres"], k["preuve"] = dec["bloc"], dec["preuve"]
        else:
            k["apres"], k["preuve"] = b, ""
    return cands


def shares(cands: list[dict], key: str) -> tuple[dict[str, float], float]:
    """Parts par bloc (% exprimés) et couverture = part du vote rattachée à un bloc."""
    s = {b: 0.0 for b in ("G", "CD", "ED")}
    for k in cands:
        if k[key]:
            s[k[key]] += k["pct"]
    return s, sum(s.values())


def served_coverage() -> dict[str, float]:
    """Couverture DÉJÀ obtenue par la chaîne, lue dans les parts servies (`circo.json`).

    Elle inclut la réparation de lignée de `cross_type_ridge` (un candidat codé « Autre »
    reprend le bloc où il était codé antérieurement), qui rattrape à elle seule 34 circos —
    Nadeau, Molac, Dupont-Aignan, Gokel, Beaudet, Sempastous. On la COMPOSE avec la table
    plutôt que de la réimplémenter : recalculer la couverture depuis la seule nuance officielle
    la perdrait et ferait régresser une vingtaine de circonscriptions.
    """
    if not SERVED_CIRCO.exists():
        return {}
    a = json.loads(SERVED_CIRCO.read_text())
    return {cid: a["r24G"][i] + a["r24CD"][i] + a["r24ED"][i]
            for i, cid in enumerate(a["id"]) if a["r24G"][i] is not None}


def build() -> list[dict]:
    sets, table, res = block_sets(), table_by_key(), load_results()
    served = served_coverage()
    out = []
    for cid, cands in sorted(res.items()):
        cands = classify(cands, sets, table, cid)
        av, cov_raw = shares(cands, "avant")
        ap, _ = shares(cands, "apres")
        # Part récupérée PAR LA TABLE (candidats sans bloc que l'on attribue explicitement).
        gained = sum(k["pct"] for k in cands if k["avant"] is None and k["apres"])
        # Point de départ = ce que la chaîne couvre déjà (lignée incluse) quand on le connaît ;
        # sinon la nuance brute, pour les 76 circos absentes du backtest 2024.
        base = served.get(cid, cov_raw)
        out.append({"circo": cid, "couverture_avant": round(base, 2),
                    "couverture_apres": round(min(100.0, base + gained), 2),
                    "origine_avant": "servi" if cid in served else "nuance officielle",
                    **{f"r24{b}_nuance": round(av[b], 2) for b in av},
                    **{f"r24{b}_attribue": round(ap[b], 2) for b in ap},
                    "recupere": round(gained, 2)})
    return out


def write_coverage(rows: list[dict]) -> None:
    """Sert la couverture RÉELLE des 577 circos (`report_app/2027/data/coverage.json`).

    Mesurée sur le fichier officiel du ministère, attribution appliquée — donc connue pour
    TOUTES les circonscriptions, y compris les 76 absentes du backtest 2024. Cela remplace
    l'ancienne estimation (somme des parts servies + report départemental pour les circos sans
    référence), qui devinait là où l'on peut mesurer.
    """
    COVERAGE_OUT.parent.mkdir(parents=True, exist_ok=True)
    # On sert les DEUX couvertures. `cov_avant` décrit le modèle tel qu'il est ENTRAÎNÉ
    # aujourd'hui ; `cov_apres` ce qu'il sera une fois la table passée dans la chaîne. Tant que
    # `summary.attribution_applied` est absent ou faux, c'est `cov_avant` qui gouverne le
    # marquage : les déviations 2027 servies proviennent encore de l'ancienne nomenclature, et
    # lever le marquage publierait une prévision périmée sous prétexte que la donnée 2024 est
    # réparée. La reconstruction (`report_data_2027`) pose l'estampille et bascule le marquage.
    payload = {
        "source": "resultats-definitifs-par-circonscriptions-legislatives.csv (1er tour 2024)",
        "attribution": str(TABLE),
        "cov_avant": {r["circo"]: r["couverture_avant"] for r in rows},
        "cov_apres": {r["circo"]: r["couverture_apres"] for r in rows},
    }
    COVERAGE_OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path, help="exporter le détail par circo")
    ap.add_argument("--no-serve", action="store_true",
                    help="ne pas réécrire report_app/2027/data/coverage.json")
    a = ap.parse_args()

    rows = build()
    table = load_table()
    moved = [r for r in rows if r["recupere"] > 0.05]
    print(f"{len(rows)} circonscriptions | table : {len(table)} décisions "
          f"({sum(1 for t in table.values() if t['bloc'] != 'NA')} attributions, "
          f"{sum(1 for t in table.values() if t['bloc'] == 'NA')} non-attributions assumées)\n")
    print(f"{len(moved)} circos corrigées :\n")
    print(f"  {'circo':<7}{'couverture':>22}   {'gauche (parts exprimées)':>26}")
    for r in sorted(moved, key=lambda x: -x["recupere"]):
        print(f"  {r['circo']:<7}  {r['couverture_avant']:5.1f} % → {r['couverture_apres']:5.1f} %"
              f"   (+{r['recupere']:4.1f})      {r['r24G_nuance']:5.1f} % → {r['r24G_attribue']:5.1f} %")

    if not a.no_serve:
        write_coverage(rows)
        print(f"\ncouverture des {len(rows)} circos → {COVERAGE_OUT}")

    if a.csv:
        with a.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\ndétail par circo → {a.csv}")


if __name__ == "__main__":
    main()
