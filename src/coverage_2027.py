"""Couverture de la nomenclature de blocs, par circonscription — garde-fou de publication.

Le modèle range chaque candidat dans l'un de trois blocs (Gauche, Centre+Droite, Extrême
Droite). Les nuances **régionalistes / autonomistes** du ministère (`REG` et les codes « divers »
locaux) n'appartiennent à aucun des trois : les voix qu'elles portent tombent dans « Autre » et
sortent du modèle. Là où une telle force domine — Guyane, Martinique, Nouvelle-Calédonie,
Polynésie, Corse — la prédiction ne porte plus que sur une fraction de l'électorat, et le score
de jouabilité qui en découle n'est pas publiable. Cas extrême, Cayenne (ZC-01) : **25,6 %**
seulement du vote exprimé de 2024 est rattaché à un bloc, la gauche y pèse 1,2 % dans nos
données — alors qu'elle détient le siège.

`cross_type_ridge._apply_candidate_lineage` répare déjà une partie du problème : un candidat
codé « Autre » récupère le bloc où le ministère l'avait codé à un scrutin ANTÉRIEUR (le cas
Rimane, DVG en 2022 → REG en 2024). Cette réparation est sans effet sur un élu codé régionaliste
depuis toujours : il n'y a aucun codage antérieur à retrouver.

Ce module ne corrige pas la nomenclature — il **mesure le trou** et marque les circos concernées,
pour que ni la carte ni l'export ne publient un chiffre que le premier contradicteur venu peut
renverser en citant le nom du député sortant. La correction de fond (rattacher les forces
autonomistes à un bloc, territoire par territoire) demande une décision politique explicite
ET une reconstruction complète du pipeline : voir `REMEDIATION` en bas de fichier.

**Mesure.** Couverture d'une circo = part du vote exprimé de 2024 réellement rattachée à l'un des
trois blocs, servie par `coverage.json` et produite par `src/attribution_2027.py` : elle compose
ce que la chaîne couvre déjà (réparation de lignée incluse) avec la table d'attribution des voix
régionalistes. Elle est MESURÉE sur le fichier officiel du ministère pour les **577** circos, y
compris les 76 absentes du backtest 2024 — l'ancienne version, faute de ce fichier, estimait
celles-là par la couverture de leur département.

**Seuil.** Pas un chiffre rond posé à la main : une circo est marquée quand la part de vote NON
rattachée dépasse la demi-largeur à 90 % la plus large que le modèle s'accorde lui-même
(`summary.circo_halfwidth_90`, soit ±9,7 pts sur le Centre+Droite → seuil de couverture 90,3 %).
Au-delà, l'erreur de nomenclature domine à elle seule l'incertitude annoncée : la prédiction est
hors de son enveloppe validée. Sur les données servies, cela marque 15 circos mesurées + 3 par
report départemental, et laisse 1 inconnue (Wallis-et-Futuna) — tout le reste de la métropole est
au-dessus de 89,6 %.

    python3 -u -m src.coverage_2027        # audit lisible des circos marquées
"""

from __future__ import annotations

import json
from pathlib import Path

SERVED = Path("report_app/2027/data")
COVERAGE = SERVED / "coverage.json"

# Étiquettes de fiabilité (miroir exact de js/coverage.js).
OK, LOW, UNKNOWN = "mesuree", "faible", "inconnue"
LABELS = {
    OK: "mesurée",
    LOW: "non fiable — nomenclature de blocs incomplète",
    UNKNOWN: "inconnue — aucune référence 2024 sur le territoire",
}


def threshold(summary: dict) -> float:
    """Couverture minimale exigée = 100 − la plus large demi-largeur à 90 % servie.

    En deçà, la part de vote non rattachée à un bloc dépasse l'incertitude que le modèle
    s'accorde : l'erreur de nomenclature domine, la prédiction sort de son enveloppe validée.
    """
    hw = summary["circo_halfwidth_90"]
    return round(100.0 - max(float(v) for v in hw.values()), 3)


def applied(summary: dict) -> bool:
    """La table d'attribution est-elle passée dans la CHAÎNE (et pas seulement mesurée) ?

    Estampille posée par `report_data_2027` à la reconstruction. Tant qu'elle manque, les
    déviations 2027 servies descendent de l'ancienne nomenclature : la donnée 2024 a beau être
    réparée, la PRÉVISION ne l'est pas, et le marquage doit rester au niveau d'avant.
    """
    return bool(summary.get("attribution_applied"))


def coverage(arr: dict, summary: dict | None = None) -> tuple[list[float | None], list[str | None]]:
    """(couverture %, origine) par circo, dans l'ordre servi.

    Lit `coverage.json` (mesure officielle, cf. `attribution_2027`) et choisit la couverture
    AVANT ou APRÈS attribution selon que la chaîne a été reconstruite (`applied`). Une circo
    absente du fichier ressort `None` : inconnue, ce qui n'est pas la même chose que mauvaise.
    """
    raw = json.loads(COVERAGE.read_text()) if COVERAGE.exists() else {}
    after = applied(summary or {})
    cov = raw.get("cov_apres" if after else "cov_avant", {})
    # Le bloc « Autre » est-il désormais MODÉLISÉ (4e bloc servi) ? Si oui, le vote hors-axe
    # n'est plus « non couvert » : il est prédit comme les autres (avec sa propre incertitude
    # conforme). La couverture d'une circo redevient alors complète — le garde-fou de grisage,
    # simple palliatif de l'absence de modèle sur ces territoires, se retire de lui-même.
    autre_modeled = "dAU" in arr
    vals: list[float | None] = []
    srcs: list[str | None] = []
    for i, cid in enumerate(arr["id"]):
        v = cov.get(cid)
        if v is not None and autre_modeled:
            # Le vote hors-axe est désormais un bloc prédit : la circo est couverte à 100 %
            # (l'incertitude propre à l'Autre est portée par sa fourchette conforme, pas par un
            # grisage). Les circos sans AUCUNE référence (v=None, ex. Wallis) restent « inconnues ».
            v = 100.0
        vals.append(round(float(v), 3) if v is not None else None)
        srcs.append(("mesure" if after else "mesure (avant reconstruction)")
                    if v is not None else None)
    return vals, srcs


def flag(cov: float | None, thr: float) -> str:
    """Étiquette de fiabilité d'une circo depuis sa couverture."""
    if cov is None:
        return UNKNOWN
    return LOW if cov < thr else OK


def audit(arr: dict, summary: dict) -> list[dict]:
    """Une ligne par circo NON publiable (marquée `faible` ou `inconnue`), la pire d'abord."""
    thr = threshold(summary)
    vals, srcs = coverage(arr, summary)
    out = []
    for i, cid in enumerate(arr["id"]):
        f = flag(vals[i], thr)
        if f == OK:
            continue
        out.append({"circo": cid, "nm": arr["nm"][i], "dept": arr["dept"][i],
                    "couverture": vals[i], "origine": srcs[i], "fiabilite": f})
    return sorted(out, key=lambda r: (r["couverture"] is not None, r["couverture"] or 0))


def load() -> tuple[dict, dict]:
    return (json.loads((SERVED / "circo.json").read_text()),
            json.loads((SERVED / "summary.json").read_text()))


def main() -> None:
    arr, summary = load()
    thr = threshold(summary)
    rows = audit(arr, summary)
    if not applied(summary):
        print("  ⓘ Table d'attribution MESURÉE mais pas encore dans la chaîne : le marquage "
              "reste au niveau d'avant\n    (les déviations 2027 servies datent de l'ancienne "
              "nomenclature). Lancez ./rebuild_2027.sh.")
    hw = summary["circo_halfwidth_90"]
    worst = max(hw, key=lambda k: hw[k])
    print(f"Seuil de couverture : {thr} % (= 100 − {hw[worst]}, demi-largeur 90 % la plus "
          f"large, bloc {worst})")
    if not COVERAGE.exists():
        print(f"  ⚠ {COVERAGE} absent — lancez `python3 -m src.attribution_2027`.")
    print(f"{len(arr['id'])} circos → {len(rows)} non publiables :\n")
    for r in rows:
        cov = "  n/a " if r["couverture"] is None else f"{r['couverture']:6.1f}"
        org = f"({r['origine']})" if r["origine"] else "(aucune référence)"
        print(f"  {r['circo']:<6} {r['nm']:<26} {cov} % {org:<18} {LABELS[r['fiabilite']]}")
    ok = len(arr["id"]) - len(rows)
    print(f"\n{ok} circos publiables ({100 * ok / len(arr['id']):.1f} %).")


# ── Correction de fond (NON appliquée ici) ────────────────────────────────────────────────
# Marquer n'est qu'un garde-fou : la carte reste aveugle sur ces territoires. La vraie
# correction est de rattacher les forces autonomistes à un bloc, ce qui suppose :
#   1. une décision politique explicite, territoire par territoire — MDES/Guyane insoumise et
#      Tavini (Polynésie) à gauche, les loyalistes calédoniens au centre-droit, la mouvance
#      nationaliste corse scindée (Corsica Libera à gauche, Femu a Corsica au centre) : il n'y a
#      pas de règle mécanique « REG → un bloc », un mapping uniforme serait faux ;
#   2. une reconstruction complète : `data/predictions_2027.csv` et `data/report/bv_master.parquet`
#      doivent être régénérés (`./rebuild_2027.sh`), la nomenclature entrant dans la CIBLE
#      d'entraînement autant que dans les prédicteurs décalés ;
#   3. une revalidation : le backtest 2024 doit être rejoué, ces territoires n'étant plus neutres.
# Tant que ce n'est pas fait, ces circos doivent rester marquées.
REMEDIATION = "cf. bloc de commentaire ci-dessus : mapping par territoire + rebuild + revalidation"


if __name__ == "__main__":
    main()
