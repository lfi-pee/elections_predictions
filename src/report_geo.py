"""Découpe les contours bureau de vote en GeoJSON simplifiés par département.

`scan`   : repère les bureaux sans contour (territoires particuliers) →
           `data/report/contourless.json`. Indépendant de la table maître.
`export` : lit `bv_master.parquet`, simplifie chaque polygone, attache les
           propriétés légères nécessaires à la carte + au moteur de scénario
           client, et écrit `report_app/data/bv/<dept>.geojson` (un par dept).

Propriétés volontairement minimales (octets = fluidité) : prédictions par bloc
(le client recalcule bloc en tête + marge sous scénario), marge, disputé,
inscrits, point de bascule ED, nom de commune. Le détail lourd (réel, intervalles,
SHAP) est servi à la demande par `report_shap.py`.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import pandas as pd
from shapely.geometry import mapping, shape

CONTOURS = Path("data/geo/contours-bv.geojson")
MASTER = Path("data/report/bv_master.parquet")
CACHE = Path("data/report")
WHY_LEFT = Path("data/report/why_left.json")
OUT = Path("report_app/data/bv")
PRED_CSV = Path("data/predictions_with_intervals.csv")

SIMPLIFY_TOL = 0.00015
PRECISION = 5

# --------------------------------------------------------------------------------------
# Communes désynchronisées des contours
# --------------------------------------------------------------------------------------
# Les contours data.gouv sont figés sur le REU du 1er juin 2022 ; les prédictions portent
# la numérotation des scrutins récents. Une commune qui a RENUMÉROTÉ ses bureaux depuis
# (Bordeaux : 1101 → 1001, 1201 → 1021, 1301 → 1041…) n'a donc plus les mêmes clés des
# deux côtés du `join` par `codeBureauVote`. Deux dégâts, dont le second est le pire :
#
#   1. la quasi-totalité de ses bureaux tombe dans le `continue` silencieux ;
#   2. les rares codes qui coïncident PAR ACCIDENT sont servis — à Bordeaux, 18 sur 153,
#      dont les propriétés (prédictions, inscrits, plancher) sont celles d'un bureau et la
#      géométrie celle d'un AUTRE. Le contour 33063_1101 comptait 686 inscrit·es en 2022 ;
#      la carte y affichait les 1 349 du bureau numéroté 1101 en 2024.
#
# Le TÉMOIN n'est pas la couverture, c'est l'écart d'INSCRITS. Deux raisons :
#
#   · la couverture est AVEUGLE au redécoupage. Une commune qui refond ses bureaux sans
#     changer d'espace de codes — 17 bureaux fusionnés en 11 — garde 100 % de couverture
#     tout en n'ayant plus aucun appariement valide : son 0001 de 2024 ne décrit plus le
#     territoire du 0001 de 2022. C'est le cas le plus RÉPANDU : 79 communes, 829 features ;
#   · la couverture ACCUSE À TORT. Une couverture partielle est le cas NORMAL d'une commune
#     qui a seulement créé des bureaux : ses codes appariés désignent bien le même bureau
#     des deux côtés. Saint-Victoret n'a que 42 % de couverture et un appariement juste.
#
# On compare donc, sur les codes que la jointure APPARIE, les inscrits d'un scrutin d'avant
# le millésime des contours et d'un scrutin d'après (`general_results.parquet`, qui porte les
# 56 scrutins par bureau sous le même `id_brut_miom` que les contours). Il faut DEUX
# statistiques, parce qu'elles ne voient pas la même chose.
#
# 1. L'ÉCART MÉDIAN dit « la plupart des appariements sont faux ». Sur les 1 826 communes
#    dont l'ensemble des codes est identique d'un millésime à l'autre — donc présumées
#    intactes — il vaut 1,9 % en médiane, 5,7 % au 95e centile, 10,5 % au 99e.
#
# 2. La PART DES BUREAUX FRANCHEMENT FAUX dit « une partie des appariements est fausse », ce
#    que la médiane, justement robuste, cache. Bordeaux le montre : ses 18 codes coïncidant
#    par accident ont des écarts de 0, 2, 3, 3, 5, 5, 5, 8, 11, 14, puis 18, 24, 27, 29, 30,
#    51, 97 et 108 % — médiane 12,4 %, SOUS le seuil, alors que huit bureaux sur dix-huit
#    sont grossièrement faux. La moitié basse n'est pas un signe de justesse : ses bureaux
#    faisant tous entre 600 et 1 400 inscrit·es, un appariement au hasard tombe juste une
#    fois sur deux. Sur les communes intactes, cette part vaut 0,0 % jusqu'au 90e centile et
#    0,9 % au 95e : FRAC_FAUX_MAX est vingt-cinq fois ce 95e centile.
#
# Au-delà de l'un ou l'autre seuil, l'appariement est déclaré faux et la commune sort de la
# carte des bureaux — elle reste servie à l'échelle communale, où aucun code de bureau
# n'intervient. Mieux vaut aucun contour qu'un contour faux : les propriétés d'un bureau
# posées sur le polygone d'un autre font un chiffre FAUX, pas un chiffre absent.
#
# Mêmes témoins et mêmes seuils que prep_elections.construire_crosswalk_renumerotation dans
# devoirs_maison, qui s'en sert pour DÉCLENCHER un alignement ordonné et répare ainsi
# 30 communes sur 131 au lieu de les écarter. Cet export-ci n'a pas de quoi remapper : il ne
# peut que refuser.
ECART_MAX = 0.15  # 99e centile de l'écart médian des communes intactes
ECART_BUREAU_MAX = 0.20  # au-delà, l'écart d'UN bureau n'est plus une dérive de listes
FRAC_FAUX_MAX = 0.25  # 25 × le 95e centile de cette part sur les communes intactes
CODES_APPARIES_MIN = 5  # sous 5 codes appariés, ces témoins ne sont pas fiables
# Troisième témoin : un alignement ordonné qui contredit l'identité (cf. plus bas).
COUV_EXAMEN = 0.90  # on ne réaligne que les communes au rattachement lacunaire
ECART_ALIGNEMENT_MAX = 0.06  # 95e centile de l'écart des communes intactes
COUT_TROU = 0.35  # coût d'un bureau laissé non apparié par l'alignement
GENERAL = Path("data/elections/agregees/general_results.parquet")
SCRUTIN_AVANT = ("2022_legi_t1", "2022_pres_t1")  # d'avant le millésime des contours
SCRUTIN_APRES = "2024_euro_t1"


def codes_contours() -> set[str]:
    """Codes de bureau présents dans le fichier de contours.

    Passe LÉGÈRE : on ne demande à ijson que la propriété utile, il saute les géométries —
    ce qui économise l'essentiel des 645 Mo du fichier."""
    with CONTOURS.open("rb") as f:
        return set(ijson.items(f, "features.item.properties.codeBureauVote"))


def communes_desynchronisees() -> dict[str, float]:
    """{code commune → part de bureaux franchement faux} pour les communes dont un code ne
    désigne plus le contour qu'il nomme (cf. le commentaire ci-dessus).

    Renvoie un dict vide si `general_results.parquet` manque : sans témoin on ne devine pas —
    mais on le DIT, plutôt que de servir un appariement non vérifié en silence."""
    if not GENERAL.exists():
        print(
            f"  ⚠ {GENERAL} absent : l'appariement bureau ↔ contour n'est PAS vérifié "
            "(un code renuméroté depuis 2022 serait servi sur le contour d'un autre bureau)"
        )
        return {}
    g = pd.read_parquet(GENERAL, columns=["id_election", "id_brut_miom", "inscrits"])
    apres = g[g["id_election"] == SCRUTIN_APRES].set_index("id_brut_miom")["inscrits"]
    avant = pd.Series(dtype="float64")
    for cle in SCRUTIN_AVANT:
        s = g[g["id_election"] == cle].set_index("id_brut_miom")["inscrits"]
        avant = pd.concat([avant, s[~s.index.isin(avant.index)]])
    if apres.empty or avant.empty:
        print(f"  ⚠ {GENERAL} : scrutins de référence absents — appariement non vérifié")
        return {}
    avec = codes_contours()
    # Codes que la jointure apparie ET que les deux millésimes mesurent : les seuls sur
    # lesquels l'écart d'inscrits dit quelque chose.
    communs = apres.index.intersection(avant.index)
    communs = communs[communs.isin(avec)]
    a = avant[communs].astype(float)
    ecart = (a - apres[communs].astype(float)).abs() / a.clip(lower=1)
    par_com = ecart.groupby(communs.str.slice(0, 5)).agg(
        median="median", size="size", frac_faux=lambda s: (s > ECART_BUREAU_MAX).mean()
    )
    assez = par_com["size"] >= CODES_APPARIES_MIN
    vise = par_com[
        assez
        & ((par_com["median"] > ECART_MAX) | (par_com["frac_faux"] > FRAC_FAUX_MAX))
    ]
    faux = vise["frac_faux"].to_dict()
    # Troisième témoin, pour les communes que les deux premiers laissent passer : l'identité
    # est aussi démentie quand un ALIGNEMENT ORDONNÉ des deux listes de codes, à effectifs
    # concordants, place les bureaux AUTREMENT qu'elle. Quatre communes du corpus ne se
    # voient que comme ça (27375, 38553, 77296, 85222) — c'est ce qui permet à devoirs_maison
    # de les réparer. On ne le tente que là où le rattachement par code est déjà lacunaire :
    # l'alignement est quadratique, et une commune entièrement retrouvée n'a rien à y gagner.
    faux.update(
        _alignement_contredit_identite(
            apres, avant, avec, set(faux), par_com["median"].to_dict()
        )
    )
    return faux


def _aligner(anciens: list[str], nouveaux: list[str], ia, ib) -> list[tuple[str, str, float]]:
    """Alignement ordonné : apparie les deux listes dans l'ordre, en tolérant qu'un bureau
    soit créé ou supprimé d'un côté. Coût d'un couple = écart relatif d'inscrits."""
    n, m = len(anciens), len(nouveaux)
    cout = [
        [abs(ia[a] - ib[b]) / max(ia[a], 1) for b in nouveaux] for a in anciens
    ]
    d = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        d[i][0] = i * COUT_TROU
    for j in range(1, m + 1):
        d[0][j] = j * COUT_TROU
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(
                d[i - 1][j - 1] + cout[i - 1][j - 1],
                d[i - 1][j] + COUT_TROU,
                d[i][j - 1] + COUT_TROU,
            )
    i, j, couples = n, m, []
    while i > 0 and j > 0:
        if d[i][j] == d[i - 1][j - 1] + cout[i - 1][j - 1]:
            couples.append((anciens[i - 1], nouveaux[j - 1], cout[i - 1][j - 1]))
            i, j = i - 1, j - 1
        elif d[i][j] == d[i - 1][j] + COUT_TROU:
            i -= 1
        else:
            j -= 1
    return couples[::-1]


def _alignement_contredit_identite(
    apres, avant, avec, deja: set[str], ecart_identite: dict[str, float]
) -> dict[str, float]:
    """Communes où un alignement ordonné cohérent place les bureaux autrement que l'identité.

    « Autrement » ne suffit pas : il faut que l'alignement soit MEILLEUR. Deux bureaux
    d'effectifs voisins se permutent sans que rien ne le trahisse, et une commune dont
    l'identité concorde déjà (Oullins-Pierre-Bénite : 1,5 % d'écart médian) n'a rien à
    réparer — la déplacer sur la foi d'une permutation gratuite lui ferait perdre sa carte
    pour rien. On exige donc que l'identité, elle, SORTE du seuil des communes intactes."""
    import statistics

    par_com: dict[str, list[str]] = {}
    for c in apres.index:
        par_com.setdefault(c[:5], []).append(c)
    trouve: dict[str, float] = {}
    for com, nouveaux in par_com.items():
        if com in deja or len(nouveaux) < CODES_APPARIES_MIN:
            continue
        tot = float(apres[nouveaux].sum())
        if tot <= 0:
            continue
        if sum(float(apres[c]) for c in nouveaux if c in avec) / tot >= COUV_EXAMEN:
            continue  # rattachement par code complet : rien à gagner à réaligner
        anciens = sorted(c for c in avec if c[:5] == com and c in avant.index)
        if len(anciens) < CODES_APPARIES_MIN:
            continue
        if ecart_identite.get(com, 0.0) <= ECART_ALIGNEMENT_MAX:
            continue  # l'identité concorde : rien à contredire
        couples = _aligner(anciens, sorted(nouveaux), avant, apres)
        if not couples:
            continue
        ecart = statistics.median(e for _, _, e in couples)
        deplaces = sum(1 for a, b, _ in couples if a != b)
        if ecart <= ECART_ALIGNEMENT_MAX and deplaces:
            trouve[com] = ecart
    return trouve


def _annoncer_desync(desync: dict[str, float]) -> None:
    if not desync:
        return
    pires = sorted(desync.items(), key=lambda kv: -kv[1])[:5]
    detail = ", ".join(f"{c} ({f:.0%} de bureaux faux)" for c, f in pires)
    print(
        f"  ✂ {len(desync)} commune(s) dont l'appariement bureau ↔ contour est démenti par "
        f"les inscrits (renumérotation ou redécoupage depuis 2022) : leurs bureaux ne sont "
        f"PAS servis. Pires écarts : {detail}"
    )


def _pred_locations() -> set[str]:
    return set(pd.read_csv(PRED_CSV, usecols=["location"]).location.unique())


def scan() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    have = set()
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            have.add(feat["properties"]["codeBureauVote"])
    contourless = sorted(_pred_locations() - have)
    (CACHE / "contourless.json").write_text(json.dumps(contourless))
    print(f"scan: {len(have)} contours, {len(contourless)} bureaux sans contour")


def _round_geom(geom: dict) -> dict:
    g = shape(geom).simplify(SIMPLIFY_TOL, preserve_topology=True)

    def r(coords):
        return [[round(x, PRECISION), round(y, PRECISION)] for x, y in coords]

    m = mapping(g)
    if m["type"] == "Polygon":
        m = {"type": "Polygon", "coordinates": [r(ring) for ring in m["coordinates"]]}
    else:
        m = {
            "type": "MultiPolygon",
            "coordinates": [[r(ring) for ring in poly] for poly in m["coordinates"]],
        }
    return m


def export() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    why = json.loads(WHY_LEFT.read_text()) if WHY_LEFT.exists() else {}
    m = pd.read_parquet(MASTER).set_index("location")
    props = m[
        [
            "pred_G",
            "pred_CD",
            "pred_ED",
            "pred_AB",
            "margin",
            "unc",
            "inscrits",
            "ed_tip",
            "mob",
            "abst_floor",
            "act_AB",
            "libelle_commune",
            "code_departement",
            "lag_fallback",
        ]
    ]
    desync = communes_desynchronisees()
    _annoncer_desync(desync)
    by_dept: dict[str, list[dict]] = defaultdict(list)
    kept = 0
    with CONTOURS.open("rb") as f:
        for feat in ijson.items(f, "features.item"):
            loc = feat["properties"]["codeBureauVote"]
            if loc not in props.index or loc[:5] in desync:
                continue
            row = props.loc[loc]
            by_dept[row.code_departement].append(
                {
                    "type": "Feature",
                    "geometry": _round_geom(feat["geometry"]),
                    "properties": {
                        "l": loc,
                        "n": row.libelle_commune,
                        "pg": round(float(row.pred_G), 1),
                        "pc": round(float(row.pred_CD), 1),
                        "pe": round(float(row.pred_ED), 1),
                        "pa": round(float(row.pred_AB), 1),
                        "m": round(float(row.margin), 1),
                        "u": round(float(row.unc), 0),
                        "t": round(float(row.ed_tip), 1),
                        "mv": int(row.mob),
                        "i": int(row.inscrits),
                        "ab": int(round(row.inscrits * row.pred_AB / 100)),
                        # Conjunctural abstainers (predicted − historical floor) — the
                        # denominator γ is read against, so the hover shows the SAME
                        # left-share as the click panel (mob / conjunctural), never the
                        # mobilizable-over-all-abstainers ratio.
                        "cj": int(
                            round(
                                row.inscrits
                                * max(0.0, row.pred_AB - row.abst_floor)
                                / 100
                            )
                        ),
                        "w": why.get(loc, ""),
                        # Lower-confidence prediction: lag features fell back to
                        # the commune aggregate (own-BV history missing or from a
                        # reused precinct). Emitted only when true, to keep size down.
                        **({"fb": 1} if bool(row.lag_fallback) else {}),
                    },
                }
            )
            kept += 1
    for dept, feats in by_dept.items():
        fc = {"type": "FeatureCollection", "features": feats}
        (OUT / f"{dept}.geojson").write_text(
            json.dumps(fc, ensure_ascii=False, separators=(",", ":"))
        )
    size = sum(p.stat().st_size for p in OUT.glob("*.geojson")) / 1e6
    print(f"export: {kept} bureaux, {len(by_dept)} départements, {size:.1f} Mo")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "scan"
    {"scan": scan, "export": export}[mode]()
