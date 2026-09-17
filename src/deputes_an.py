"""Députés en exercice (XVIIe législature) par circonscription, depuis l'open data de
l'Assemblée nationale — la dimension « sortant » de la négociation 2027.

Source : `AMO10_deputes_actifs_mandats_actifs_organes` (data.assemblee-nationale.fr), un JSON
par acteur avec son mandat de député EN COURS (lieu d'élection : département + n° de circo) et
son groupe politique EN COURS. On en tire, pour chacune des 577 circonscriptions, le député
actuel et son groupe (sigle AN : LFI-NFP, SOC, ECOS, GDR, EPR, DEM, HOR, DR, UDDPLR, RN, LIOT,
NI). C'est l'état ACTUEL de l'Assemblée (remplacements et partielles compris), pas le résultat
de 2024 : ce qui compte pour négocier, c'est qui siège aujourd'hui.

    python3 -u -m src.deputes_an            # lit le zip en cache (le télécharge s'il manque)
                                            # → data/nuance/deputes_2026.csv

Le CSV dérivé est versionné (petit, relu par negotiation_2027) ; le zip source (~5 Mo) reste
en cache local hors dépôt.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.request
import zipfile
from pathlib import Path

URL = ("https://data.assemblee-nationale.fr/static/openData/repository/17/amo/"
       "deputes_actifs_mandats_actifs_organes/AMO10_deputes_actifs_mandats_actifs_organes.json.zip")
CACHE = Path("data/report/AMO10_deputes_actifs.json.zip")
OUT = Path("data/nuance/deputes_2026.csv")

# Codes département de l'AN → préfixe des identifiants de circonscription du site (nomenclature
# du ministère de l'Intérieur : Z? pour l'outre-mer, ZZ pour les Français de l'étranger).
DEPT_MAP = {"099": "ZZ", "971": "ZA", "972": "ZB", "973": "ZC", "974": "ZD", "975": "ZS",
            "976": "ZM", "977": "ZX", "978": "ZX", "986": "ZW", "987": "ZP", "988": "ZN"}

# Groupes AN → bloc du modèle (G / CD / ED) et, à gauche, pôle (LFI / autre gauche).
GROUP_BLOC = {"LFI-NFP": "G", "SOC": "G", "ECOS": "G", "GDR": "G",
              "EPR": "CD", "DEM": "CD", "HOR": "CD", "DR": "CD", "UDDPLR": "ED", "RN": "ED",
              "LIOT": None, "NI": None}
FIELDS = ["circo", "nom", "prenom", "groupe", "groupe_libelle", "qualite", "bloc",
          "debut_mandat", "cause_mandat"]


def circo_id(num_dept: str, num_circo: str) -> str:
    d = DEPT_MAP.get(num_dept, num_dept)
    return f"{d}-{int(num_circo):02d}"


def _load_zip() -> zipfile.ZipFile:
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        print(f"  téléchargement {URL}")
        urllib.request.urlretrieve(URL, CACHE)
    return zipfile.ZipFile(CACHE)


def build() -> list[dict]:
    z = _load_zip()
    names = z.namelist()
    organes = {}
    for n in names:
        if "/organe/" in n and n.endswith(".json"):
            o = json.loads(z.read(n))["organe"]
            organes[o["uid"]] = (o.get("libelleAbrev"), o.get("libelle"))
    rows = []
    for n in names:
        if "/acteur/" not in n or not n.endswith(".json"):
            continue
        a = json.loads(z.read(n))["acteur"]
        ms = a["mandats"]["mandat"]
        ms = ms if isinstance(ms, list) else [ms]
        circ = gp = None
        debut = cause = qual = ""
        for m in ms:
            if m.get("dateFin"):
                continue  # seuls les mandats EN COURS
            if m.get("typeOrgane") == "ASSEMBLEE":
                lieu = m["election"]["lieu"]
                circ = circo_id(lieu["numDepartement"], lieu["numCirco"])
                debut, cause = m.get("dateDebut", ""), m["election"].get("causeMandat", "")
            elif m.get("typeOrgane") == "GP":
                gp = organes.get(m["organes"]["organeRef"], (None, None))
                qual = (m.get("infosQualite") or {}).get("codeQualite", "")
        if circ is None:
            continue
        sig = gp[0] if gp else "NI"
        ident = a["etatCivil"]["ident"]
        rows.append(dict(circo=circ, nom=ident["nom"], prenom=ident["prenom"], groupe=sig,
                         groupe_libelle=(gp[1] if gp else "Non inscrit"), qualite=qual,
                         bloc=GROUP_BLOC.get(sig) or "", debut_mandat=debut, cause_mandat=cause))
    rows.sort(key=lambda r: r["circo"])
    return rows


def load() -> dict[str, dict]:
    """{circo: ligne} depuis le CSV versionné."""
    with OUT.open(encoding="utf-8") as f:
        return {r["circo"]: r for r in csv.DictReader(f)}


def main() -> None:
    rows = build()
    assert len(rows) == 577, f"{len(rows)} députés en exercice (577 attendus)"
    assert len({r["circo"] for r in rows}) == 577, "doublon de circonscription"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    from collections import Counter
    print(f"  {len(rows)} députés → {OUT}")
    print("  groupes :", dict(Counter(r["groupe"] for r in rows).most_common()))


if __name__ == "__main__":
    main()
