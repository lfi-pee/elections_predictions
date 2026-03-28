from __future__ import annotations

import requests
from pathlib import Path

DATA_DIR = Path("data")
ELECTIONS_DIR = DATA_DIR / "elections"
ELECTIONS_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS: list[tuple[str, str]] = [
    # --- Présidentielles ---
    (
        "https://static.data.gouv.fr/resources/resultats-elections-presidentielles-depuis-2002/20181115-095451/resultats-elections-presidentielles-depuis-2002.csv",
        "presidentielles_historique_2002_2017.csv",
    ),
    (
        "https://static.data.gouv.fr/resources/resultats-du-premier-tour-de-lelection-presidentielle-2022-par-commune-et-par-departement/20220413-153144/04-resultats-par-commune.csv",
        "presidentielles_2022_t1.csv",
    ),
    (
        "https://static.data.gouv.fr/resources/resultats-du-second-tour-de-lelection-presidentielle-2022/20220425-091118/04-t2-resultats-par-commune.csv",
        "presidentielles_2022_t2.csv",
    ),
    # --- Législatives ---
    (
        "https://static.data.gouv.fr/resources/resultats-elections-legislatives-depuis-2002/20181115-095149/resultats-elections-legislatives-depuis-2002.csv",
        "legislatives_historique_2002_2017.csv",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-legislatives-des-12-et-19-juin-2022-resultats-definitifs-du-premier-tour/20220614-192729/resultats-par-niveau-subcom-t1-france-entiere.txt",
        "legislatives_2022_t1.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-legislatives-des-12-et-19-juin-2022-resultats-definitifs-du-second-tour/20220621-175945/resultats-par-niveau-subcom-t2-france-entiere.txt",
        "legislatives_2022_t2.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-legislatives-des-30-juin-et-7-juillet-2024-resultats-definitifs-du-1er-tour/20240711-075056/resultats-definitifs-par-communes.csv",
        "legislatives_2024_t1.csv",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-legislatives-des-30-juin-et-7-juillet-2024-resultats-definitifs-du-2nd-tour/20240710-170606/resultats-definitifs-par-commune.csv",
        "legislatives_2024_t2.csv",
    ),
    # --- Municipales ---
    (
        "https://static.data.gouv.fr/resources/resultats-elections-municipales-depuis-2001/20181115-095313/resultats-elections-municipales-depuis-2001.csv",
        "municipales_historique_2001_2014.csv",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-municipales-2020-resultats/20200525-133805/2020-05-18-resultats-communes-de-moins-de-1000.txt",
        "municipales_2020_t1_moins_1000.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-municipales-2020-resultats/20200525-133704/2020-05-18-resultats-communes-de-1000-et-plus.txt",
        "municipales_2020_t1_plus_1000.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/municipales-2020-resultats-2nd-tour/20200629-192436/2020-06-29-resultats-t2-communes-de-moins-de-1000-hab.txt",
        "municipales_2020_t2_moins_1000.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/municipales-2020-resultats-2nd-tour/20200629-192435/2020-06-29-resultats-t2-communes-de-1000-hab-et-plus.txt",
        "municipales_2020_t2_plus_1000.txt",
    ),
    # --- Européennes ---
    (
        "https://static.data.gouv.fr/resources/resultats-des-elections-europeennes-2019/20190531-144212/resultats-definitifs-par-commune.txt",
        "europeennes_2019.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/resultats-des-elections-europeennes-du-9-juin-2024/20240613-154634/resultats-definitifs-par-commune.csv",
        "europeennes_2024.csv",
    ),
    # --- Régionales ---
    (
        "https://static.data.gouv.fr/resources/elections-regionales-2015-et-des-assemblees-de-corse-de-guyane-et-de-martinique-par-communes-resultats-tour-1-1/20160121-102246/Reg_15_Resultats_Communes_T1_c.xlsx",
        "regionales_2015_t1.xlsx",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-regionales-2015-et-des-assemblees-de-corse-de-guyane-et-de-martinique-par-communes-resultats-tour-2-1/20160121-104121/Reg_15_Resultats_Communes_T2c.xlsx",
        "regionales_2015_t2.xlsx",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-regionales-2021-resultats-du-1er-tour-1/20210713-100212/reg-resultats-par-niveau-subcom-t1-france-entiere-2021-07-12-18h41.txt",
        "regionales_2021_t1.txt",
    ),
    (
        "https://static.data.gouv.fr/resources/elections-regionales-2021-resultats-du-2eme-tour/20210713-095756/reg-resultats-par-niveau-subcom-t2-france-entiere-2021-07-12-09h15.txt",
        "regionales_2021_t2.txt",
    ),
    # Européennes 2014 only available as xlsx
    (
        "https://www.data.gouv.fr/storage/f/2014-05-30T10-29-10/euro-2014-resultats-communes-c.xlsx",
        "europeennes_2014.xlsx",
    ),
]


def download_file(url: str, dest_fname: str) -> bool:
    dest = ELECTIONS_DIR / dest_fname
    if dest.exists():
        print(f"  SKIP (exists): {dest_fname}")
        return True
    print(f"  Downloading {dest_fname}...")
    res = requests.get(url, stream=True, timeout=120)
    if res.status_code != 200:
        print(f"  FAILED ({res.status_code}): {url}")
        return False
    with open(dest, "wb") as f:
        for chunk in res.iter_content(chunk_size=65536):
            f.write(chunk)
    size_mb = dest.stat().st_size / 1_000_000
    print(f"  OK: {dest_fname} ({size_mb:.1f} MB)")
    return True


def main() -> None:
    ok, fail = 0, 0
    for url, fname in DOWNLOADS:
        if download_file(url, fname):
            ok += 1
        else:
            fail += 1
    print(f"\nDone: {ok} ok, {fail} failed out of {ok + fail}")


if __name__ == "__main__":
    main()
