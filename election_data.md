# Elections Data Dictionary

All election result files are in `data/elections/`. All poll files are in `data/polls/`.

## Aggregated Dataset (Bureau de Vote level, 1999–2026)

**Source**: [Données des élections agrégées](https://www.data.gouv.fr/fr/datasets/donnees-des-elections-agregees/) — Ministère de l'Intérieur, consolidated by data.gouv.fr.

Located in `data/elections/agregees/`:

| File | Format | Size | Content |
|------|--------|------|---------|
| `general_results.parquet` | Parquet | 68 MB | Participation, blank/null votes per bureau de vote |
| `candidats_results.parquet` | Parquet | 154 MB | Per-candidate votes per bureau de vote |

**Coverage**: 3,162,440 rows across **56 elections** (both tables join on `id_election` + `id_brut_miom`):

| id_election | Election | Rows |
|---|---|---:|
| `1999_euro_t1` | Européennes 1999 | 63,737 |
| `2001_cant_t1` | Cantonales 2001 T1 | 31,772 |
| `2001_cant_t2` | Cantonales 2001 T2 | 21,218 |
| `2002_pres_t1` | Présidentielle 2002 T1 | 64,141 |
| `2002_pres_t2` | Présidentielle 2002 T2 | 64,141 |
| `2002_legi_t1` | Législatives 2002 T1 | 63,394 |
| `2002_legi_t2` | Législatives 2002 T2 | 57,575 |
| `2004_cant_t1` | Cantonales 2004 T1 | 32,106 |
| `2004_cant_t2` | Cantonales 2004 T2 | 24,817 |
| `2004_euro_t1` | Européennes 2004 | 64,761 |
| `2004_regi_t1` | Régionales 2004 T1 | 62,573 |
| `2004_regi_t2` | Régionales 2004 T2 | 61,966 |
| `2007_pres_t1` | Présidentielle 2007 T1 | 65,617 |
| `2007_pres_t2` | Présidentielle 2007 T2 | 65,617 |
| `2007_legi_t1` | Législatives 2007 T1 | 65,618 |
| `2007_legi_t2` | Législatives 2007 T2 | 51,779 |
| `2008_cant_t1` | Cantonales 2008 T1 | 32,646 |
| `2008_cant_t2` | Cantonales 2008 T2 | 17,689 |
| `2008_muni_t1` | Municipales 2008 T1 | 25,926 |
| `2008_muni_t2` | Municipales 2008 T2 | 11,862 |
| `2009_euro_t1` | Européennes 2009 | 66,583 |
| `2010_regi_t1` | Régionales 2010 T1 | 66,222 |
| `2010_regi_t2` | Régionales 2010 T2 | 65,842 |
| `2011_cant_t1` | Cantonales 2011 T1 | 32,934 |
| `2011_cant_t2` | Cantonales 2011 T2 | 27,023 |
| `2012_pres_t1` | Présidentielle 2012 T1 | 67,932 |
| `2012_pres_t2` | Présidentielle 2012 T2 | 67,932 |
| `2012_legi_t1` | Législatives 2012 T1 | 67,932 |
| `2012_legi_t2` | Législatives 2012 T2 | 62,633 |
| `2014_euro_t1` | Européennes 2014 | 68,246 |
| `2014_muni_t1` | Municipales 2014 T1 | 68,169 |
| `2014_muni_t2` | Municipales 2014 T2 | 22,480 |
| `2015_dpmt_t1` | Départementales 2015 T1 | 65,781 |
| `2015_dpmt_t2` | Départementales 2015 T2 | 60,975 |
| `2015_regi_t1` | Régionales 2015 T1 | 67,776 |
| `2015_regi_t2` | Régionales 2015 T2 | 67,776 |
| `2017_pres_t1` | Présidentielle 2017 T1 | 69,242 |
| `2017_pres_t2` | Présidentielle 2017 T2 | 69,242 |
| `2017_legi_t1` | Législatives 2017 T1 | 69,242 |
| `2017_legi_t2` | Législatives 2017 T2 | 68,767 |
| `2019_euro_t1` | Européennes 2019 | 69,297 |
| `2020_muni_t1` | Municipales 2020 T1 | 68,941 |
| `2020_muni_t2` | Municipales 2020 T2 | 19,594 |
| `2021_dpmt_t1` | Départementales 2021 T1 | 66,016 |
| `2021_dpmt_t2` | Départementales 2021 T2 | 62,063 |
| `2021_regi_t1` | Régionales 2021 T1 | 68,611 |
| `2021_regi_t2` | Régionales 2021 T2 | 68,611 |
| `2022_pres_t1` | Présidentielle 2022 T1 | 69,682 |
| `2022_pres_t2` | Présidentielle 2022 T2 | 69,682 |
| `2022_legi_t1` | Législatives 2022 T1 | 69,682 |
| `2022_legi_t2` | Législatives 2022 T2 | 69,355 |
| `2024_euro_t1` | Européennes 2024 | 70,104 |
| `2024_legi_t1` | Législatives 2024 T1 | 70,102 |
| `2024_legi_t2` | Législatives 2024 T2 | 61,615 |
| `2026_muni_t1` | Municipales 2026 T1 | 70,003 |
| `2026_muni_t2` | Municipales 2026 T2 | 17,398 |

---

## Individual Election Files (Per-Commune)

### Présidentielles

| File | Years | Granularity | Source |
|------|-------|-------------|--------|
| `presidentielles_historique_2002_2017.csv` | 2002-2017 | National aggregate (1 row/year) | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-elections-presidentielles-depuis-2002/20181115-095451/resultats-elections-presidentielles-depuis-2002.csv) |
| `presidentielles_2007_bvot.txt` | 2007 T1+T2 | Par bureau de vote | [data.gouv.fr](https://static.data.gouv.fr/resources/election-presidentielle-2007-resultats-par-bureaux-de-vote/20151001-154056/PR07_Bvot_T1T2.txt) |
| `presidentielles_2012.xls` | 2012 T1+T2 | Par commune | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/election-presidentielle-2012-resultats-572126/) |
| `presidentielles_2017_t1.xls` | 2017 T1 | Par commune | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-1er-tour-par-communes/) |
| `presidentielles_2017_t2.xls` | 2017 T2 | Par commune | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/election-presidentielle-des-23-avril-et-7-mai-2017-resultats-definitifs-du-2nd-tour-par-communes/) |
| `presidentielles_2022_t1.csv` | 2022 T1 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-du-premier-tour-de-lelection-presidentielle-2022-par-commune-et-par-departement/20220413-153144/04-resultats-par-commune.csv) |
| `presidentielles_2022_t2.csv` | 2022 T2 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-du-second-tour-de-lelection-presidentielle-2022/20220425-091118/04-t2-resultats-par-commune.csv) |

### Législatives

| File | Years | Granularity | Source |
|------|-------|-------------|--------|
| `legislatives_historique_2002_2017.csv` | 2002-2017 | National aggregate (1 row/year) | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-elections-legislatives-depuis-2002/20181115-095149/resultats-elections-legislatives-depuis-2002.csv) |
| `legislatives_2012.xls` | 2012 | Par département/circ. | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-legislatives-2012-resultats-572077/) |
| `legislatives_2017_t1.xlsx` | 2017 T1 | Par commune | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-legislatives-des-11-et-18-juin-2017-resultats-par-communes-du-1er-tour/) |
| `legislatives_2017_t2.xlsx` | 2017 T2 | Par commune | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-legislatives-des-11-et-18-juin-2017-resultats-du-2nd-tour/) |
| `legislatives_2022_t1.txt` | 2022 T1 | Par commune (subcom) | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-legislatives-des-12-et-19-juin-2022-resultats-definitifs-du-premier-tour/20220614-192729/resultats-par-niveau-subcom-t1-france-entiere.txt) |
| `legislatives_2022_t2.txt` | 2022 T2 | Par commune (subcom) | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-legislatives-des-12-et-19-juin-2022-resultats-definitifs-du-second-tour/20220621-175945/resultats-par-niveau-subcom-t2-france-entiere.txt) |
| `legislatives_2024_t1.csv` | 2024 T1 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-legislatives-des-30-juin-et-7-juillet-2024-resultats-definitifs-du-1er-tour/20240711-075056/resultats-definitifs-par-communes.csv) |
| `legislatives_2024_t2.csv` | 2024 T2 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-legislatives-des-30-juin-et-7-juillet-2024-resultats-definitifs-du-2nd-tour/20240710-170606/resultats-definitifs-par-commune.csv) |

### Municipales

| File | Years | Granularity | Source |
|------|-------|-------------|--------|
| `municipales_historique_2001_2014.csv` | 2001-2014 | National aggregate (1 row/year) | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-elections-municipales-depuis-2001/20181115-095313/resultats-elections-municipales-depuis-2001.csv) |
| `municipales_2008_1.xls` | 2008 | Par commune (group 1) | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2008-resultats-572154/) |
| `municipales_2008_2.xls` | 2008 | Par commune (group 2) | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2008-resultats-572152/) |
| `municipales_2008_3.xls` | 2008 | Par commune (group 3) | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2008-resultats-572150/) |
| `municipales_2014_t1_plus_1000.txt` | 2014 T1 | Communes ≥ 1000 hab | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2014-resultats-1er-tour/) |
| `municipales_2014_t1_moins_1000.txt` | 2014 T1 | Communes < 1000 hab | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2014-resultats-1er-to-0/) |
| `municipales_2014_t2_plus_1000.txt` | 2014 T2 | Communes ≥ 1000 hab | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2014-resultats-2eme-tour/) |
| `municipales_2014_t2_moins_1000.txt` | 2014 T2 | Communes < 1000 hab | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-municipales-2014-resultats-2eme-to-0/) |
| `municipales_2020_t1_moins_1000.txt` | 2020 T1 | Communes < 1000 hab | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-municipales-2020-resultats/20200525-133805/2020-05-18-resultats-communes-de-moins-de-1000.txt) |
| `municipales_2020_t1_plus_1000.txt` | 2020 T1 | Communes ≥ 1000 hab | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-municipales-2020-resultats/20200525-133704/2020-05-18-resultats-communes-de-1000-et-plus.txt) |
| `municipales_2020_t2_moins_1000.txt` | 2020 T2 | Communes < 1000 hab | [data.gouv.fr](https://static.data.gouv.fr/resources/municipales-2020-resultats-2nd-tour/20200629-192436/2020-06-29-resultats-t2-communes-de-moins-de-1000-hab.txt) |
| `municipales_2020_t2_plus_1000.txt` | 2020 T2 | Communes ≥ 1000 hab | [data.gouv.fr](https://static.data.gouv.fr/resources/municipales-2020-resultats-2nd-tour/20200629-192435/2020-06-29-resultats-t2-communes-de-1000-hab-et-plus.txt) |
| `municipales_2026_candidatures_t1.csv` | 2026 T1 | All candidates per commune (888K rows, 144 MB) | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-municipales-2026-listes-candidates-au-premier-tour/20260313-152615/municipales-2026-candidatures-france-entiere-tour-1-2026-03-13.csv) |

> [!NOTE]
> **2026 Municipales T1 candidate name enrichment**: The T1 results file on data.gouv.fr does not include candidate names in its wide-format columns (`Nom candidat N` are empty for all communes). The `municipales_2026_candidatures_t1.csv` file was used to extract head-of-list names (`Tête de liste = OUI`) and populate the `nom`/`prenom` fields in `candidats_results.parquet` for the `2026_muni_t1` election. This brought coverage from 0.4% to 99.5% (remaining gaps are Polynésie française communes, retrieved from a separate candidatures file not downloaded here).

### Européennes

| File | Years | Granularity | Source |
|------|-------|-------------|--------|
| `europeennes_2009_dept_01_49.xls` | 2009 | Depts 01-49 | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-europeennes-2009-resultats-571327/) |
| `europeennes_2009_dept_50_95.xls` | 2009 | Depts 50-95+OM | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-europeennes-2009-resultats-571329/) |
| `europeennes_2009_3.xls` | 2009 | Additional | [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/elections-europeennes-2009-resultats-571331/) |
| `europeennes_2014.xlsx` | 2014 | Par commune | [data.gouv.fr](https://www.data.gouv.fr/storage/f/2014-05-30T10-29-10/euro-2014-resultats-communes-c.xlsx) |
| `europeennes_2019.txt` | 2019 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-des-elections-europeennes-2019/20190531-144212/resultats-definitifs-par-commune.txt) |
| `europeennes_2024.csv` | 2024 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/resultats-des-elections-europeennes-du-9-juin-2024/20240613-154634/resultats-definitifs-par-commune.csv) |

### Régionales

| File | Years | Granularity | Source |
|------|-------|-------------|--------|
| `regionales_2015_t1.xlsx` | 2015 T1 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-regionales-2015-et-des-assemblees-de-corse-de-guyane-et-de-martinique-par-communes-resultats-tour-1-1/20160121-102246/Reg_15_Resultats_Communes_T1_c.xlsx) |
| `regionales_2015_t2.xlsx` | 2015 T2 | Par commune | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-regionales-2015-et-des-assemblees-de-corse-de-guyane-et-de-martinique-par-communes-resultats-tour-2-1/20160121-104121/Reg_15_Resultats_Communes_T2c.xlsx) |
| `regionales_2021_t1.txt` | 2021 T1 | Par commune (subcom) | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-regionales-2021-resultats-du-1er-tour-1/20210713-100212/reg-resultats-par-niveau-subcom-t1-france-entiere-2021-07-12-18h41.txt) |
| `regionales_2021_t2.txt` | 2021 T2 | Par commune (subcom) | [data.gouv.fr](https://static.data.gouv.fr/resources/elections-regionales-2021-resultats-du-2eme-tour/20210713-095756/reg-resultats-par-niveau-subcom-t2-france-entiere-2021-07-12-09h15.txt) |

---

## Polls

### Présidentielle (all years)

Source: [nsppolls/nsppolls](https://github.com/nsppolls/nsppolls) + [Wikipedia scraping](https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27%C3%A9lection_pr%C3%A9sidentielle_fran%C3%A7aise)

Located in `data/polls/presidentielle/{year}/`:

- **2002**: `premier-tour-mars.csv`, `premier-tour-avril.csv`, `second-tour-chirac-jospin.csv`, `second-tour-chirac-lepen.csv`
- **2007**: `premier-tour.csv`, `second-tour-royal-sarkozy.csv`
- **2012**: `premier-tour.csv`, `premier-tour-janvier.csv`, `premier-tour-fevrier.csv`, `premier-tour-mars.csv`, `premier-tour-avril.csv`, `second-tour-hollande-sarkozy.csv`
- **2017**: `premier-tour-officiel.csv`, `premier-tour-avec-bayrou.csv`, `premier-tour-sans-bayrou.csv`, `premier-tour-sans-jadot.csv`, + 8 second-tour hypotheses
- **2022**: `nsppolls_presidentielle_2022.csv` (1.8 MB, full nsppolls dataset), 11 `k*.csv` files (sub-hypotheses), + 5 second-tour hypotheses
- **2027**: 13 files scraped from Wikipedia (65+ first-round polls across multiple hypotheses, + second-round matchups)

### Législatives (2007–2024)

Source: Wikipedia scraping. Located in `data/polls/legislatives/`:

- **2007**: 8 files (22 first-round rows, 10 second-round, + constituency polls)
- **2012**: 8 files (22 first-round rows, + constituency and seat projections)
- **2017**: 6 files (27 first-round rows, + seat projections)
- **2022**: 36 files (470 first-round rows, + constituency and demographic breakdowns)
- **2024**: 12 files (54 first-round rows, + seat projections)

### Législatives 2002

Source: Wikipedia FR scraping. Located in `data/polls/legislatives/`:

- **2002**: 5 files (51 first-round rows, + second-round matchups and seat projections)

### Européennes (2009–2024)

Source: Wikipedia scraping. Located in `data/polls/europeennes/`:

- **2009**: 3 files (35 national rows, + regional)
- **2014**: 2 files (69 national rows)
- **2019**: 7 files (84 rows pre-list, 27 post-list, + seat projections)
- **2024**: 1 file (25 rows)

### Régionales

- `data/polls/regionales/nsppolls_regionales_2021.csv` — Source: [nsppolls](https://github.com/nsppolls/nsppolls)
- `data/polls/regionales/regionales_2015_sondages_*.csv` — 24 files from Wikipedia FR (national + per-region polls)
- `data/polls/regionales/regionales_2010_sondages_*.csv` — 32 files from Wikipedia EN (national polls + per-region results)
- `data/polls/regionales/regionales_2004_sondages_*.csv` — 29 files from Wikipedia FR (per-region results, no dedicated polls)

### Départementales

- `data/polls/departementales/departementales_2015_sondages_*.csv` — 6 files from Wikipedia FR (18 national voting intention polls + results)
- `data/polls/departementales/departementales_2021_sondages_*.csv` — 4 files from Wikipedia FR (mainly results, sparse polling)

---

## Known Gaps

- **Européennes 2004 polls**: No dedicated Wikipedia polling page exists. Pre-election polls not available as open data.
- **Municipales polls**: Not available as open data for nationwide coverage.
- **Party popularity barometers** (ongoing tracking between elections): Published by Ifop, Ipsos, ELABE, Odoxa but only as PDF reports, not open CSV.
