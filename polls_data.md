# Polls Data (`data/polls/`)

## Presidential Polls (`data/polls/presidentielle/`)

Voting intention polls (sondages d'intentions de vote) for French presidential elections.

### Source 1: depuis1958/sondages (2002–2022)

- **Repo**: https://github.com/depuis1958/sondages
- **Coverage**: 2002, 2007, 2012, 2017, 2022
- **Format**: CSV, one row per poll, columns are candidates
- **Columns**: `sondeur, source, date début, date fin, échantillon, <candidate1>, <candidate2>, ...`
- **Files per election**:
  - `premier-tour*.csv` — First round voting intentions (various hypotheses for 2017/2022)
  - `second-tour-*.csv` — Second round matchup polls

| Year | First Round Polls | Second Round Files | Key Candidates |
|------|------------------|--------------------|----------------|
| 2002 | ~33 rows (mars+avril) | Chirac-Jospin, Chirac-Le Pen | Chirac, Jospin, Le Pen |
| 2007 | 44 rows | Royal-Sarkozy (108 rows) | Sarkozy, Royal, Bayrou, Le Pen |
| 2012 | ~88 rows (jan-avril) | Hollande-Sarkozy (100 rows) | Hollande, Sarkozy, Le Pen, Mélenchon |
| 2017 | ~110 rows (multiple hypotheses) | 7 matchup files | Macron, Le Pen, Fillon, Mélenchon |
| 2022 | ~20 rows + hypothesis files (k*.csv) | 5 matchup files | Macron, Le Pen, Zemmour, Pécresse, Mélenchon |

### Source 2: nsppolls (2022, detailed)

- **Repo**: https://github.com/nsppolls/nsppolls
- **File**: `presidentielle/2022/nsppolls_presidentielle_2022.csv`
- **Coverage**: 2020-06 to 2022-04 (8863 rows)
- **Format**: Long format (one row per candidate per poll), rich metadata
- **Columns**: `candidat, parti, intentions, erreur_sup, erreur_inf, id, nom_institut, commanditaire, debut_enquete, fin_enquete, echantillon, population, rolling, media, tour, hypothese, sous_echantillon`
- **Notes**: Most comprehensive dataset. Includes confidence intervals, sample sizes, and polling methodology info.

## Regional Polls (`data/polls/regionales/`)

### Source: nsppolls (2021)

- **Repo**: https://github.com/nsppolls/nsppolls
- **File**: `nsppolls_regionales_2021.csv`
- **Coverage**: 2021 regional elections, 13 regions, 1416 rows
- **Format**: Long format, converted from JSON
- **Columns**: `region_name, nom_institut, commanditaire, debut_enquete, fin_enquete, echantillon, population, tour, hypothese, tete_liste, parti, intentions`

### Régionales 2015 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Élections_régionales_françaises_de_2015
- **Key files**:
  - `regionales_2015_sondages_2.csv` — National voting intentions 1st round (18 rows, 19 cols: LO/NPA, FG, EELV, PS/PRG, LR, FN, etc.)
  - `regionales_2015_sondages_0.csv` — Election results summary (62 rows)
  - `regionales_2015_sondages_6.csv` to `_22.csv` — Per-region polls (~15-22 rows each, covering all 13 regions)
  - `regionales_2015_sondages_23.csv` — Seat projections by region (47 rows)
- **Coverage**: 2015 regional elections, national + per-region polls

### Régionales 2010 — Source: Wikipedia EN (scraped 2026-03-27)

- **Source page**: https://en.wikipedia.org/wiki/2010_French_regional_elections
- **Key files**:
  - `regionales_2010_sondages_2.csv` — National polling 1st round (9 rows x 16 cols, institutes: OpinionWay, Ifop, CSA, TNS Sofres)
  - `regionales_2010_sondages_3.csv` — National polling 2nd round (9 rows x 6 cols)
  - `regionales_2010_sondages_4.csv` to `_31.csv` — Per-region results and polls (~28 tables, covering all regions)
- **Coverage**: 2009–March 2010, national + per-region

### Régionales 2004 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Élections_régionales_françaises_de_2004
- **Key files**:
  - `regionales_2004_sondages_0.csv` — National results summary (49 rows)
  - `regionales_2004_sondages_2.csv` — National results by party (17 rows x 12 cols)
  - `regionales_2004_sondages_4.csv` to `_29.csv` — Per-region detailed results (~25 tables)
- **Coverage**: 2004 regional elections, per-region results
- **Note**: No dedicated polling page exists; tables are mainly election results, not pre-election polls

## European Election Polls (`data/polls/europeennes/`)

### Européennes 2024 — Source: Wikipedia (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Sondages_sur_les_élections_européennes_de_2024
- **File**: `europeennes_2024_france_0.csv` (main voting intentions table)
- **Coverage**: ~25 polls from late 2023 to April 2024
- **Columns**: `Institut, Date, PCF, LFI, PS, EELV, LC, ENS, LR, RN, REC`

### Européennes 2019 — Source: Wikipedia (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Sondages_sur_les_élections_européennes_de_2019
- **Key files**:
  - `europeennes_2019_france_1.csv` — Polls after list submission (27 rows, 27 parties)
  - `europeennes_2019_france_2.csv` — Polls before list submission (84 rows, 22 parties)
  - `europeennes_2019_france_4.csv` — Seat projections (28 rows)
- **Coverage**: 2018–May 2019
- **Columns**: `Sondeur, Date, Échantillon, PRC, LO, FI, PCF, ..., RN, ..., Autres`
- **Other files**: `_0` (vote results summary), `_3`/`_5` (seat projections), `_6` (late projections)

### Européennes 2014 — Source: Wikipedia EN (scraped 2026-03-27)

- **Source page**: https://en.wikipedia.org/wiki/Opinion_polling_for_the_2014_European_Parliament_election_in_France
- **Key files**:
  - `europeennes_2014_france_0.csv` — Main voting intentions (69 rows, 23 cols)
  - `europeennes_2014_france_3.csv` — Late polls (3 rows)
- **Coverage**: 2013–May 2014
- **Columns**: `Polling firm, Fieldwork date, Sample size, Abs., LO, NPA, FG, PS PRG, DVG, EELV, MoDem UDI, UMP, MPF, DLR, FN, ...`

### Européennes 2009 — Source: Wikipedia EN (scraped 2026-03-27)

- **Source page**: https://en.wikipedia.org/wiki/Opinion_polling_for_the_2009_European_Parliament_election_in_France
- **Key files**:
  - `europeennes_2009_france_1.csv` — National voting intentions (35 rows: LO, NPA, FG, PS, EE, MoDem, UMP, etc.)
  - `europeennes_2009_france_5.csv` — Regional polls East (7 rows)
  - `europeennes_2009_france_8.csv` — Regional polls additional (7 rows)
- **Coverage**: 2008–May 2009

## Legislative Election Polls (`data/polls/legislatives/`)

### Législatives 2024 — Source: Wikipedia (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_les_élections_législatives_françaises_de_2024
- **Key files**:
  - `legislatives_2024_sondages_0.csv` — First round national voting intentions (54 rows, blocs: EXG, NFP, DVG, ECO, DVC, ENS, DVD, LR, RN, UPF, REC)
  - `legislatives_2024_sondages_10.csv` — Seat projections (40 rows)
  - `legislatives_2024_sondages_1.csv` to `_9.csv` — Second round matchup polls by scenario
  - `legislatives_2024_sondages_11.csv` — Seat projection summary by alliance
- **Coverage**: June–July 2024 (snap elections after EU election)

### Législatives 2022 — Source: Wikipedia (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_les_élections_législatives_françaises_de_2022
- **Key files**:
  - `legislatives_2022_sondages_0.csv` — First round national voting intentions (469 rows, 32 cols: LO-NPA, NUPES, DVG-FGR, ECO, DVC, Ensemble, UDC/LR-UDI, DVD, UPF, RN, REC, Autres)
  - `legislatives_2022_sondages_1.csv` — Second round / seat projection data (427 rows, 31 cols)
  - `legislatives_2022_sondages_2.csv` to `_9.csv` — Seat projections by scenario
  - `legislatives_2022_sondages_10.csv` to `_26.csv` — Constituency-level polls (per-circo)
  - `legislatives_2022_sondages_27.csv` to `_32.csv` — Second round head-to-head polls
  - `legislatives_2022_sondages_33.csv` — Demographic breakdown (by profession)
  - `legislatives_2022_sondages_34.csv` — Detailed breakdown (215 rows, 30 cols)
  - `legislatives_2022_sondages_35.csv` — Age-based breakdown (102 rows, 30 cols)
- **Coverage**: April–June 2022

### Législatives 2017 — Source: Wikipedia (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_les_élections_législatives_françaises_de_2017
- **Key files**:
  - `legislatives_2017_sondages_0.csv` — First round national voting intentions (27 rows, 14 cols: EXG, LFI, PCF, ECO, PS, LREM, UDI-LR, DLF, FN, Autres)
  - `legislatives_2017_sondages_1.csv` — Second round bloc projections (4 rows)
  - `legislatives_2017_sondages_2.csv` — Constituency-level polls (27 rows)
  - `legislatives_2017_sondages_3.csv` — Seat projection scenarios (15 rows, majority likelihood)
  - `legislatives_2017_sondages_4.csv` — Seat projections by bloc (20 rows)
  - `legislatives_2017_sondages_5.csv` — Regional breakdown (12 rows)
- **Coverage**: April–June 2017

### Législatives 2012 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_les_élections_législatives_françaises_de_2012
- **Key files**:
  - `legislatives_2012_sondages_0.csv` — First round national voting intentions (22 rows, 28 cols: LO/NPA, FG, EELV, PS/PRG, etc.)
  - `legislatives_2012_sondages_1.csv` — Constituency-level polls 1st round (14 rows)
  - `legislatives_2012_sondages_2.csv` — Constituency-level polls detail (12 rows)
  - `legislatives_2012_sondages_3.csv` — Constituency-level polls 2nd round (24 rows)
  - `legislatives_2012_sondages_4.csv` — Constituency-level 2nd round detail (22 rows)
  - `legislatives_2012_sondages_5.csv` — Seat projections (6 rows, 35 cols)
- **Coverage**: April–June 2012

### Législatives 2002 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_les_élections_législatives_françaises_de_2002
- **Key files**:
  - `legislatives_2002_sondages_0.csv` — First round national voting intentions (51 rows, 69 cols: EXG, PCF, PS, DVG, MDC, Les Verts, UDF, RPR/UMP, DL, FN, MNR, etc.)
  - `legislatives_2002_sondages_1.csv` — Second round Gauche plurielle vs RPR-UDF-DL (7 rows)
  - `legislatives_2002_sondages_2.csv` — Second round three-way matchup (6 rows)
  - `legislatives_2002_sondages_3.csv` — Cohabitation preference polls (4 rows)
  - `legislatives_2002_sondages_4.csv` — Seat projections by bloc (6 rows, PCF, PS-DVG, Verts, UDF, RPR/UMP-DVD)
- **Coverage**: April–June 2002

### Législatives 2007 — Source: Wikipedia EN (scraped 2026-03-27)

- **Source page**: https://en.wikipedia.org/wiki/Opinion_polling_for_the_2007_French_legislative_election
- **Key files**:
  - `legislatives_2007_sondages_0.csv` — First round national voting intentions (22 rows, 108 cols: LO, LCR, PCF, MRC, PS, PRG, MoDem, UMP, FN, etc.)
  - `legislatives_2007_sondages_1.csv` — Second round national polls (10 rows, 50 cols)
  - `legislatives_2007_sondages_2.csv` — Second round PS/PCF/LV vs UMP/NC (19 rows)
  - `legislatives_2007_sondages_3.csv` — Three-way projections (5 rows)
  - `legislatives_2007_sondages_15.csv` to `_23.csv` — Constituency-level polls
- **Coverage**: April–June 2007

## Présidentielle 2027 Polls (`data/polls/presidentielle/2027/`)

### Présidentielle 2027 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Liste_de_sondages_sur_l%27élection_présidentielle_française_de_2027
- **Key files**:
  - `presidentielle_2027_sondages_0.csv` — Hypothesis 1 first round (13 rows, 14 cols: Arthaud, Mélenchon, Roussel, Tondelier, PS, etc.)
  - `presidentielle_2027_sondages_1.csv` — Hypothesis 2 first round (65 rows, 17 cols, most data)
  - `presidentielle_2027_sondages_2.csv` — Hypothesis 3 first round (34 rows, 16 cols)
  - `presidentielle_2027_sondages_3.csv` — Hypothesis 4 first round (24 rows, 15 cols)
  - `presidentielle_2027_sondages_4.csv` — Hypothesis 5 first round (8 rows)
  - `presidentielle_2027_sondages_5.csv` to `_12.csv` — Second round matchups (Attal-Bardella, Mélenchon-Bardella, Philippe-Bardella, Philippe-Le Pen, etc.)
- **Coverage**: 2024–ongoing

## Départementales Polls (`data/polls/departementales/`)

### Départementales 2015 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Élections_départementales_françaises_de_2015
- **Key files**:
  - `departementales_2015_sondages_2.csv` — National voting intentions 1st round (18 rows, 12 cols: LO/NPA, FG, EELV, PS/PRG, MoDem, UDI, UMP, FN)
  - `departementales_2015_sondages_0.csv` — Election overview (48 rows)
  - `departementales_2015_sondages_3.csv` — Per-department results (183 rows)
  - `departementales_2015_sondages_4.csv` — Results by party (17 rows)
  - `departementales_2015_sondages_5.csv` — Second round analysis (10 rows)
  - `departementales_2015_sondages_6.csv` — Per-department president changes (100 rows)
- **Coverage**: Dec 2014–March 2015

### Départementales 2021 — Source: Wikipedia FR (scraped 2026-03-27)

- **Source page**: https://fr.wikipedia.org/wiki/Élections_départementales_françaises_de_2021
- **Key files**:
  - `departementales_2021_sondages_0.csv` — Election overview (25 rows)
  - `departementales_2021_sondages_2.csv` — Seat changes by party (19 rows)
  - `departementales_2021_sondages_3.csv` — Per-department president changes (95 rows)
  - `departementales_2021_sondages_4.csv` — Results by party nuance (72 rows)
- **Coverage**: June 2021
- **Note**: Sparse polling data — mainly election results, not pre-election polls

## Not Available as Open Data

- **Européennes 2004 polls**: No dedicated Wikipedia polling page exists. Pre-election polls not available as open data.
- **Municipal polls**: Not available as open data for nationwide coverage.
- **Party popularity barometers** (ongoing tracking between elections): Published by Ifop, Ipsos, ELABE, Odoxa but only as PDF reports, not open CSV.
