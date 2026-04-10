# Demographic Data Dictionary

All demographic data files are stored in `data/demographics/`. Each demographic token carries two dates:
- **`date_float`** (reference date): What the data describes (e.g. Census 2021 centred ~2019.5).
- **`availability_date`** (publication date): When the data was actually published and usable. The router uses this for temporal causality — a token is invisible to the model when predicting elections before its publication.

For election/poll tokens: `availability_date == date_float`. For demographics, publication lags by 1–3 years.

## 1. Recensement de la Population — Census (`data/demographics/census/`)

**Source**: [INSEE — Recensement de la population](https://www.insee.fr/fr/information/2008354) — rolling annual census since 2006.

**Methodology**: Since 2004, INSEE replaced the traditional exhaustive census with a continuous rolling survey. Communes <10,000 inhabitants are exhaustively surveyed once every 5 years (rotating 1/5 per year). Communes ≥10,000 receive an 8% dwelling sample annually (40% over 5 years). Each published vintage pools the 5 most recent annual surveys. Results are therefore 5-year rolling averages, not point-in-time snapshots.

**Available vintages**: `2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022` (17 directories; 2006–2008 have incompatible column schemas and are skipped by the loader).

**Geographic levels**: Commune (all ~35,000), IRIS (~16,000 sub-commune units of ~2,000 residents for cities >5,000 pop; communes <5,000 = 1 IRIS).

**Geography alignment**: All vintages are republished on the **current** commune boundaries (géographie au 01/01/2025), so longitudinal joins are safe without conversion tables.

> [!IMPORTANT]
> **Temporal interpretation**: Vintage 2021 uses survey data from 2017–2021, centered around ~2019.5. Token `date_float = vintage_year - 1.5`. Token `availability_date`: Census vintage N is published ~June of year N+3 (e.g. Census 2021 → published June 2024 → `availability_date = 2024.5`). Compare vintages ≥5 years apart for meaningful signals.

### Theme: Population (POP)

**Download**: `https://www.insee.fr/fr/statistiques` → search "Base infracommunale (IRIS) — Population" or "Évolution et structure de la population — Commune"

**Bulk download page**: [Données infracommunales — IRIS](https://www.insee.fr/fr/statistiques?debut=0&geographie=IRIS&theme=73&categorie=4)

| Indicator | INSEE Variable | Description | Electoral relevance |
|---|---|---|---|
| Total population | `P{YY}_POP` | Population municipale | Normalisation, urbanity proxy |
| % aged 18–24 | derived from `P{YY}_POP1524` | Young adults | Youth turnout, left-leaning tendency |
| % aged 25–39 | derived from `P{YY}_POP2539` | Active young adults | Urban/mobile population |
| % aged 40–59 | derived from `P{YY}_POP4059` | Established adults | Core voting block |
| % aged 60–74 | derived from `P{YY}_POP6074` | Pre-retirees/young retirees | High turnout, conservative tendency |
| % aged 75+ | derived from `P{YY}_POP75P` | Elderly | Incumbency bias, high turnout |
| % immigrants | `P{YY}_POP_IMM` / `P{YY}_POP` | Born foreign abroad | Immigration salience proxy |
| Sex ratio | `P{YY}_POPH` / `P{YY}_POPF` | Men / Women | Minor, but gender gap in vote |

**File format**: CSV/Excel, one row per commune (or IRIS), columns = indicators by vintage year.

### Theme: Activité (ACT) — Employment & Socioprofessional Categories

**Download**: Search "Activité des résidents — Commune" or IRIS equivalent.

| Indicator | INSEE Variable | Description | Electoral relevance |
|---|---|---|---|
| Activity rate | `P{YY}_ACT1564` / `P{YY}_POP1564` | % 15–64 economically active | Labour market integration |
| Unemployment rate | `P{YY}_CHOM1564` / `P{YY}_ACT1564` | % active who are unemployed | **Strong** predictor of protest vote, FN/RN |
| % Cadres & professions intellectuelles | `C{YY}_ACT_CSP3` | Managers, professionals | Urban, centre-left/Macronist |
| % Professions intermédiaires | `C{YY}_ACT_CSP4` | Technicians, supervisors | Swing voters |
| % Employés | `C{YY}_ACT_CSP5` | Clerical / service workers | LFI/PS tendency |
| % Ouvriers | `C{YY}_ACT_CSP6` | Blue-collar workers | **Strong** RN predictor |
| % Agriculteurs | `C{YY}_ACT_CSP1` | Farmers | Rural right, specific municipal dynamics |
| % Artisans, commerçants | `C{YY}_ACT1564_CS2` | Self-employed, shopkeepers | Centre-right tendency |

### Theme: Formation (FOR) — Education / Diplomas

**Download**: Search "Diplômes — Formation — Commune" or IRIS equivalent.

| Indicator | INSEE Variable | Description | Electoral relevance |
|---|---|---|---|
| % sans diplôme | `P{YY}_NSCOL15P_DIPLMIN` | No diploma or CEP only | Deprivation, protest vote |
| % CAP/BEP | `P{YY}_NSCOL15P_CAPBEP` | Vocational certificate | Working class proxy |
| % Baccalauréat | `P{YY}_NSCOL15P_BAC` | High school diploma | Middle class |
| % Bac+2 | `P{YY}_NSCOL15P_SUP2` | Higher vocational (BTS/DUT) | Intermediate |
| % Bac+3/4 | `P{YY}_NSCOL15P_SUP34` | Bachelor/Master 1 | Upper-middle |
| % Bac+5+ | `P{YY}_NSCOL15P_SUP5` | Master 2 / Grande école / PhD | **Strong** predictor of Macronist / EELV vote |

### Theme: Logement (LOG) — Housing

**Download**: Search "Logement — Commune" or IRIS equivalent.

| Indicator | INSEE Variable | Description | Electoral relevance |
|---|---|---|---|
| % owner-occupied | `P{YY}_RP_PROP` / `P{YY}_RP` | Homeowners | Incumbency / stability preference |
| % renters | `P{YY}_RP_LOC` / `P{YY}_RP` | Private renters | Urban, mobile |
| % HLM (social housing) | `P{YY}_RP_LOCHLM` / `P{YY}_RP` | Social housing tenants | **Strong** left-vote predictor |
| % vacant dwellings | `P{YY}_LOGVAC` / `P{YY}_LOG` | Empty housing stock | Rural decline proxy |

### Theme: Familles-Ménages (FAM)

**Download**: Search "Couples, familles, ménages — Commune" or IRIS equivalent.

| Indicator | INSEE Variable | Description | Electoral relevance |
|---|---|---|---|
| Avg household size | `C{YY}_MEN` / `P{YY}_POP_MEN` | Persons per household | Urbanity proxy |
| % single-person households | `C{YY}_MEN_ISOL` / `C{YY}_MEN` | Living alone | Urban isolation, abstention risk |
| % single-parent families | `C{YY}_FAM_MONO` / `C{YY}_FAM` | Single parents | Deprivation proxy |

---

## 2. État Civil — Vital Statistics (`data/demographics/etat_civil/`)

**Source**: [INSEE — État civil](https://www.insee.fr/fr/statistiques?theme=71) — exhaustive administrative records from all mairies.

**Methodology**: Exhaustive administrative extraction from birth/death/marriage certificates. Unchanged methodology. No sampling or estimation.

**Available years**: Annually since 1968 (births), daily since 2020 (deaths by département).

**Geographic levels**: Commune (births, deaths, marriages). Département (daily deaths).

**Download**: Search "Naissances, décès et mariages — État civil" on insee.fr. Daily deaths: [Nombre de décès quotidiens](https://www.insee.fr/fr/information/4470857).

| Indicator | Source | Frequency | Granularity | Electoral relevance |
|---|---|---|---|---|
| Birth rate | État civil births / pop | Annual | Commune | Population dynamics, young families |
| Death rate | État civil deaths / pop | Annual | Commune | Ageing signal |
| Natural balance | births − deaths | Annual | Commune | Growth vs decline / demographic vitality |
| Daily deaths | Fichier décès | Daily | Département | COVID shock / mortality anomalies |

---

---

## 4. Répertoire Sirene — Business Registry (`data/demographics/sirene/`)

**Source**: [INSEE — Sirene](https://www.insee.fr/fr/information/3591226) — open data since 2017.

**Methodology**: Administrative register of all economic units (businesses, associations, public bodies). Monthly updates.

**Geographic levels**: Establishment-level (geocoded address) → aggregable to commune.

**Available**: Monthly snapshots since 2017. Historical stocks reconstructed to ~2008.

**Download**: [sirene.fr](https://www.sirene.fr/) or [data.gouv.fr](https://www.data.gouv.fr/fr/datasets/base-sirene-des-entreprises-et-de-leurs-etablissements-siren-siret/)

| Indicator (aggregated) | Source field | Description | Electoral relevance |
|---|---|---|---|
| Business density | count(SIRET) / pop | Establishments per 1,000 inhabitants | Economic vitality |
| % primary sector | NAF rev.2 section A | Agriculture / extraction | Rural identity |
| % secondary sector | NAF rev.2 sections B–F | Manufacturing / construction | Industrial base → RN vote |
| % tertiary sector | NAF rev.2 sections G–U | Services | Urban / post-industrial |
| Business creation rate | DCRET in year / stock | Annual new registrations | Entrepreneurial dynamism |

---

## Integration Priority

For the token-based architecture (see `archi.md`), each indicator becomes a token:
```
[date_float, election_type, commune_code, indicator_name, "", "Demographics", value]
```

**Priority order for integration**:

| Priority | Source | Indicators | Status |
|---|---|---|---|
| **P0** | Census ACT | `Taux_Chomage`, `Pct_Ouvriers`, `Pct_Cadres`, `Pct_Employes`, `Pct_Prof_Intermediaires`, `Pct_Agriculteurs`, `Pct_Artisans`, `Pct_Emploi_Agriculture`, `Pct_Emploi_Industrie`, `Pct_Emploi_Construction`, `Pct_Emploi_Tertiaire`, `Pct_Retraites` | ✅ 12 indicators (17 vintages) |
| **P1** | Census FOR | `Pct_Sans_Diplome`, `Pct_CAP_BEP`, `Pct_Bac`, `Pct_Bac_Plus_2`, `Pct_Bac_Plus_3_4`, `Pct_Bac_Plus_5` | ✅ 6 indicators (17 vintages) |
| **P1** | Census POP | `Pct_Age_0_14`, `Pct_Age_18_24`, `Pct_Age_30_44`, `Pct_Age_45_59`, `Pct_Age_60_Plus`, `Pct_Immigres` | ✅ 6 indicators (17 vintages) |
| **P2** | Census LOG | `Pct_Proprietaires`, `Pct_HLM`, `Pct_Locataires`, `Pct_Logements_Vacants` | ✅ 4 indicators (16 vintages, missing 2011) |
| **P2** | Census FAM | `Pct_Menages_Seuls`, `Pct_Familles_Monoparentales` | ✅ 2 indicators (13 vintages, missing 2010-2013) |
| **P3** | État civil | Natural balance | ⬜ Not yet |
| **P3** | Sirene | Sector breakdown, business density | ⬜ Not yet |

**Current token count (all census, multi-vintage)**: up to 30 indicators × ~35,000 communes × ~17 vintages ≈ **~17.9M tokens** (actual count varies by vintage due to column availability).

---

## Known Gaps & Caveats

- **Filosofi (REMOVED)**: Filosofi was discontinued after the 2021 vintage. The abolition of *taxe d'habitation* on main residences broke its data linkage. INSEE is developing **Résil** (Répertoire Statistique des Individus et des Logements) as a replacement, but it is not yet available. Income/poverty indicators (`Revenu_Median`, `Taux_Pauvrete`) are no longer in the pipeline.
- **Census 2020 disruption**: The 2020 annual survey was suspended due to COVID-19. INSEE adjusted the 2020 and 2021 vintages using specific correction methods. Compare with caution (prefer ≥6-year gaps for this period).
- **Census 2006–2008**: These vintages have incompatible column schemas and are skipped by the loader. Census 2009 only produces `Taux_Chomage`.
- **IRIS boundary changes**: IRIS contours are re-drawn every few years. Some IRIS codes are renamed/split/merged. If using IRIS level, use INSEE's [table de passage IRIS](https://www.insee.fr/fr/information/7672015) to harmonize across years.
- **Sirene**: Requires significant aggregation work (millions of raw establishments → commune-level indicators). Lower priority.
