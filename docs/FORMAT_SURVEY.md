# Format survey: CGM and pump data in the wild

A survey of 2026-10-21 of what CGM and insulin-pump data exists outside this library: the exports
vendors let users download, the open and DIY ecosystems, and public datasets and files people post.
Three research agents covered one slice each; their claims carry a source URL or a file and line in a
public parser. Where an agent ran a real or public sample through this library, the outcome is given
as **ran**; otherwise it is an inference. Sample files read were public, and only their structure was
read: no person's data enters this repository.

Supported when this was written (0.13.0 plus `main`): Dexcom Clarity CSV (mg/dL, EU mmol/L), LibreView
CSV (mg/dL, EU mmol/L), Medtronic CareLink CSV (Guardian Connect; MiniMed pump, verified on the 720G),
Nightscout (API JSON, exporter CSV), CGMacros, D1NAMO, BIG IDEAs. Loop is planned (RM11).

---

## 1. Already broken: drift in formats the library claims

These are exports of formats the library parses that fail, or worse, parse wrong. Format drift is a
patch (`CLAUDE.md` §12).

| What | Outcome | Evidence |
|---|---|---|
| MiniMed 780G CareLink: daily SmartGuard totals read as a bolus at 00:00 | **wrong number, silent** in 0.13.0; fixed on `main` (`faacbd1`); F20 open | ran; aidR `inst/extdata/carelink_example.csv` (randomized values) |
| CareLink, US: comma-delimited, `MM-dd-yyyy hh:mm:ss AM` preamble | detected as MEDTRONIC, then the header reads as one column | ran; github.com/bewest/decoding-carelink, github.com/thedamon/opendatabetes |
| CareLink in mmol/L (`Sensor Glucose (mmol/L)`, also `(mmol/l)`) | pump export fails; a CGM-only one is not detected | ran on an edited copy; NightscoutLoader `DBResultMedtronicNew.java` L25-51 |
| CareLink, Spanish (Paradigm Veo): `BWZ Carb Input (exchanges)` holds decimals | fails on an `i64` dtype | ran; HUPA-UCM raw folder (Mendeley 10.17632/3hbcscwz44.1, CC BY 4.0) |
| LibreView, US Libre 3: 12-hour `MM-DD-YYYY hh:mm AM` timestamps | fails: no `%I %p` timestamp format | ran; github.com/jeffreyruoss/cgm-food-repsonse-assessment |
| LibreView "Patient report" layout: one extra line before the header | fails at column lookup | ran; github.com/cafoala/diametrics `libre_amer_01.csv` |
| LibreView, German and Spanish (`Gerätezeitstempel`, `Sello de tiempo del dispositivo`; Spanish uses `;` and CR line endings) | Spanish ran: detected as LIBRE, then fails. German inferred to do the same: the `FreeStyle Libre` pattern sits in the data rows | ran (Spanish, HUPA-UCM); peisenh/loop-cr-review `lcr/readers/libreview.py` L42-65 |
| Clarity, German: `;`-delimited, every field quoted, `Zeitstempel (JJJJ-MM-TTThh:mm:ss)`, `Glukosewert (mg/dL)` | `UnknownFormatError` | ran; aidR `dexcom_clarity_example.csv` |
| Clarity, Spanish (`Marca temporal (AAAA-MM-DDThh:mm:ss)`) | not detected (inferred) | HUPA-UCM raw folder, per the DiaData script |
| Clarity mmol/L, UK G6 with 10 metadata rows | parses, warns about the count | ran; github.com/MRCIEU/GLU |

**Localization is the theme.** Clarity and LibreView write their headers in the account's language,
confirmed for German and Spanish; French, Italian and Dutch are unconfirmed. Detection is English-only.
The schema already has an `aliases` mechanism for header spellings, so a language is aliases plus a
detection pattern, not a new format, as long as units and record-type vocabularies are covered too.

## 2. New formats, ranked

### Aggregators first: one parser, many devices

| Format | Covers | Layout | Priority |
|---|---|---|---|
| **Tidepool JSON export** | Tandem, Omnipod, Medtronic, Dexcom, Libre, Loop, xDrip+ uploads | JSON array of datums with `type` (`cbg`, `smbg`, `basal`, `bolus`, `wizard`, `food`…); `time` UTC ISO, `timezoneOffset` minutes; glucose stored mmol/L. Basal: `deliveryType`, `rate` (U/h), `duration` (ms), optional `percent`. Spec: developer.tidepool.org/data-model; tidepool-org/export `userDataHandler.mjs` L72-98 | **High** |
| **Glooko / diasend ZIP** | Omnipod 5/DASH, CamAPS, Control-IQ, YpsoPump, Accu-Chek SmartGuide (EU) | ZIP of per-type CSVs (`cgm_data_N`, `bg_data`, `Insulin data/{basal,bolus}`…), row 1 `Name:,Date Range:`, localized headers, local time with no offset. aidR `glooko_example.zip` | **High** |
| **Apple Health export.xml** | anything writing HealthKit: Dexcom and Libre apps, Loop, Trio, xDrip4iOS | `<Record type="HKQuantityTypeIdentifierBloodGlucose"…>`, insulin with a basal/bolus reason; offsets in timestamps. Several apps write the same reading, so it needs per-`sourceName` dedup | **High** |

Tidepool basal is already the library's shape: a rate in force for a span is what `BASAL_RT` rows
with `basal_rate` hold since 0.13.0, and `fold_insulin_onto_grid` integrates it. A Tidepool parser
emits those rows; it needs no new insulin representation. HealthKit and Loop write a temp basal as
the units delivered over a start–end span, which maps to an `INS_FAST`-like dose with
`insulin_duration`, or to a rate over the span; that choice is a design call.

### Single devices and apps

| Format | Layout | Priority |
|---|---|---|
| **Tandem Source ZIP** (t:slim X2, Mobi) | per-type CSVs, local ISO time, only the Bolus CSV has a BOM, bidi-isolate characters around `Device Type`. aidR `tandem_source_example.zip` | High |
| **Juggluco TSV** (Libre 2/3, Sibionics, Dexcom readers) | `Sensorid nr UnixTime YYYY-mm-ddTHH:MM:SS TZ Min <unit> Rate ChangeLabel`; unit per app setting. j-kaltes/Juggluco `exports.cpp` L35-371 | Med-high |
| **Sibionics GS1 app** | `.xls` name, OOXML bytes, sheet `Sensor Glucose`, offset inside the time cell. daedalus/agp_tool `examples/` | Med-high (Excel) |
| **xDrip+ / Spike SiDiary CSV** | `DAY;TIME;UDT_CGMS;BG_LEVEL;CH_GR;BOLUS;REMARK`, no timezone. xDrip `DatabaseUtil.java` L242-258 | Med (needs a caller-stated timezone) |
| **Libre desktop software / Libre Pro TSV** | tab-separated, line 1 the patient name, `ID, Time, Record Type, Historic Glucose (mmol/L)`; English, Spanish `;` and tab variants. nahog/freestyle-libre-parser-viewer | Med (old research data) |
| **AAPS Open Humans ZIP** | one JSON array per Room table (`GlucoseValues.json`, `TemporaryBasals.json`…), ms timestamps plus `utcOffset`. AndroidAPS `OpenHumansUploaderPlugin.kt` L409-494 | Med |
| Diabeloop YourLoops ZIP | `cbg.csv` `timestamp,timezoneOffSet,value,units`; bolus, basal, meals. aidR | Med |
| Eversense DMS XLSX | sheet layout unconfirmed | Med, blocked on a sample |
| CareSens Air CSV | layout changed 2025-02, no sample found | Med, blocked on a sample |
| Lingo CSV (Abbott, OTC) | 5-minute values **clipped to 55-200 mg/dL**; columns unconfirmed | Med: the clip is censoring, like Dexcom High/Low |
| Medtronic iPro2 | tab-separated, UTF-16 or UTF-8, header on row 11, Excel serial dates. MRCIEU/GLU | Low-med |
| HL7 FHIR CGM IG | Observations, LOINC `99504-3` / `105272-9`; glucose only, no insulin or carbs. build.fhir.org/ig/HL7/cgm | Med-low (easy, few holders) |
| Open mHealth, Health Connect, Nightscout API v3, mylife Cloud, Smart Pix, t:connect, OpenAPS rig logs, Gluroo, Glimp, Diabox, Sugarmate | glucose-only, no public spec, retired, or reachable through Nightscout | Low |

Report-only vendors, which offer no file of readings: Medtrum EasyView, AiDEX/LinX, Syai, Ottai,
Sinocare iCan, and the SmartGuide app. Dexcom Stelo and the Libre 3 app export nothing of their own;
their data reaches a file through Clarity and LibreView.

## 3. Public datasets

MetaboNet (metabo-net.org, arXiv 2601.11505) is the harmonised corpus the forecasting models train on.
Tags: **[pub]** in MetaboNet's public subset, **[DUA]** in it behind an agreement, **[–]** not in it.

| Dataset | Access | Size, devices | Layout | Priority |
|---|---|---|---|---|
| **AZT1D** [pub] | Mendeley 10.17632/gk9m674wcx.1, CC BY 4.0, 776 MB | 25 subj, 6-8 wk; G6, t:slim X2 Control-IQ | per-subject CSV `EventDateTime, CGM, Basal (U/h), TotalBolusInsulinDelivered, CarbSize, DeviceMode` (some basal ×1000) | **High** |
| **HUPA-UCM** [pub] | Mendeley 10.17632/3hbcscwz44.1, CC BY 4.0, 411 MB | 25 subj, ≥14 days; Libre 2, Fitbit, Medtronic/Roche pumps | preprocessed `;` CSV, plus **raw Spanish vendor exports** (LibreView, Libre desktop, CareLink, Clarity, Roche): the best open set of real format variants for fixtures | **High** |
| **Jaeb family** [pub]: DCLP3, DCLP5, IOBP2, PEDAP, Replace-BG, CTR3, FLAIR; [–]: CITY, WISDM, Severe Hypo | public.jaeb.org, click-through form | 100-440 subj each; G4-G6, Control-IQ, iLet | pipe-delimited `.txt` keyed `PtID`; time column and format vary by study (day-first, `%m/%d/%Y %I:%M:%S %p`, date-only cells, or days since enrolment) | **High**: one family parser covers about eight MetaboNet sets |
| Shanghai T1DM/T2DM [pub, T1 only] | figshare 10.6084/m9.figshare.20444397, CC BY 4.0 | 12 T1 + 100 T2; 15-min | mixed `.xls`/`.xlsx`, 7 header variants, Chinese headers, free-text diet | Med (Excel) |
| T1D-UOM [pub] | github.com/sharpic/manchestercscoordinateddiabetesstudy, CC BY 4.0 | 17 files | per-stream folders, mmol/L; README's date format contradicts the data (day-first) | Med |
| OhioT1DM [DUA] | by email | 12 subj; Medtronic | XML per subject: `glucose_level`, `basal`, `temp_basal`, `bolus`, `meal`… `<event ts value>` | Med |
| T1DEXI / T1DEXIP [DUA] | Vivli | 497 / 245 subj; G6 | CDISC SDTM `.xpt` in a deflate64 zip | Med (slow access, SDTM reader) |
| BrisT1D [pub] | data.bris, CC BY 4.0 (raw part restricted) | 19 subj | processed per-subject CSV, mmol/L; the raw part is vendor exports | Med |
| Hall 2018 [–] | PLOS Biology S1, CC BY 4.0 | 57 mostly non-diabetic; G4 | TSV `DisplayTime, GlucoseValue, subjectId, InternalTime` | Med |
| DiaTrend [DUA], T1DiabetesGranada [–], Colás 2019 [–], Broll, Weinstock | Synapse, Zenodo by permission, PLOS | | Excel; CGM-only; time of day only | Low-med |
| **MetaboNet itself** | metabo-net.org, login | 3,135 subj | one parquet on a 5-min grid; carbs and insulin summed per bin, zero and missing indistinguishable | High value, but its 0-vs-null loss means it comes in as a derived, flagged corpus, never as raw data |

Parsers worth reading before building any of these: github.com/replicahealth/GluPredKit
`glupredkit/parsers/` (11 corpora), github.com/IrinaStatsLab/Awesome-CGM, github.com/Beyza-Cinar/DiaData
`data_integration.py`.

## 4. Questions this raises for the person

- **Excel.** Sibionics, Shanghai, DiaTrend, Eversense and Tidepool XLSX are Excel; reading it means
  `fastexcel` or similar. The core is Polars-only (charter, minimal core), so this is behind the
  `cli` extra at most, or not at all.
- **Formats with no timezone** (Glooko, Tandem Source, mylife, xDrip+ SiDiary) need a caller-stated
  timezone, the way RM19 made the caller state an exporter's date order; never the machine's own.
- **Datasets with no calendar date** (Replace-BG and Severe Hypo: days since enrolment; Colás: time of
  day only). A base date is an invention; how to represent relative time is a design question.
- **Censored readings**: Lingo clips at 55/200 mg/dL, Dexcom writes High/Low. One flag for "the device
  says beyond this bound", instead of a number, would serve both.

## 5. Suggested order

1. **Drift fixes, patch-level** (§1): CareLink comma delimiter and mmol/L; LibreView 12-hour time and
   the patient-report preamble; CareLink decimal carb exchanges. About a day together, each with a
   fixture built from a public sample's structure.
2. **Localized Clarity and LibreView** (German, Spanish) through header aliases and detection patterns,
   using HUPA-UCM's raw exports and aidR's German Clarity as layouts. About a day.
3. **Tidepool JSON**, which emits the `BASAL_RT` rows the fold already reads. About 2 days.
4. **AZT1D** and the **Jaeb family**, on the corpus machinery, with RM7's lazy ingest for size.
5. **Glooko ZIP**, **Apple Health**, **Tandem Source**, **Juggluco**, by demand.
