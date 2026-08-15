# Research corpora — a source category beside vendor exports

A design proposal. The library today absorbs **vendor exports**: one file, one person, one device,
glucose plus a handful of events. Research datasets — CGMacros, D1NAMO, Loop — are a different kind
of source, and they break that shape along several axes at once. This document names the categories,
writes down the design, and records what it must not cost us. The work itself was filed as RM7–RM13 in
[ROADMAP.md](ROADMAP.md).

**Shipped in 0.10.0:** the source categories (RM12), faceted output (RM9), the extended schema
(RM8), CGMacros (RM10) and D1NAMO (RM13). Their entries are in
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md). Still open: the lazy `scan_csv` ingest path (RM7) and
Loop (RM11), which needs an iterator form of `parse_corpus` rather than the eager mapping that fits
CGMacros and D1NAMO.

Appendix C was written before the archives were read end to end, and implementing D1NAMO falsified
three of its claims. They are corrected in place below and marked **[corrected 0.10.0]**.

---

## Why these are not format drift

The library already has declarative knobs for vendor exports that mutate over time, and
[NEW_SCHEMA.md](NEW_SCHEMA.md) is emphatic that you reach for them before writing a new module: a
renamed column is an `aliases` entry, a regional unit is `derive_schema`, an extra metadata row is a
`data_start_line` override.

None of those apply here, and the reason is worth stating precisely. Every knob we have changes how
*one* frame is produced from *one* file. These datasets change the **cardinality on both sides** —
how many files come in, and how many frames go out:

- **CGMacros** records two CGM sensors on the same wrist over the same ten days. `Libre GL` and
  `Dexcom GL` are not two spellings of one column; they are two independent measurements of the same
  quantity, disagreeing by design. One file in, two frames out.
- **D1NAMO** gives each subject a directory of separate files, one per modality — glucose, insulin,
  meals — plus continuous physiological streams in two further archives. Many files in, one frame out
  per subject. Its two subsets do not even share a schema.
- **Loop** puts roughly a thousand participants in one shared file set keyed by `PtID`. There is no
  per-person file to parse. Many files in, many frames out, and the subject list is not visible
  without reading the data.

Cardinality is not something a schema declaration can express. That is what makes this a design
question rather than a drift question, and it is why the next section names categories instead of
adding another knob.

The corpora also carry channels the unified schema has no home for — macronutrients, wearable
streams, meal photographs. That part *is* expressible, and it is a separate concern from the
cardinality one. The proposal keeps them separate.

---

## Source categories

Everything the library supports today is one shape: **one file, one subject, one device**. That
assumption is not written down anywhere, because until now nothing violated it. It is baked into
`parse_file(path) -> UnifiedFormat` and into `detect_format`, which reads a text prefix of a single
file.

Research datasets need the assumption named, because they break it in two independent ways — how
many *files* come in, and how many *subjects* are in them. Those are orthogonal, so they give three
categories rather than a spectrum:

| Category | Input | Entry point | Examples |
|---|---|---|---|
| **Export** | one file, one subject | `parse_file(path) -> UnifiedFormat` | Dexcom, Libre, Medtronic, the Nightscout exporter CSV |
| **Bundle** | several files, **one** subject, each file a different *modality* | `parse_bundle(...) -> UnifiedFormat` | Nightscout entries + treatments + profile; an app API pull; one D1NAMO subject directory |
| **Corpus** | many subjects | `parse_corpus(root) -> Mapping[str, UnifiedFormat]` | D1NAMO, CGMacros, Loop |

The categories compose rather than stack: **a corpus's member is a bundle or an export.** So
`parse_corpus` is built out of `parse_bundle`, not implemented a third time. That composition is most
of the value of naming them.

**The bundle category already exists, unrecognized.** `from_nightscout_exports(entries_path,
treatments_path, profile_path) -> UnifiedFormat` (`src/cgm_format/format_parser.py:1414`) takes
several files describing one person's record and merges them into one frame with a diagonal concat.
`from_nightscout_url()` does the same across several API endpoints. Both were written as Nightscout
special cases; they are in fact the general shape, and that is why this category serves **app APIs**
as naturally as it serves research datasets — an API pull is several endpoints describing one user,
which is a bundle by any other name.

A bundle merges to **one** frame because its files are different modalities of the same record —
glucose here, insulin there, meals in a third. That is a diagonal concat, exactly what
`_process_nightscout` already does. It is not a corpus, and the distinction is not pedantic: merging
two subjects the way you merge two modalities is the silent-corruption failure described below.

### Corpora split again, by where subject identity lives

- **Subject-per-directory** — each subject is a directory (D1NAMO's `diabetes_subset/001/`,
  CGMacros' `CGMacros-001/`). Subjects are enumerable from the filesystem before reading any data.
- **Subject-as-key-column** — all subjects share one file set, distinguished by a column (Loop's
  `PtID`). Subjects can only be enumerated by scanning a column, which for Loop means scanning
  ~111 million rows.

That difference is not cosmetic: it decides whether the corpus entry point can list its members
cheaply, and therefore whether it can return an eager mapping or must return an iterator.

### What this means for detection

`detect_format` matches patterns against the first N lines of decoded text. A bundle or a corpus has
no single text to sniff — what identifies it is **directory shape**: does `CGMacros-001/CGMacros-001.csv`
exist, is there a `diabetes_subset/`, are there six `LOOPDeviceCGM*.txt` siblings. So the registry
needs a second detection mechanism keyed on paths, sitting beside the existing text-prefix one, not a
new list of patterns fed to the existing loop.

### Where downloaders live

**Corpus downloaders are `scripts/` utilities behind the `dev` extra, never shipped in the package.**
Fetching a static research dataset once is a setup chore for a developer, not a runtime feature of a
consuming app — and the archives are large enough (CGMacros 627 MB, Loop 1.63 GB) that shipping the
capability invites a library user to pull them by accident. `sugar-sugar` already does this correctly:
`download-cgmacros` is a script in the *consuming app*, not in this library.

The existing `nightscout_downloader.py` is not a counterexample. It fetches **the user's own live
data** from their own server, which is a genuine runtime feature; a corpus fetch is a one-time
acquisition of someone else's published archive. Different purpose, different home.

---

## The organising decision

**Facets become frames. Measurements become an extended schema.**

Dual-device and multi-subject look like two problems and are one: a single source yields N frames
that differ by a *facet* — which device, which person — not by content. So both get the same
treatment, and identity lives in the **mapping key**, never in a column.

### Why a `subject_id` column is the wrong answer

This is mechanical, not a matter of taste. Every stage downstream of the parser assumes it is
looking at a single time series:

- `_postprocess_unified` sorts by `datetime`. With 45 subjects in one frame, that interleaves them.
- `detect_and_assign_sequences` then splits on time gaps — and having interleaved the subjects, it
  splices them into shared sequences. Nothing errors. The output is simply wrong.
- `interpolate_gaps` would then invent rows bridging one person's Tuesday to another's Thursday.

Making a `subject_id` column safe means teaching `FormatProcessor` to group by it at every stage.
`FormatProcessor` is the vendor-agnostic half of the library, and
[PHILOSOPHY.md](PHILOSOPHY.md) states that a new sensor format requires zero changes to it. A schema
column that only works if the processor learns about it is not an additive change; it is a new
contract wearing an additive change's clothes.

Keeping identity outside the frame costs nothing by comparison. A `dict[str, UnifiedFormat]` holds
exactly the same information, and every frame in it is independently valid — sequences,
interpolation and synchronization all work per frame with no changes at all.

### Precedent

This is not a new idea in the codebase. `from_nightscout_exports()`
(`src/cgm_format/format_parser.py:1414`) and `from_nightscout_url()` are already dataset-level entry
points that take more than one file and bypass `detect_format()` entirely. A keyed multi-frame entry
point is the same move, generalized.

---

## Tracks are alternative views, never shards

When a CGMacros file splits into a `libre` frame and a `dexcom` frame, the rows that belong to
neither device — meals, macronutrients, heart rate, photo annotations — are **replicated into
both**. Each frame is a complete, self-contained view of those ten days as seen through one sensor.

The consequence matters more than the rule: **concatenating two track frames double-counts every
meal.** A consumer who reasonably assumes tracks are shards of one dataset, and stacks them, gets
carbohydrate totals that are exactly twice reality, with no error raised anywhere. This is the single
most likely misuse of the design, so it belongs in the docstrings and in
[UNIFIED_FORMAT.md](UNIFIED_FORMAT.md), not only here.

Tracks are alternatives. Pick one, or compare them. Never add them.

---

## The extended schema

The measurement channels are the easy half, and they are genuinely additive:

```
CGM_SCHEMA_EXTENDED = derive_schema(CGM_SCHEMA, append_data_columns=(...))
```

carrying food decomposition (protein, fat, fiber, calories), wearable streams (heart rate, METs,
activity calories, steps), and `annotations` — a `Utf8` column holding a JSON object for anything
that has no typed home, such as a meal-photo path.

Three properties this must have:

**`CGM_SCHEMA` is not modified.** Existing consumers see exactly the frame they see today. The
extended schema is an opt-in target, not a replacement.

**`ExtraColumnError` is not relaxed.** It would be tempting to make validation accept "at least these
columns" so an extended frame passes the core check. That error guards a real invariant — it is how
we notice a parser smuggling a column through — and weakening it to buy convenience trades a
correctness check for ergonomics. Extended frames validate against the extended schema, and a
narrowing helper downcasts to the core one.

**Annotations serialize deterministically.** JSON with sorted keys and stable float formatting.
Without that, two runs produce different bytes for the same data, and the round-trip and idempotency
guarantees — which are byte-level — become flaky rather than false, which is worse.

### The thing that makes this non-trivial

`format_processor.py` refers to `CGM_SCHEMA` at 24 sites. Most are gated on `validation_mode` and
fail loudly. Two are not:

- `format_processor.py:410` — `CGM_SCHEMA.validate_columns(result, enforce=True)`
- `format_processor.py:1302` — `CGM_SCHEMA.validate_dataframe(result_df, enforce=True)`

Both enforce unconditionally, ignoring `validation_mode`, and enforcement drops extra columns
(`interface/schema.py:410`, `dataframe.select(expected_columns.keys())`). An extended frame handed to
the processor therefore comes back **silently narrowed** — the macronutrients and the annotations are
gone, no warning, no error, in the middle of the pipeline.

So the extended schema is not a schema-file change with a parser following behind. Threading a
schema parameter through the parser and processor *is* the work. RM8 owns it.

---

## What stays out

- **No new core dependency.** Polars only. In particular, the Loop archive is deflate64, which the
  standard library cannot open — that is a user-side extraction step or at most a `scripts/` helper,
  never an import in the core.
- **No resampling in a parser.** CGMacros ships on a 1-minute grid; the library's default grid is
  5 minutes. Regridding is the processor's job and is lossy, so the parser emits native cadence and
  the caller passes `expected_interval_minutes`.
- **No silent substitution.** Where a corpus offers a second-best value in place of a missing one —
  Loop's per-patient timezone offset standing in for a per-row one that is absent for most patients —
  it is offered explicitly and warned about, per `CLAUDE.md` §2. A substitution the caller cannot see
  is the failure mode this library exists to prevent.

---

## Appendix A — CGMacros ground truth

PhysioNet `cgmacros/1.0.0`, CC BY-NC-SA, open access, no credentialing. Downloaded by
`sugar-sugar`'s `download-cgmacros`. Every figure below was read off the actual files, not the
published data dictionary.

45 subjects in directories `CGMacros-001` … `CGMacros-049` (the numbering has gaps), one CSV each,
687,580 rows in total, 1,706 meals and 3,197 image paths. `Libre GL` is populated on essentially
every row; `Dexcom GL` on about 92%. Both are mg/dL. The authors interpolated both series onto a
common **1-minute grid**, so neither is at its device's native cadence (Libre Pro samples every
15 minutes, Dexcom G6 Pro every 5).

The header of subject 001:

```
Unnamed: 0,Timestamp,Libre GL,Dexcom GL,HR,Calories (Activity),METs,Meal Type,Calories,Carbs,Protein,Fat,Fiber,Amount Consumed ,Image path
```

**That header is not stable across the 45 subjects — there are 9 distinct variants.** This is drift
*within a single release*, and it is what will break a parser written against subject 001 and tested
only on subject 001:

| Variant | Subjects affected | Handling |
|---|---|---|
| `METs` vs `Intensity` | 11 use `Intensity` | `aliases` |
| `Unnamed: 0` present or absent | 9 have it | dropped as an extra |
| `Amount Consumed ` with a trailing space | 1 | header strip, then `aliases` |
| `Amount Consumed` **absent entirely** | 2 | added as a typed null — see below |
| an extra `Sugar` column | 1 | declared optional, or dropped |
| `Steps` + `RecordIndex` in place of `Calories (Activity)` | 1 | `Steps` maps to an extended column; `RecordIndex` dropped |
| `Dexcom GL` / `Libre GL` in swapped order | 1 | harmless; Polars is name-keyed |

The two subjects missing `Amount Consumed` hit the gotcha recorded in
[NEW_SCHEMA.md](NEW_SCHEMA.md): a `.select(pl.col(X))` is evaluated even when an upstream `.filter()`
leaves zero rows, so an absent column raises `ColumnNotFound` regardless of whether any row would
have used it. This is exactly how the newer LibreView export used to crash the Libre insulin
sub-frame. The fix is `aliases` plus `normalize_headers`, and typed nulls for genuinely absent
columns — not a guard around each select.

`Meal Type` carries ten raw spellings for four meals: `Breakfast`/`breakfast`, `Lunch`/`lunch`,
`Dinner`/`dinner`, and `Snack`/`Snacks`/`snack`/`snack 1`. Normalizing case and plural is
straightforward; the raw string is worth keeping in `annotations` so the normalization stays
inspectable.

**1,553 rows carry a photo with no meal attached** — the meal-*end* photograph — against 1,644 rows
with both, and 62 meals with no photo at all. Annotation-only rows are therefore the majority of
photo rows, which is why `annotations` cannot simply hang off a `CARBS_IN` event. A row that is only
an annotation has to survive `_postprocess_unified` with no glucose and no carbs.

Timestamps in the files read `2020-05-01 10:30:00`, while the published data dictionary describes
them as `Month/Day/Year HH:MM`. Probe with a format tuple rather than trusting either. Dates are
shifted by 365–720 days for de-identification, so absolute dates are meaningless but intervals are
preserved.

`METs` is stored multiplied by 10, per the data dictionary.

## Appendix B — Loop ground truth

**These figures are second-hand and must be verified before any code is written.** They come from
the JAEB dataset-560 page and from published third-party parsers, not from the authoritative
`DataGlossary.rtf` shipped inside the archive. The column lists in particular were reconstructed
from other people's `usecols` arguments.

The Loop observational study (NCT03838900, Jaeb Center for Health Research) is distributed as a
1.63 GB **deflate64** zip expanding to roughly 20.5 GB of pipe-delimited `.txt`, behind a
click-through data use agreement. Participants share one file set keyed by `PtID`. CGM data is split
across six files of about 2.3 GB each, roughly 111 million rows, with a reported ~35% exact
duplicates. Glucose is in mmol/L (Tidepool stores mmol/L only, and this export inherits that).

The single fact that shapes the design: **a 2.3 GB file cannot go through the current ingest path at
all.** `parse_file` reads bytes, `decode_raw_data` returns a `str`, and `detect_format` splits that
string. The whole table would have to exist in memory as a Python string before parsing begins. Loop
therefore waits on the lazy `scan_csv` path (RM7); it is not a `_process_*` branch that happens to be
large.

Two further traps worth recording now:

- `TmZnOffset` is reportedly null for around 63% of patients, so local time would have to be
  reconstructed from a per-patient offset in `PtRoster`. That is a substitution, and it is opt-in
  with a warning or it does not happen.
- Carbohydrates live in two mutually exclusive eras — `LOOPDeviceWizard.txt` for 2017 to mid-2018,
  `LOOPDeviceFood.txt` afterwards — with an overlap of patients but not of time.

There is no in-house prior art to lean on. `glucose_data_processing`, the sister library, contains
only `base_converter`, `dexcom_g6_converter`, `freestyle_libre3_converter` and `format_detector`
locally; it has no Loop converter.

## Appendix C — D1NAMO ground truth

Zenodo record `5651217`, version 1.2.0, DOI `10.5281/zenodo.5651217`. Open access, anonymous
download, **CC BY-SA 4.0** — note the *share-alike*, which is stricter than the CC-BY most datasets
carry and which is a reason to prefer a synthetic fixture over an excerpt. Paper: Dubosson et al.,
*Informatics in Medicine Unlocked*, 2018.

Six zips, ~11.2 GB total. The figures below were verified by reading the archives directly.

**Cohort.** 9 type-1 diabetes subjects (`001`–`009`) and 20 healthy subjects, roughly 4 days each —
47 recording sessions across the diabetes subjects, 84 across the healthy ones. One gotcha worth
encoding in a test: the healthy subset's twelfth subject directory is literally named
**`012_diabetes`**, so a subject-id pattern that assumes three digits will drop or mis-key it.

**Layout — the motivating example for the bundle category.** Each subject is a directory of files,
one per *modality*:

```
diabetes_subset_pictures-glucose-food-insulin/
  001/ … 009/
    glucose.csv  food.csv  insulin.csv  food_pictures/001.jpg …

healthy_subset_pictures-glucose-food/
  001/ … 011/  012_diabetes/  013/ … 020/
    glucose.csv  food.csv  annotations.csv  food_pictures/000.jpg …

diabetes_subset_sensor_data/   <subject>/sensor_data/<session>/<session>_{Accel,BB,Breathing,RR,Summary}.csv
diabetes_subset_ecg_data/      <subject>/sensor_data/<session>/<session>_ECG.csv
```

Session directories are named `2014_10_01-10_09_39` and every file inside is prefixed with that name.
**ECG ships in a separate zip from the other four Zephyr streams** but under an identical tree, so the
two must be merged by session-directory name — a bundle assembled across archives, not within one.

**The two subsets are not one format.** They differ by more than units:

| | diabetes | healthy |
|---|---|---|
| `food.csv` header | `picture,description,calories,balance,quality,datetime` | `date,time,picture,description,calories,balance,quality` |
| `insulin.csv` | present | **absent** |
| `annotations.csv` | absent | present — parsed; start becomes a row, end kept in the annotation |
| `glucose.csv` `type` values | `cgm` (5-min automatic) / `manual` | `BB,AB,BL,AL,BD,AD` (before/after meal) and empty |
| photo numbering | 1-based (`001.jpg`) | 0-based (`000.jpg`) |

`glucose.csv` shares one header — `date,time,glucose,type,comments` — and is **mmol/L** in both.

**The healthy subset has no CGM at all**, only four to six fingersticks a day. Those are calibration-
style readings, not sensor readings, and mapping them to `EGV_READ` would misrepresent them; `CALIBRAT`
is the honest target, exactly as Libre strip readings already map there. Whether a subset with no
continuous glucose belongs in this library at all is a scoping question, not a parsing one — RM13
records it as a decision rather than assuming an answer.

**There is no carbohydrate column anywhere in D1NAMO.** Meals carry `calories`, plus human-assigned
`balance` and `quality` labels. Our `carbs` column therefore stays null — a genuine "the source did
not say", not a zero. This makes D1NAMO's food data entirely dependent on the extended schema (RM8):
without a `calories` column the meal rows carry nothing but a timestamp and an annotation.

**Timestamp conventions are mixed within one subject directory**, which is the trap most likely to
produce silently wrong data:

| File | Literal | Format |
|---|---|---|
| `glucose.csv`, `insulin.csv` (diabetes) | `2014-10-01` / `19:14:00` | `%Y-%m-%d` / `%H:%M:%S` |
| `glucose.csv` (healthy) | `2014-10-01` / `11:35` | `%Y-%m-%d` / `%H:%M` — no seconds |
| `food.csv` (diabetes) | `2014:10:01 19:27:49` | `%Y:%m:%d %H:%M:%S` — EXIF-style colons in the date |
| all Zephyr streams | `01/10/2014 10:09:39.417` | `%d/%m/%Y %H:%M:%S.%f` — **day-first** |
| session directory name | `2014_10_01-10_09_39` | `%Y_%m_%d-%H_%M_%S` |

A naive parse without an explicit format misreads the day-first Zephyr timestamps as month-first for
every day of the month ≤ 12. No timezone information anywhere; times are local Swiss time, October
2014, before that year's DST change.

**Sampling rates**, measured from the files: ECG 250 Hz, accelerometer 100 Hz, `Summary.csv` 1 Hz with
35 columns, breathing **25 Hz**. The paper is reported to state 18 Hz for breathing; the file
disagrees, and the file wins. Recorded here so nobody re-derives the discrepancy.

`Summary.csv` is riddled with sentinel values that must not be read as measurements: `-3276.8` for an
absent skin-temperature sensor, `65535` for noise/GSR/HRV, `-128` for RSSI and TxPower, `6553.5` for
core temperature. `BB.csv` and `RR.csv` were observed with an **empty `Time` column on every data
row**, which would make them useless for alignment — checked in one session only, so verify before
relying on it.

**Meal photographs** live in `<subject>/food_pictures/`, and `food.csv`'s `picture` column holds the
bare filename resolving against that sibling directory. 352 JPEGs across the corpus. Diabetes subject
`005` has none at all, so a parser must tolerate a `food_pictures/` directory that is empty. Stray
`.DS_Store` files sit inside several directories and need filtering.

Real dirt, **[corrected 0.10.0]** — every item verified against the archives, and the distribution
matters more than the list:

- **All of it is in the healthy subset.** The diabetes subset's `glucose.csv` is clean: 8,055 `cgm`
  and 166 `manual` rows, every value a plain decimal in 2.2–22.2 mmol/L.
- `7:0` — a `:` typed for a `.` — is a single row, in subject **017**.
- The leading zeros (`08.2`, `05.4`, …) are **all in `012_diabetes`**, the same subject as the
  directory-name trap. Both traps in one subject.
- `No information` appears in `balance`/`quality` (3 rows) and the corrupt `8 Balance""` once; 102
  food rows carry empty balance/quality.
- `description` is free text containing commas, so a real CSV reader is mandatory.

**A "dangling `picture` reference" is not a missing file.** Ten rows across healthy subjects 002,
004, 007 and 013 hold *words* — `lunch`, `diner`, `breakfast` — where a filename belongs. That is
"the source said something we cannot resolve", a different report from diabetes subject 005's nine
rows with an empty `picture` cell ("the source did not say"). Both instances are real, so the
distinction is testable against data rather than only against a synthetic.

**`insulin.csv`'s header was never recorded here:** `date,time,fast_insulin,slow_insulin,comment`.
It maps straight onto `insulin_fast` / `insulin_slow`.

**Diabetes subject 005 carries the literal `NA` in every `food.csv` `datetime` cell** — the only
subject in the corpus that does, and absent from every prior survey of this dataset. Its meals
cannot be placed on a timeline; its glucose and insulin are unaffected.

Public parsers worth reading before writing ours: `IrinaStatsLab/Awesome-CGM`
(`R/Dubosson2018/preprocessor.r`, the reference harmonization — filters `type == "cgm"`, ×18 to
mg/dL), `IrinaStatsLab/GlucoBench` (`exploratory_analysis/dubosson.ipynb`, the fullest glucose+insulin
join), and `PSI-TAMU/D1NAMO` (`preprocess.py`, CGM-window alignment of the Zephyr streams).

---

## Fixtures

`data/.gitignore` ignores `input/` outright (finding F2 in [dogfooding.md](dogfooding.md)), so any
new fixture needs `git add -f`. None of the three real datasets travels with the repo: CGMacros is
CC BY-NC-SA, D1NAMO is CC BY-**SA** (share-alike, so a derived excerpt carries obligations onward),
and Loop is DUA-bound. Size alone would rule out the last two regardless.

Each format therefore gets a committed **synthetic** fixture, following the existing
`Clarity_Export_synthetic.csv` and `FreeStyle_Libre_3_synthetic.csv` precedent, so CI exercises the
path on every run. Synthetics must reproduce the *dirt*, not the happy path — that is the whole point
of having them:

- **CGMacros**: at minimum an `Intensity` variant and one file missing `Amount Consumed`, the two
  cases that break a parser written against subject 001.
- **D1NAMO**: a subject directory per subset, since the subsets differ in schema; a `012_diabetes`-style
  directory name; the EXIF-colon timestamp in `food.csv`; a `food_pictures/` directory that is empty;
  and a `picture` reference with no corresponding file.

Real data stays local behind the skip-if-absent pattern (`tests/test_libre_eu.py:41-55`). Tests must
not hardcode the `../sugar-sugar/` path where CGMacros currently happens to live, nor assume any
corpus has been downloaded.
