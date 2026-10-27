# Roadmap history

Shipped roadmap items, moved here from [ROADMAP.md](ROADMAP.md) with the rationale intact. Nothing
is deleted from the roadmap; it is relocated. Newest first.

An item arrives here when the work shipped — keep the `RMn` id it had, and add what actually
changed and where, so a later reader can find the code without a git archaeology session.

---

## RM3 — `data/` layout does not match the house convention

**Severity:** low · **Status:** shipped 2026-10-27, in the release after 0.14.1 · **Owner:** lead

The house layout puts committed fixtures in `assets/` and reserves `data/` for git-ignored
`input/` · `interim/` · `output/`. This repo commits `data/input/` and writes generated output to
`data/parsed/`, `data/cli_test_output/` and `data/cli_examples_output/`.

Deferred on purpose, and out of scope for any task that is not this task: every integration test
path, the `.gitignore` ignore-all (F2: `input/` is not allowlisted), the sdist exclusion and `tests/conftest.py`'s
Nightscout download target are built on the current names and all work today. Doing it properly
means one change that moves the fixtures, updates every test path, rewrites the allowlist, and fixes
the docs that name these directories — planned as a unit, not smuggled into an unrelated edit.

Legality: no effect on anything a consumer imports, so a patch — but it touches `data/input/` paths
that examples and docs quote, so the docs move with it.

**What changed.** Desk #71 chose option 1, the house layout. Committed fixtures moved with `git mv`
to `assets/fixtures/` (`data/comparison/` became `assets/fixtures/comparison/`); `data/` is ignored
whole; generated output goes to `data/output/` (`parsed/`, `cli_test/`, `cli_examples/`,
`processed_examples/`), with `data/interim/` reserved. Tests take their paths from `tests/paths.py`;
`tests/conftest.py` downloads a Nightscout pull into `data/input/` and reads the committed JSON from
`assets/fixtures/` when there is none. The sdist excludes `assets/` as it does `data/`. F2 resolved.

---

## RM22 — exports of formats the library claims, failing on real files

**Severity:** high · **Status:** shipped 2026-10-22 in 0.14.1 (`ed6accc`, `5c2142d`, `cd2dc21`) · **Owner:** lead

Found by the format survey ([FORMAT_SURVEY.md](FORMAT_SURVEY.md) §1), each seen failing on a public
sample:

1. LibreView, US Libre 3: 12-hour `MM-DD-YYYY hh:mm AM` timestamps (no `%I %p` format).
2. LibreView "Patient report" layout: one extra line before the header.
3. CareLink, US: comma-delimited, the parser assumes `;`.
4. CareLink in mmol/L (`Sensor Glucose (mmol/L)`, also `(mmol/l)`).
5. CareLink, Spanish: `BWZ Carb Input (exchanges)` holds decimals, read as `i64`.

Format drift, so a **patch** each (`CLAUDE.md` §12): widen what is read, never what is emitted. Each
fixture is synthetic, built at test time from a committed synthetic export reshaped into the public
sample's layout, never a copy of a person's file. Asked as desk #63; its default is this item, built
while the person is away.

---

## RM1 — schema-regen helper prints instead of logging

**Severity:** low · **Status:** shipped 2026-10-22 in 0.14.1 (`b4080c8`) · **Owner:** lead

`interface/schema.py:519` does `print(f"✓ Regenerated {schema_file}")`. The house rule is
standard-library `logging` for library diagnostics; `print` belongs only to CLI output the user asked
to see. The helper is called from `scripts/regenerate_all_schemas.py`, so the honest fix is either a
`logger.info` here, or moving the message to the script that owns the user-facing output.

Legality: pure legibility, so a patch.

---

## RM19 — let a caller state the exporter CSV's date order

**Severity:** medium · **Status:** shipped 2026-10-21 in 0.14.0 (`bdbfe60`); the `cgm-cli --date-order` flag was not built · **Owner:** triage

Motivated by **S6** (`docs/FEEDBACK_HISTORY.md`), relayed from `sugar-sugar`. Queued by the
person on 2026-10-21 for unattended work ("Add the fix to unattended to-dos"), answering the
question whether a warning is enough or callers should be able to say the date order outright.

0.12.3 reads a nightscout-exporter CSV in whichever locale reads every row. One case is left that
the text cannot settle: a file on a 12-hour clock in which no day is past the 12th reads completely
both month-first (en-US) and day-first (en-AU, en-NZ, en-IN). It is read month first, as before,
with one warning. A day-first export of that shape therefore still comes back with day and month
swapped on every row, and only the warning says it might have.

Shape: an optional `date_order` parameter, a new `EnumLiteral` (`MONTH_FIRST`, `DAY_FIRST`) whose
default `None` keeps 0.12.3's inference. Given, it restricts `_exporter_datetime_format` to the
formats of that order, so an ambiguous file is read the way the caller says, with no warning, and
rows that do not read that way are counted as now. It belongs on `parse_nightscout` and
`from_nightscout_exports`. Open for the build: how `parse_to_unified` / `parse_file`, which take no
vendor options, reach it (a dedicated exporter entry point rather than class state, since every
method is a stateless classmethod), and whether `cgm-cli parse` grows a `--date-order` flag.

Making the order **required** was the other answer, and it is rejected: it breaks every existing
caller, which is a major, and most files (any US export, and any day-first one with a day past the
12th) need no help.

**Legality:** a new optional parameter and a new enum beside what exists, so **minor**.

---

## RM16 — MiniMed 720G pump exports parse

**Severity:** high · **Status:** shipped 2026-10-21 in 0.13.0 (`5d916df`, `f991caa` (tie order), `e6c48f1` (audit)) · **Owner:** lead

The proposal is [PLAN_pump.md](PLAN_pump.md) §2 and §3.1; this item is the parser half. All four real
720G CareLink exports in the tree fail to parse today (`MalformedDataError` on dtype inference), and
the shipped Medtronic path misreads two columns on any CareLink file: `Basal Rate (U/h)` becomes an
`insulin_slow` dose, and fingersticks become `EGV_READ` glucose (255 of the Guardian fixture's
22,404 "readings" are fingersticks).

Shape: a new variant format `MEDTRONIC_PUMP` (the `DEXCOM_EU` / `LIBRE_EU` pattern) targeting
`CGM_SCHEMA_EXTENDED`, because the target schema is per format. Section-aware read, every column
`Utf8`. Boluses paired by `Bolus Number`, placed at the programmed time with the delivered amount, a
square part carrying a new extended `insulin_duration`. Fingersticks become a new `BG_METER` event
type carrying the value (the person's answer to Q2), on the Guardian path too. The effective basal rate (percent temps
applied, every suspend 0 until `NORMAL_PUMPING`, ties resolved by `Index`) as `BASAL_RT` rows in a
new extended `basal_rate` (U/h). Priming and rows after a clock change left out, each counted in one
warning. A synthetic 720G fixture from a new scrub script, expected values written by
hand.

Two pump behaviours are assumptions, stated in the plan's §3.4 (a percent temp delivers schedule ×
percent; a cancelled square bolus reports its actual duration), the second watched by a warning.

**Legality:** additive (a new format, two extended columns, two event types), so **minor**.
`insulin_slow` losing the Medtronic basal rate, and fingersticks leaving `glucose`, are repairs with
their own changelog lines.

---

## RM17 — fold pump insulin onto the glucose grid

**Severity:** high · **Status:** shipped 2026-10-21 in 0.13.0 (`df57287`, `e6c48f1` (audit)) · **Owner:** lead

The proposal is [PLAN_pump.md](PLAN_pump.md) §3.2. `fold_insulin_onto_grid(cgm, pump)` takes the two
frames apart, never bundles or re-synchronizes them (bundling first lets pump rows move the CGM's
grid, F19), and returns one row per grid point of each CGM sequence: glucose as the CGM had it,
`insulin_fast` = instant doses in the bin + spread doses' share + the basal rate integrated over the
bin, which is `rate × I / 60` for a constant rate (`I` = `expected_interval_minutes`, no literal 12).
Basal is **not** folded into `insulin_fast`: every model trained in `glucose_forecasting_metabonet`
reads basal (U/h) and bolus (U) as separate channels, and the person ruled that real training inputs
take precedence. It travels as the bin's mean `basal_rate`; the fold is an opt-in `fold_basal=True`.
Insulin and carbs come from a `source` option, `PUMP` / `CGM` / `MERGE`. Merge pairs entries at most
`merge_pair_slots` apart (default 1, the person's choice; beyond it both are kept), one-to-one, onto
the pump's slot; it takes a 0 as "did not say", averages only two non-zeros, and warns with both
values beyond a tolerance.
Null outside the pump's coverage, 0 where the pump was suspended; insulin outside every sequence
reported with its total, never attached; overlapping monthly exports deduplicated. A terminal view:
inputs untouched, the output refuses to be folded again.

Checked on real data over the five days where a Dexcom export and the
pump record overlap; no more exports are coming.

**Legality:** additive (a new classmethod, a new enum), so **minor**, riding RM16's release.

---

## RM21 — `to_sugarone_ready_df`: pump basal in the models' input shape

**Severity:** high · **Status:** shipped 2026-10-21 in 0.13.0 (`4622f51`) · **Owner:** lead

`to_ml_ready_df` leaves `Basal Rate (U/h)` empty unless `insulin_slow` is copied into it, and reads a
unified frame the fold's output is not. The person: "keep this output as legacy, build a
to_sugarone_ready_df output node meeting the format the trained models expect". A new classmethod
renders RM17's output as the models in `glucose_forecasting_metabonet` read it (`sequence_id`, `User
ID`, `Timestamp`, `Event Type`, `Glucose (mg/dL)`, `Basal Rate (U/h)` from `basal_rate`, `Bolus
Insulin (U)` from `insulin_fast`, `Carbohydrates (g)`), the exact columns and order taken from that
repo's `src/common/data/columns.py` and loader at build time; nulls stay null. `to_ml_ready_df` is
unchanged. [PLAN_pump.md](PLAN_pump.md) §3.5.

**Legality:** additive (a new classmethod), so **minor**, riding RM16's release.

---

## RM18 — the collapse keeps the first dose on a grid point and drops the rest

**Severity:** medium · **Status:** shipped 2026-10-21 in 0.13.0 (`9bb3447`, `e6c48f1` (sensor-only glucose)) · **Owner:** lead

`_coalesce_duplicate_timestamps`, behind `to_data_only_df(drop_duplicates=True)`, keeps the first
non-null value per column per grid point, so a second bolus or carb entry on the same point is lost,
for every vendor. Amounts (`carbs`, `insulin_slow`, `insulin_fast`) are summed when at least one is
non-null, else null (Polars sums an all-null group to `0.0`, and a null amount means "not recorded").
[PLAN_pump.md](PLAN_pump.md) §2.6 and §3.3.

**Legality:** a repair of a silent loss, so a **patch**, with its own changelog line.

---

## RM2 — hardcoded version fallback in `__init__.py`

**Severity:** low · **Status:** shipped 2026-10-21 in 0.13.0 (`b9b4304`) · **Owner:** lead

`src/cgm_format/__init__.py:33` carries `__version__ = "0.10.0"` as a fallback when
`importlib.metadata.version("cgm-format")` raises. That is a second source of truth for the version,
and it is the one that gets read when the package is not installed — exactly when it is most likely
to be stale. It already has been: the literal still said `0.9.0` at the moment the 0.10.0 bump
landed, which is finding F3's second half.

The fix is an editable install (`uv sync`) so metadata is always present, then dropping the literal.
Worth confirming first that nothing in CI or the docs build imports the package without installing
it.

Legality: patch.

---

## RM5 — European Libre mmol/L variant, and Libre scan/strip parsing

**Severity:** medium · **Status:** shipped 2026-08-13 in 0.9.0 · **Owner:** —

This never sat on the active roadmap. It was implemented from a real LibreView export
(`data/input/JonGrove_glucose_13-8-2026.csv`, local, gitignored) in the same change that
found it. Recorded here so the id is reserved and the code is findable.

**Legality:** minor. A new `SupportedCGMFormat.LIBRE_EU` member, two appended ketone columns on a
derived vendor schema, and newly parsed rows of existing unified event types (`EGV_READ`,
`CALIBRAT`). Nothing a consumer already reads was removed or redefined. `EGV_READ` still means
sensor glucose.

**What shipped:**

- `formats/libre_eu.py` / `formats/libre_eu.json` — `LIBRE_EU_SCHEMA` derived from `LIBRE_SCHEMA`
  via `derive_schema` (renames + units + `append_data_columns` for `Historic Ketone mmol/L` and
  `Scan Ketone mmol/L`). Detection pattern `"Historic Glucose mmol/L"`.
- Registry: `LIBRE_EU` before `LIBRE` in `FORMAT_DETECTION_PATTERNS` (`formats/supported.py`), same
  reason `DEXCOM_EU` precedes `DEXCOM`.
- `FormatParser._process_libre(..., european=True)` and a shared
  `FormatParser._glucose_to_canonical` used by both Dexcom and Libre. Canonical unit named once as
  `CANONICAL_GLUCOSE_UNIT` in `formats/unified.py`.
- Record Type 1 (scan glucose) merges into `EGV_READ` with historic readings. Record Type 2 (strip)
  parses as `CALIBRAT`. All Libre files, not only the mmol/L variant.
- Tests: `tests/test_libre_eu.py` (skips when the local fixture is absent); scan coverage and
  `TIME_DUPLICATE` flagging on the committed `FreeStyle_Libre_3_synthetic.csv`.
- Docs: `docs/CHANGELOG.md` 0.9.0, `docs/NEW_SCHEMA.md` (the old "zero parser changes" mmol/L
  claim), `docs/PIPELINE.md`.

**Left open, not this item:** F1 in `docs/dogfooding.md` (whether colliding scan/historic timestamps
should be reconciled rather than flagged); F2 (new `data/input/` fixtures cannot be committed
without `git add -f`); RM6 (Libre ketone columns sit on the vendor schema and are dropped at
parse — unified has nowhere to put them).

## RM8 — extended unified schema (macros, wearables, annotations)

**Severity:** high · **Status:** shipped 2026-08-15 in 0.10.0 · **Owner:** —

**Legality:** minor. `CGM_SCHEMA_EXTENDED` is a new schema **beside** an unmodified `CGM_SCHEMA`;
`Quality.TRACK_MERGE` is a new flag beside the existing ones; no column was removed, renamed or
retyped, and `ExtraColumnError` was not relaxed. Every existing entry point keeps its shape.

**What shipped:**

- `CGM_SCHEMA_EXTENDED` in `formats/unified.py`, built with `derive_schema(append_data_columns=...)`
  — food (`calories`, `protein`, `fat`, `fiber`), wearable (`heart_rate`, `breathing_rate`,
  `acceleration`, `mets`, `activity_calories`, `steps`), `ketones`, and `annotations` last. The core
  schema is an exact prefix of the extended one, which is what lets a frame be narrowed by
  projection alone.
- **The schema seam.** `FormatProcessor.schema` is a `ClassVar`; all 24 `CGM_SCHEMA` usages in
  `format_processor.py` became `cls.schema`, **including the 7 ungated structural reads** (`:168`,
  `:410`, `:483`, `:680`, `:1081`, `:1230`, `:1302`) that were silently narrowing an extended frame
  mid-pipeline. `ExtendedFormatProcessor` carries the extended schema; `to_core_df()` narrows.
- A third unconditional enforcement the roadmap never counted, `format_parser.py:276` in
  `_postprocess_unified`, now takes a schema defaulted from the new `UNIFIED_TARGET_SCHEMA` registry.
- `to_csv_string` / `to_csv_file` became classmethods — as staticmethods naming `FormatParser`
  literally, a subclass override of `validation_mode` had no effect.
- `annotations_to_json`: one deterministic serializer (sorted keys, `allow_nan=False`, `None` for
  nothing-to-record). Load-bearing because `annotations` joins the sort keys and the byte-level
  round-trip.
- `UNIFIED_EXTENDED` registered **first** in `FORMAT_DETECTION_PATTERNS`, `derive_schema` exported,
  `exercise` unit `"seconds"` → `"s"`.
- Both narrowing failure modes were demonstrated failing against pre-fix code before the fix was
  claimed (`CLAUDE.md` §2). Recorded in `tests/test_extended_schema_narrowing.py`, including the
  trap that a first draft fell into: an all-`EGV_READ` fixture never reaches the narrowing branch.

**Left open, not this item:** RM6's first decision — what `event_type` a ketone row carries. The
column ships and its unit is settled (D8: mmol/L, not routed through `_glucose_to_canonical`); the
parser mapping needs a real export with populated ketone cells.

## RM9 — faceted parse output (multi-track, multi-subject)

**Severity:** high · **Status:** shipped 2026-08-15 in 0.10.0 · **Owner:** —

**Legality:** minor. Two new entry points and a new typed exception raised only by a *newly
registered* format. `parse_file` on every previously supported format behaves exactly as before.

**What shipped:**

- `parse_tracks(path)` and `parse_corpus(root)`, both `-> dict[str, UnifiedFormat]`, declared on
  `CGMParser` and implemented on `FormatParser`. Corpus keys are flat composite strings
  (`"CGMacros-001/libre"`); `/` is public contract, and a subject id may contain `_` but never `/`
  — D1NAMO's `012_diabetes` is exactly why.
- `MultiTrackSourceError`, raised by `parse_to_unified` for a multi-track source, naming both tracks
  and the entry point that returns them (D5). No silent sensor pick.
- **Tracks are alternatives, never shards.** Non-sensor rows are replicated into every track, and a
  test quantifies the misuse: concatenating two CGMacros track frames doubles the meal count and the
  carbohydrate total exactly, with nothing raised.
- The opt-in synthetic `mean` track (D5a), never in `parse_tracks`' default output. Verified on real
  data: rows flagged `TRACK_MERGE` are exactly those where both sensors reported, every mean lies
  between its two inputs, and single-sensor rows are deliberately **not** flagged because they are
  that sensor's real reading.
- Identity never enters the frame — the reason is mechanical and is now in the docstrings and
  `docs/PHILOSOPHY.md`.

## RM10 — CGMacros format support

**Severity:** medium · **Status:** shipped 2026-08-15 in 0.10.0 · **Owner:** —

**Legality:** minor. A new `SupportedCGMFormat.CGMACROS` member plus registry entries. Nothing a
consumer already reads changed.

**What shipped:**

- `formats/cgmacros.py` / `.json`, `_process_cgmacros`, and entries in all six registries.
- **All 9 real header variants** absorbed declaratively: `METs` vs `Intensity` by alias, the
  trailing-space `"Amount Consumed "` by header strip then alias, the 2 subjects lacking
  `Amount Consumed` by typed nulls added up front (the `ColumnNotFound` gotcha — a `select` is
  evaluated even when an upstream filter leaves zero rows). `Unnamed: 0`, `RecordIndex` and `Sugar`
  are dropped rather than smuggled into `annotations`.
- `METs` divided by 10; meal labels normalized from ten raw spellings with the raw string kept in
  `annotations`; annotation-only rows (the *majority* of photo rows) carry `OTHEREVT`.
- **No resampling.** Native cadence is 1 minute; callers pass `expected_interval_minutes`.
- Two pre-existing CLI bugs fixed here because this is what made them visible: `report` iterated a
  hardcoded three-format list (silently omitting `DEXCOM_EU`, `LIBRE_EU`, `MEDTRONIC`,
  `NIGHTSCOUT`), and `validate` checked every format against `CGM_SCHEMA` regardless of its target.

**Left open, not this item:** the `bio.csv`, `microbes.csv` and `gut_health_test.csv` cohort files.
They are per-subject attributes, not a time series, and have no home in a frame keyed by timestamp.

## RM12 — source categories: export, bundle, corpus

**Severity:** medium · **Status:** shipped 2026-08-15 in 0.10.0 · **Owner:** —

**Legality:** minor. A new enum, two new registries, one new concrete classmethod. `SupportedCGMFormat`
kept its shape — the category is a sidecar dict, not a field, because the enum is public API.

**What shipped:**

- `FormatCategory` (`EXPORT` / `BUNDLE` / `CORPUS`) and `FORMAT_CATEGORY`, exhaustive over
  `SupportedCGMFormat` and asserted so.
- `detect_path_format(path)` + `PATH_DETECTION_PROBES` — glob patterns only, never callables.
  Probes are **conjunctive**, and an empty probe tuple never matches: `all()` over an empty tuple is
  vacuously true, so without that guard an unprobed format would swallow the first directory offered.
- `parse_bundle(paths)` and `merge_bundle_frames(frames)` on `CGMParser`, generalizing what
  `from_nightscout_exports` was already doing. Column order is canonicalized against the registered
  unified schemas rather than taken from the concat, which is order-dependent for disjoint columns.
- `from_nightscout_exports` keeps its name and now warns that `profile_path` is accepted and ignored
  — the profile holds settings, not readings, so it has no rows to contribute.
- **A bundle merges modalities, never subjects**, and the library cannot tell them apart. Documented
  in the docstrings and the charter, with a test demonstrating the corruption rather than faking a
  check that cannot exist.

## RM13 — D1NAMO dataset support

**Severity:** medium · **Status:** shipped 2026-08-15 in 0.10.0 · **Owner:** —

**Legality:** minor. Two new `SupportedCGMFormat` members plus registry entries.

**All three open decisions were settled by reading the archives, not by inference:**

1. **Stream scope** — annotation bundle only. The Zephyr waveforms (ECG 250 Hz, accelerometer
   100 Hz) are declined explicitly in `formats/d1namo.py`, per D7. Only 2 of the 6 zips (730 MB of
   11.2 GB) are needed.
2. **The healthy subset is in scope**, with fingersticks mapped to `CALIBRAT` (D6). A test asserts
   that all 20 real subjects produce **zero** `EGV_READ` rows — mapping a fingerstick to a sensor
   trace is the misrepresentation that decision prevents.
3. **Two formats, not one with a flag** — settled by evidence: `food.csv` is a genuinely different
   column set between subsets (split `date`+`time` vs one EXIF-style `datetime`), `insulin.csv`
   exists only in diabetes, `annotations.csv` only in healthy, and the `type` vocabulary is disjoint.
   `derive_schema` patches names and units, not a different set of columns.

**What shipped:** `formats/d1namo.py` / `.json`, `_process_d1namo_subject`, `_parse_d1namo_corpus`,
both formats registered, `scripts/download_d1namo.py` behind the `dev` extra (never shipped in the
package), and synthetic fixtures reproducing every real defect.

**Ground truth corrected while implementing** — `docs/RESEARCH_CORPORA.md` Appendix C was partly
second-hand and wrong in three places:

- `insulin.csv`'s header (`date,time,fast_insulin,slow_insulin,comment`) was never recorded; it maps
  straight onto `insulin_fast` / `insulin_slow`.
- The corrupt literals are **healthy-subset-only**; the diabetes subset's glucose is clean. `7:0` is
  in subject 017, the leading zeros (`08.2`) are all in `012_diabetes` — the same subject as the
  directory-name trap.
- A "dangling `picture` reference" is not a missing file: those cells hold words (`lunch`, `diner`,
  `breakfast`) where a filename belongs. Reported separately from a blank cell, which is the
  "did not say" case.

**Found in the data, absent from every prior survey:** diabetes subject 005 carries the literal `NA`
in *every* `food.csv` `datetime` cell — the only subject that does. Its meals cannot be placed on a
timeline and are dropped with a prominent warning; its glucose and insulin are unaffected.

**Left open, not this item:** the Zephyr streams (RM7-adjacent — the extended schema declares
`heart_rate`, `breathing_rate` and `acceleration`, but waveform→column conversion waits for a
requested use case).
