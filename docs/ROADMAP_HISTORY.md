# Roadmap history

Shipped roadmap items, moved here from [ROADMAP.md](ROADMAP.md) with the rationale intact. Nothing
is deleted from the roadmap; it is relocated. Newest first.

An item arrives here when the work shipped — keep the `RMn` id it had, and add what actually
changed and where, so a later reader can find the code without a git archaeology session.

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
