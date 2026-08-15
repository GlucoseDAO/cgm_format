# Changelog

What actually shipped, newest first. Cross-repo integration changes made on our side are recorded
here too, so agents working in `sugar-sugar` or `glucosedao` are not surprised by a schema or CLI
change they did not ask for.

Version numbers are the `version` field of `pyproject.toml` at the commit. Entries before 0.8.4 were
reconstructed from git history on 2026-08-13 and are summaries of what the commits did, not
contemporaneous release notes — treat the commit as the authority where the two disagree.

Legality sizing (see `CLAUDE.md` §8): additive → minor, removal/retype/rename → major, legibility →
patch.

## 0.10.0 — 2026-08-15

**Minor.** Everything is additive: no column was removed, renamed or retyped, every existing entry
point keeps its shape, and no existing flag changed meaning. Two new source categories and three new
registered formats.

`CGM_SCHEMA` keeps its exact column names, dtypes and order — which is what a consumer reading a
frame depends on — but two of its *metadata* fields did change, and the blanket phrase "untouched"
would have hidden them: `quality`'s `enum` constraint gained `32` for the new `TRACK_MERGE` bit
(additive), and `exercise` now declares its unit as `"s"` rather than `"seconds"`. Both are visible
in the regenerated `formats/unified.json`.

### The extended schema (RM8)

`CGM_SCHEMA_EXTENDED` sits **beside** `CGM_SCHEMA`, adding food (`calories`, `protein`, `fat`,
`fiber`), wearable (`heart_rate`, `breathing_rate`, `acceleration`, `mets`, `activity_calories`,
`steps`), `ketones`, and `annotations`. The core schema is an exact prefix of it, so an extended
frame narrows to a core one by projection alone (`FormatProcessor.to_core_df`).

`FormatProcessor.schema` is now a `ClassVar` and every one of its 24 `CGM_SCHEMA` usages reads it —
including 7 ungated structural reads that were **silently dropping extra columns mid-pipeline**.
`ExtendedFormatProcessor` carries the extended schema. `to_csv_string` / `to_csv_file` became
classmethods; as staticmethods naming `FormatParser` literally, a subclass override of
`validation_mode` had no effect. Call sites are unchanged.

`Quality.TRACK_MERGE` (bit 32) is new. `annotations_to_json` is the single deterministic serializer
for the `annotations` column — sorted keys, `None` for nothing-to-record — which matters because
that column joins the sort keys and the byte-level round-trip guarantee.

Also: `derive_schema` is now exported, and `exercise` declares unit `"s"` rather than `"seconds"`,
the spelling `UNIT_CONVERSIONS` is actually keyed on.

### Source categories and faceted output (RM12, RM9)

`FormatCategory` (`EXPORT` / `BUNDLE` / `CORPUS`) names an assumption the library had never written
down: one file, one subject, one device.

- `parse_bundle(paths)` — several files, **one** subject, each a different modality.
  `from_nightscout_exports` was already this shape and keeps its name.
- `parse_tracks(path)` — one file, several independent measurements of the same quantity.
- `parse_corpus(root, track=…, subjects=…)` — many subjects, keyed `"subject"` or `"subject/track"`.
- `detect_path_format(path)` identifies a directory by shape, beside the existing text-prefix
  detection. Driven by `PATH_DETECTION_PROBES`: glob patterns only, never callables.

**Reaching one subject.** Naming BUNDLE as a category is not much use if the only way into a
bundle-shaped corpus returns all 29 of its subjects, so three pieces close that:

- **`parse_bundle` accepts a subject directory as a member.** `parse_bundle([root / "001"])` parses
  that whole subject — for a corpus whose members are folders, the directory *is* the bundle.
  Naming the files individually cannot work there and was never meant to: `parse_file` refuses a
  bare D1NAMO `glucose.csv` precisely because one modality is not a record. `parse_subject_directory`
  is the per-subject step, named so a subclass can override it without reimplementing the merge.
- **`list_subjects(root)`** returns frozen `SubjectEntry` records — subject id, format, path,
  modality files present, and a `TrackCoverage` per glucose track (values, rows, first, last). The
  ids are read off the corpus rather than derived, which matters because none of them are
  predictable: D1NAMO's healthy subset holds a directory named `012_diabetes` and CGMacros runs
  `001`–`049` with four numbers missing. Coverage counts cells the *source* filled, deliberately not
  the same number as readings the schema can hold — D1NAMO ships a glucose cell reading `7:0`, which
  is counted here and dropped-with-a-warning by the parser, and collapsing the two would hide that.
  Measured on the published corpora: 1.4s for CGMacros' 45 subjects against 12.8s for the equivalent
  `parse_corpus`.
- **`parse_corpus(root, subjects=[...])`** prunes before parsing, so one id costs one subject's work.
  It composes with `track=`. An id that is not in the corpus **raises**, listing what is available —
  the same lesson `track=` taught earlier in this release: a filter that silently selects nothing
  makes a typo indistinguishable from a subject with no data.

`detect_subject_format(path)` is the third detector, one directory level below `detect_path_format`,
driven by its own `SUBJECT_PATH_PROBES` registry. Two registries rather than one because the shapes
differ by exactly that level, and a probe answering both questions would let a corpus root parse as a
single person. When a subject probe fails on something that is in fact a root, the error says so and
names `parse_corpus`.

`CGMACROS_TRACK_COLUMNS` now maps track name → raw column in one place, read by both the parser and
the coverage reader, so the two cannot drift into disagreeing about which sensor a track name means.

**Two things a consumer must know.** A bundle merges *modalities*, never *subjects* — two people's
files concatenate just as cleanly as two modalities and nothing raises, so the caller owns that
guarantee. And **tracks are alternatives, never shards**: non-sensor rows are replicated into every
track, so concatenating two of them double-counts every meal.

`parse_file` on a multi-track source now raises `MultiTrackSourceError` naming both tracks rather
than silently picking a sensor.

### Research corpora (RM10, RM13)

**CGMacros** — 45 subjects, two concurrent sensors, macronutrients and meal photos. All 9 real
header variants are absorbed declaratively. The opt-in synthetic `mean` track averages the two
sensors and flags every synthesized row `TRACK_MERGE`; rows with only one contributing sensor are
deliberately not flagged, because they are that sensor's real reading.

**D1NAMO** — two registered formats, not one with a flag: the subsets differ in `food.csv`'s column
set, in which modality files exist, and in a disjoint glucose `type` vocabulary. Fingersticks map to
`CALIBRAT`, never `EGV_READ` — the healthy subset has no CGM at all. `carbs` stays null throughout
because D1NAMO records no carbohydrate anywhere.

Neither parser resamples: CGMacros' native cadence is 1 minute, and callers pass
`expected_interval_minutes`.

### CLI

- New `cgm-cli corpus <root> --out <dir>`, with `--subject <id>` (repeatable) to parse a selection.
  `cgm-cli detect` accepts a directory.
- New `cgm-cli subjects <root>` — the ids `--subject` accepts, with each track's value count, row
  count, coverage percentage and span. A subject whose glucose could not be read is shown as
  `unreadable` rather than as a plausible zero.
- **Fixed:** `cgm-cli report` iterated a hardcoded three-format list, so `DEXCOM_EU`, `LIBRE_EU`,
  `MEDTRONIC` and `NIGHTSCOUT` were **already** silently absent from its output before this release.
- **Fixed:** `cgm-cli validate` checked every format against `CGM_SCHEMA` regardless of the schema it
  is actually parsed into.
- Processing commands (`pipeline`, `process`, `batch`) now pick their processor from the frame's
  shape instead of hardcoding the core one.

### Surface consistency

Auditing the API and CLI against the docs turned up four mismatches, all fixed before release:

- **`MultiTrackSourceError` was caught nowhere in the CLI.** Five commands (`parse`, `process`,
  `pipeline`, `validate`, `info`) handed a shell user a message naming `FormatParser.parse_tracks(...)`
  — correct advice for a library caller, useless at a prompt. They now print the refusal plus the
  `cgm-cli corpus` invocation that works. `validate` additionally reported it as a *validation
  failure*, which told the user their data was broken when it was not.
- **`parse_corpus(root, track=...)` silently ignored the track on a single-track corpus**, returning
  every subject while the caller believed they had filtered. Now refuses. Found by running a README
  example verbatim.
- **`cgm-cli info <directory>`** surfaced a raw `Is a directory` errno instead of naming `detect` or
  `corpus`.
- `docs/NEW_SCHEMA.md` still described **four** registries; there are eight, six of them exhaustive
  over `SupportedCGMFormat`. The checklist now lists each one and notes that `detect_format` is
  disjunctive while the two path detectors are conjunctive.
- **Docstrings and docs that outlived what they described.** `parse_bundle`'s public docstring said
  `parse_corpus` and `parse_tracks` were "not implemented yet" — both ship in this release, and that
  text is what `help(FormatParser.parse_bundle)` prints. `docs/PHILOSOPHY.md` carried the same
  parenthetical for `parse_corpus`, and both it and `docs/NEW_SCHEMA.md` described the `cgm-cli
  report` hardcoded-list bug in the present tense after it was fixed. The CLI note is kept but
  rewritten to say what is actually true: the CLI is written against the registries by hand rather
  than derived from them, so it *can* fall behind, and it did.
- The `parse_file` refusal on a bare D1NAMO `glucose.csv` told the caller to pass the subject
  directory to `parse_bundle` — which then raised `IsADirectoryError`, because `parse_bundle` took
  files only. The advice is now true rather than the message being softened.
- Stale line pointers: `docs/ROADMAP.md` RM1/RM2 and `CLAUDE.md` §5 cited `interface/schema.py:477`
  and `__init__.py:27`; the code moved to `:519` and `:33`, and RM2 still quoted the version fallback
  as `0.9.0`.

### Docs

`docs/PHILOSOPHY.md` gained a Source Categories section and corrected two claims: the processor is
now schema-parameterized (the vendor boundary is intact), and "the CLI requires zero changes" was
already false. RM8, RM9, RM10, RM12 and RM13 moved to `docs/ROADMAP_HISTORY.md`. RM6 is narrowed to
its remainder — the `ketones` column ships, but what `event_type` a ketone row carries still needs a
real export with populated cells.

## 0.9.0 — 2026-08-13

- European / mmol/L FreeStyle Libre exports (`LIBRE_EU`, RM5): glucose columns relabeled mmol/L, two
  ketone columns appended. Detected ahead of `LIBRE` the same way `DEXCOM_EU` sits ahead of `DEXCOM`.
- Libre historic and scan readings both parse as `EGV_READ`. Libre files now yield more glucose
  rows than before — a consumer comparing row counts across versions should expect the scan
  readings that used to be dropped. Duplicate timestamps (a scan landing on a historic minute) are
  flagged `TIME_DUPLICATE`, not dropped.
- Libre strip (finger-prick) readings parse as `CALIBRAT`.
- Shared `_glucose_to_canonical` helper: both Dexcom and Libre scale glucose from the unit declared
  in the vendor schema. `derive_schema` can append data columns.

## 0.8.4 — 2026-07-24

- Schema-drift primitives: column aliases, `derive_schema`, declarative units.
- LibreView column rename supported.

## 0.8.3 — 2026-07-18 … 2026-07-23

- Newer Clarity/Dexcom exports with an extra metadata row now parse.
- Frictionless suppression cap fixed — a bounded suppression list stopped silently swallowing more
  than it was meant to.
- Added the agent guide (`CLAUDE.md`, `AGENTS.md` symlink) and `docs/NEW_SCHEMA.md`.

## 0.8.2 — 2026-04-22

- European Dexcom G7 export support.

## 0.8.1 — 2026-04-10

- Version bump and housekeeping.

## 0.7.2 — 2026-04-04 … 2026-04-07

- **Nightscout support**, dual-path: the JSON API (`parse_nightscout`, `from_nightscout_exports`,
  `from_nightscout_url`, combining entries + treatments) and the nightscout-exporter combined CSV
  with `# CGM ENTRIES` / `# TREATMENTS` section headers.
- `api_secret` auth added alongside `token`.
- The built-in Nightscout API CSV endpoints were dropped as unsupported — headerless 5-column
  entries, and treatments returns JSON regardless. `data/input/nightscout_entries.csv` stays as a
  negative control.
- **Medtronic parser**: basal insulin, co-occurring events, exports fixed.
- Gap thresholds harmonized with `glucose_data_processing` (`SMALL_GAP_MAX_MINUTES = 15`, a grid
  multiple; the previous `19` made `interpolate_gaps` and `synchronize_timestamps` disagree on
  borderline gaps).
- Interpolation gap detection always computes from `original_datetime`, never from a mutated
  `datetime` — this is the idempotency anchor.
- Inputs moved under `data/input/`; docs added.

## 0.7.0 — 2025-12-12

- CLI tool (`cgm-cli`), synthetic data fixtures, test coverage.
- Exports, READMEs and tests brushed up.

## 0.6.x — 2025-12-09 … 2025-12-12

- **Schema change**, idempotent processing, rigorous tests (0.6.0).
- Docs update (0.6.1), fixes (0.6.2).

## 0.5.1 — 2025-12-03

- Immutable (frozen-dataclass) schemas plus bugfixes.
- Better tests.

## 0.4.x — 2025-11-30 … 2025-12-01

- `Quality` field reworked in the unified format (0.4.0).
- Format-processor and interface fixes (0.4.1), inference-prep update (0.4.2), further processor
  fixes (0.4.3), docs (0.4.4).

## 0.3.x — 2025-11-27 … 2025-11-30

- Separation of concerns in the inference preprocessor (0.3.7).
- Export fixes (0.3.6), QoL functions (0.3.5), `pyproject` fix (0.3.3), restructuring and cleanup
  (0.3.2).

## 0.2.x — 2025-11-26

- Medtronic format, work in progress (0.2.2).
- Docs and version bump (0.2.1).

## 0.1.1 — 2025-10-08 … 2025-10-30

- MVP: interface, schemas, two supported formats, `FormatParser` implemented, `CGMProcessor` planned.
- `FormatProcessor` implemented.
- Data marked as sensor calibration after a gap longer than 2 h 45 min.
