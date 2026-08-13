# Changelog

What actually shipped, newest first. Cross-repo integration changes made on our side are recorded
here too, so agents working in `sugar-sugar` or `glucosedao` are not surprised by a schema or CLI
change they did not ask for.

Version numbers are the `version` field of `pyproject.toml` at the commit. Entries before 0.8.4 were
reconstructed from git history on 2026-08-13 and are summaries of what the commits did, not
contemporaneous release notes — treat the commit as the authority where the two disagree.

Legality sizing (see `CLAUDE.md` §8): additive → minor, removal/retype/rename → major, legibility →
patch.

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
