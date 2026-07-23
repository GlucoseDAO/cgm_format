# Agent Guidelines — cgm-format

`cgm-format` converts vendor-specific Continuous Glucose Monitoring exports (Dexcom, Libre,
Medtronic, Nightscout) into one standardized, well-typed Polars DataFrame for ML training and
inference. Single package, `src/` layout (`src/cgm_format/`, hatchling build). **Polars is the only
core dependency**; everything else (`typer`, `rich`, `httpx`, `pandas`, `pyarrow`, `frictionless`)
lives behind the `cli` / `dev` extras. GitHub: `GlucoseDAO/cgm_format`.

This is the **single source of truth** for working in this repo (`AGENTS.md` is a symlink to it). The
design *rationale* lives in **[docs/PHILOSOPHY.md](docs/PHILOSOPHY.md)**; stage/schema/usage detail in
**[docs/PIPELINE.md](docs/PIPELINE.md)**, **[docs/UNIFIED_FORMAT.md](docs/UNIFIED_FORMAT.md)**,
**[docs/USAGE.md](docs/USAGE.md)**. Keep this file in sync with the code — see *Self-correction*.

**This library is in active development.** No frozen constitution, no full backward-compat
obligation. Prefer additive changes, but a breaking change to the schema, CLI, or public API is fine
when it's the right call — bump the version in `pyproject.toml` and note it. What you must *not* break
casually are the runtime invariants (idempotency, losslessness, deterministic ordering): those are
correctness, not compatibility, and they're covered by tests.

## Project map

The two public entry classes:
- `FormatParser` (`format_parser.py`) — Stages 1–3: decode raw bytes, detect vendor format, parse to
  a unified Polars DataFrame.
- `FormatProcessor` (`format_processor.py`) — Stages 4–6: sequence detection, gap interpolation,
  timestamp synchronization, inference prep, data-only export. All methods are `@classmethod` (no
  mutable instance state).

Supporting modules:
- `formats/unified.py` — `UnifiedEventType`, `Quality` flags, `CGM_SCHEMA` (the canonical
  `CGMSchemaDefinition`).
- `formats/{dexcom,dexcom_eu,libre,medtronic,nightscout}.py` — vendor column enums, detection
  patterns, schemas.
- `formats/supported.py` — `FORMAT_DETECTION_PATTERNS`, `SCHEMA_MAP`, `KNOWN_ISSUES_TO_SUPPRESS`.
- `interface/cgm_interface.py` — abstract `CGMParser` / `CGMProcessor`, all exception types,
  `ProcessingWarning`, constants, the `SupportedCGMFormat` enum.
- `interface/schema.py` — `CGMSchemaDefinition`, `ColumnSchema`, `EnumLiteral`, Frictionless export
  helpers.
- `cgm_cli.py` — Typer CLI (`cgm-cli`, owned by `[project.scripts]`).
- `nightscout_downloader.py` — `download_nightscout()` over `httpx` (JSON-only; `token` + `api_secret`
  auth).

`scripts/` holds one-off utilities (schema regen, scrub scripts) — not part of the installed package.
`examples/` shows usage patterns — keep them runnable; they're documentation.

## Build & test commands

`uv` is the package manager. **Always run via `uv run`** — never bare `pytest`/`python`/`cgm-cli`
(system Python lacks the package → `ModuleNotFoundError`). Use `uv sync` / `uv add`, never
`uv pip install`.

```bash
uv sync --extra dev                          # FIRST: install/sync deps (dev includes cli + pytest)
uv run pytest                                # full suite  (SLOW, ~15 min — see below)
uv run pytest -vvv tests/test_format_parser.py   # one file, verbose (iterate like this)
uv run cgm-cli --help                        # explore the CLI
uv run cgm-cli detect|parse|pipeline <file>  # detect / parse-to-unified / full 6-stage run
```

- Sync the extras with `--extra dev` (which pulls in `cli`). **Never** `uv run --extra <x> ...` — sync
  first, then run.
- **The full suite takes ~15 min.** Give it a 600s+ timeout or run it in the background; scope to one
  file while iterating.
- Tests are **integration tests against real data** in `data/input/` — do not mock unless truly
  unavoidable.
- `uv lock --upgrade` only updates `uv.lock`; raise the lower-bound version constraints in
  `pyproject.toml` by hand to match the newly resolved versions.

## Architecture: the 6-stage pipeline

| Stage | Class · method | Description |
|-------|----------------|-------------|
| 1 | `FormatParser.decode_raw_data` | Strip BOM, fix encoding artifacts |
| 2 | `FormatParser.detect_format` | Pattern-match header to `SupportedCGMFormat` |
| 3 | `FormatParser.parse_*` | Vendor CSV → unified Polars DataFrame |
| 4 | `FormatProcessor.detect_and_assign_sequences` | Split on large gaps, assign `sequence_id` |
| 5 | `FormatProcessor.interpolate_gaps` / `synchronize_timestamps` | Fill small gaps, snap to 5-min grid |
| 6 | `FormatProcessor.prepare_for_inference` / `to_data_only_df` | Quality checks, drop service columns |

Hard boundary: the **parser** knows vendors (BOM, metadata rows, record types, timestamp formats) and
emits `UnifiedFormat`; the **processor** operates only on `UnifiedFormat` and is vendor-agnostic. A
new sensor requires zero processor changes.

## Unified format schema

Canonical output is a Polars DataFrame conforming to `CGM_SCHEMA` (`formats/unified.py`):

- **Service columns** (metadata): `sequence_id`, `original_datetime`, `quality`, `event_type`
- **Data columns** (signal): `datetime`, `glucose`, `carbs`, `insulin_slow`, `insulin_fast`, `exercise`

`get_polars_schema(data_only=True)` / `to_data_only_df()` strip service columns for ML consumption.
`quality` is a bitwise `Quality` flag (`OUT_OF_RANGE`, `SENSOR_CALIBRATION`, `IMPUTATION`,
`TIME_DUPLICATE`, `SYNCHRONIZATION`); `0` = good. `event_type` holds 8-char `UnifiedEventType` codes
(`"EGV_READ"`, `"CALIBRAT"`, `"CARBS_IN"`, …).

**Validation vs enforcement.** Validation (`enforce=False`) raises typed errors
(`MissingColumnError`, `ExtraColumnError`, `ColumnOrderError`, `ColumnTypeError`). Enforcement
(`enforce=True`) fixes the frame — adds null columns, casts, reorders, drops extras, stable-sorts.
Internal stages enforce; external-facing APIs default to validate. `ValidationMethod` has four modes
(`INPUT`, `OUTPUT`, `INPUT_FORCED`, `OUTPUT_FORCED`); forced variants raise, non-forced only warn.

## Coding standards

- **Type hints mandatory** (Python ≥ 3.10 syntax); **`pathlib.Path`** for paths; **absolute imports
  only** (`from cgm_format.formats.unified import ...`), never relative.
- **No inline imports** — the sole exception is a guarded import of an *optional* dependency
  (`httpx`, `frictionless`, `pandas`) that raises a clear `ImportError` with install instructions if
  missing. The core `FormatParser` / `FormatProcessor` must import cleanly with only Polars installed.
- **Dependency tiers are load-bearing** (PHILOSOPHY: *Minimal core*): never add a hard dependency to
  the core. Heavy packages go behind the `cli`/`dev` extras plus a guarded import. Prefer Polars over
  Pandas everywhere in the core.
- **Polars idiom**: prefer expressions (`with_columns`, `filter`, `group_by`, `join_asof`) over Python
  loops, and lazyframes (`scan_*`, `sink_*`) on large data paths — keep hot paths in the Rust engine.
  Always compute *from* `original_datetime`, never from a mutated `datetime` (the idempotency anchor).
- **Avoid nested try/except** — it hides the real error. Let typed exceptions propagate
  (`UnknownFormatError`, `MalformedDataError`, `ZeroValidInputError`, …); wrap only where there's a
  genuine recovery path (probing the next timestamp format, or the guarded optional-dep import).
  `format_supported()` is the sanctioned soft check that returns a bool instead of raising.
- **`typer` for the CLI**; the user-facing `cgm-cli` command is owned by `[project.scripts]`
  (`cgm_format.cgm_cli:main`). Don't rename it to dodge a stale `uv run` wrapper — `uv sync` after a
  dependency bump instead.
- **`logging` for library diagnostics — never `print`.** `print` is only for CLI output the user
  asked to see. *(Tech debt: `interface/schema.py:423` prints from a schema-regen helper — move it to
  `logging` or the CLI layer when you're next in there.)*
- **Heed terminal warnings — deprecations especially.** They signal an API moved since training. Read
  and fix them; don't paper over them.
- **No placeholder paths or fabricated example values** in code (`/my/custom/path/`, dummy digests).
- **Refactor internals aggressively** — no dead code or old APIs kept for nostalgia. Internals are
  free to change; the schema shape, CLI surface, and public API are the contract (breaking them is
  allowed but deliberate and versioned — see the active-development note).
- **Version comes only from `importlib.metadata.version("cgm-format")`** (which reads
  `pyproject.toml`) — never hardcode a version string. *(Tech debt: `__init__.py:27` hardcodes a
  `"0.8.3"` dev-fallback; the right fix is an editable install (`uv sync`) so metadata is always
  present, then drop the literal.)*

### Type system (this repo's idiom)

- **Schemas are frozen dataclasses.** `CGMSchemaDefinition` / `ColumnSchema` are
  `@dataclass(frozen=True)`; `CGM_SCHEMA` is the single canonical instance and never mutates at
  runtime. Column names bind to Polars dtypes there — dtypes are the source of truth; there is no
  separate type language that can drift.
- **New vendor column vocabularies subclass `EnumLiteral`** (`interface/schema.py`), so members
  compare/serialize as plain strings (`event_type == "EGV_READ"` works in code and CSV round-trips).
- **`Quality` is a bitwise `Flag`** stored as `Int64` — combine with `|`, test with `&`,
  `Quality(0)` is good. It is *not* a categorical column:
  `df.filter((pl.col("quality") & Quality.IMPUTATION.value) != 0)`.
- **`__all__` is used deliberately here** to curate the public surface — keep it in sync with the
  actual imports in `__init__.py`, and never rely on `import *`. Register new public enums/schemas in
  both the import block and `__all__` (per the vendor checklist below).

## The load-bearing invariants (keep these; prove them with tests)

These are the pipeline's correctness contract. Changing one is a real behavior change — gate it on a
test demonstrating the new behavior and update PHILOSOPHY.md.

- **Idempotency.** Every op yields the same result run once or ten times. `original_datetime` is
  write-once (created at parse, never overwritten); `detect_and_assign_sequences` resets to 0 then
  reassigns from scratch; quality flags are additive via `|`. Re-running is a bit-level no-op.
- **Losslessness.** `synchronize_timestamps` keeps all rows (pure timestamp transform);
  `interpolate_gaps` only *adds* rows (marked `IMPUTATION`, plus `SYNCHRONIZATION` when snapped);
  sequence detection is pure annotation. Nothing silently drops or edits original rows.
- **Commutativity of grid ops.** `interpolate_gaps` (snap-to-grid) and `synchronize_timestamps` share
  one grid calculation rooted in each sequence's first `original_datetime`, so their order doesn't
  matter — see *Gap thresholds* below.
- **Deterministic row order.** Parquet/CSV bytes depend on row order. The schema defines a total
  ordering (sequence, time, quality, event type, then data columns); apply stable sorting after any
  `concat`/`merge`. Never derive emitted rows from `set`/`dict` iteration or unstable polars
  (`mode()`/`unique()` without a tie-break). Prefer explicit sort keys and first-occurrence dedup.
  Every new ordering gets a test.

## Testing

- **Run via uv, always** (see *Build & test*). Use `-vvv` when diagnosing.
- **Real data + ground truth.** Integration tests against real vendor files in `data/input/`.
  Exercise the actual parse/process paths — do not mock a data transformation. Compute expected values
  at runtime from the fixture rather than hardcoding a count read off a dump.
- **Meaningful assertions.** Prefer relationships, aggregates, and set equality over
  existence/count-only checks. Hardcoding *domain constants* (vocabulary members, the 15-min gap
  threshold) is fine; hardcoding *row/unique counts* inspected from data is not.
- **Idempotency, round-trip, and every new ordering get a real test** — extend the dedicated
  `test_idempotency.py` / `test_roundtrip_datetime.py` suites when you touch those paths.
- **Be resilient to changing data.** For optional data features (a specific treatment type in a live
  Nightscout pull), `pytest.skip()` rather than asserting the fixture contains it.
- **Avoid the AI test anti-patterns**: happy-path-only, counts derived from inspecting data, mocking
  the transformation under test, and claiming a test "would have caught" a bug without first showing
  it fail on the buggy code.
- `tests/conftest.py` loads `.env` via `python-dotenv` and provides a session-scoped
  `nightscout_data_dir` fixture that downloads Nightscout JSON from `NIGHTSCOUT_URL` (optional
  `NIGHTSCOUT_TOKEN` / `NIGHTSCOUT_API_SECRET`) into `data/input/`, cached; `--nightscout-redownload`
  forces refresh.

## Adding a new vendor format

1. **Create `formats/<vendor>.py`**: file-layout constants, detection patterns, column enums
   (subclassing `EnumLiteral`), and a `CGMSchemaDefinition` for the raw CSV columns.
2. **Register in `supported.py`**: `FORMAT_DETECTION_PATTERNS`, `SCHEMA_MAP`, and optionally
   `KNOWN_ISSUES_TO_SUPPRESS` for vendor CSV quirks Frictionless would flag.
3. **Add the enum value** to `SupportedCGMFormat` in `interface/cgm_interface.py`.
4. **Implement parsing**: a `_process_<vendor>` method + dispatch branch in `FormatParser`. It must
   map vendor→unified columns, probe/normalize timestamps, handle edge cases (out-of-range markers,
   metadata rows, variable-length records), and return a frame passing
   `CGM_SCHEMA.validate_dataframe(enforce=True)`.
5. **Export public symbols** from `__init__.py` (import block **and** `__all__`).
6. **Write real-data integration tests** in `tests/`: detection, parsing, round-trip, full pipeline.

The processor, schema validation, and CLI need zero changes — they only see `UnifiedFormat`.

## Gap thresholds & grid-aligned measurement

- **`SMALL_GAP_MAX_MINUTES = EXPECTED_INTERVAL_MINUTES * 3 = 15`** separates "small" (fillable) from
  "large" (sequence-splitting) gaps. Aligned with the sister lib
  [`glucose_data_processing`](https://github.com/GlucoseDAO/glucose_data_processing)
  (`small_gap_max_minutes=15`).
- **A grid multiple matters.** With `snap_to_grid=True` (default) raw timestamps are projected onto
  the 5-min grid before measuring gaps, so effective gaps are multiples of 5. A threshold that is
  itself a grid multiple (15) gives clean, deterministic fill/skip decisions; the old `19` was not a
  grid multiple and made `interpolate_gaps` and `synchronize_timestamps` disagree on borderline gaps.
- **Commutativity** comes from `_interpolate_sequence` projecting both endpoints of each gap onto the
  grid via `calculate_grid_point()`, then measuring grid-position distance, before applying the
  `> expected_interval` / `<= SMALL_GAP_MAX_MINUTES` thresholds. Only active when `snap_to_grid=True`.
- **Operator convention** (both libs): sequence splits use `> threshold` (a gap *at* threshold stays
  in the same sequence); interpolation fill uses `<= threshold` (a gap *at* threshold IS filled).

## Known pitfalls

- **Encoding artifacts.** Dexcom/Libre files ship UTF-8 BOM (and double/triple-encoded variants);
  `decode_raw_data` normalizes them. Test new formats with raw `bytes`, not pre-decoded strings.
- **Dexcom variable-length rows.** Non-EGV rows omit trailing columns; Frictionless flags them.
  Suppressed via `KNOWN_ISSUES_TO_SUPPRESS` — not data corruption.
- **`Quality` is a bitwise Flag, not an enum.** Values are ints; test with `&` (see above).
- **`frictionless` is optional.** CLI `report`/`validate` use it if present, degrade gracefully
  otherwise. Import it inside functions or guard with `HAS_FRICTIONLESS`; core parser/processor never
  depend on it.
- **`httpx` is optional.** `nightscout_downloader` needs it (in `cli`/`dev` extras); import inside
  functions and raise a clear `ImportError` with install instructions if missing.
- **Nightscout dual-path.** (1) JSON API path — `parse_nightscout()` / `from_nightscout_exports()` /
  `from_nightscout_url()`, combining entries + treatments, `token`/`api_secret` auth. (2)
  nightscout-exporter CSV — a combined file with `# CGM ENTRIES` / `# TREATMENTS` section headers,
  auto-detected by `detect_format()`. `_process_nightscout` dispatches both. The built-in Nightscout
  API CSV endpoints are **not** supported (headerless 5-col entries; treatments returns JSON anyway) —
  `data/input/nightscout_entries.csv` is kept as a negative control. JSON files do **not** go through
  `detect_format()`.

## Format-drift & known-issue handling

Vendor exports mutate over time (new metadata rows, extra columns, encoding variants). When a real
export no longer fits, that's a **schema gap to widen additively**, not a data error.

- Expected Frictionless quirks are suppressed centrally in `KNOWN_ISSUES_TO_SUPPRESS`
  (`formats/supported.py`) — not scattered through parser branches. **Keep suppressions bounded and
  specific**; never blanket-suppress a whole error class.
- For tolerance thresholds, favor a **static floor + dynamic tolerance + a warning** over a hard
  reject, so a slightly-drifted-but-valid export still parses while genuinely broken data is caught.
- A compatibility fix for a new export variant is a **patch bump**.

## Reporting a finding — dogfood it first

Before reporting an idempotency/losslessness/ordering "bug", build a *real, sensible* example against
the actual code paths and show it fails. A loss that is mechanically possible but has no realistic CGM
instantiation is noise, not a finding — walk the data model with a domain eye first. A demonstrated
failure on the current code beats a plausible-looking mechanistic claim.

## Documentation & prose style

- Natural, human prose. Avoid AI tells (em-dash pile-ups, filler transitions, marketing voice). Never
  hallucinate docs or overpromise an unimplemented feature.
- Keep the `README` usable; deep detail lives in `docs/`. New markdown (other than this file /
  `README` / the `AGENTS.md` symlink) goes in `docs/`.
- Describe the library honestly: it *absorbs vendor chaos and emits a trustworthy DataFrame*. Don't
  imply it measures or diagnoses anything.

## Data & assets conventions

- **`data/` is git-ignored by ignore-all + allowlist** (`*`, then `!input/`, `!input/**`). Generated
  output, local dumps, and downloaded Nightscout pulls stay ignored; only `data/input/` fixtures are
  committed (and excluded from the sdist). To commit another top-level subtree under `data/`, add
  explicit `!<dir>/` and `!<dir>/**` lines.
- **Anything over ~5 MB that must travel goes through Git LFS**: `git lfs install` once,
  `git lfs track "<path>"`, commit the pointer, never the raw blob.

**LFS history gotcha:** a blob committed *before* `git lfs track` stays in every past commit even
after the pointer replaces it at HEAD, so the pack still ships it. Detect:

```bash
git lfs ls-files                       # what LFS tracks at HEAD
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '/^blob/ && $3 > 5000000 {print $3, $4}' | sort -rn   # large blobs anywhere in history
```

History-rewriting (`git lfs migrate import`, force-push) is the **user's** call — surface a found blob,
don't rewrite history yourself. Tree operations, pushing, releases, and stash-discarding are the
user's domain.

## Learned workspace facts & preferences

- Source layout `src/cgm_format/` (hatchling, `tool.hatch.build.targets.wheel.packages`).
- Optional-dep groups: `cli` (typer, rich, httpx, pandas, pyarrow, frictionless), `dev` (cli + pytest
  + python-dotenv).
- `download_nightscout()` always fetches JSON (entries, treatments, profile); `token` = query param,
  `api_secret` = SHA1-hashed header.
- When upgrading deps (`uv lock --upgrade`), also raise the lower-bound constraints in
  `pyproject.toml` to match the newly resolved versions.
- Tests should be resilient to changing data — `pytest.skip()` for optional data features rather than
  hard assertions about specific content.

## Self-correction

When outdated API knowledge causes a real crash or logic failure, fix the code *and* update this file
(and the affected `docs/`) with the correct pattern so the next agent doesn't repeat it. Update the
guide immediately whenever code is refactored — stale guidance is worse than none.
