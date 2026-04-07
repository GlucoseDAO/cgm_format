## Project overview

`cgm-format` is a Python library for converting vendor-specific Continuous Glucose Monitoring (CGM) data (Dexcom, Libre, Medtronic, Nightscout) into a standardized unified format for ML training and inference.

The package lives under `src/cgm_format/` (PEP 517 src layout). The two main public classes are:
- `FormatParser` (`format_parser.py`) — Stages 1–3: decode raw bytes, detect vendor format, parse to unified Polars DataFrame.
- `FormatProcessor` (`format_processor.py`) — Stages 4–6: sequence detection, gap interpolation, timestamp synchronization, inference preparation, data-only export.

Supporting modules:
- `formats/unified.py` — `UnifiedEventType`, `Quality` flags, `CGM_SCHEMA` (the canonical `CGMSchemaDefinition`).
- `formats/dexcom.py`, `formats/libre.py`, `formats/medtronic.py`, `formats/nightscout.py` — vendor-specific column enums, detection patterns, schemas.
- `formats/supported.py` — `FORMAT_DETECTION_PATTERNS`, `SCHEMA_MAP`, `KNOWN_ISSUES_TO_SUPPRESS`.
- `interface/cgm_interface.py` — abstract base classes `CGMParser` / `CGMProcessor`, all exception types, `ProcessingWarning`, constants.
- `interface/schema.py` — `CGMSchemaDefinition`, `ColumnSchema`, `EnumLiteral`, Frictionless export helpers.
- `cgm_cli.py` — Typer CLI entry-point (`cgm-cli`).
- `nightscout_downloader.py` — `download_nightscout()` and `download_and_parse_nightscout()` helpers using `httpx`.

GitHub repo: GlucoseDAO/cgm_format

## Build and test commands

`uv` is used as the package manager. **Always run commands via `uv run`** — never use bare `pytest`, `python`, or `cgm-cli` directly. The project uses a src layout with hatchling; the package and its dependencies (including `polars`) are only available inside the uv-managed virtual environment. Running bare `pytest` picks up the system Python, which does not have the project installed, and fails with `ModuleNotFoundError`.

```bash
uv sync --extra dev # FIRST: install/sync all dependencies (dev includes cli + pytest)
uv run pytest                 # run the full test suite
uv run pytest tests/test_format_parser.py   # run a specific test file
uv run cgm-cli --help         # explore the CLI
uv run cgm-cli detect <file>  # detect format of a CGM CSV
uv run cgm-cli parse  <file>  # parse to unified format (stdout or -o <out.csv>)
uv run cgm-cli pipeline <file>  # run full 6-stage pipeline
```

If tests fail with `ModuleNotFoundError: No module named 'polars'` or `No module named 'cgm_format'`, run `uv sync --extra dev` first.

Tests are **integration tests that use real data** in `data/input/` — do not mock unless absolutely required.

## Code style guidelines

- Always use type hints (Python ≥ 3.10 syntax).
- Prefer Polars over Pandas everywhere; use `pl.DataFrame`, `pl.LazyFrame`, polars expressions.
- Use `pathlib.Path` for file paths, `typer` for CLI, polars for DataFrames.
- Functional style; avoid unnecessary mutability and duplication.
- No relative imports — always use absolute imports (`from cgm_format.formats.unified import ...`).
- Avoid excessive `try/except` blocks. Let exceptions propagate unless there is a clear recovery strategy.
- Do NOT hardcode version strings in `__init__.py`; version is read from package metadata via `importlib.metadata.version`.
- All new vendor-specific enums should subclass `EnumLiteral` from `interface/schema.py`.

## Architecture: the 6-stage pipeline

| Stage | Class | Description |
|-------|-------|-------------|
| 1 | `FormatParser.decode_raw_data` | Strip BOM, fix encoding artifacts |
| 2 | `FormatParser.detect_format` | Pattern-match header to `SupportedCGMFormat` |
| 3 | `FormatParser.parse_*` | Vendor-specific CSV → unified Polars DataFrame |
| 4 | `FormatProcessor.detect_and_assign_sequences` | Split on large gaps, assign `sequence_id` |
| 5 | `FormatProcessor.interpolate_gaps` / `synchronize_timestamps` | Fill small gaps, snap to 5-min grid |
| 6 | `FormatProcessor.prepare_for_inference` / `to_data_only_df` | Quality checks, drop service columns |

All processing operations are **idempotent** — running them twice produces the same result.

## Unified format schema

The canonical output is a Polars DataFrame conforming to `CGM_SCHEMA` (`formats/unified.py`):

**Service columns** (metadata): `sequence_id`, `original_datetime`, `quality`, `event_type`  
**Data columns**: `datetime`, `glucose`, `carbs`, `insulin_slow`, `insulin_fast`, `exercise`

`quality` is a bitwise `Quality` flag (`OUT_OF_RANGE`, `SENSOR_CALIBRATION`, `IMPUTATION`, `TIME_DUPLICATE`, `SYNCHRONIZATION`); `0` means `GOOD_QUALITY`.

`event_type` stores 8-char string codes defined in `UnifiedEventType` (e.g. `"EGV_READ"`, `"CALIBRAT"`, `"CARBS_IN"`).

## Adding a new vendor format

1. Create `src/cgm_format/formats/<vendor>.py` with column enums subclassing `EnumLiteral`, detection patterns, and a `CGMSchemaDefinition`.
2. Register detection patterns in `formats/supported.py` (`FORMAT_DETECTION_PATTERNS`, `SCHEMA_MAP`).
3. Add a `SupportedCGMFormat.<VENDOR>` enum value to `interface/cgm_interface.py`.
4. Implement the parse branch in `FormatParser` (Stages 1–3).
5. Export new public symbols from `src/cgm_format/__init__.py` and add to `__all__`.
6. Add real-data integration tests in `tests/`.

## Gap thresholds and grid-aligned gap measurement

### SMALL_GAP_MAX_MINUTES = 15 (3 intervals)

The gap threshold that separates "small" (fillable) from "large" (sequence-splitting) gaps is `SMALL_GAP_MAX_MINUTES = EXPECTED_INTERVAL_MINUTES * 3 = 15` minutes. This value is aligned with the sister library [`glucose_data_processing`](https://github.com/GlucoseDAO/glucose_data_processing) which uses the same `small_gap_max_minutes=15` default.

**Why a grid multiple matters:** `interpolate_gaps` uses grid-aligned gap measurement when `snap_to_grid=True` (the default). Raw timestamps are projected onto the 5-minute grid before measuring gaps, so effective gap sizes are always multiples of 5 (0, 5, 10, 15, 20, ...). A threshold that is itself a grid multiple (15) produces clean, deterministic fill/skip decisions. The previous threshold of 19 was not a grid multiple, which caused borderline instability: a raw gap of 18.7 min would round to 20 min on the grid (exceeding 19), while the same gap measured on raw timestamps would be below 19. This made `interpolate_gaps` and `synchronize_timestamps` disagree on whether to fill such gaps.

### Grid-aligned gap measurement for commutativity

`_interpolate_sequence` computes effective gap sizes by projecting both endpoints of each gap onto the 5-minute grid via `calculate_grid_point()`, then measuring the distance between grid positions. This ensures that `interpolate_gaps → synchronize_timestamps` and `synchronize_timestamps → interpolate_gaps` see identical gap sizes and produce identical results (commutativity). The approach is:

1. For each pair of consecutive glucose readings, compute the nearest grid point for both timestamps.
2. Measure the gap as the difference between grid positions (always a multiple of `expected_interval_minutes`).
3. Apply the `> expected_interval_minutes` and `<= SMALL_GAP_MAX_MINUTES` thresholds to the grid-aligned gap.

This is only active when `snap_to_grid=True`. When `snap_to_grid=False`, raw timestamp differences are used (no grid alignment), so commutativity with `synchronize_timestamps` is not guaranteed.

### Comparison operators

Both `cgm_format` and `glucose_data_processing` use the same operator convention:
- **Sequence splits:** `> threshold` (strictly greater → gap AT the threshold stays in the same sequence)
- **Interpolation fill:** `<= threshold` (less-or-equal → gap AT the threshold IS filled)

## Known pitfalls

### Encoding artifacts in vendor CSVs

Dexcom and Libre files frequently ship with UTF-8 BOM (and double/triple-encoded variants). `FormatParser.decode_raw_data` normalises all known variants. When adding new format support, always test with raw `bytes` inputs, not pre-decoded strings.

### Dexcom variable-length rows

Dexcom exports use variable-length CSV rows: non-EGV event rows omit trailing columns (Transmitter ID, Transmitter Time, Glucose Value). Frictionless validation flags these as errors. They are suppressed via `KNOWN_ISSUES_TO_SUPPRESS` in `formats/supported.py`. Do not treat them as data corruption.

### `Quality` is a bitwise Flag, not an enum

`quality` column values are integers, not enum members. Use bitwise AND to test individual flags:
```python
df.filter((pl.col("quality") & Quality.IMPUTATION.value) != 0)
```
`GOOD_QUALITY = Quality(0)` — zero means no flags set.

### Schema validation modes

`ValidationMethod` has four modes (`INPUT`, `OUTPUT`, `INPUT_FORCED`, `OUTPUT_FORCED`). `FormatParser` defaults to `INPUT` validation; pass `ValidationMethod.OUTPUT` to validate the produced unified DataFrame. Forced variants raise on any violation; non-forced variants only warn.

### `frictionless` is an optional dependency

The CLI `report` and `validate` commands use `frictionless` if available, but degrade gracefully without it. Import it inside functions (not at module level) or guard with `HAS_FRICTIONLESS`. The core `FormatParser` / `FormatProcessor` do not depend on it.

### Nightscout dual-file architecture

Nightscout data comes from two separate API endpoints: entries (glucose) and treatments (insulin, carbs, temp basals). Unlike single-file vendor formats, Nightscout requires combining these two sources. Use `FormatParser.parse_nightscout(entries_data, treatments_data)` for the combined parse. The single-file `parse_from_bytes` / `detect_format` path works for entries-only (glucose data). Nightscout treatments CSV (`nightscout_treatments.csv`) is a supplementary file — it cannot be detected or parsed standalone. Both JSON and CSV inputs are auto-detected.

### `httpx` is an optional dependency

The `nightscout_downloader` module requires `httpx` for HTTP requests. It is included in the `cli` and `dev` optional dependency groups. Import it inside functions and raise a clear `ImportError` with install instructions if missing.

## Learned workspace facts

- Source layout: `src/cgm_format/` (hatchling build, `tool.hatch.build.targets.wheel.packages`).
- Test data lives in `data/input/` (excluded from sdist). Tests use real files — no mocking.
- `scripts/` contains one-off utilities (`regenerate_all_schemas.py`, scrub scripts); they are not part of the installed package.
- `examples/` shows library usage patterns; keep them runnable as documentation.
- The `cgm-cli` entry point is defined in `[project.scripts]` in `pyproject.toml`; the implementation is `cgm_format.cgm_cli:main`.
- Optional dependency groups: `cli` (typer, rich, httpx, pandas, pyarrow, frictionless), `dev` (pytest + cli + python-dotenv).
- `uv lock --upgrade` only updates `uv.lock`; `pyproject.toml` minimum version bounds must be bumped manually if you want to raise them.
- `tests/conftest.py` loads `.env` via `python-dotenv` and provides a session-scoped `nightscout_data_dir` fixture that downloads Nightscout data from `NIGHTSCOUT_URL` (and optional `NIGHTSCOUT_TOKEN`) into `data/input/`. Files are cached; pass `--nightscout-redownload` to force refresh.
- `data/.gitignore` uses an ignore-all + allowlist pattern (`*` then `!input/`, `!input/**`). To commit a new top-level subdirectory under `data/`, add explicit `!<dir>/` and `!<dir>/**` entries.
- Nightscout JSON entries are detected and parsed through the standard single-file pipeline (`detect_format` → `parse_from_bytes` / `parse_from_string` / `parse_file`), not only through `parse_nightscout()`.
- `NightscoutApiFormat = Literal["json", "csv"]` in `nightscout_downloader.py`; both `download_nightscout()` and `download_and_parse_nightscout()` accept an `api_format` parameter (default `"json"`).

## Learned User Preferences

- When upgrading dependencies (`uv lock --upgrade`), also raise the lower-bound version constraints in `pyproject.toml` to match the newly resolved versions.
- Tests should be resilient to changing data — use `pytest.skip()` for optional data features (e.g. specific treatment types) rather than hard assertions that assume specific data content.
