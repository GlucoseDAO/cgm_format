## Project overview

`cgm-format` is a Python library for converting vendor-specific Continuous Glucose Monitoring (CGM) data (Dexcom, Libre) into a standardized unified format for ML training and inference.

The package lives under `src/cgm_format/` (PEP 517 src layout). The two main public classes are:
- `FormatParser` (`format_parser.py`) — Stages 1–3: decode raw bytes, detect vendor format, parse to unified Polars DataFrame.
- `FormatProcessor` (`format_processor.py`) — Stages 4–6: sequence detection, gap interpolation, timestamp synchronization, inference preparation, data-only export.

Supporting modules:
- `formats/unified.py` — `UnifiedEventType`, `Quality` flags, `CGM_SCHEMA` (the canonical `CGMSchemaDefinition`).
- `formats/dexcom.py`, `formats/libre.py` — vendor-specific column enums, detection patterns, schemas.
- `formats/supported.py` — `FORMAT_DETECTION_PATTERNS`, `SCHEMA_MAP`, `KNOWN_ISSUES_TO_SUPPRESS`.
- `interface/cgm_interface.py` — abstract base classes `CGMParser` / `CGMProcessor`, all exception types, `ProcessingWarning`, constants.
- `interface/schema.py` — `CGMSchemaDefinition`, `ColumnSchema`, `EnumLiteral`, Frictionless export helpers.
- `cgm_cli.py` — Typer CLI entry-point (`cgm-cli`).

GitHub repo: GlucoseDAO/cgm_format

## Build and test commands

`uv` is used as the package manager. **Always run commands via `uv run`** — never use bare `pytest`, `python`, or `cgm-cli` directly. The project uses a src layout with hatchling; the package and its dependencies (including `polars`) are only available inside the uv-managed virtual environment. Running bare `pytest` picks up the system Python, which does not have the project installed, and fails with `ModuleNotFoundError`.

```bash
uv sync --extra dev           # FIRST: install/sync all dependencies (dev includes pytest, typer, rich, pandas, pyarrow, frictionless)
uv run pytest                 # run the full test suite
uv run pytest tests/test_format_parser.py   # run a specific test file
uv run cgm-cli --help         # explore the CLI
uv run cgm-cli detect <file>  # detect format of a CGM CSV
uv run cgm-cli parse  <file>  # parse to unified format (stdout or -o <out.csv>)
uv run cgm-cli pipeline <file>  # run full 6-stage pipeline
```

If tests fail with `ModuleNotFoundError: No module named 'polars'` or `No module named 'cgm_format'`, run `uv sync --extra dev` first.

Tests are **integration tests that use real data** in `data/` — do not mock unless absolutely required.

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

## Learned workspace facts

- Source layout: `src/cgm_format/` (hatchling build, `tool.hatch.build.targets.wheel.packages`).
- Test data lives in `data/` (excluded from sdist). Tests use real files — no mocking.
- `scripts/` contains one-off utilities (`regenerate_all_schemas.py`, scrub scripts); they are not part of the installed package.
- `examples/` shows library usage patterns; keep them runnable as documentation.
- The `cgm-cli` entry point is defined in `[project.scripts]` in `pyproject.toml`; the implementation is `cgm_format.cgm_cli:main`.
- Optional dependency groups: `extra` (pandas, pyarrow, frictionless), `cli` (typer, rich + extra), `dev` (pytest + cli).
- `uv lock --upgrade` only updates `uv.lock`; `pyproject.toml` minimum version bounds must be bumped manually if you want to raise them.

## Learned User Preferences

- When upgrading dependencies (`uv lock --upgrade`), also raise the lower-bound version constraints in `pyproject.toml` to match the newly resolved versions.
