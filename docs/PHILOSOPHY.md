# Design Philosophy

This document describes the design principles behind `cgm-format`. Every decision — from schema definitions to error handling — serves the same goal: **make CGM data trustworthy for machine learning**. These principles should guide any new sensor implementation.

## Core Principle: Data You Can Trust

Medical sensor data passes through many hands before it reaches a model. Vendor exports contain encoding artifacts, variable-length rows, proprietary column names, and undocumented edge cases. The library exists to absorb that chaos and emit a single, predictable, well-typed DataFrame — every time.

---

## Type Safety at Every Layer

Types are the first line of defense against data corruption.

**Polars dtypes as the source of truth.** The schema system (`CGMSchemaDefinition`) binds column names to Polars dtypes (`Int64`, `Float64`, `Datetime`, `Utf8`). Validation and enforcement operate on these dtypes directly — there is no separate "type language" that might drift from the actual DataFrame.

**Semantic type aliases.** `UnifiedFormat = pl.DataFrame` adds no runtime cost but tells every function signature *what kind* of DataFrame is expected. `InferenceResult`, `ValidationResult` are tuple aliases that document return contracts.

**String-safe enums.** `EnumLiteral(str, Enum)` members compare equal to plain strings, hash like strings, and serialize without the `ClassName.MEMBER` noise. This means `event_type == "EGV_READ"` works in both Python code and CSV round-trips.

**Frozen schema objects.** `CGMSchemaDefinition` is a `@dataclass(frozen=True)`. Once defined, a schema cannot be mutated — there is exactly one `CGM_SCHEMA` instance for unified data, and it never changes at runtime.

**Bitwise quality flags.** `Quality` is a Python `Flag` enum stored as `Int64`. Individual flags combine with bitwise OR, and testing uses bitwise AND. This is more expressive than a categorical column (a single cell can be both `IMPUTATION | SYNCHRONIZATION`) and cheaper than a multi-column boolean encoding.

---

## Idempotency

Every processing operation must produce the same output whether it runs once or ten times. This is non-negotiable for pipelines where retries, caching, and reprocessing are normal.

**`original_datetime` is the anchor.** Parsing creates this column as a copy of `datetime` before any modification. All downstream operations — grid alignment, gap detection, calibration marking — compute from `original_datetime`, never from `datetime`. Since `original_datetime` is never written after creation, repeated processing always starts from the same reference.

**Sequence detection resets before recomputing.** `detect_and_assign_sequences` sets all `sequence_id` values to 0 and then reassigns from scratch. Calling it twice with the same parameters produces identical output.

**Grid alignment is deterministic.** Both `interpolate_gaps` (in snap-to-grid mode) and `synchronize_timestamps` use the same grid calculation rooted in the first `original_datetime` of each sequence. This makes their order irrelevant: interpolate-then-sync and sync-then-interpolate converge to the same result.

**Quality flags are additive.** Flags are combined with bitwise OR. Re-applying an operation adds the same flag that is already set — a no-op at the bit level.

**Stable sorting guarantees deterministic row order.** The schema defines a total ordering over all columns (sequence, time, quality, event type, then data columns). After any concat or merge, sorting by these keys produces a unique row order.

---

## Separation of Concerns

The pipeline has a hard boundary between vendor-specific and vendor-agnostic code.

**Parser (Stages 1–3):** Knows about BOM marks, Dexcom metadata rows, Libre record types, and vendor timestamp formats. Its job is to emit a `UnifiedFormat` DataFrame. Once that DataFrame exists, the vendor is irrelevant.

**Processor (Stages 4–6):** Operates only on `UnifiedFormat`. Sequences, interpolation, synchronization, calibration marking, and inference preparation are vendor-agnostic. A new sensor format requires zero changes to the processor.

**Registry (`supported.py`):** Detection patterns, schema mappings, and known validation suppressions live in a central registry — not scattered across parser branches. Adding a vendor means adding entries here, not modifying core logic.

---

## Schema System

The schema is the contract between parser output and processor input.

**Service columns vs data columns.** Service columns (`sequence_id`, `original_datetime`, `quality`, `event_type`) carry metadata for processing. Data columns (`datetime`, `glucose`, `carbs`, `insulin_slow`, `insulin_fast`, `exercise`) carry the signal. `get_polars_schema(data_only=True)` and `to_data_only_df()` strip service columns for ML consumption.

**Validation vs enforcement.** Validation (`enforce=False`) checks the DataFrame and raises typed errors (`MissingColumnError`, `ExtraColumnError`, `ColumnOrderError`, `ColumnTypeError`). Enforcement (`enforce=True`) fixes the DataFrame — adds missing columns with nulls, casts types, reorders, removes extras, and applies stable sorting. Internal pipeline stages use enforcement; external-facing APIs default to validation.

**Frictionless as an export, not a dependency.** The schema can generate Frictionless Table Schema JSON for interoperability, but the core library does not depend on the `frictionless` package at runtime. Import it inside functions or guard with availability checks.

---

## Lossless Operations

Processing should add information, not destroy it.

**Synchronization keeps all rows.** Timestamp rounding maps each source row to its nearest grid point. No rows are dropped; the operation is purely a timestamp transform.

**Interpolation adds rows.** New rows are inserted in gaps, marked with `IMPUTATION` (and `SYNCHRONIZATION` if snapped to grid). Original rows are never removed or modified.

**Sequence detection is annotation.** Assigning `sequence_id` does not alter, merge, or delete any row. It is a pure labeling operation.

**`original_datetime` is write-once.** Created during parsing, never overwritten. Any operation that modifies `datetime` leaves `original_datetime` untouched.

---

## Quality Tracking: Two Levels

**Row-level: `Quality` flags.** Fine-grained, per-row quality indicators stored as an integer column. Flags include `OUT_OF_RANGE`, `SENSOR_CALIBRATION`, `IMPUTATION`, `TIME_DUPLICATE`, and `SYNCHRONIZATION`. Multiple flags combine via bitwise OR. Zero means good quality.

**Sequence-level: `ProcessingWarning` flags.** Coarse-grained, returned from `prepare_for_inference()`. Summarize the overall state of the selected sequence: `TOO_SHORT`, `CALIBRATION`, `OUT_OF_RANGE`, `IMPUTATION`, `TIME_DUPLICATES`. These tell the caller whether the data is usable, without requiring inspection of every row.

---

## Functional Style

**Classmethods over instances.** `FormatProcessor` methods are all `@classmethod`. There is no mutable instance state to manage, no initialization ordering to get wrong, and no thread-safety concerns from shared state. Configuration parameters are method arguments.

**Polars expressions over Python loops.** Data transformations use `with_columns`, `filter`, `group_by`, `join_asof`, and expression chains. This keeps hot paths in Polars' Rust engine and makes operations declarative.

**Immutability by default.** Schemas are frozen dataclasses. Enums are immutable. DataFrames are returned as new objects (Polars DataFrames are inherently immutable). The pipeline reads data in, transforms it through a chain, and returns new data out.

---

## Error Handling

**Typed exceptions for predictable failures.** `UnknownFormatError` for unrecognized formats, `MalformedDataError` (and its subclasses) for structural CSV problems, `ZeroValidInputError` for empty data. Callers can catch exactly what they expect.

**Let exceptions propagate.** The library avoids wrapping every block in try/except. If a Polars operation fails because the data is genuinely broken, that error should reach the caller with its original context. Recovery logic is reserved for cases where there is a real alternative path (e.g., trying the next timestamp format during probing).

**`format_supported()` for soft checks.** When you just want to know "can this file be parsed?" without an exception, use `format_supported()`. It returns a boolean and swallows internal errors — a pragmatic escape hatch for batch processing.

**Validation modes control strictness.** `ValidationMethod.INPUT` warns on schema mismatch; `ValidationMethod.INPUT_FORCED` raises. The caller decides the tolerance level.

---

## Dependency Philosophy

**Minimal core.** The only runtime dependency is Polars. If you can `import polars`, you can parse and process CGM data. Everything else is optional.

**Layered extras.** `cli` adds typer/rich/httpx for the command-line tool plus pandas/pyarrow/frictionless for interop and validation. `dev` extends `cli` and adds pytest for testing.

**No version in code.** `__version__` reads from `importlib.metadata.version("cgm-format")`, which pulls from `pyproject.toml`. There is exactly one place where the version is defined.

**Guard optional imports.** Functions that use optional packages import them locally and raise `ImportError` with a helpful message if the package is missing. Module-level imports of optional dependencies are forbidden.

---

## Adding a New Sensor Format

These principles translate into a concrete checklist:

1. **Create `formats/<vendor>.py`**: Define file layout constants, detection patterns, column enums (subclassing `EnumLiteral`), and a `CGMSchemaDefinition` for the raw CSV columns.

2. **Register in `supported.py`**: Add entries to `FORMAT_DETECTION_PATTERNS`, `SCHEMA_MAP`, and optionally `KNOWN_ISSUES_TO_SUPPRESS` for vendor CSV quirks that Frictionless would flag.

3. **Add enum value**: Extend `SupportedCGMFormat` in `interface/cgm_interface.py`.

4. **Implement parsing**: Add a `_process_<vendor>` method in `format_parser.py` and a dispatch branch in `parse_to_unified`. The method must:
   - Map vendor columns to unified columns
   - Probe and normalize timestamps
   - Handle vendor-specific edge cases (out-of-range markers, metadata rows, variable-length records)
   - Return a DataFrame that passes `CGM_SCHEMA.validate_dataframe(enforce=True)`

5. **Export public symbols**: Add new enums and schemas to `__init__.py` and `__all__`.

6. **Write integration tests**: Use real vendor CSV files in `data/`. Test detection, parsing, round-trip serialization, and full pipeline execution. Do not mock.

The processor, schema validation, CLI, and all downstream consumers require zero changes — they only see `UnifiedFormat`.
