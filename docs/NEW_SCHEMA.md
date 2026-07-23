# Adding a New Format — Checklist & Reference

How format support is organized in `cgm-format`, how `FormatParser` turns a vendor file into the
unified DataFrame, and a step-by-step checklist for adding a new vendor. Written from the code as it
stands (`src/cgm_format/`); if you change the wiring, update this file (see CLAUDE.md *Self-correction*).

---

## What "a supported format" actually is

A format is not one thing — it's a small set of coordinated pieces spread across five files. A format
is "supported" when **all** of these exist and agree:

| # | Piece | Lives in | Purpose |
|---|-------|----------|---------|
| 1 | A `SupportedCGMFormat` enum member | `interface/cgm_interface.py` | The identity token threaded through the pipeline |
| 2 | A vendor module | `formats/<vendor>.py` | Raw column enums, detection patterns, timestamp formats, file-layout constants, a `CGMSchemaDefinition` |
| 3 | Registry entries | `formats/supported.py` | Maps the enum → detection patterns, → schema, → detection line count, → known-issue suppressions |
| 4 | A parse branch | `format_parser.py` | `_process_<vendor>()` + a dispatch arm in `parse_to_unified()` that maps vendor rows → unified rows |
| 5 | Public exports | `__init__.py` (import block **and** `__all__`) | Makes the schema/enums importable; this repo curates `__all__` deliberately |

Plus, non-code-but-expected:
- **Real-data integration tests** in `tests/` (detection, parse, round-trip, full pipeline).
- A committed **fixture** under `data/input/` (the only committed `data/` subtree).
- Optionally, a generated **`formats/<vendor>.json`** Frictionless schema (via the module's
  `regenerate_schema_json()`; batch-regen with `scripts/regenerate_all_schemas.py`).

The registry (`supported.py`) is the single place detection/validation logic is centralized — adding a
vendor means *adding entries there*, never editing the detection loop or the processor.

---

## How `FormatParser` works (Stages 1–3)

`FormatParser` (subclass of the `CGMParser` ABC) is a set of `@classmethod`s — no instance state. The
public entry points all funnel into the same three stages:

```
parse_file(path) ─┐
parse_from_bytes ─┼─▶ decode_raw_data ──▶ detect_format ──▶ parse_to_unified ──▶ UnifiedFormat
parse_base64     ─┘        (Stage 1)        (Stage 2)          (Stage 3)
parse_from_string ───────────────────────▶ (skips Stage 1; assumes decoded text)
```

**Stage 1 — `decode_raw_data(bytes|str) -> str`.** Normalizes known encoding artifacts (double/triple
-encoded UTF-8 BOM, quoted BOM — see `ENCODING_ARTIFACTS`) then decodes with `utf-8-sig`. A `str` input
is returned unchanged. Always test new formats with raw **`bytes`**, since BOM handling only triggers on
bytes.

**Stage 2 — `detect_format(text) -> SupportedCGMFormat`.** Splits the first `DETECTION_LINE_COUNT`
lines and returns the **first** format whose *any* pattern appears in *any* of those lines. Raises
`UnknownFormatError` if nothing matches.
- **Iteration order is load-bearing.** `FORMAT_DETECTION_PATTERNS` is a plain dict; Python preserves
  insertion order, and detection returns on first match. `DEXCOM_EU` is registered **before** `DEXCOM`
  on purpose: the EU export also matches generic Dexcom patterns, so the more specific `mmol/L` check
  must win first. Put more-specific formats earlier.
- `DETECTION_LINE_COUNT = max(per-format data_start_line) * 2`, so detection reads far enough to see
  the header even for formats where the header sits below metadata rows.
- Detection works on the **raw string**, before any CSV parsing, to sidestep vendor CSV quirks.

**Stage 3 — `parse_to_unified(text, format_type) -> UnifiedFormat`.** A dispatch `if/elif` on
`format_type` calling the vendor's `_process_<vendor>()`. Note `DEXCOM_EU` reuses `_process_dexcom(...,
european=True)` — a **variant** format sharing one parser.

Each `_process_<vendor>()` follows the same shape:
1. `pl.read_csv(StringIO(text), ...)` with vendor-specific dialect knobs
   (`skip_rows`/`skip_rows_after_header`/`skip_lines`, `separator`, `truncate_ragged_lines=True`,
   `infer_schema_length`, `schema_overrides`).
2. Clean column names (strip whitespace, smart quotes, BOM leftovers).
3. Probe the timestamp format once via `_probe_timestamp_format(df, col, FORMATS_TUPLE)` — tries each
   candidate format string, returns the first that parses. Vendors ship multiple date layouts across
   versions/locales, so `*_TIMESTAMP_FORMATS` is always a tuple.
4. Build one sub-frame **per event type** (glucose, insulin fast/slow, carbs, exercise), each selecting
   its columns and adding literal `event_type` (a `UnifiedEventType` code) and `quality` (int flag).
5. `pl.concat(frames, how="diagonal")` (diagonal aligns differing column sets, filling nulls).
6. Add `sequence_id = 0` and return `cls._postprocess_unified(unified)`.

**`_postprocess_unified()` is the shared convergence point** — every vendor path ends here, so this is
where the unified contract is enforced:
- Raises `ZeroValidInputError` if no rows survived.
- Creates `original_datetime` as a copy of `datetime` **if absent** (the write-once idempotency anchor).
- Sorts by `datetime`.
- Runs `CGM_SCHEMA.validate_dataframe(enforce=True)` — adds any missing schema columns as typed nulls,
  casts to schema dtypes, drops extras, reorders to schema order, and stable-sorts by
  `get_stable_sort_keys()`.
- Ensures `sequence_id` exists (defaults to `0`).

> **Note the sequence-assignment split.** The `CGMParser` ABC docstrings say Stage 3 assigns
> `sequence_id`, but the real `FormatParser` leaves it `0` (unassigned). Actual sequence detection is a
> **processor** step — `FormatProcessor.detect_and_assign_sequences` (Stage 4). A newly parsed frame has
> `sequence_id == 0` everywhere until the processor runs. Don't rely on the ABC docstring here.

`format_supported(raw)` is the soft check: it runs decode + detect in a `try/except` and returns a bool
instead of raising — use it for batch triage.

---

## The unified target contract

Your parser's job is to emit rows conforming to `CGM_SCHEMA` (`formats/unified.py`). After
`_postprocess_unified`, every row has all of these columns:

**Service columns** (metadata, `get_stable_sort_keys()` order): `sequence_id` (`Int64`),
`original_datetime` (`Datetime('ms')`), `quality` (`Int64`), `event_type` (`Utf8`).
**Data columns** (the signal): `datetime` (`Datetime('ms')`), `glucose` (`Float64`, mg/dL), `carbs`
(`Float64`, g), `insulin_slow` (`Float64`, U), `insulin_fast` (`Float64`, U), `exercise` (`Int64`, s).

- **`event_type`** is one of the 8-char `UnifiedEventType` codes (`"EGV_READ"`, `"CALIBRAT"`,
  `"CARBS_IN"`, `"INS_FAST"`, `"INS_SLOW"`, `"XRCS_LTE/MED/HVY"`, alert/health codes, `"OTHEREVT"`).
- **`quality`** is a bitwise `Quality` flag stored as int: `0` = good; combine with `|`, test with `&`.
  Parsers typically emit `0`, or `Quality.OUT_OF_RANGE.value` for sensor High/Low markers. The
  `IMPUTATION` / `SYNCHRONIZATION` / `TIME_DUPLICATE` / `SENSOR_CALIBRATION` flags are added later by
  the processor.
- **Units are unified**: glucose in **mg/dL** (convert mmol/L with `MMOL_TO_MGDL = 18.0182`), exercise
  in **seconds**. Do the conversion in the parser — the processor and schema assume unified units.
- One source row often becomes one unified row of a single kind (a carb row has `carbs` set and
  `glucose`/`insulin_*` null). That's expected; diagonal concat + schema enforcement fills the nulls.

---

## The schema system (`CGMSchemaDefinition`)

Every format — vendor *and* unified — is described by a frozen `@dataclass` `CGMSchemaDefinition`
(`interface/schema.py`). It binds column names to **Polars dtypes** (the single source of truth) and
carries the file layout:

- `service_columns` / `data_columns`: tuples of `ColumnSchema` TypedDicts (`name`, `dtype`,
  `description`, optional `unit`, optional `constraints`). Column names are usually members of an
  `EnumLiteral` subclass so they compare as plain strings.
- `header_line`, `data_start_line`, `metadata_lines`: 1-indexed file geometry. These auto-generate a
  Frictionless **dialect** (`commentRows` for metadata, `headerRows` when the header isn't line 1) via
  `_generate_dialect` — returns `None` for a standard "header on line 1, no metadata" CSV.
- `primary_key`: used by the unified schema (all data columns) to define true-duplicate rows.

Key methods: `get_polars_schema(data_only=?)`, `get_column_names`, `get_stable_sort_keys` (the total
ordering that makes row output deterministic), `get_cast_expressions`, `to_frictionless_schema` /
`export_to_json`, and **`validate_dataframe(df, enforce=?)`**:
- `enforce=False` (validation): raises typed errors — `MissingColumnError`, `ExtraColumnError`,
  `ColumnOrderError`, `ColumnTypeError` (all subclasses of `MalformedDataError`).
- `enforce=True` (enforcement): fixes the frame — adds null columns, casts (numeric casts use
  `strict=False` to tolerate nulls), drops extras, reorders, stable-sorts. Internal stages enforce;
  external-facing APIs default to validate.

`EnumLiteral(str, Enum)` is the string-safe enum base: members `==` plain strings and serialize without
`ClassName.MEMBER` noise, so `event_type == "EGV_READ"` works in code and after a CSV round-trip. All
vendor column/event vocabularies subclass it. (`LibreRecordType` is an `int, Enum` because Libre keys
rows by a numeric Record Type.)

### How Frictionless validation works

Validation of a **raw vendor file** against its **raw vendor schema** is a separate, optional layer
from the Polars `CGM_SCHEMA` enforcement inside the parser. It's driven by the CLI (`cgm-cli validate`
and `cgm-cli report`) and lives in `cgm_cli.py::_validate_with_frictionless`:

1. Look up the vendor `CGMSchemaDefinition` in `SCHEMA_MAP[format_type]`.
2. `schema.to_frictionless_schema()` → a Frictionless Table Schema dict; `schema.get_dialect()` →
   the CSV dialect (`commentRows`/`headerRows`) auto-derived from `header_line`/`metadata_lines`.
3. Build a Frictionless `Resource(path, schema, dialect)` and call `resource.validate()`.
4. Walk `report.tasks[*].errors`; each error is passed to `_should_suppress_error(...)`. If **every**
   error is suppressed, the file is reported valid (`N known issues suppressed`); otherwise it's invalid.

`frictionless` is an **optional** dependency (`cli`/`dev` extras). Both commands degrade gracefully when
it's absent (`HAS_FRICTIONLESS` guard), so the core library never imports it.

### Known-issue suppression

`KNOWN_ISSUES_TO_SUPPRESS[format]` (in `supported.py`) lists Frictionless-flagged quirks that are
*expected*, not corruption — e.g. Dexcom's variable-length rows (`missing-cell`), text `High`/`Low`
glucose markers (`type-error`), the BOM `incorrect-label`, Medtronic `-------` placeholders, Nightscout
section-separator `blank-row`s. Entries are `(error-type, field-name-or-None, value-or-None)`; a
`None` field/value means "match any". `_should_suppress_error` returns `True` when an error's
`type` + `fieldName` (+ optionally `cell`) match a rule. Keep suppressions **bounded and specific** —
never blanket-suppress an error class.

A rule may carry an **optional 4th element** that caps how many times it fires **per file**, e.g.
`('constraint-error', 'Timestamp (YYYY-MM-DDThh:mm:ss)', None, 1)` — tolerate exactly one drifted
blank-timestamp metadata row from a newer Clarity export, but let a second occurrence surface as a real
error. `_validate_with_frictionless` keeps a per-file tally and passes it to `_should_suppress_error`, so
the cap is enforced across all of a file's errors; uncapped (3-element) rules suppress unconditionally.

---

## Checklist — adding a new vendor format

Work top to bottom; each step references the pattern to copy.

- [ ] **1. Register the identity.** Add a member to `SupportedCGMFormat` in
  `interface/cgm_interface.py` (e.g. `MYVENDOR = "myvendor"`).

- [ ] **2. Create `formats/<vendor>.py`.** Mirror an existing module (`dexcom.py` is the fullest
  example; `libre.py` for a header-below-metadata layout; `nightscout.py` for JSON+CSV dual paths).
  Define:
  - [ ] File-layout constants: `<V>_HEADER_LINE`, `<V>_DATA_START_LINE`, `<V>_METADATA_LINES`.
  - [ ] `<V>_TIMESTAMP_FORMATS` — a **tuple** of candidate `strptime` formats (probing tries each).
  - [ ] `<V>_DETECTION_PATTERNS` — a list of strings that appear in the header/first lines and are
    **unique** to this vendor. Make them specific enough not to collide with other formats.
  - [ ] Column-name enum(s) subclassing `EnumLiteral`; any event/record-type vocabularies.
  - [ ] A `<V>_SCHEMA = CGMSchemaDefinition(...)` describing the raw file (service + data columns, dtypes,
    units, constraints, and the file-layout constants).
  - [ ] A module-level `regenerate_schema_json()` that calls the shared helper with `__file__`.

- [ ] **3. Wire the registry** in `formats/supported.py` — add your format to **all four** dicts:
  `SCHEMA_MAP`, `FORMAT_DETECTION_PATTERNS`, `FORMAT_DETECTION_LINE_COUNT`, and a
  `KNOWN_ISSUES_TO_SUPPRESS` entry (`[]` if none). **Order matters** in `FORMAT_DETECTION_PATTERNS`:
  place more-specific formats before ones whose patterns they'd also match.

- [ ] **4. Implement parsing** in `format_parser.py`:
  - [ ] Import the vendor's columns/constants at the top (grouped with the other format imports).
  - [ ] Add a `_process_<vendor>(cls, text_data) -> UnifiedFormat` classmethod: read CSV → clean column
    names → `_probe_timestamp_format` → build per-event-type sub-frames with literal `event_type` +
    `quality` → `pl.concat(how="diagonal")` → add `sequence_id=0` → `return
    cls._postprocess_unified(unified)`. Convert units to mg/dL / seconds here. Wrap the body so
    `pl.exceptions.PolarsError` becomes a `MalformedDataError` (via `cls._truncate_error_message`).
  - [ ] Add the dispatch arm in `parse_to_unified()`.

- [ ] **5. Export** the new schema and public enums from `__init__.py` — add to the import block **and**
  the `__all__` list (keep the two in sync; no `import *`).

- [ ] **6. Regenerate the JSON schema** (optional but conventional): call the module's
  `regenerate_schema_json()`, or run `scripts/regenerate_all_schemas.py`, and commit `formats/<vendor>.json`.

- [ ] **7. Add a real fixture** under `data/input/` and commit it (that subtree is allowlisted in
  `data/.gitignore`; add `!<dir>/` + `!<dir>/**` if you introduce a new top-level `data/` subtree).

- [ ] **8. Write real-data integration tests** in `tests/` (no mocking). The four things a supported
  format must demonstrably do — **recognized, parses, round-trips, Frictionless-validatable** — each get
  an explicit assertion:
  - [ ] **Recognized.** `detect_format(decode_raw_data(fixture_bytes))` returns your enum, *and* the
    existing detection tests still pass — i.e. your patterns don't hijack another vendor's fixture and
    no other vendor hijacks yours. (See `tests/test_format_detection_validation.py`; detection order in
    `FORMAT_DETECTION_PATTERNS` is the usual culprit.)
  - [ ] **Parses.** `FormatParser.parse_file(fixture)` returns a frame that passes
    `CGM_SCHEMA.validate_dataframe(df, enforce=False)` (schema-clean: right columns, order, dtypes) with
    the expected event types present and values in sane ranges. Compute expectations at runtime; don't
    hardcode row counts.
  - [ ] **Round-trips & is idempotent.** `to_csv_string(df)` → `parse_from_string(...)` reproduces the
    frame (losslessly), and parsing twice yields identical output. Extend `test_roundtrip_datetime.py` /
    `test_idempotency.py`.
  - [ ] **Frictionless-validatable.** Run the vendor schema through Frictionless the way the CLI does —
    `SCHEMA_MAP[YOUR_FORMAT].to_frictionless_schema()` builds without error, its `get_dialect()` matches
    your `header_line`/`metadata_lines`, and validating the **raw fixture** yields *either* a clean
    report *or* only errors covered by your `KNOWN_ISSUES_TO_SUPPRESS` entries (assert the residual
    `error_count == 0`). The quickest end-to-end check is `uv run cgm-cli validate <fixture>` and
    `uv run cgm-cli report <fixture>`. Confirm the run doesn't crash and no *unexpected* error survives
    suppression. If a vendor quirk is version-dependent (a metadata row that comes and goes), give its
    rule a count cap (4th element) so exactly the tolerated number is suppressed and extras still fail.
  - [ ] **Pipeline.** The full `FormatProcessor` chain (sequences → interpolate → synchronize →
    prepare_for_inference) runs on the parsed fixture without error.

- [ ] **9. Verify** end-to-end: `uv sync --extra dev`, then `uv run pytest tests/test_<vendor>.py -vvv`
  and `uv run cgm-cli detect|parse|validate|report <fixture>` (the full suite is ~15 min — scope to your
  file while iterating).

The processor, schema validation, and CLI need **no changes** — they only ever see `UnifiedFormat`.

---

## Gotchas worth internalizing

- **Detection order is a real bug source.** A new format with loose patterns can shadow or be shadowed
  by an existing one. Always add a cross-detection test against the other fixtures.
- **`str` vs `bytes`.** BOM/encoding normalization only runs on `bytes`. Test the byte path.
- **Units are the parser's responsibility.** The schema won't convert mmol/L→mg/dL or duration→seconds
  for you; do it before `_postprocess_unified`.
- **Timestamps vary by version and locale.** Use a probing tuple, never a single hardcoded format.
- **`sequence_id` is `0` after parsing.** Real sequences come from the processor, not the parser —
  contra the ABC docstring.
- **Suppressions must stay specific, never blanket.** Match a concrete `(error-type, field, value)`
  so genuinely broken files still fail. For a quirk that should be tolerated only a bounded number of
  times per file, add the 4th-element count cap rather than an uncapped rule.
- **The Frictionless schema is auto-derived — check the dialect.** `to_frictionless_schema()` and the
  `commentRows`/`headerRows` dialect come straight from your `header_line`/`data_start_line`/
  `metadata_lines`. If those file-geometry constants are wrong, the validator reads the wrong row as the
  header and every column errors at once.
