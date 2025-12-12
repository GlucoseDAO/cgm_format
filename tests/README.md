# Tests

This directory contains pytest tests for the cgm_format package. All tests are designed to work with real data files and validate the complete processing pipeline.

## Test Organization

The test suite is organized into 8 test files covering different aspects of the CGM data processing pipeline:

1. **Format Detection & Validation** (`test_format_detection_validation.py`) - Validates format detection and Frictionless schema compliance
2. **Format Parsing** (`test_format_parser.py`) - Tests parsing from vendor formats to unified format
3. **Schema Definitions** (`test_schema.py`) - Tests schema structure and Frictionless conversion
4. **Format Processing** (`test_format_processor.py`) - Tests data processing operations (interpolation, synchronization)
5. **Utility Functions** (`test_utils.py`) - Tests helper functions like split_glucose_events
6. **Idempotency** (`test_idempotency.py`) - Validates idempotent and commutative operations
7. **Datetime Roundtrip** (`test_roundtrip_datetime.py`) - Tests datetime type preservation
8. **Integration Pipeline** (`test_integration_pipeline.py`) - End-to-end pipeline tests on real data

## Running Tests

From the project root:

```bash
# Run all tests
uv run pytest tests/

# Run with verbose output
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_format_processor.py

# Run specific test class
uv run pytest tests/test_format_processor.py::TestFormatDetection

# Run specific test
uv run pytest tests/test_format_processor.py::TestFormatDetection::test_all_files_detected

# Run tests matching a pattern
uv run pytest tests/ -k "idempotency"
```

## Test Files

### test_format_detection_validation.py

**Purpose:** Validates format detection and Frictionless schema compliance for all CGM formats.

**Key Features:**
- Parametrized tests run on all CSV files in `data/` directory
- Validates format detection for Dexcom, Libre, and Unified formats
- Uses Frictionless library to validate CSV files against their schemas
- Suppresses known vendor format issues (e.g., Dexcom's "Low"/"High" text markers)

**Test Classes:**
- **TestFormatDetection** - Tests that all CSV files can be correctly format-detected
  - `test_detect_format()` - Verifies each file is detected as a known format
  
- **TestSchemaValidation** - Validates CSV files against their Frictionless schemas
  - `test_validate_against_schema()` - Validates files with known vendor issues suppressed
  - Supports Dexcom metadata rows via `commentRows` dialect
  - Supports Libre header rows via `headerRows` dialect
  
- **TestSchemaDefinitions** - Tests schema structure
  - `test_schema_has_columns()` - Verifies all schemas have service and data columns
  - `test_schema_converts_to_frictionless()` - Tests Frictionless schema generation

**Known Vendor Issues Suppressed:**
- Dexcom: Variable-length rows (missing transmitter columns for non-EGV events)
- Dexcom: "Low"/"High" text markers instead of numeric glucose values
- Dexcom: UTF-8 BOM marker in headers

---

### test_format_parser.py

**Purpose:** Tests parsing from vendor formats (Dexcom/Libre) to unified format.

**Key Features:**
- Tests all convenience parsing methods (from file, bytes, string, base64)
- Validates unified format schema compliance
- Tests roundtrip: vendor → unified → CSV → unified
- Tests sequence detection with large gaps
- No mocking - uses real data files from `data/` directory

**Test Classes:**
- **TestFormatDetection** - Format detection for all data files
  - `test_all_files_detected()` - Tests all supported files can be decoded and format-detected
  - `test_format_counts_reasonable()` - Verifies we have at least one Dexcom or Libre file
  - `test_format_supported()` - Tests `format_supported()` method correctly identifies formats

- **TestUnifiedParsing** - Parsing to unified format
  - `test_parse_all_to_unified()` - Tests all supported files parse to unified format
  - `test_unified_format_schema()` - Validates unified format has expected columns
  - `test_datetime_column_type()` - Verifies datetime column has correct Polars type
  - `test_glucose_values_reasonable()` - Checks glucose values are in valid range (20-500 mg/dL)

- **TestSaveToDirectory** - Saving parsed files to `data/parsed/`
  - `test_save_all_parsed_files()` - Parses and saves all files to parsed directory
  - `test_parsed_files_can_be_read_back()` - Tests roundtrip: parse → save → read → compare

- **TestConvenienceMethods** - Convenience parsing methods
  - `test_parse_from_file()` - Tests `parse_file()` convenience method
  - `test_parse_from_bytes()` - Tests `parse_from_bytes()` method
  - `test_parse_from_string()` - Tests `parse_from_string()` method

- **TestInputHelpers** - Various input sources
  - `test_parse_file()` - Tests parsing from file path
  - `test_parse_base64()` - Tests parsing from base64-encoded data
  - `test_parse_file_matches_parse_from_bytes()` - Verifies different methods produce same results

- **TestErrorHandling** - Error conditions and edge cases
  - `test_unknown_format_error()` - Tests UnknownFormatError is raised for invalid formats
  - `test_malformed_data_error()` - Tests MalformedDataError is raised for malformed data
  - `test_error_message_truncation()` - Ensures huge error messages are truncated to prevent log overflow

- **TestEndToEndPipeline** - Complete pipeline integration
  - `test_full_pipeline_integration()` - Tests read → decode → detect → parse → save → verify

---

### test_schema.py

**Purpose:** Tests schema definitions and Frictionless conversion for all CGM formats.

**Key Features:**
- Tests EnumLiteral custom enum class that serializes cleanly
- Validates Frictionless schema generation with dialects
- Tests column order matches actual CSV file structure (critical for parsing)
- Regression tests for specific bugs (enum serialization, column order)

**Test Classes:**
- **TestEnumLiteral** - Tests custom EnumLiteral enum class
  - `test_enum_str_representation()` - Verifies enums serialize as values, not `<Enum.NAME: 'value'>`
  - `test_enum_string_comparison()` - Tests enum equality with strings
  - `test_enum_json_serialization()` - Validates clean JSON serialization

- **TestFrictionlessSchemaGeneration** - Schema generation
  - `test_basic_schema_generation()` - Tests schema generates valid Frictionless format
  - `test_schema_with_primary_key()` - Tests primary key specification for unified format
  - `test_dexcom_dialect_with_comment_rows()` - Tests Dexcom commentRows dialect
  - `test_libre_dialect_with_header_rows()` - Tests Libre headerRows dialect

- **TestDexcomColumnOrder** - Critical column order validation
  - `test_column_order_matches_csv()` - Verifies Dexcom schema column order matches actual CSV files
  - `test_timestamp_comes_second()` - **CRITICAL:** Timestamp must be column 2 (after Index)

- **TestLibreColumnOrder** - Libre column order validation
  - `test_libre_column_order()` - Verifies Libre schema matches actual CSV structure

- **TestSchemaValidation** - Schema validation tests
  - `test_relative_path_handling()` - Tests Frictionless requires relative paths for security
  - `test_enum_values_in_constraints()` - Validates enum constraints use values, not enum objects

- **TestSchemaConsistency** - Internal consistency checks
  - `test_all_columns_have_types()` - Every column has a valid Frictionless type
  - `test_column_count_consistency()` - Service + data columns = total fields
  - `test_no_duplicate_column_names()` - No duplicate column names

- **TestRegressionPrevention** - Prevents specific bugs
  - `test_enum_not_in_field_names()` - Field names must not contain enum class names
  - `test_primary_key_matches_fields()` - Primary key fields exist in schema
  - `test_timestamp_not_in_data_columns_for_dexcom()` - Timestamp must be in service_columns

---

### test_format_processor.py

**Purpose:** Comprehensive tests for FormatProcessor implementation (interpolation, synchronization, inference preparation).

**Key Features:**
- Tests all FormatProcessor class methods (now all classmethods)
- Tests processing operations on synthetic and real data
- Tests idempotency of synchronize_timestamps on all real datasets
- Tests calibration gap detection and quality marking
- Tests sequence detection and splitting logic

**Test Classes:**
- **Basic Tests** - Initialization and constants
  - `test_processor_initialization()` - Tests class configuration constants
  - `test_constants_match_documentation()` - Validates constants match PIPELINE.md documentation

- **Synchronize Timestamps Tests** - Timestamp rounding and glucose interpolation
  - `test_synchronize_timestamps_basic()` - Tests basic timestamp rounding to nearest minute
  - `test_synchronize_timestamps_glucose_interpolation()` - Tests glucose value interpolation during sync
  - `test_synchronize_timestamps_discrete_events_shifted()` - Tests carbs/insulin shifted to nearest timestamp
  - `test_synchronize_timestamps_multiple_sequences()` - Tests sync works with multiple sequences
  - `test_synchronize_timestamps_is_lossless()` - **Integration test:** Validates lossless sync on all real datasets

- **Interpolate Gaps Tests** - Gap detection and interpolation
  - `test_interpolate_gaps_no_gaps()` - Tests no interpolation when no gaps exist
  - `test_interpolate_gaps_with_small_gap()` - Tests interpolation of gaps < 19 minutes
  - `test_interpolate_gaps_with_snap_to_grid()` - **Parametrized:** Tests grid-aligned interpolation with various starting times and gap sizes (8, 13, 18, 23 minutes) and different timestamp offsets (0-300 seconds in 10-second steps)

- **Prepare for Inference Tests** - Sequence selection, truncation, quality checks
  - `test_prepare_for_inference_keeps_only_latest_sequence()` - Tests only the latest sequence is kept
  - `test_prepare_for_inference_success()` - Tests successful inference preparation
  - `test_prepare_for_inference_truncation()` - Tests sequences are truncated to maximum duration
  - `test_prepare_for_inference_truncation_keeps_latest()` - **Critical:** Verifies truncation keeps LATEST data, not oldest
  - `test_prepare_for_inference_warnings_after_truncation()` - **Critical bug fix:** Warnings calculated on truncated data only

- **Calibration Gap Tests** - Tests calibration gap detection (>= 2:45:00)
  - `test_calibration_gap_marks_next_24_hours_as_sensor_calibration()` - Tests gaps >= 2:45:00 mark next 24 hours as poor quality
  - `test_calibration_gap_exactly_at_threshold()` - Tests gap exactly at threshold triggers marking
  - `test_calibration_gap_below_threshold_no_marking()` - Tests gaps below threshold don't trigger marking

- **Sequence Detection Tests** - Tests large gap sequence splitting
  - `test_large_gap_creates_new_sequence()` - Tests gaps > SMALL_GAP_MAX_MINUTES create new sequences
  - `test_multiple_existing_sequences_with_internal_gaps()` - Tests splitting sequences with internal large gaps
  - `test_small_vs_large_gap_handling()` - Tests small gaps interpolated, large gaps split sequences

- **TestSequenceDetection Class** - Edge cases for sequence detection
  - `test_large_gap_creates_new_sequence()` - Tests glucose-only gap detection logic
  - `test_multiple_existing_sequences_with_internal_gaps()` - Tests splitting multiple sequences with gaps
  - `test_glucose_gap_with_event_bridge()` - **Critical:** Tests non-glucose events don't bridge glucose gaps

- **Full Pipeline Tests** - End-to-end processing
  - `test_full_pipeline()` - Tests interpolate → prepare → convert pipeline
  - `test_full_pipeline_with_synchronization()` - Tests interpolate → synchronize → prepare → convert

---

### test_utils.py

**Purpose:** Tests utility methods for FormatProcessor.

**Key Features:**
- Tests `split_glucose_events()` static method
- Validates glucose-only vs non-glucose event separation
- Tests data integrity and edge cases

**Test Classes:**
- **TestSplitGlucoseEvents** - Tests split_glucose_events() static method
  - `test_split_basic()` - Tests basic split functionality, verifies columns preserved
  - `test_split_glucose_only_contains_egv()` - Tests glucose DataFrame contains only EGV_READ events
  - `test_split_events_excludes_glucose()` - Tests events DataFrame excludes EGV_READ
  - `test_split_no_data_loss()` - Tests total rows equals original (no data loss)
  - `test_split_empty_events()` - Tests split when DataFrame has no non-glucose events
  - `test_split_empty_glucose()` - Tests split when DataFrame has no glucose events
  - `test_split_with_imputation_events()` - Tests split correctly handles EGV_READ and other events

---

### test_idempotency.py

**Purpose:** Tests idempotency and commutativity of processing operations.

**Key Features:**
- Parametrized tests run on all supported CSV files
- Tests multiple applications of operations produce same result
- Tests different operation orders produce same final result
- Validates strict output constraints (no large gaps, no NULL glucose, sharp timestamps)

**Test Classes:**
- **TestProcessingIdempotency** - Tests idempotent and commutative operations
  - `test_triple_sync_idempotency()` - Tests sync → sync → sync is idempotent
  - `test_triple_interpolate_idempotency()` - Tests interpolate → interpolate → interpolate is idempotent
  - `test_interpolate_sync_interpolate_idempotency()` - Tests interpolate → sync → interpolate
  - `test_sync_interpolate_sync_idempotency()` - Tests sync → interpolate → sync
  - `test_processing_commutativity()` - **Critical:** Tests both chains (interpolate→sync→interpolate and sync→interpolate→sync) produce identical results
  - `test_triple_sequence_detection_idempotency()` - Tests detect_and_assign_sequences is idempotent

**Helper Functions:**
- `check_no_large_gaps()` - Verifies no gaps > TOLERANCE_INTERVAL_MINUTES within sequences (glucose-only by default)
- `check_no_null_glucose_egv()` - Verifies no NULL glucose values for EGV events
- `check_seconds_are_zero()` - Verifies all datetime values have 00 seconds (sharp timestamps)

---

### test_roundtrip_datetime.py

**Purpose:** Tests datetime type preservation through roundtrip conversions.

**Key Features:**
- Tests roundtrip: vendor CSV → DataFrame → CSV → DataFrame
- Tests both `datetime` and `original_datetime` columns
- Tests Polars and Pandas DataFrame equality after roundtrip
- Parametrized tests run on all real data files

**Test Classes:**
- **TestDatetimeRoundtrip** - Datetime type preservation tests
  - `test_dataframe_equality_polars()` - Tests Polars DataFrame equality after roundtrip on minimal CSV
  - `test_dataframe_equality_pandas()` - Tests Pandas DataFrame equality after roundtrip on minimal CSV
  - `test_unified_format_roundtrip_datetime_type()` - **Parametrized:** Tests both datetime columns preserve type through roundtrip
  - `test_real_file_roundtrip_datetime_type()` - **Parametrized:** Tests all real files preserve datetime types through: vendor CSV → unified → CSV → unified
  - `test_unified_format_all_datetime_formats()` - Tests various ISO 8601 datetime formats can be parsed

**Datetime Columns Tested:**
- `datetime` - Synchronized/processed timestamp
- `original_datetime` - Original timestamp from source data

---

### test_integration_pipeline.py

**Purpose:** Full end-to-end pipeline integration tests on real data.

**Key Features:**
- Parametrized tests run on each file individually (clear pass/fail per file)
- No mocking - uses actual data files from `data/` directory
- Tests complete pipeline: parse → interpolate → synchronize → prepare → convert
- Tests error handling, data consistency, and timestamp ordering

**Test Classes:**
- **TestFullPipelineIntegration** - End-to-end pipeline tests
  - `test_pipeline_single_file()` - **Parametrized:** Tests full pipeline on each file individually
    - Stage 1-3: Parse vendor format to unified
    - Stage 4: Interpolate gaps
    - Stage 5: Synchronize timestamps
    - Stage 6: Prepare for inference (quality checks)
    - Stage 7: Convert to data-only format
  
  - `test_pipeline_summary()` - Generates summary report of all files (always passes)
  
  - `test_pipeline_single_file_detailed()` - Detailed test with extensive validation
    - Validates parsed data structure and date range
    - Validates sequences and their durations
    - Validates synchronization and timestamp intervals
    - Validates inference preparation and warning flags
    - Validates glucose statistics (range 50-300 mg/dL mean)
  
  - `test_pipeline_error_handling()` - Tests error conditions
    - FileNotFoundError for invalid paths
    - Empty DataFrame handling
  
  - `test_pipeline_data_consistency()` - Tests data consistency
    - Verifies timestamps are sorted
    - Verifies each sequence has sorted timestamps
    - Verifies we have at least one sequence

**Pipeline Stages Tested:**
1. **Parse:** Vendor CSV → Unified DataFrame
2. **Interpolate:** Fill small gaps (< 19 minutes) with linear interpolation
3. **Synchronize:** Round timestamps to nearest 5-minute grid
4. **Prepare:** Select latest sequence, truncate to max duration, quality checks
5. **Convert:** Extract data-only columns, remove duplicates

---

## Test Data

All tests use real CGM data files from the `data/` directory:
- Tests automatically discover and parametrize over all CSV files
- Unsupported formats (e.g., Medtronic Guardian Connect) are skipped using `format_supported()`
- Parsed output is saved to `data/parsed/` for inspection

## Test Principles

1. **Real Data:** Tests use actual CGM data files, not mocks
2. **Parametrization:** Tests run on all files individually for clear pass/fail
3. **Idempotency:** Operations can be applied multiple times safely
4. **Commutativity:** Different operation orders produce same results
5. **Lossless:** Processing preserves all data (no unexpected data loss)
6. **Type Safety:** Datetime types preserved through all conversions
7. **Schema Compliance:** All output validates against defined schemas

All tests use relative paths and work from any directory.

