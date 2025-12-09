# Tests

This directory contains pytest tests for the cgm_format package.

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
```

## Test Coverage

### test_format_detection_validation.py

Format detection and Frictionless schema validation for all CGM formats:

- **TestFormatDetection** - Format detection for all CSV files, verifies correct format identification
- **TestSchemaValidation** - Validates CSV files against Frictionless schemas with known vendor issues suppressed
- **TestSchemaDefinitions** - Validates schema structure and Frictionless conversion

### test_integration_pipeline.py

Full end-to-end pipeline integration tests on real data:

- **TestFullPipelineIntegration** - Complete pipeline: parse → interpolate → synchronize → prepare → convert (parametrized per file)
- Tests error handling, data consistency, and timestamp ordering
- No mocking - uses actual data files

### test_format_processor.py

Comprehensive FormatProcessor implementation tests:

- Processor initialization and constant validation
- **synchronize_timestamps()** - Timestamp rounding, glucose interpolation, discrete event handling
- **interpolate_gaps()** - Gap interpolation with/without snap_to_grid, various gap sizes
- **prepare_for_inference()** - Latest sequence selection, truncation, quality checks
- **mark_calibration_periods()** - Calibration gap detection and quality marking
- Full pipeline integration with synchronization
- Validation on all real datasets (lossless test)

### test_format_converter.py (test_format_parser.py)

Format parser and converter tests:

- **TestFormatDetection** - Format detection for all data files
- **TestUnifiedParsing** - Parsing to unified format with schema validation
- **TestSaveToDirectory** - Saving parsed files and roundtrip verification
- **TestConvenienceMethods** - Convenience parsing methods (parse_file, parse_base64)
- **TestInputHelpers** - Various input sources (file, bytes, string, base64)
- **TestSequenceDetection** - Large gap detection, sequence splitting, non-glucose event bridge tests
- **TestErrorHandling** - Error conditions and edge cases
- **TestEndToEndPipeline** - Complete pipeline integration

### test_roundtrip_datetime.py

Datetime type preservation through roundtrip conversions:

- **TestDatetimeRoundtrip** - Polars and Pandas DataFrame equality after roundtrip
- Tests both `datetime` and `original_datetime` columns
- Vendor file roundtrip tests (Dexcom/Libre → unified → CSV → unified)
- Various ISO 8601 datetime format parsing

### test_utils.py

Utility methods for FormatProcessor:

- **TestSplitGlucoseEvents** - split_glucose_events() static method
- Tests basic split, data integrity, empty cases, imputation events
- Verifies glucose-only vs events separation

### test_idempotency.py

Idempotency and commutativity of processing operations:

- **TestProcessingIdempotency** - Triple application tests (sync→sync→sync, interpolate→interpolate→interpolate)
- Cross-operation idempotency (interpolate→sync→interpolate, sync→interpolate→sync)
- **test_processing_commutativity** - Verifies both operation chains produce identical results
- Validates strict constraints: no large gaps, no NULL glucose for EGV, sharp timestamps

### test_schema.py

Schema definition and Frictionless validation tests:

- **TestEnumLiteral** - EnumLiteral serialization and behavior
- **TestFrictionlessSchemaGeneration** - Schema generation with dialects, primary keys
- **TestDexcomColumnOrder** - Column order matches actual CSV structure (critical for parsing)
- **TestLibreColumnOrder** - Libre schema column order validation
- **TestSchemaValidation** - Path handling, enum constraints
- **TestSchemaConsistency** - Column types, counts, no duplicates
- **TestRegressionPrevention** - Prevents specific bugs (enum in names, primary key mismatches)

All tests use relative paths and work from any directory.

