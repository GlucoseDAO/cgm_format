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
uv run pytest tests/test_format_parser.py

# Run specific test class
uv run pytest tests/test_format_parser.py::TestFormatDetection

# Run specific test
uv run pytest tests/test_format_parser.py::TestFormatDetection::test_all_files_detected
```

## Test Coverage

### test_format_parser.py

Comprehensive tests for the format_parser module:

1. **TestFormatDetection** - Tests format detection for all 37 data files
2. **TestUnifiedParsing** - Tests parsing to unified format with schema validation
3. **TestSaveToDirectory** - Tests saving parsed files and roundtrip verification
4. **TestConvenienceMethods** - Tests convenience parsing methods
5. **TestErrorHandling** - Tests error conditions
6. **TestEndToEndPipeline** - Tests complete pipeline integration

All tests use relative paths and work from any directory.

