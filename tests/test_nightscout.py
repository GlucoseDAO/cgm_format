"""Integration tests for Nightscout format support.

Tests use real data files in data/:
- nightscout_entries.csv   (~628 rows of SGV readings)
- nightscout_entries.json  (same data in native JSON)
- nightscout_treatments.csv  (~848 rows of treatments)
- nightscout_treatments.json (same data in native JSON)

Covers: CSV + JSON detection, entries-only parsing, combined entries+treatments
parsing, roundtrip, and full pipeline.
"""

import pytest
from pathlib import Path
from typing import ClassVar
import polars as pl

from cgm_format import FormatParser as FormatParserPrime
from cgm_format import FormatProcessor
from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    ValidationMethod,
)
from cgm_format.formats.unified import (
    CGM_SCHEMA,
    UnifiedEventType,
    Quality,
)


class FormatParser(FormatParserPrime):
    """Format parser with strict input+output validation for tests."""
    validation_mode: ClassVar[ValidationMethod] = ValidationMethod.INPUT | ValidationMethod.OUTPUT


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
ENTRIES_CSV = DATA_DIR / "nightscout_entries.csv"
ENTRIES_JSON = DATA_DIR / "nightscout_entries.json"
TREATMENTS_CSV = DATA_DIR / "nightscout_treatments.csv"
TREATMENTS_JSON = DATA_DIR / "nightscout_treatments.json"


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def entries_csv_text() -> str:
    assert ENTRIES_CSV.exists(), f"Test file not found: {ENTRIES_CSV}"
    return ENTRIES_CSV.read_text()


@pytest.fixture(scope="session")
def entries_json_text() -> str:
    assert ENTRIES_JSON.exists(), f"Test file not found: {ENTRIES_JSON}"
    return ENTRIES_JSON.read_text()


@pytest.fixture(scope="session")
def treatments_csv_text() -> str:
    assert TREATMENTS_CSV.exists(), f"Test file not found: {TREATMENTS_CSV}"
    return TREATMENTS_CSV.read_text()


@pytest.fixture(scope="session")
def treatments_json_text() -> str:
    assert TREATMENTS_JSON.exists(), f"Test file not found: {TREATMENTS_JSON}"
    return TREATMENTS_JSON.read_text()


@pytest.fixture(scope="session")
def entries_csv_bytes() -> bytes:
    assert ENTRIES_CSV.exists(), f"Test file not found: {ENTRIES_CSV}"
    return ENTRIES_CSV.read_bytes()


@pytest.fixture(scope="session")
def unified_entries_csv(entries_csv_text: str) -> pl.DataFrame:
    """Parse entries CSV to unified format (glucose only)."""
    return FormatParser.parse_from_string(entries_csv_text)


@pytest.fixture(scope="session")
def unified_entries_json(entries_json_text: str) -> pl.DataFrame:
    """Parse entries JSON to unified format (glucose only)."""
    return FormatParser.parse_from_string(entries_json_text)


@pytest.fixture(scope="session")
def unified_combined_csv(entries_csv_text: str, treatments_csv_text: str) -> pl.DataFrame:
    """Parse entries + treatments CSV to unified format."""
    return FormatParser.parse_nightscout(entries_csv_text, treatments_csv_text)


@pytest.fixture(scope="session")
def unified_combined_json(entries_json_text: str, treatments_json_text: str) -> pl.DataFrame:
    """Parse entries + treatments JSON to unified format."""
    return FormatParser.parse_nightscout(entries_json_text, treatments_json_text)


# =============================================================================
# Detection Tests
# =============================================================================

class TestNightscoutDetection:
    """Test format detection for Nightscout data."""

    def test_detect_entries_csv(self, entries_csv_text: str) -> None:
        detected = FormatParser.detect_format(entries_csv_text)
        assert detected == SupportedCGMFormat.NIGHTSCOUT

    def test_detect_entries_json(self, entries_json_text: str) -> None:
        detected = FormatParser.detect_format(entries_json_text)
        assert detected == SupportedCGMFormat.NIGHTSCOUT

    def test_format_supported_bytes(self, entries_csv_bytes: bytes) -> None:
        assert FormatParser.format_supported(entries_csv_bytes) is True

    def test_format_supported_string(self, entries_csv_text: str) -> None:
        assert FormatParser.format_supported(entries_csv_text) is True

    def test_parse_file(self) -> None:
        df = FormatParser.parse_file(ENTRIES_CSV)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


# =============================================================================
# Entries-Only Parsing Tests
# =============================================================================

class TestNightscoutEntriesParsing:
    """Test parsing of Nightscout entries to unified format."""

    def test_schema_columns(self, unified_entries_csv: pl.DataFrame) -> None:
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(unified_entries_csv.columns) == expected_columns

    def test_schema_dtypes(self, unified_entries_csv: pl.DataFrame) -> None:
        expected_schema = CGM_SCHEMA.get_polars_schema()
        for col_name, expected_dtype in expected_schema.items():
            actual_dtype = unified_entries_csv[col_name].dtype
            assert actual_dtype == expected_dtype, (
                f"Column {col_name}: expected {expected_dtype}, got {actual_dtype}"
            )

    def test_has_glucose_rows(self, unified_entries_csv: pl.DataFrame) -> None:
        glucose_rows = unified_entries_csv.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 100, f"Expected >100 glucose rows, got {len(glucose_rows)}"

    def test_glucose_values_in_range(self, unified_entries_csv: pl.DataFrame) -> None:
        glucose_vals = unified_entries_csv["glucose"].drop_nulls()
        assert glucose_vals.min() >= 20, f"Glucose min too low: {glucose_vals.min()}"
        assert glucose_vals.max() <= 500, f"Glucose max too high: {glucose_vals.max()}"

    def test_datetimes_not_null(self, unified_entries_csv: pl.DataFrame) -> None:
        null_count = unified_entries_csv["datetime"].null_count()
        assert null_count == 0, f"Found {null_count} null datetime values"

    def test_quality_flags_all_good(self, unified_entries_csv: pl.DataFrame) -> None:
        quality_vals = unified_entries_csv["quality"].unique().to_list()
        assert quality_vals == [0], f"Expected only quality=0 (GOOD), got {quality_vals}"

    def test_entries_only_has_glucose_events(self, unified_entries_csv: pl.DataFrame) -> None:
        event_types = set(unified_entries_csv["event_type"].unique().to_list())
        assert event_types == {UnifiedEventType.GLUCOSE.value}


# =============================================================================
# JSON Entries Parsing Tests
# =============================================================================

class TestNightscoutJsonEntriesParsing:
    """Test parsing Nightscout JSON entries produces same results as CSV."""

    def test_json_has_glucose_rows(self, unified_entries_json: pl.DataFrame) -> None:
        glucose_rows = unified_entries_json.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 100

    def test_json_glucose_range(self, unified_entries_json: pl.DataFrame) -> None:
        glucose_vals = unified_entries_json["glucose"].drop_nulls()
        assert glucose_vals.min() >= 20
        assert glucose_vals.max() <= 500

    def test_csv_json_row_count_match(
        self, unified_entries_csv: pl.DataFrame, unified_entries_json: pl.DataFrame
    ) -> None:
        assert len(unified_entries_csv) == len(unified_entries_json), (
            f"CSV has {len(unified_entries_csv)} rows, JSON has {len(unified_entries_json)}"
        )

    def test_json_schema_columns(self, unified_entries_json: pl.DataFrame) -> None:
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(unified_entries_json.columns) == expected_columns


# =============================================================================
# Combined Entries + Treatments Parsing Tests
# =============================================================================

class TestNightscoutCombinedParsing:
    """Test parse_nightscout() with both entries and treatments."""

    def test_has_glucose_rows(self, unified_combined_csv: pl.DataFrame) -> None:
        glucose_rows = unified_combined_csv.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 100

    def test_has_insulin_fast_rows(self, unified_combined_csv: pl.DataFrame) -> None:
        insulin_rows = unified_combined_csv.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value
        )
        assert len(insulin_rows) > 0, "No insulin_fast rows found"
        insulin_vals = insulin_rows["insulin_fast"].drop_nulls()
        assert len(insulin_vals) > 0, "Insulin rows have no insulin_fast values"
        assert insulin_vals.min() > 0, "Insulin values should be positive"

    def test_has_insulin_slow_rows(self, unified_combined_csv: pl.DataFrame) -> None:
        basal_rows = unified_combined_csv.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_SLOW.value
        )
        assert len(basal_rows) > 0, "No insulin_slow (temp basal) rows found"
        basal_vals = basal_rows["insulin_slow"].drop_nulls()
        assert len(basal_vals) > 0
        assert basal_vals.min() > 0

    def test_has_carb_rows(self, unified_combined_csv: pl.DataFrame) -> None:
        carb_rows = unified_combined_csv.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        assert len(carb_rows) > 0, "No carb rows found"
        carb_vals = carb_rows["carbs"].drop_nulls()
        assert len(carb_vals) > 0
        assert carb_vals.min() > 0

    def test_all_event_types_present(self, unified_combined_csv: pl.DataFrame) -> None:
        event_types = set(unified_combined_csv["event_type"].unique().to_list())
        assert UnifiedEventType.GLUCOSE.value in event_types
        assert UnifiedEventType.INSULIN_FAST.value in event_types
        assert UnifiedEventType.CARBOHYDRATES.value in event_types

    def test_more_rows_than_entries_only(
        self, unified_entries_csv: pl.DataFrame, unified_combined_csv: pl.DataFrame
    ) -> None:
        assert len(unified_combined_csv) > len(unified_entries_csv), (
            "Combined should have more rows than entries-only"
        )

    def test_schema_columns(self, unified_combined_csv: pl.DataFrame) -> None:
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(unified_combined_csv.columns) == expected_columns


# =============================================================================
# Combined JSON Parsing Tests
# =============================================================================

class TestNightscoutCombinedJsonParsing:
    """Test parse_nightscout() with JSON inputs."""

    def test_json_combined_has_all_event_types(self, unified_combined_json: pl.DataFrame) -> None:
        event_types = set(unified_combined_json["event_type"].unique().to_list())
        assert UnifiedEventType.GLUCOSE.value in event_types
        assert UnifiedEventType.INSULIN_FAST.value in event_types
        assert UnifiedEventType.CARBOHYDRATES.value in event_types

    def test_json_csv_combined_row_counts_similar(
        self, unified_combined_csv: pl.DataFrame, unified_combined_json: pl.DataFrame
    ) -> None:
        assert len(unified_combined_json) == len(unified_combined_csv), (
            f"JSON combined: {len(unified_combined_json)}, CSV combined: {len(unified_combined_csv)}"
        )


# =============================================================================
# Convenience API Tests
# =============================================================================

class TestNightscoutConvenienceAPIs:
    """Test convenience parse methods work with Nightscout data."""

    def test_parse_from_bytes(self, entries_csv_bytes: bytes) -> None:
        df = FormatParser.parse_from_bytes(entries_csv_bytes)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_parse_from_string(self, entries_csv_text: str) -> None:
        df = FormatParser.parse_from_string(entries_csv_text)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_parse_nightscout_with_bytes(self, entries_csv_bytes: bytes) -> None:
        df = FormatParser.parse_nightscout(entries_csv_bytes)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


# =============================================================================
# Roundtrip Tests
# =============================================================================

class TestNightscoutRoundtrip:
    """Test parse -> save -> re-parse roundtrip."""

    def test_roundtrip_csv_string(self, unified_combined_csv: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(unified_combined_csv)
        assert len(csv_str) > 0

        reparsed = FormatParser.parse_from_string(csv_str)
        assert reparsed.shape == unified_combined_csv.shape
        assert list(reparsed.columns) == list(unified_combined_csv.columns)

        detected = FormatParser.detect_format(csv_str)
        assert detected == SupportedCGMFormat.UNIFIED_CGM

    def test_roundtrip_preserves_dtypes(self, unified_combined_csv: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(unified_combined_csv)
        reparsed = FormatParser.parse_from_string(csv_str)

        for col_name in unified_combined_csv.columns:
            assert reparsed[col_name].dtype == unified_combined_csv[col_name].dtype, (
                f"Column {col_name}: original {unified_combined_csv[col_name].dtype}, "
                f"roundtrip {reparsed[col_name].dtype}"
            )


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestNightscoutPipeline:
    """Test that FormatProcessor pipeline works on Nightscout data."""

    def test_sequence_detection(self, unified_combined_csv: pl.DataFrame) -> None:
        processed = FormatProcessor.detect_and_assign_sequences(unified_combined_csv)
        assert "sequence_id" in processed.columns
        unique_sequences = processed["sequence_id"].unique()
        assert len(unique_sequences) > 0

    def test_full_pipeline(self, unified_combined_csv: pl.DataFrame) -> None:
        processed = FormatProcessor.detect_and_assign_sequences(unified_combined_csv)
        processed = FormatProcessor.interpolate_gaps(processed)
        inference_df, warnings = FormatProcessor.prepare_for_inference(processed)
        assert isinstance(inference_df, pl.DataFrame)
        assert len(inference_df) > 0
