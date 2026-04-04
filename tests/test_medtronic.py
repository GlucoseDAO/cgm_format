"""Integration tests for Medtronic Guardian Connect / CareLink format support.

Tests use the real data file data/medtronics_test_livia.csv which contains:
- Multiple device sections (Pump + Sensor) with repeated headers
- European decimal format (comma separator)
- "-------" placeholders in numeric columns
- Event Marker free-text for insulin and carb entries
- ~24,907 lines of real CareLink export data
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
from cgm_format.formats.medtronic import MedtronicColumn


class FormatParser(FormatParserPrime):
    """Format parser for testing with strict input+output validation."""
    validation_mode: ClassVar[ValidationMethod] = ValidationMethod.INPUT | ValidationMethod.OUTPUT


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MEDTRONIC_FILE = DATA_DIR / "medtronics_test_livia.csv"


@pytest.fixture(scope="session")
def medtronic_raw_bytes() -> bytes:
    """Read the Medtronic test file as raw bytes."""
    assert MEDTRONIC_FILE.exists(), f"Test file not found: {MEDTRONIC_FILE}"
    return MEDTRONIC_FILE.read_bytes()


@pytest.fixture(scope="session")
def medtronic_text(medtronic_raw_bytes: bytes) -> str:
    """Decode the Medtronic test file."""
    return FormatParser.decode_raw_data(medtronic_raw_bytes)


@pytest.fixture(scope="session")
def medtronic_unified(medtronic_raw_bytes: bytes) -> pl.DataFrame:
    """Parse Medtronic file to unified format."""
    return FormatParser.parse_from_bytes(medtronic_raw_bytes)


class TestMedtronicDetection:
    """Test format detection for Medtronic CareLink exports."""

    def test_detect_format_returns_medtronic(self, medtronic_text: str) -> None:
        detected = FormatParser.detect_format(medtronic_text)
        assert detected == SupportedCGMFormat.MEDTRONIC

    def test_format_supported(self, medtronic_raw_bytes: bytes) -> None:
        assert FormatParser.format_supported(medtronic_raw_bytes) is True

    def test_parse_file(self) -> None:
        df = FormatParser.parse_file(MEDTRONIC_FILE)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


class TestMedtronicParsing:
    """Test parsing of Medtronic CareLink CSV to unified format."""

    def test_schema_columns(self, medtronic_unified: pl.DataFrame) -> None:
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(medtronic_unified.columns) == expected_columns

    def test_schema_dtypes(self, medtronic_unified: pl.DataFrame) -> None:
        expected_schema = CGM_SCHEMA.get_polars_schema()
        for col_name, expected_dtype in expected_schema.items():
            actual_dtype = medtronic_unified[col_name].dtype
            assert actual_dtype == expected_dtype, (
                f"Column {col_name}: expected {expected_dtype}, got {actual_dtype}"
            )

    def test_has_glucose_rows(self, medtronic_unified: pl.DataFrame) -> None:
        glucose_rows = medtronic_unified.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 0, "No glucose rows found"

    def test_glucose_values_in_range(self, medtronic_unified: pl.DataFrame) -> None:
        glucose_rows = medtronic_unified.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        glucose_vals = glucose_rows["glucose"].drop_nulls()
        assert glucose_vals.min() >= 20, f"Glucose min too low: {glucose_vals.min()}"
        assert glucose_vals.max() <= 500, f"Glucose max too high: {glucose_vals.max()}"

    def test_has_insulin_rows(self, medtronic_unified: pl.DataFrame) -> None:
        insulin_rows = medtronic_unified.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value
        )
        assert len(insulin_rows) > 0, "No insulin rows found"
        insulin_vals = insulin_rows["insulin_fast"].drop_nulls()
        assert len(insulin_vals) > 0, "Insulin rows have no insulin_fast values"
        assert insulin_vals.min() > 0, "Insulin values should be positive"

    def test_has_carb_rows(self, medtronic_unified: pl.DataFrame) -> None:
        carb_rows = medtronic_unified.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        assert len(carb_rows) > 0, "No carb rows found"
        carb_vals = carb_rows["carbs"].drop_nulls()
        assert len(carb_vals) > 0, "Carb rows have no carbs values"
        assert carb_vals.min() > 0, "Carb values should be positive"

    def test_event_types_present(self, medtronic_unified: pl.DataFrame) -> None:
        event_types = set(medtronic_unified["event_type"].unique().to_list())
        assert UnifiedEventType.GLUCOSE.value in event_types
        assert UnifiedEventType.INSULIN_FAST.value in event_types
        assert UnifiedEventType.CARBOHYDRATES.value in event_types

    def test_quality_flags_all_good(self, medtronic_unified: pl.DataFrame) -> None:
        """At parse time, all rows should have quality == 0 (GOOD)."""
        quality_vals = medtronic_unified["quality"].unique().to_list()
        assert quality_vals == [0], f"Expected only quality=0 (GOOD), got {quality_vals}"

    def test_datetimes_not_null(self, medtronic_unified: pl.DataFrame) -> None:
        null_count = medtronic_unified["datetime"].null_count()
        assert null_count == 0, f"Found {null_count} null datetime values"

class TestMedtronicConvenienceAPIs:
    """Test convenience parse methods work with Medtronic data."""

    def test_parse_from_bytes(self, medtronic_raw_bytes: bytes) -> None:
        df = FormatParser.parse_from_bytes(medtronic_raw_bytes)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_parse_from_string(self, medtronic_text: str) -> None:
        df = FormatParser.parse_from_string(medtronic_text)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


class TestMedtronicRoundtrip:
    """Test parse -> save -> re-parse roundtrip."""

    def test_roundtrip_csv_string(self, medtronic_unified: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(medtronic_unified)
        assert len(csv_str) > 0

        reparsed = FormatParser.parse_from_string(csv_str)

        assert reparsed.shape == medtronic_unified.shape
        assert list(reparsed.columns) == list(medtronic_unified.columns)

        detected = FormatParser.detect_format(csv_str)
        assert detected == SupportedCGMFormat.UNIFIED_CGM

    def test_roundtrip_preserves_dtypes(self, medtronic_unified: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(medtronic_unified)
        reparsed = FormatParser.parse_from_string(csv_str)

        for col_name in medtronic_unified.columns:
            assert reparsed[col_name].dtype == medtronic_unified[col_name].dtype, (
                f"Column {col_name}: original {medtronic_unified[col_name].dtype}, "
                f"roundtrip {reparsed[col_name].dtype}"
            )


class TestMedtronicPipeline:
    """Test that FormatProcessor pipeline works on Medtronic data."""

    def test_sequence_detection(self, medtronic_unified: pl.DataFrame) -> None:
        processed = FormatProcessor.detect_and_assign_sequences(medtronic_unified)
        assert "sequence_id" in processed.columns
        unique_sequences = processed["sequence_id"].unique()
        assert len(unique_sequences) > 0

    def test_full_pipeline(self, medtronic_unified: pl.DataFrame) -> None:
        processed = FormatProcessor.detect_and_assign_sequences(medtronic_unified)
        processed = FormatProcessor.interpolate_gaps(processed)
        inference_df, warnings = FormatProcessor.prepare_for_inference(processed)
        assert isinstance(inference_df, pl.DataFrame)
        assert len(inference_df) > 0
