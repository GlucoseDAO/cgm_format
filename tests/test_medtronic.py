"""Integration tests for Medtronic Guardian Connect / CareLink format support.

Tests dynamically discover Medtronic files from data/ using detect_format,
then validate parsing, schema conformance, roundtrip, and pipeline processing.
"""

import pytest
from pathlib import Path
from typing import ClassVar, Dict, List, Tuple
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


def get_medtronic_files() -> List[Path]:
    """Discover all Medtronic-format CSV files in the data directory."""
    if not DATA_DIR.exists():
        return []
    csv_files = sorted(DATA_DIR.glob("*.csv"))
    csv_files = [f for f in csv_files if "parsed" not in str(f)]

    medtronic_files = []
    for f in csv_files:
        with open(f, 'rb') as fh:
            raw = fh.read()
        text = FormatParser.decode_raw_data(raw)
        if FormatParser.detect_format(text) == SupportedCGMFormat.MEDTRONIC:
            medtronic_files.append(f)
    return medtronic_files


MEDTRONIC_FILES = get_medtronic_files()

if not MEDTRONIC_FILES:
    pytest.skip("No Medtronic files found in data/", allow_module_level=True)


@pytest.fixture(scope="session")
def medtronic_cache() -> Dict[str, Tuple[bytes, str, pl.DataFrame]]:
    """Parse all Medtronic files once: returns {filename: (raw_bytes, text, unified_df)}."""
    cache = {}
    for f in MEDTRONIC_FILES:
        raw = f.read_bytes()
        text = FormatParser.decode_raw_data(raw)
        unified = FormatParser.parse_from_bytes(raw)
        cache[f.name] = (raw, text, unified)
    return cache


class TestMedtronicDetection:
    """Test format detection for Medtronic CareLink exports."""

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_detect_format_returns_medtronic(self, file_path: Path) -> None:
        with open(file_path, 'rb') as f:
            raw = f.read()
        text = FormatParser.decode_raw_data(raw)
        assert FormatParser.detect_format(text) == SupportedCGMFormat.MEDTRONIC

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_format_supported(self, file_path: Path) -> None:
        with open(file_path, 'rb') as f:
            raw = f.read()
        assert FormatParser.format_supported(raw) is True

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_parse_file(self, file_path: Path) -> None:
        df = FormatParser.parse_file(file_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


class TestMedtronicParsing:
    """Test parsing of Medtronic CareLink CSV to unified format."""

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_schema_columns(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(df.columns) == expected_columns

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_schema_dtypes(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        expected_schema = CGM_SCHEMA.get_polars_schema()
        for col_name, expected_dtype in expected_schema.items():
            actual_dtype = df[col_name].dtype
            assert actual_dtype == expected_dtype, (
                f"Column {col_name}: expected {expected_dtype}, got {actual_dtype}"
            )

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_has_glucose_rows(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        glucose_rows = df.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 0, "No glucose rows found"

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_glucose_values_in_range(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        glucose_vals = df.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )["glucose"].drop_nulls()
        assert glucose_vals.min() >= 20, f"Glucose min too low: {glucose_vals.min()}"
        assert glucose_vals.max() <= 500, f"Glucose max too high: {glucose_vals.max()}"

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_has_insulin_rows(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        insulin_rows = df.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value
        )
        assert len(insulin_rows) > 0, "No insulin rows found"
        insulin_vals = insulin_rows["insulin_fast"].drop_nulls()
        assert len(insulin_vals) > 0, "Insulin rows have no insulin_fast values"
        assert insulin_vals.min() > 0, "Insulin values should be positive"

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_has_carb_rows(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        carb_rows = df.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        assert len(carb_rows) > 0, "No carb rows found"
        carb_vals = carb_rows["carbs"].drop_nulls()
        assert len(carb_vals) > 0, "Carb rows have no carbs values"
        assert carb_vals.min() > 0, "Carb values should be positive"

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_event_types_present(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        event_types = set(df["event_type"].unique().to_list())
        assert UnifiedEventType.GLUCOSE.value in event_types
        assert UnifiedEventType.INSULIN_FAST.value in event_types
        assert UnifiedEventType.CARBOHYDRATES.value in event_types

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_quality_flags_all_good(self, file_path: Path, medtronic_cache) -> None:
        """At parse time, all rows should have quality == 0 (GOOD)."""
        _, _, df = medtronic_cache[file_path.name]
        quality_vals = df["quality"].unique().to_list()
        assert quality_vals == [0], f"Expected only quality=0 (GOOD), got {quality_vals}"

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_datetimes_not_null(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        null_count = df["datetime"].null_count()
        assert null_count == 0, f"Found {null_count} null datetime values"


class TestMedtronicConvenienceAPIs:
    """Test convenience parse methods work with Medtronic data."""

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_parse_from_bytes(self, file_path: Path, medtronic_cache) -> None:
        raw, _, _ = medtronic_cache[file_path.name]
        df = FormatParser.parse_from_bytes(raw)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_parse_from_string(self, file_path: Path, medtronic_cache) -> None:
        _, text, _ = medtronic_cache[file_path.name]
        df = FormatParser.parse_from_string(text)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


class TestMedtronicRoundtrip:
    """Test parse -> save -> re-parse roundtrip."""

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_roundtrip_csv_string(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        csv_str = FormatParser.to_csv_string(df)
        assert len(csv_str) > 0

        reparsed = FormatParser.parse_from_string(csv_str)
        assert reparsed.shape == df.shape
        assert list(reparsed.columns) == list(df.columns)

        detected = FormatParser.detect_format(csv_str)
        assert detected == SupportedCGMFormat.UNIFIED_CGM

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_roundtrip_preserves_dtypes(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        csv_str = FormatParser.to_csv_string(df)
        reparsed = FormatParser.parse_from_string(csv_str)

        for col_name in df.columns:
            assert reparsed[col_name].dtype == df[col_name].dtype, (
                f"Column {col_name}: original {df[col_name].dtype}, "
                f"roundtrip {reparsed[col_name].dtype}"
            )


class TestMedtronicPipeline:
    """Test that FormatProcessor pipeline works on Medtronic data."""

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_sequence_detection(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        processed = FormatProcessor.detect_and_assign_sequences(df)
        assert "sequence_id" in processed.columns
        unique_sequences = processed["sequence_id"].unique()
        assert len(unique_sequences) > 0

    @pytest.mark.parametrize("file_path", MEDTRONIC_FILES, ids=lambda p: p.name)
    def test_full_pipeline(self, file_path: Path, medtronic_cache) -> None:
        _, _, df = medtronic_cache[file_path.name]
        processed = FormatProcessor.detect_and_assign_sequences(df)
        processed = FormatProcessor.interpolate_gaps(processed)
        inference_df, warnings = FormatProcessor.prepare_for_inference(processed)
        assert isinstance(inference_df, pl.DataFrame)
        assert len(inference_df) > 0
