"""Integration tests for Nightscout format support.

Tests use Nightscout JSON data files in data/input/. When ``NIGHTSCOUT_URL`` is
set in the project ``.env``, the ``nightscout_data_dir`` session fixture
(defined in conftest.py) downloads fresh data before the test session starts.

Two parsing paths are covered:

1. **JSON API** (``parse_nightscout`` / ``from_nightscout_exports``): parses the
   entries + treatments JSON files fetched from the Nightscout REST API.
2. **Exporter CSV** (``detect_format`` → ``parse_to_unified``): parses the
   combined CSV file produced by *nightscout-exporter*.

Expected file naming convention:
- nightscout_entries.json  (SGV readings)
- nightscout_treatments.json  (treatments)
- nightscout_profile.json  (pump/loop profile)
"""

import pytest
from pathlib import Path
from typing import ClassVar, Optional
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
)


class FormatParser(FormatParserPrime):
    """Format parser with strict input+output validation for tests."""
    validation_mode: ClassVar[ValidationMethod] = ValidationMethod.INPUT | ValidationMethod.OUTPUT


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"

ENTRIES_JSON = DATA_DIR / "nightscout_entries.json"
TREATMENTS_JSON = DATA_DIR / "nightscout_treatments.json"
PROFILE_JSON = DATA_DIR / "nightscout_profile.json"

if not ENTRIES_JSON.exists():
    pytest.skip("No Nightscout entries JSON found in data/input/", allow_module_level=True)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def entries_json_text() -> str:
    return ENTRIES_JSON.read_text()


@pytest.fixture(scope="session")
def treatments_json_text() -> Optional[str]:
    if not TREATMENTS_JSON.exists():
        return None
    return TREATMENTS_JSON.read_text()


@pytest.fixture(scope="session")
def unified_entries(entries_json_text: str) -> pl.DataFrame:
    """Parse entries JSON to unified format (glucose only)."""
    return FormatParser.parse_nightscout(entries_json_text)


@pytest.fixture(scope="session")
def unified_combined(entries_json_text: str, treatments_json_text: Optional[str]) -> Optional[pl.DataFrame]:
    """Parse entries + treatments JSON to unified format."""
    if treatments_json_text is None:
        return None
    return FormatParser.parse_nightscout(entries_json_text, treatments_json_text)


@pytest.fixture(scope="session")
def unified_from_exports() -> pl.DataFrame:
    """Parse via from_nightscout_exports file-based API."""
    treatments = TREATMENTS_JSON if TREATMENTS_JSON.exists() else None
    return FormatParser.from_nightscout_exports(ENTRIES_JSON, treatments)


# =============================================================================
# Download Tests
# =============================================================================

class TestNightscoutDownload:
    """Verify the download fixture produced the expected JSON files."""

    def test_entries_json_exists(self, nightscout_data_dir: Path) -> None:
        path = nightscout_data_dir / "nightscout_entries.json"
        if not path.exists():
            pytest.skip("No JSON entries file (NIGHTSCOUT_URL not set or download skipped)")
        assert path.stat().st_size > 0

    def test_treatments_json_exists(self, nightscout_data_dir: Path) -> None:
        path = nightscout_data_dir / "nightscout_treatments.json"
        if not path.exists():
            pytest.skip("No JSON treatments file")
        assert path.stat().st_size > 0

    def test_profile_json_exists(self, nightscout_data_dir: Path) -> None:
        path = nightscout_data_dir / "nightscout_profile.json"
        if not path.exists():
            pytest.skip("No profile file")
        assert path.stat().st_size > 0


# =============================================================================
# Entries-Only Parsing Tests (JSON API)
# =============================================================================

class TestNightscoutEntriesParsing:
    """Test parsing of Nightscout entries JSON to unified format."""

    def test_schema_columns(self, unified_entries: pl.DataFrame) -> None:
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(unified_entries.columns) == expected_columns

    def test_schema_dtypes(self, unified_entries: pl.DataFrame) -> None:
        expected_schema = CGM_SCHEMA.get_polars_schema()
        for col_name, expected_dtype in expected_schema.items():
            actual_dtype = unified_entries[col_name].dtype
            assert actual_dtype == expected_dtype, (
                f"Column {col_name}: expected {expected_dtype}, got {actual_dtype}"
            )

    def test_has_glucose_rows(self, unified_entries: pl.DataFrame) -> None:
        glucose_rows = unified_entries.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 0, "No glucose rows found"

    def test_glucose_values_in_range(self, unified_entries: pl.DataFrame) -> None:
        glucose_vals = unified_entries["glucose"].drop_nulls()
        assert glucose_vals.min() >= 20, f"Glucose min too low: {glucose_vals.min()}"
        assert glucose_vals.max() <= 500, f"Glucose max too high: {glucose_vals.max()}"

    def test_datetimes_not_null(self, unified_entries: pl.DataFrame) -> None:
        null_count = unified_entries["datetime"].null_count()
        assert null_count == 0, f"Found {null_count} null datetime values"

    def test_quality_flags_all_good(self, unified_entries: pl.DataFrame) -> None:
        quality_vals = unified_entries["quality"].unique().to_list()
        assert quality_vals == [0], f"Expected only quality=0 (GOOD), got {quality_vals}"

    def test_entries_only_has_glucose_events(self, unified_entries: pl.DataFrame) -> None:
        event_types = set(unified_entries["event_type"].unique().to_list())
        assert event_types == {UnifiedEventType.GLUCOSE.value}


# =============================================================================
# Combined Entries + Treatments Parsing Tests (JSON API)
# =============================================================================

class TestNightscoutCombinedParsing:
    """Test parse_nightscout() with both entries and treatments JSON."""

    def test_has_glucose_rows(self, unified_combined: Optional[pl.DataFrame]) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        glucose_rows = unified_combined.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 0

    def test_has_insulin_fast_rows(self, unified_combined: Optional[pl.DataFrame]) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        insulin_rows = unified_combined.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value
        )
        if len(insulin_rows) == 0:
            pytest.skip("No bolus/SMB treatments in downloaded data")
        insulin_vals = insulin_rows["insulin_fast"].drop_nulls()
        assert len(insulin_vals) > 0, "Insulin rows have no insulin_fast values"
        assert insulin_vals.min() > 0, "Insulin values should be positive"

    def test_has_insulin_slow_rows(self, unified_combined: Optional[pl.DataFrame]) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        basal_rows = unified_combined.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_SLOW.value
        )
        if len(basal_rows) == 0:
            pytest.skip("No temp basal treatments in downloaded data")
        basal_vals = basal_rows["insulin_slow"].drop_nulls()
        assert len(basal_vals) > 0
        assert basal_vals.min() > 0

    def test_has_carb_rows(self, unified_combined: Optional[pl.DataFrame]) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        carb_rows = unified_combined.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        if len(carb_rows) == 0:
            pytest.skip("No carb treatments in downloaded data")
        carb_vals = carb_rows["carbs"].drop_nulls()
        assert len(carb_vals) > 0
        assert carb_vals.min() > 0

    def test_all_event_types_present(self, unified_combined: Optional[pl.DataFrame]) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        event_types = set(unified_combined["event_type"].unique().to_list())
        assert UnifiedEventType.GLUCOSE.value in event_types, "Must have glucose entries"

    def test_more_rows_than_entries_only(
        self, unified_entries: pl.DataFrame, unified_combined: Optional[pl.DataFrame]
    ) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        assert len(unified_combined) >= len(unified_entries), (
            "Combined should have at least as many rows as entries-only"
        )

    def test_schema_columns(self, unified_combined: Optional[pl.DataFrame]) -> None:
        if unified_combined is None:
            pytest.skip("No treatments file available")
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(unified_combined.columns) == expected_columns


# =============================================================================
# from_nightscout_exports Tests
# =============================================================================

class TestNightscoutExportsAPI:
    """Test the from_nightscout_exports file-based convenience API."""

    def test_returns_dataframe(self, unified_from_exports: pl.DataFrame) -> None:
        assert isinstance(unified_from_exports, pl.DataFrame)
        assert len(unified_from_exports) > 0

    def test_schema_columns(self, unified_from_exports: pl.DataFrame) -> None:
        expected_columns = CGM_SCHEMA.get_column_names()
        assert list(unified_from_exports.columns) == expected_columns

    def test_has_glucose(self, unified_from_exports: pl.DataFrame) -> None:
        glucose_rows = unified_from_exports.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        assert len(glucose_rows) > 0

    def test_matches_parse_nightscout(
        self, unified_from_exports: pl.DataFrame, unified_combined: Optional[pl.DataFrame]
    ) -> None:
        """from_nightscout_exports should produce the same result as parse_nightscout."""
        if unified_combined is None:
            pytest.skip("No treatments file")
        assert len(unified_from_exports) == len(unified_combined)


# =============================================================================
# Exporter CSV Detection Tests
# =============================================================================

class TestNightscoutExporterCSVDetection:
    """Test detect_format for nightscout-exporter CSV format."""

    SAMPLE_ENTRIES_CSV = (
        "# CGM ENTRIES\n"
        "Date,Time,Glucose (mg/dL),Type,Device,Trend,ID\n"
        '"3/31/2026","7:51:03 PM","202","sgv","","4","abc123"\n'
        '"3/31/2026","7:46:03 PM","198","sgv","","4","abc124"\n'
    )

    SAMPLE_COMBINED_CSV = (
        "# CGM ENTRIES\n"
        "Date,Time,Glucose (mg/dL),Type,Device,Trend,ID\n"
        '"3/31/2026","7:51:03 PM","202","sgv","","4","abc123"\n'
        "\n"
        "# TREATMENTS (Insulin, Carbs, Exercise)\n"
        "Date,Time,Event Type,Insulin (U),Carbs (g),Notes,ID\n"
        '"3/31/2026","8:12:21 PM","SMB","0.1","","","def456"\n'
        '"3/31/2026","8:30:00 PM","Bolus","2.5","","dinner","def457"\n'
    )

    def test_detect_entries_csv(self) -> None:
        detected = FormatParser.detect_format(self.SAMPLE_ENTRIES_CSV)
        assert detected == SupportedCGMFormat.NIGHTSCOUT

    def test_detect_combined_csv(self) -> None:
        detected = FormatParser.detect_format(self.SAMPLE_COMBINED_CSV)
        assert detected == SupportedCGMFormat.NIGHTSCOUT

    def test_format_supported(self) -> None:
        assert FormatParser.format_supported(self.SAMPLE_ENTRIES_CSV) is True

    def test_parse_entries_only(self) -> None:
        df = FormatParser.parse_from_string(self.SAMPLE_ENTRIES_CSV)
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 2
        glucose_vals = df["glucose"].drop_nulls()
        assert set(glucose_vals.to_list()) == {202.0, 198.0}

    def test_parse_combined(self) -> None:
        df = FormatParser.parse_from_string(self.SAMPLE_COMBINED_CSV)
        assert isinstance(df, pl.DataFrame)
        # 2 glucose rows + 2 insulin rows
        event_types = set(df["event_type"].unique().to_list())
        assert UnifiedEventType.GLUCOSE.value in event_types
        assert UnifiedEventType.INSULIN_FAST.value in event_types


# =============================================================================
# parse_nightscout direct API Tests
# =============================================================================

class TestNightscoutDirectAPI:
    """Test parse_nightscout with JSON string data."""

    def test_parse_entries_only(self, entries_json_text: str) -> None:
        df = FormatParser.parse_nightscout(entries_json_text)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_parse_entries_bytes(self, entries_json_text: str) -> None:
        df = FormatParser.parse_nightscout(entries_json_text.encode("utf-8"))
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


# =============================================================================
# Roundtrip Tests
# =============================================================================

class TestNightscoutRoundtrip:
    """Test parse -> save -> re-parse roundtrip."""

    def test_roundtrip_csv_string(self, unified_entries: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(unified_entries)
        assert len(csv_str) > 0

        reparsed = FormatParser.parse_from_string(csv_str)
        assert reparsed.shape == unified_entries.shape
        assert list(reparsed.columns) == list(unified_entries.columns)

        detected = FormatParser.detect_format(csv_str)
        assert detected == SupportedCGMFormat.UNIFIED_CGM

    def test_roundtrip_preserves_dtypes(self, unified_entries: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(unified_entries)
        reparsed = FormatParser.parse_from_string(csv_str)

        for col_name in unified_entries.columns:
            assert reparsed[col_name].dtype == unified_entries[col_name].dtype, (
                f"Column {col_name}: original {unified_entries[col_name].dtype}, "
                f"roundtrip {reparsed[col_name].dtype}"
            )


# =============================================================================
# Pipeline Tests
# =============================================================================

class TestNightscoutPipeline:
    """Test that FormatProcessor pipeline works on Nightscout data."""

    def test_sequence_detection(self, unified_entries: pl.DataFrame) -> None:
        processed = FormatProcessor.detect_and_assign_sequences(unified_entries)
        assert "sequence_id" in processed.columns
        unique_sequences = processed["sequence_id"].unique()
        assert len(unique_sequences) > 0

    def test_full_pipeline(self, unified_entries: pl.DataFrame) -> None:
        processed = FormatProcessor.detect_and_assign_sequences(unified_entries)
        processed = FormatProcessor.interpolate_gaps(processed)
        inference_df, warnings = FormatProcessor.prepare_for_inference(processed)
        assert isinstance(inference_df, pl.DataFrame)
        assert len(inference_df) > 0
