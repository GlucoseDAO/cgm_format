"""Integration tests for Nightscout format support.

Tests use Nightscout data files in data/input/. When ``NIGHTSCOUT_URL`` is set
in the project ``.env``, the ``nightscout_data_dir`` session fixture (defined
in conftest.py) downloads fresh data in both JSON and CSV formats before the
test session starts.  Existing files are reused unless ``--nightscout-redownload``
is passed.

Expected file naming convention:
- nightscout_entries.csv / nightscout_entries.json  (SGV readings)
- nightscout_treatments.csv / nightscout_treatments.json  (treatments)

Covers: CSV + JSON detection, entries-only parsing, combined entries+treatments
parsing, roundtrip, and full pipeline.
"""

import pytest
from pathlib import Path
from typing import ClassVar, List, Optional, Tuple
import polars as pl

from cgm_format import FormatParser as FormatParserPrime
from cgm_format import FormatProcessor
from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    UnknownFormatError,
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
DATA_DIR = PROJECT_ROOT / "data" / "input"


# =============================================================================
# File Discovery
# =============================================================================

def _discover_nightscout_files(data_dir: Path) -> Tuple[List[Path], List[Path]]:
    """Discover Nightscout entries and treatments files in *data_dir*.

    Uses format detection for CSV files, filename convention + content sniffing
    for JSON files.  Returns (entries_files, treatments_files).
    """
    if not data_dir.exists():
        return [], []

    entries: List[Path] = []
    treatments: List[Path] = []

    for f in sorted(data_dir.glob("*.csv")):
        if "parsed" in str(f):
            continue
        with open(f, "rb") as fh:
            raw = fh.read()
        text = FormatParserPrime.decode_raw_data(raw)
        try:
            detected = FormatParserPrime.detect_format(text)
        except UnknownFormatError:
            detected = None

        first_line = text.split("\n", 1)[0]
        if detected == SupportedCGMFormat.NIGHTSCOUT:
            if "eventType" in first_line:
                treatments.append(f)
            else:
                entries.append(f)
        elif detected is None and "eventType" in first_line and "created_at" in first_line:
            treatments.append(f)

    for f in sorted(data_dir.glob("*.json")):
        text = f.read_text(errors="replace")[:2000]
        stripped = text.strip()
        if not stripped.startswith("["):
            continue
        if '"sgv"' in stripped:
            entries.append(f)
        elif '"eventType"' in stripped:
            treatments.append(f)

    return entries, treatments


# Module-level discovery for parameterisation.  The conftest session fixture
# has not run yet at collection time, so we rely on whatever files are already
# present on disk.  When NIGHTSCOUT_URL is configured, the first test run
# will download files, and subsequent runs will pick them up here.
ENTRIES_FILES, TREATMENTS_FILES = _discover_nightscout_files(DATA_DIR)

if not ENTRIES_FILES:
    pytest.skip("No Nightscout entries files found in data/input/", allow_module_level=True)


def _find_treatments_for(entries_path: Path) -> Optional[Path]:
    """Find a treatments file matching an entries file (same extension)."""
    ext = entries_path.suffix
    matches = [t for t in TREATMENTS_FILES if t.suffix == ext]
    return matches[0] if matches else None


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session", params=ENTRIES_FILES, ids=lambda p: p.name)
def entries_path(request: pytest.FixtureRequest, nightscout_data_dir: Path) -> Path:
    return request.param


@pytest.fixture(scope="session")
def entries_text(entries_path: Path) -> str:
    return entries_path.read_text()


@pytest.fixture(scope="session")
def entries_bytes(entries_path: Path) -> bytes:
    return entries_path.read_bytes()


@pytest.fixture(scope="session")
def unified_entries(entries_text: str) -> pl.DataFrame:
    """Parse entries to unified format (glucose only)."""
    return FormatParser.parse_from_string(entries_text)


@pytest.fixture(scope="session")
def treatments_path(entries_path: Path) -> Optional[Path]:
    return _find_treatments_for(entries_path)


@pytest.fixture(scope="session")
def unified_combined(entries_text: str, treatments_path: Optional[Path]) -> Optional[pl.DataFrame]:
    """Parse entries + treatments to unified format. None if no treatments file."""
    if treatments_path is None:
        return None
    treatments_text = treatments_path.read_text()
    return FormatParser.parse_nightscout(entries_text, treatments_text)


# =============================================================================
# Download Tests
# =============================================================================

class TestNightscoutDownload:
    """Verify the download fixture produced the expected files."""

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

    def test_entries_csv_exists(self, nightscout_data_dir: Path) -> None:
        path = nightscout_data_dir / "nightscout_entries.csv"
        if not path.exists():
            pytest.skip("No CSV entries file")
        assert path.stat().st_size > 0

    def test_treatments_csv_exists(self, nightscout_data_dir: Path) -> None:
        path = nightscout_data_dir / "nightscout_treatments.csv"
        if not path.exists():
            pytest.skip("No CSV treatments file")
        assert path.stat().st_size > 0

    def test_profile_json_exists(self, nightscout_data_dir: Path) -> None:
        path = nightscout_data_dir / "nightscout_profile.json"
        if not path.exists():
            pytest.skip("No profile file")
        assert path.stat().st_size > 0


# =============================================================================
# Detection Tests
# =============================================================================

class TestNightscoutDetection:
    """Test format detection for Nightscout data."""

    def test_detect_entries(self, entries_text: str) -> None:
        detected = FormatParser.detect_format(entries_text)
        assert detected == SupportedCGMFormat.NIGHTSCOUT

    def test_format_supported_bytes(self, entries_bytes: bytes) -> None:
        assert FormatParser.format_supported(entries_bytes) is True

    def test_format_supported_string(self, entries_text: str) -> None:
        assert FormatParser.format_supported(entries_text) is True

    def test_parse_file(self, entries_path: Path) -> None:
        df = FormatParser.parse_file(entries_path)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0


# =============================================================================
# Entries-Only Parsing Tests
# =============================================================================

class TestNightscoutEntriesParsing:
    """Test parsing of Nightscout entries to unified format."""

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
# Combined Entries + Treatments Parsing Tests
# =============================================================================

class TestNightscoutCombinedParsing:
    """Test parse_nightscout() with both entries and treatments."""

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
        """Check which event types are present (informational, always passes)."""
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
# Convenience API Tests
# =============================================================================

class TestNightscoutConvenienceAPIs:
    """Test convenience parse methods work with Nightscout data."""

    def test_parse_from_bytes(self, entries_bytes: bytes) -> None:
        df = FormatParser.parse_from_bytes(entries_bytes)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_parse_from_string(self, entries_text: str) -> None:
        df = FormatParser.parse_from_string(entries_text)
        assert isinstance(df, pl.DataFrame)
        assert len(df) > 0

    def test_parse_nightscout_entries_only(self, entries_text: str) -> None:
        df = FormatParser.parse_nightscout(entries_text)
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
