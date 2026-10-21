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

import logging
import re
import pytest
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, ClassVar, Optional
import polars as pl

from cgm_format import FormatParser as FormatParserPrime
from cgm_format import FormatProcessor
from cgm_format import NightscoutDateOrder
from cgm_format.interface.cgm_interface import (
    MalformedDataError,
    SupportedCGMFormat,
    ValidationMethod,
    ZeroValidInputError,
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


# =============================================================================
# Heterogeneous treatments JSON — S2 (docs/FEEDBACK.md)
# =============================================================================

class TestTreatmentsJsonSchema:
    """Treatments JSON must not depend on polars' 100-row dtype inference window.

    Nightscout treatments are heterogeneous by design, so a column is routinely
    null for a long opening run. Inference typed such a column ``Null`` and the
    first real value raised ``ComputeError``, which took down every Nightscout
    URL import on 0.12.0. These build the payload in the test rather than reading
    a fixture, because the defect is a property of the *record ordering* and no
    committed fixture pins that.
    """

    NULL_RUN: ClassVar[int] = 105  # > polars' infer_schema_length default of 100

    @staticmethod
    def _entries(count: int = 5) -> list[dict]:
        return [
            {"type": "sgv", "dateString": f"2026-08-28T{h:02d}:00:00.000Z", "sgv": 120}
            for h in range(count)
        ]

    def _treatments(self, carbs: object, insulin: object) -> list[dict]:
        """A long carbs-free basal run, then one meal bolus carrying the values."""
        rows: list[dict] = [
            {"eventType": "Temp Basal", "created_at": "2026-08-28T00:00:00.000Z",
             "rate": 0.5, "duration": 30, "carbs": None, "insulin": None}
            for _ in range(self.NULL_RUN)
        ]
        rows.append({"eventType": "Meal Bolus", "created_at": "2026-08-28T05:00:00.000Z",
                     "carbs": carbs, "insulin": insulin})
        return rows

    def _parse(self, treatments: list[dict]) -> pl.DataFrame:
        import json
        return FormatParserPrime.parse_nightscout(
            json.dumps(self._entries()), json.dumps(treatments)
        )

    def test_null_run_longer_than_inference_window_parses(self) -> None:
        """The reported failure: 105 null carbs, then a real one."""
        df = self._parse(self._treatments(carbs=47, insulin=2.5))

        # The carb entry must survive, with the value the payload carried.
        carbs = df.filter(pl.col("carbs").is_not_null())["carbs"].to_list()
        assert carbs == [47.0], f"expected the meal's 47 g to survive, got {carbs}"

        # And the basal run must not have been dropped to make that work.
        assert df.filter(pl.col("insulin_slow").is_not_null().and_(pl.col("insulin_slow") > 0)).height \
            == self.NULL_RUN

    def test_numeric_strings_are_kept_not_silently_dropped(self) -> None:
        """Some uploaders send numbers as strings; those events are real events.

        Inference types such a column String, after which the downstream
        ``cast(Float64, strict=False)`` nulls it and the event vanishes with no
        warning — a silent loss, which §5 forbids.
        """
        df = self._parse(self._treatments(carbs="47", insulin="2.5"))
        assert df.filter(pl.col("carbs").is_not_null())["carbs"].to_list() == [47.0]
        assert df.filter(pl.col("insulin_fast").is_not_null())["insulin_fast"].to_list() == [2.5]

    def test_booleans_do_not_become_doses(self) -> None:
        """bool is an int subclass in Python; coercing it would fabricate a dose."""
        df = self._parse(self._treatments(carbs=True, insulin=True))
        assert df.filter(pl.col("carbs").is_not_null()).height == 0
        assert df.filter(pl.col("insulin_fast").is_not_null()).height == 0

    def test_empty_treatments_still_carry_the_declared_columns(self) -> None:
        from cgm_format.formats.nightscout import NIGHTSCOUT_TREATMENTS_JSON_DTYPES
        empty = FormatParserPrime._parse_nightscout_treatments_json("[]")
        assert set(empty.columns) == set(NIGHTSCOUT_TREATMENTS_JSON_DTYPES)


class TestDownloaderRedirects:
    """download_nightscout must follow redirects — S3 (docs/FEEDBACK.md).

    httpx defaults ``follow_redirects`` to False (unlike requests), so an
    instance that redirects — http to https, apex to www, a PaaS hostname to its
    canonical domain — failed at ``raise_for_status`` instead of working.

    This captures the client's construction kwargs rather than mocking the parse:
    the transformation under test is untouched, and the boundary being pinned is
    exactly the argument that was missing.
    """

    def test_client_is_constructed_with_follow_redirects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        httpx = pytest.importorskip("httpx", reason="httpx is an optional (cli/dev) dependency")
        from cgm_format import nightscout_downloader

        captured: dict[str, object] = {}

        def spy(*args: object, **kwargs: object) -> object:
            captured.update(kwargs)
            raise RuntimeError("stop before any network call")

        # httpx is imported inside download_nightscout (the guarded optional-dependency
        # pattern), so the name to patch lives on httpx itself, not on the downloader.
        monkeypatch.setattr(httpx, "Client", spy)

        with pytest.raises(RuntimeError, match="stop before any network call"):
            nightscout_downloader.download_nightscout(
                "https://example.invalid", output_dir=Path("/tmp"), days=1
            )

        assert captured.get("follow_redirects") is True, (
            "download_nightscout must pass follow_redirects=True; httpx defaults it to False, "
            f"so a redirecting instance fails. Client kwargs were: {sorted(captured)}"
        )


class TestJsonRouting:
    """A JSON payload must reach the JSON parser — S4 (docs/FEEDBACK.md).

    Routing used to turn on whether ``"sgv"`` appeared in the first 2000
    characters, so the payloads most in need of explanation — an empty account,
    an auth failure — fell through to the CSV parser and were reported as a
    missing ``'Glucose (mg/dL)'`` column.
    """

    def test_empty_account_reaches_the_zero_input_guard(self) -> None:
        """`[]` is what a Nightscout account with no data returns, with a 200."""
        with pytest.raises(ZeroValidInputError, match="empty"):
            FormatParserPrime.parse_nightscout("[]")

    def test_auth_failure_surfaces_the_server_message(self) -> None:
        with pytest.raises(MalformedDataError, match="Unauthorized") as excinfo:
            FormatParserPrime.parse_nightscout('{"status":401,"message":"Unauthorized"}')
        # The old failure named a CSV column instead of the server's reason.
        assert "Glucose (mg/dL)" not in str(excinfo.value)

    def test_entries_without_sgv_in_the_first_records_still_parse(self) -> None:
        """`cal`/`mbg` records ahead of the first sgv used to misroute the array."""
        import json
        records = [{"type": "cal", "dateString": "2026-08-28T00:00:00.000Z", "slope": 1.0}
                   for _ in range(30)]
        records += [{"type": "sgv", "dateString": f"2026-08-28T{h:02d}:10:00.000Z", "sgv": 100 + h}
                    for h in range(6)]
        expected = sorted(float(r["sgv"]) for r in records if r["type"] == "sgv")

        df = FormatParserPrime.parse_nightscout(json.dumps(records))
        assert sorted(df.filter(pl.col("glucose").is_not_null())["glucose"].to_list()) == expected

    def test_exporter_csv_is_still_routed_to_the_csv_parser(self) -> None:
        """The JSON branch must not swallow the CSV path it shares an entry point with."""
        exporter_csv = DATA_DIR / "nightscout-data-with-treatments.csv"
        if not exporter_csv.exists():
            pytest.skip(f"{exporter_csv.name} not present")
        assert FormatParserPrime._is_nightscout_entries_json(exporter_csv.read_text()) is False

    def test_a_bare_number_is_not_mistaken_for_json(self) -> None:
        """json.loads("123") succeeds, so the opener check is what keeps CSV out."""
        assert FormatParserPrime._is_nightscout_entries_json("123") is False
        assert FormatParserPrime._is_nightscout_entries_json("Glucose (mg/dL)\n120") is False


class TestNightscoutSafetyAndTolerance:
    """S5 items (docs/FEEDBACK.md): what the download leaves behind, and what it drops."""

    def test_read_token_is_not_in_the_error_message(self) -> None:
        """httpx puts the request URL in HTTPStatusError, and the token rides the URL."""
        httpx = pytest.importorskip("httpx", reason="httpx is an optional (cli/dev) dependency")
        from cgm_format.nightscout_downloader import _raise_for_status_without_token

        secret = "SUPERSECRET-abc123"
        request = httpx.Request(
            "GET", f"https://ns.example.invalid/api/v1/entries.json?count=10&token={secret}"
        )
        response = httpx.Response(401, request=request, text="Unauthorized")

        with pytest.raises(httpx.HTTPStatusError) as excinfo:
            _raise_for_status_without_token(response)

        assert secret not in str(excinfo.value), "the read token leaked into the exception message"
        assert "<redacted>" in str(excinfo.value)

    def test_healthy_response_passes_through(self) -> None:
        httpx = pytest.importorskip("httpx", reason="httpx is an optional (cli/dev) dependency")
        from cgm_format.nightscout_downloader import _raise_for_status_without_token
        request = httpx.Request("GET", "https://ns.example.invalid/api/v1/entries.json")
        _raise_for_status_without_token(httpx.Response(200, request=request, text="[]"))

    def test_records_without_a_type_field_are_kept(self) -> None:
        """Some xDrip+/Loop uploaders omit `type`; every row used to be skipped."""
        import json
        records = [{"dateString": f"2026-08-28T{h:02d}:00:00.000Z", "sgv": 100 + h}
                   for h in range(6)]
        expected = sorted(float(r["sgv"]) for r in records)

        df = FormatParserPrime.parse_nightscout(json.dumps(records))
        assert sorted(df["glucose"].drop_nulls().to_list()) == expected

    def test_sgv_zero_is_not_treated_as_absent(self) -> None:
        """0 is a sensor-error value, not a missing reading; `or` dropped it silently."""
        import json
        records = [{"type": "sgv", "dateString": "2026-08-28T00:00:00.000Z", "sgv": 0},
                   {"type": "sgv", "dateString": "2026-08-28T00:05:00.000Z", "sgv": 120}]

        df = FormatParserPrime.parse_nightscout(json.dumps(records))
        assert sorted(df["glucose"].drop_nulls().to_list()) == [0.0, 120.0]

    def test_raw_download_is_not_left_behind(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """Without output_dir the raw JSON is a patient's record and must not survive."""
        import json
        seen: dict[str, Path] = {}

        def fake_download(base_url: str, output_dir: Path, **kwargs: object) -> tuple[Path, Path, Path]:
            seen["dir"] = Path(output_dir)
            entries = Path(output_dir) / "nightscout_entries.json"
            entries.write_text(json.dumps(
                [{"type": "sgv", "dateString": f"2026-08-28T{h:02d}:00:00.000Z", "sgv": 110}
                 for h in range(6)]))
            treatments = Path(output_dir) / "nightscout_treatments.json"
            treatments.write_text("[]")
            profile = Path(output_dir) / "nightscout_profile.json"
            profile.write_text("{}")
            return entries, treatments, profile

        monkeypatch.setattr(
            "cgm_format.nightscout_downloader.download_nightscout", fake_download
        )
        df = FormatParserPrime.from_nightscout_url("https://ns.example.invalid")

        assert len(df) > 0, "the parse must still work"
        assert not seen["dir"].exists(), f"raw patient JSON left behind in {seen['dir']}"

    def test_output_dir_is_still_honoured(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A caller who asks to keep the files must still get them."""
        import json

        def fake_download(base_url: str, output_dir: Path, **kwargs: object) -> tuple[Path, Path, Path]:
            entries = Path(output_dir) / "nightscout_entries.json"
            entries.write_text(json.dumps(
                [{"type": "sgv", "dateString": "2026-08-28T00:00:00.000Z", "sgv": 110}]))
            treatments = Path(output_dir) / "nightscout_treatments.json"
            treatments.write_text("[]")
            profile = Path(output_dir) / "nightscout_profile.json"
            profile.write_text("{}")
            return entries, treatments, profile

        monkeypatch.setattr(
            "cgm_format.nightscout_downloader.download_nightscout", fake_download
        )
        FormatParserPrime.from_nightscout_url("https://ns.example.invalid", output_dir=tmp_path)
        assert (tmp_path / "nightscout_entries.json").exists()


# =============================================================================
# Exporter CSV dates written in a non-US locale (S6)
# =============================================================================

EXPORTER_CSV = DATA_DIR / "nightscout-data-with-treatments.csv"

# The committed exporter fixture was written by a US browser: "4/7/2026","7:59:42 PM".
_US_ROW = re.compile(r'^"(\d{1,2}/\d{1,2}/\d{4})","(\d{1,2}:\d{2}:\d{2} [AP]M)",(.*)$')
_US_FORMAT = "%m/%d/%Y %I:%M:%S %p"


def _en_gb(moment: datetime) -> tuple[str, str]:
    """What toLocaleDateString()/toLocaleTimeString() write in an en-GB browser: 07/04/2026, 19:59:42."""
    return f"{moment:%d/%m/%Y}", f"{moment:%H:%M:%S}"


def _en_au(moment: datetime) -> tuple[str, str]:
    """What they write in an en-AU browser: 7/4/2026, 7:59:42 pm — day first, 12-hour clock."""
    hour = moment.hour % 12 or 12
    return f"{moment.day}/{moment.month}/{moment.year}", f"{hour}:{moment:%M:%S} {'pm' if moment.hour >= 12 else 'am'}"


def _relocale(text: str, render: Callable[[datetime], tuple[str, str]],
              shift: timedelta = timedelta(0)) -> tuple[str, set[datetime]]:
    """Rewrite every dated row of the US fixture in another locale, optionally moved by SHIFT.

    Returns the new CSV and the moments its CGM entries carry, read off the fixture with the
    standard library rather than through the parser under test.
    """
    out: list[str] = []
    entries: set[datetime] = set()
    in_entries = False
    for line in text.splitlines():
        if line.startswith("#"):
            in_entries = "CGM ENTRIES" in line
        m = _US_ROW.match(line)
        if m is None:
            out.append(line)
            continue
        moment = datetime.strptime(f"{m[1]} {m[2]}", _US_FORMAT) + shift
        if in_entries:
            entries.add(moment)
        day, clock = render(moment)
        out.append(f'"{day}","{clock}",{m[3]}')
    return "\n".join(out) + "\n", entries


@pytest.fixture(scope="module")
def exporter_csv_text() -> str:
    if not EXPORTER_CSV.exists():
        pytest.skip(f"{EXPORTER_CSV.name} not present")
    return EXPORTER_CSV.read_text()


class TestExporterLocaleDates:
    """S6 (docs/FEEDBACK.md): the exporter writes dates in the browser's locale.

    The date probe used to keep the first format that did not raise, and with ``strict=False`` none
    ever raises, so US month-first won every file: day-first dates past the 12th became null and
    were dropped, and those up to the 12th were read with day and month swapped.
    """

    @staticmethod
    def _glucose_moments(df: pl.DataFrame) -> set[datetime]:
        rows = df.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
        return set(rows["datetime"].to_list())

    def test_en_gb_export_parses_like_the_us_one(self, exporter_csv_text: str) -> None:
        """24-hour day-first: the shape en-GB, en-IE and fr-FR browsers write."""
        gb_text, expected = _relocale(exporter_csv_text, _en_gb)

        gb = FormatParser.parse_from_string(gb_text)
        us = FormatParser.parse_from_string(exporter_csv_text)

        assert self._glucose_moments(gb) == expected
        assert gb.equals(us), "the same readings written in another locale must give the same frame"

    def test_day_first_12_hour_export_past_the_12th(self, exporter_csv_text: str) -> None:
        """en-AU, en-NZ, en-IN: day first on a 12-hour clock, every day unambiguous."""
        au_text, expected = _relocale(exporter_csv_text, _en_au, shift=timedelta(days=10))

        au = FormatParser.parse_from_string(au_text)
        us = FormatParser.parse_from_string(exporter_csv_text)

        assert self._glucose_moments(au) == expected
        assert au["datetime"].null_count() == 0, "a treatment row came through with no time"
        assert au.height == us.height

    def test_ambiguous_day_first_export_is_read_month_first_and_says_so(
        self, exporter_csv_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Every day is 12 or under, so the text alone cannot tell 7 April from 4 July."""
        au_text, written = _relocale(exporter_csv_text, _en_au)
        assert all(m.day <= 12 for m in written), "the fixture must stay ambiguous for this test"

        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            au = FormatParser.parse_from_string(au_text)

        month_first = {datetime(m.year, m.day, m.month, m.hour, m.minute, m.second) for m in written}
        assert self._glucose_moments(au) == month_first
        ambiguity = [r for r in caplog.records if "ambiguous" in r.getMessage()]
        assert len(ambiguity) == 1, [r.getMessage() for r in caplog.records]
        assert "month first" in ambiguity[0].getMessage()

    def test_an_unreadable_date_is_counted_not_kept_as_null(
        self, exporter_csv_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A row no format reads is dropped with a count, never emitted with a null time."""
        broken = exporter_csv_text.replace('"4/7/2026","7:20:22 PM","SMB"', '"31/31/2026","7:20:22 PM","SMB"', 1)
        assert broken != exporter_csv_text, "the fixture row this test breaks has moved"

        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            df = FormatParser.parse_from_string(broken)

        assert df["datetime"].null_count() == 0
        dropped = [r for r in caplog.records if "no format" in r.getMessage()]
        assert len(dropped) == 1, [r.getMessage() for r in caplog.records]
        assert "1 of" in dropped[0].getMessage()

    def test_us_export_is_read_as_before(
        self, exporter_csv_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The fixture spans 6 and 7 April with 12-hour times, so it is ambiguous too — and is read
        month first, as before, with the one warning saying so; nothing else is reported."""
        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            us = FormatParser.parse_from_string(exporter_csv_text)

        assert {d for d in us["datetime"].dt.date().unique().to_list()} == {date(2026, 4, 6), date(2026, 4, 7)}
        assert [r for r in caplog.records if "no format" in r.getMessage()] == []


def _iso(moment: datetime) -> tuple[str, str]:
    """Year first, 24-hour: the one shape no stated order makes ambiguous."""
    return f"{moment:%Y-%m-%d}", f"{moment:%H:%M:%S}"


class TestExporterDateOrder:
    """RM19: a caller who knows the export's locale can say the date order outright.

    0.12.3 infers the order from the file; a file on a 12-hour clock with no day past the 12th
    reads completely both ways and is read month first with a warning. ``date_order`` settles it.
    """

    @staticmethod
    def _glucose_moments(df: pl.DataFrame) -> set[datetime]:
        rows = df.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
        return set(rows["datetime"].to_list())

    @staticmethod
    def _ambiguity(caplog: pytest.LogCaptureFixture) -> list[str]:
        return [r.getMessage() for r in caplog.records if "ambiguous" in r.getMessage()]

    def test_day_first_settles_the_ambiguous_export(
        self, exporter_csv_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        au_text, written = _relocale(exporter_csv_text, _en_au)
        assert all(m.day <= 12 for m in written), "the fixture must stay ambiguous for this test"

        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            au = FormatParser.parse_nightscout(au_text, date_order=NightscoutDateOrder.DAY_FIRST)

        assert self._glucose_moments(au) == written
        assert au["datetime"].null_count() == 0
        assert self._ambiguity(caplog) == []

    def test_month_first_stated_reads_as_inferred_without_the_warning(
        self, exporter_csv_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        inferred = FormatParser.parse_from_string(exporter_csv_text)
        caplog.clear()

        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            stated = FormatParser.parse_nightscout(exporter_csv_text, date_order=NightscoutDateOrder.MONTH_FIRST)

        assert stated.equals(inferred)
        assert self._ambiguity(caplog) == []

    def test_no_order_keeps_the_inference_and_its_warning(
        self, exporter_csv_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The combined exporter file, sections and all, now goes through parse_nightscout too."""
        au_text, _ = _relocale(exporter_csv_text, _en_au)

        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            direct = FormatParser.parse_nightscout(au_text)
            detected = FormatParser.parse_from_string(au_text)

        assert direct.equals(detected)
        assert len(self._ambiguity(caplog)) == 2, "one warning per parse"

    def test_a_wrong_stated_order_is_refused_by_name(self, exporter_csv_text: str) -> None:
        """A day-first export past the 12th read as month first reads no row at all."""
        gb_text, _ = _relocale(exporter_csv_text, _en_gb, shift=timedelta(days=10))

        with pytest.raises(MalformedDataError, match="MONTH_FIRST|month_first"):
            FormatParser.parse_nightscout(gb_text, date_order=NightscoutDateOrder.MONTH_FIRST)

    def test_year_first_dates_read_under_either_order(self, exporter_csv_text: str) -> None:
        iso_text, written = _relocale(exporter_csv_text, _iso)

        for order in NightscoutDateOrder:
            assert self._glucose_moments(FormatParser.parse_nightscout(iso_text, date_order=order)) == written

    def test_from_nightscout_exports_takes_the_order(self, exporter_csv_text: str, tmp_path: Path) -> None:
        au_text, written = _relocale(exporter_csv_text, _en_au)
        path = tmp_path / "export.csv"
        path.write_text(au_text)

        df = FormatParser.from_nightscout_exports(path, date_order=NightscoutDateOrder.DAY_FIRST)

        assert self._glucose_moments(df) == written
        assert df.equals(FormatParser.parse_nightscout(au_text, date_order=NightscoutDateOrder.DAY_FIRST))

    def test_treatments_in_the_file_and_beside_it_is_refused(
        self, exporter_csv_text: str, treatments_json_text: Optional[str]
    ) -> None:
        if treatments_json_text is None:
            pytest.skip("no treatments JSON")
        assert "# TREATMENTS" in exporter_csv_text, "the fixture lost its treatments section"

        with pytest.raises(MalformedDataError, match="TREATMENTS"):
            FormatParser.parse_nightscout(exporter_csv_text, treatments_json_text)

    def test_an_order_given_for_json_says_it_is_unused(
        self, entries_json_text: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            stated = FormatParser.parse_nightscout(entries_json_text, date_order=NightscoutDateOrder.DAY_FIRST)

        assert stated.equals(FormatParser.parse_nightscout(entries_json_text))
        assert [r for r in caplog.records if "date_order" in r.getMessage()], "the ignored order went unsaid"
