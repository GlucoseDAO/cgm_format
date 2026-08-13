"""Tests for European / mmol/L FreeStyle Libre support.

The mmol/L fixture (`JonGrove_glucose_13-8-2026.csv`) is local-only — data/input/
is gitignored — so tests that need it skip when the file is absent. Scan-glucose
coverage on the committed `FreeStyle_Libre_3_synthetic.csv` always runs.
"""

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import ClassVar, Literal

import polars as pl
import pytest

from cgm_format import FormatParser as FormatParserPrime
from cgm_format import FormatProcessor
from cgm_format.formats.supported import FORMAT_DETECTION_PATTERNS
from cgm_format.formats.unified import CGM_SCHEMA, MMOL_TO_MGDL, Quality, UnifiedEventType
from cgm_format.interface.cgm_interface import SupportedCGMFormat, ValidationMethod
from cgm_format.formats.libre import LibreColumn, LibreRecordType
from cgm_format.formats.libre_eu import LibreEUColumn


class FormatParser(FormatParserPrime):
    """Format parser for testing with strict input+output validation."""
    validation_mode: ClassVar[ValidationMethod] = ValidationMethod.INPUT | ValidationMethod.OUTPUT


PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"

JONGROVE = DATA_DIR / "JonGrove_glucose_13-8-2026.csv"
SYNTHETIC_LIBRE = DATA_DIR / "FreeStyle_Libre_3_synthetic.csv"
SYNTHETIC_DEXCOM = DATA_DIR / "Clarity_Export_synthetic.csv"
LIBREVIEW = DATA_DIR / "Libreview2026-07-23-17_55_12.csv"

LIBRE_TIMESTAMP_FORMAT = "%d-%m-%Y %H:%M"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")


@pytest.fixture(scope="module")
def jongrove_parsed() -> pl.DataFrame:
    _skip_if_missing(JONGROVE)
    return FormatParser.parse_file(JONGROVE)


@pytest.fixture(scope="module")
def synthetic_libre_parsed() -> pl.DataFrame:
    _skip_if_missing(SYNTHETIC_LIBRE)
    return FormatParser.parse_file(SYNTHETIC_LIBRE)


def _read_libre_raw(path: Path) -> pl.DataFrame:
    """Read a Libre CSV independently of FormatParser (skip metadata row)."""
    return pl.read_csv(
        path,
        skip_rows=1,
        infer_schema_length=None,
        truncate_ragged_lines=True,
    )


def _independent_egv_bag(
    path: Path,
    historic_col: str,
    scan_col: str,
    mmol: bool,
) -> Counter:
    """Multiset of (datetime, rounded-glucose) for Record Type 0 and 1.

    Computed from the raw CSV, not from the parser under test.
    """
    raw = _read_libre_raw(path)
    record = pl.col("Record Type").cast(pl.Int64)
    historic = (
        raw.filter(record == LibreRecordType.HISTORIC_GLUCOSE.value)
        .select([
            pl.col("Device Timestamp").str.strptime(
                pl.Datetime("ms"), LIBRE_TIMESTAMP_FORMAT
            ).alias("datetime"),
            pl.col(historic_col).cast(pl.Float64, strict=False).alias("glucose"),
        ])
    )
    scan = (
        raw.filter(record == LibreRecordType.SCAN_GLUCOSE.value)
        .select([
            pl.col("Device Timestamp").str.strptime(
                pl.Datetime("ms"), LIBRE_TIMESTAMP_FORMAT
            ).alias("datetime"),
            pl.col(scan_col).cast(pl.Float64, strict=False).alias("glucose"),
        ])
    )
    combined = pl.concat([historic, scan])
    if mmol:
        combined = combined.with_columns(
            (pl.col("glucose") * MMOL_TO_MGDL).alias("glucose")
        )
    return Counter(
        (dt, None if g is None else round(g, 4))
        for dt, g in zip(
            combined["datetime"].to_list(),
            combined["glucose"].to_list(),
        )
    )


def _parsed_egv_bag(df: pl.DataFrame) -> Counter:
    egv = df.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
    return Counter(
        (dt, None if g is None else round(g, 4))
        for dt, g in zip(egv["datetime"].to_list(), egv["glucose"].to_list())
    )


def _record_type_count(path: Path, record_type: int) -> int:
    raw = _read_libre_raw(path)
    return raw.filter(
        pl.col("Record Type").cast(pl.Int64) == record_type
    ).height


def _colliding_historic_scan_timestamps(path: Path) -> set[str]:
    raw = _read_libre_raw(path)
    record = pl.col("Record Type").cast(pl.Int64)
    historic_ts = set(
        raw.filter(record == LibreRecordType.HISTORIC_GLUCOSE.value)[
            "Device Timestamp"
        ].to_list()
    )
    scan_ts = set(
        raw.filter(record == LibreRecordType.SCAN_GLUCOSE.value)[
            "Device Timestamp"
        ].to_list()
    )
    return historic_ts & scan_ts


def _parse_libre_timestamp(raw_timestamp: str) -> datetime:
    return pl.Series([raw_timestamp]).str.strptime(
        pl.Datetime("ms"), LIBRE_TIMESTAMP_FORMAT
    )[0]


def _assert_keep_first_at(
    marked: pl.DataFrame,
    timestamp: datetime,
    on: Literal["datetime", "original_datetime"],
) -> None:
    at_ts = marked.filter(pl.col(on) == timestamp)
    dup_bit = Quality.TIME_DUPLICATE.value
    clean = at_ts.filter((pl.col("quality") & dup_bit) == 0)
    duped = at_ts.filter((pl.col("quality") & dup_bit) != 0)
    assert at_ts.height >= 2
    assert clean.height == 1
    assert duped.height == at_ts.height - 1


class TestLibreEuDetectionOrder:
    """LIBRE_EU must win over LIBRE the same way DEXCOM_EU wins over DEXCOM."""

    def test_libre_eu_precedes_libre(self) -> None:
        keys = list(FORMAT_DETECTION_PATTERNS)
        assert keys.index(SupportedCGMFormat.LIBRE_EU) < keys.index(
            SupportedCGMFormat.LIBRE
        )


class TestLibreEuFixture:
    """Real mmol/L LibreView export. Skips when the local fixture is absent."""

    def test_detects_as_libre_eu(self) -> None:
        _skip_if_missing(JONGROVE)
        raw = JONGROVE.read_bytes()
        fmt = FormatParser.detect_format(FormatParser.decode_raw_data(raw))
        assert fmt == SupportedCGMFormat.LIBRE_EU

    def test_does_not_detect_as_libre(self) -> None:
        _skip_if_missing(JONGROVE)
        raw = JONGROVE.read_bytes()
        fmt = FormatParser.detect_format(FormatParser.decode_raw_data(raw))
        assert fmt != SupportedCGMFormat.LIBRE

    def test_schema_clean(self, jongrove_parsed: pl.DataFrame) -> None:
        CGM_SCHEMA.validate_dataframe(jongrove_parsed, enforce=False)

    def test_egv_count_matches_historic_plus_scan(self, jongrove_parsed: pl.DataFrame) -> None:
        expected = (
            _record_type_count(JONGROVE, LibreRecordType.HISTORIC_GLUCOSE.value)
            + _record_type_count(JONGROVE, LibreRecordType.SCAN_GLUCOSE.value)
        )
        actual = jongrove_parsed.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        ).height
        assert actual == expected

    def test_glucose_values_match_independent_mmol_conversion(
        self, jongrove_parsed: pl.DataFrame
    ) -> None:
        expected = _independent_egv_bag(
            JONGROVE,
            LibreEUColumn.HISTORIC_GLUCOSE.value,
            LibreEUColumn.SCAN_GLUCOSE.value,
            mmol=True,
        )
        assert _parsed_egv_bag(jongrove_parsed) == expected

    def test_sampled_glucose_in_sane_mgdl_range(self, jongrove_parsed: pl.DataFrame) -> None:
        g = jongrove_parsed.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value)[
            "glucose"
        ].drop_nulls()
        assert g.len() > 0
        assert g.min() > 0
        # Unconverted mmol/L readings cluster around 3–15; converted mg/dL does not.
        assert 40.0 <= g.median() <= 400.0

    def test_insulin_and_carbs_present(self, jongrove_parsed: pl.DataFrame) -> None:
        types = set(jongrove_parsed["event_type"].unique().to_list())
        insulin_count = _record_type_count(JONGROVE, LibreRecordType.INSULIN.value)
        food_count = _record_type_count(JONGROVE, LibreRecordType.FOOD.value)
        if insulin_count > 0:
            assert UnifiedEventType.INSULIN_FAST.value in types or (
                UnifiedEventType.INSULIN_SLOW.value in types
            )
        if food_count > 0:
            assert UnifiedEventType.CARBOHYDRATES.value in types

    def test_strip_rows_skipped_when_absent(self, jongrove_parsed: pl.DataFrame) -> None:
        strip_count = _record_type_count(JONGROVE, LibreRecordType.STRIP_GLUCOSE.value)
        if strip_count == 0:
            pytest.skip("No Record Type 2 (strip) rows in this fixture")
        calib = jongrove_parsed.filter(
            pl.col("event_type") == UnifiedEventType.CALIBRATION.value
        )
        assert calib.height == strip_count

    def test_idempotent_parse(self, jongrove_parsed: pl.DataFrame) -> None:
        again = FormatParser.parse_file(JONGROVE)
        assert jongrove_parsed.equals(again)

    def test_roundtrip_csv_string(self, jongrove_parsed: pl.DataFrame) -> None:
        csv_str = FormatParser.to_csv_string(jongrove_parsed)
        reparsed = FormatParser.parse_from_string(csv_str)
        assert reparsed.shape == jongrove_parsed.shape
        assert list(reparsed.columns) == list(jongrove_parsed.columns)
        assert FormatParser.detect_format(csv_str) == SupportedCGMFormat.UNIFIED_CGM

    def test_pipeline_runs(self, jongrove_parsed: pl.DataFrame) -> None:
        sequenced = FormatProcessor.detect_and_assign_sequences(jongrove_parsed)
        interpolated = FormatProcessor.interpolate_gaps(sequenced)
        synced = FormatProcessor.synchronize_timestamps(interpolated)
        inference_df, _warnings = FormatProcessor.prepare_for_inference(synced)
        assert inference_df.height > 0
        CGM_SCHEMA.validate_dataframe(inference_df, enforce=False)

    def test_frictionless_validatable(self) -> None:
        from cgm_format.cgm_cli import HAS_FRICTIONLESS, _validate_with_frictionless

        if not HAS_FRICTIONLESS:
            pytest.skip("frictionless not installed")
        _skip_if_missing(JONGROVE)
        _ok, _msg, error_count, _suppressed = _validate_with_frictionless(
            JONGROVE, SupportedCGMFormat.LIBRE_EU, suppress_known=True
        )
        assert error_count == 0


class TestCrossDetection:
    """A new format must not hijack existing fixtures, or be hijacked by them."""

    def test_synthetic_libre_still_libre(self) -> None:
        _skip_if_missing(SYNTHETIC_LIBRE)
        raw = SYNTHETIC_LIBRE.read_bytes()
        fmt = FormatParser.detect_format(FormatParser.decode_raw_data(raw))
        assert fmt == SupportedCGMFormat.LIBRE

    def test_synthetic_dexcom_not_libre_eu(self) -> None:
        _skip_if_missing(SYNTHETIC_DEXCOM)
        raw = SYNTHETIC_DEXCOM.read_bytes()
        fmt = FormatParser.detect_format(FormatParser.decode_raw_data(raw))
        assert fmt != SupportedCGMFormat.LIBRE_EU
        assert fmt in (
            SupportedCGMFormat.DEXCOM,
            SupportedCGMFormat.DEXCOM_EU,
        )

    def test_libreview_still_libre(self) -> None:
        _skip_if_missing(LIBREVIEW)
        raw = LIBREVIEW.read_bytes()
        fmt = FormatParser.detect_format(FormatParser.decode_raw_data(raw))
        assert fmt == SupportedCGMFormat.LIBRE
        assert fmt != SupportedCGMFormat.LIBRE_EU


class TestLibreScanGlucose:
    """Scan readings (Record Type 1) join the EGV_READ series.

    Covered on the committed synthetic fixture so CI sees the behaviour.
    """

    def test_scan_rows_present_as_egv(self, synthetic_libre_parsed: pl.DataFrame) -> None:
        scan_count = _record_type_count(
            SYNTHETIC_LIBRE, LibreRecordType.SCAN_GLUCOSE.value
        )
        historic_count = _record_type_count(
            SYNTHETIC_LIBRE, LibreRecordType.HISTORIC_GLUCOSE.value
        )
        assert scan_count > 0, "synthetic fixture is expected to contain scan rows"
        egv_count = synthetic_libre_parsed.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        ).height
        assert egv_count == historic_count + scan_count

    def test_scan_values_survive(self, synthetic_libre_parsed: pl.DataFrame) -> None:
        expected = _independent_egv_bag(
            SYNTHETIC_LIBRE,
            LibreColumn.HISTORIC_GLUCOSE.value,
            LibreColumn.SCAN_GLUCOSE.value,
            mmol=False,
        )
        assert _parsed_egv_bag(synthetic_libre_parsed) == expected

    def test_colliding_timestamps_flagged_time_duplicate(
        self, synthetic_libre_parsed: pl.DataFrame
    ) -> None:
        colliding = _colliding_historic_scan_timestamps(SYNTHETIC_LIBRE)
        assert len(colliding) > 0, "synthetic fixture is expected to have collisions"
        marked = FormatProcessor.mark_time_duplicates(synthetic_libre_parsed)
        _assert_keep_first_at(
            marked, _parse_libre_timestamp(next(iter(colliding))), on="datetime"
        )

    def test_colliding_timestamps_flagged_after_processor_chain(
        self, synthetic_libre_parsed: pl.DataFrame
    ) -> None:
        colliding = _colliding_historic_scan_timestamps(SYNTHETIC_LIBRE)
        assert len(colliding) > 0, "synthetic fixture is expected to have collisions"
        sequenced = FormatProcessor.detect_and_assign_sequences(synthetic_libre_parsed)
        interpolated = FormatProcessor.interpolate_gaps(sequenced)
        synced = FormatProcessor.synchronize_timestamps(interpolated)
        marked = FormatProcessor.mark_time_duplicates(synced)
        source_dt = _parse_libre_timestamp(next(iter(colliding)))
        source_pair = marked.filter(pl.col("original_datetime") == source_dt)
        assert source_pair.height >= 2
        synced_times = source_pair["datetime"].unique()
        assert synced_times.len() == 1
        _assert_keep_first_at(marked, synced_times[0], on="datetime")

    def test_synthetic_pipeline_full_chain(
        self, synthetic_libre_parsed: pl.DataFrame
    ) -> None:
        sequenced = FormatProcessor.detect_and_assign_sequences(synthetic_libre_parsed)
        interpolated = FormatProcessor.interpolate_gaps(sequenced)
        synced = FormatProcessor.synchronize_timestamps(interpolated)
        inference_df, _warnings = FormatProcessor.prepare_for_inference(synced)
        assert inference_df.height > 0
        CGM_SCHEMA.validate_dataframe(inference_df, enforce=False)

    def test_synthetic_roundtrip_and_idempotent_parse(
        self, synthetic_libre_parsed: pl.DataFrame
    ) -> None:
        again = FormatParser.parse_file(SYNTHETIC_LIBRE)
        assert synthetic_libre_parsed.equals(again)
        csv_str = FormatParser.to_csv_string(synthetic_libre_parsed)
        reparsed = FormatParser.parse_from_string(csv_str)
        assert reparsed.shape == synthetic_libre_parsed.shape
        assert FormatParser.detect_format(csv_str) == SupportedCGMFormat.UNIFIED_CGM

    def test_strip_skipped_when_absent(self, synthetic_libre_parsed: pl.DataFrame) -> None:
        strip_count = _record_type_count(
            SYNTHETIC_LIBRE, LibreRecordType.STRIP_GLUCOSE.value
        )
        if strip_count == 0:
            pytest.skip("No Record Type 2 (strip) rows in the synthetic fixture")
        calib = synthetic_libre_parsed.filter(
            pl.col("event_type") == UnifiedEventType.CALIBRATION.value
        )
        assert calib.height == strip_count
