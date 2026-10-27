"""MiniMed pump CareLink exports (MEDTRONIC_PUMP, RM16).

The synthetic fixture, assets/fixtures/medtronic_pump_synthetic.csv, is hand-authored by
scripts/make_synthetic_medtronic_pump.py, one or two rows per kind of record a real 720G export
carries. Every expected value below is worked out by hand from that script's row list, never by
running the parser. Real exports, when present under data/input/, are checked for the properties that
hold whatever they contain.
"""

import csv
import logging
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import List, Optional, Tuple

import polars as pl
import pytest

from cgm_format import FormatParser
from cgm_format.formats.unified import CGM_SCHEMA_EXTENDED, UnifiedEventType
from cgm_format.formats.medtronic import MedtronicColumn
from cgm_format.formats.medtronic_pump import MedtronicPumpColumn
from cgm_format.interface.cgm_interface import SupportedCGMFormat, UnknownFormatError
from tests.paths import FIXTURES_DIR, LOCAL_INPUT_DIR

SYNTHETIC = FIXTURES_DIR / "medtronic_pump_synthetic.csv"
GUARDIAN = FIXTURES_DIR / "medtronics_test_livia.csv"

pytestmark = pytest.mark.skipif(not SYNTHETIC.exists(), reason="synthetic pump fixture absent")


def at(hhmmss: str) -> datetime:
    return datetime.strptime(f"1961-04-12 {hhmmss}", "%Y-%m-%d %H:%M:%S")


def rows_of(df: pl.DataFrame, event: UnifiedEventType, *cols: str) -> List[Tuple]:
    return (
        df.filter(pl.col("event_type") == event.value)
        .sort("datetime", *cols)
        .select("datetime", *cols)
        .rows()
    )


@pytest.fixture(scope="module")
def synthetic(caplog_module: List[logging.LogRecord]) -> pl.DataFrame:
    return FormatParser.parse_file(SYNTHETIC)


@pytest.fixture(scope="module")
def caplog_module() -> List[logging.LogRecord]:
    """The warnings the module-scoped parse emits (caplog is function-scoped)."""
    records: List[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append  # type: ignore[method-assign]
    logger = logging.getLogger("cgm_format.format_parser")
    logger.addHandler(handler)
    yield records
    logger.removeHandler(handler)


def real_pump_exports() -> List[Path]:
    found = []
    for f in sorted(LOCAL_INPUT_DIR.glob("*.csv")):
        try:
            if FormatParser.detect_format(FormatParser.decode_raw_data(f.read_bytes())) == SupportedCGMFormat.MEDTRONIC_PUMP:
                found.append(f)
        except UnknownFormatError:
            continue
    return found


class TestDetection:
    def test_pump_export_is_medtronic_pump(self) -> None:
        text = FormatParser.decode_raw_data(SYNTHETIC.read_bytes())
        assert FormatParser.detect_format(text) == SupportedCGMFormat.MEDTRONIC_PUMP

    @pytest.mark.skipif(not GUARDIAN.exists(), reason="Guardian fixture absent")
    def test_guardian_export_stays_medtronic(self) -> None:
        text = FormatParser.decode_raw_data(GUARDIAN.read_bytes())
        assert FormatParser.detect_format(text) == SupportedCGMFormat.MEDTRONIC


class TestSyntheticParse:
    def test_frame_is_extended_unified(self, synthetic: pl.DataFrame) -> None:
        CGM_SCHEMA_EXTENDED.validate_dataframe(synthetic, enforce=False)

    def test_nothing_is_insulin_slow(self, synthetic: pl.DataFrame) -> None:
        # Pump basal and boluses are rapid-acting; the old Medtronic path put the basal rate here.
        assert synthetic["insulin_slow"].null_count() == synthetic.height
        assert UnifiedEventType.INSULIN_SLOW.value not in synthetic["event_type"].to_list()

    def test_effective_basal_rate(self, synthetic: pl.DataFrame) -> None:
        assert rows_of(synthetic, UnifiedEventType.BASAL_RATE, "basal_rate") == pytest.approx([
            (at("00:00:00"), 1.0),
            (at("01:00:00"), 0.5),   # Percent temp, 50 % of the logged 1,000
            (at("01:30:00"), 0.4),   # schedule change inside it: 50 % of 0,800
            (at("02:00:00"), 0.8),   # the temp's hour is over
            (at("03:00:00"), 2.0),   # Rate temp, logged as the effective rate
            (at("03:30:00"), 0.8),
            (at("04:00:00"), 1.1),   # two rates in one second: the later one
            (at("05:00:00"), 0.0),   # user suspend
            (at("05:15:00"), 1.1),   # resumed
            (at("06:00:00"), 0.0),   # alarm suspend: no 0 logged, still 0
            (at("06:20:00"), 1.1),   # resumed: back to the last logged rate
            (at("16:00:00"), 1.0),
        ])

    def test_boluses(self, synthetic: pl.DataFrame) -> None:
        got = rows_of(synthetic, UnifiedEventType.INSULIN_FAST, "insulin_fast", "insulin_duration")
        assert got == [
            (at("07:00:00"), 3.0, None),   # normal, at its programmed time
            (at("08:00:00"), 1.5, 60.0),   # dual square part, over an hour
            (at("08:00:00"), 2.0, None),   # dual normal part
            (at("10:00:00"), 1.0, 20.0),   # square cancelled at 20 min, reporting 20
            (at("11:00:00"), 1.5, 30.0),   # square short of volume, reporting 30
            (at("13:00:00"), 4.5, None),   # wizard bolus
        ]

    def test_carbs_meter_and_sensor(self, synthetic: pl.DataFrame) -> None:
        assert rows_of(synthetic, UnifiedEventType.CARBOHYDRATES, "carbs") == [(at("13:00:00"), 45.0)]
        assert rows_of(synthetic, UnifiedEventType.GLUCOSE_METER, "glucose") == [
            (at("13:00:00"), 120.0),
            (at("14:00:00"), 95.0),
        ]
        assert rows_of(synthetic, UnifiedEventType.GLUCOSE, "glucose") == [(at("00:05:00"), 110.0)]

    def test_warnings_say_what_was_left_out(
        self, synthetic: pl.DataFrame, caplog_module: List[logging.LogRecord]
    ) -> None:
        messages = [r.getMessage() for r in caplog_module if r.levelno == logging.WARNING]
        assert any("1 bolus row(s) of an unknown type" in m and "Mystery" in m for m in messages)
        assert any("1 bolus(es) with no delivered volume" in m for m in messages)
        assert any("1 square bolus(es) delivered less than programmed" in m for m in messages)
        assert any("2 priming row(s), 10.3 U" in m for m in messages)
        assert not any("clock was changed" in m for m in messages)

    def test_clock_change_is_reported_not_shifted(self, caplog: pytest.LogCaptureFixture) -> None:
        text = FormatParser.decode_raw_data(SYNTHETIC.read_bytes())
        lines = text.split("\n")
        new_device_time = MedtronicPumpColumn.get_all_columns().index(MedtronicPumpColumn.NEW_DEVICE_TIME)
        target = next(i for i, line in enumerate(lines) if ";16:00:00;" in line)
        cells = lines[target].split(";")
        cells[new_device_time] = "1961/04/12 17:00:00"
        lines[target] = ";".join(cells)
        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            df = FormatParser.parse_from_string("\n".join(lines))
        assert any("clock was changed 1 time(s)" in r.getMessage() for r in caplog.records)
        assert rows_of(df, UnifiedEventType.BASAL_RATE, "basal_rate")[-1] == (at("16:00:00"), 1.0)

    def test_roundtrip_through_unified_csv(self, synthetic: pl.DataFrame) -> None:
        again = FormatParser.parse_from_string(synthetic.write_csv())
        assert again.equals(synthetic)

    def test_parse_is_deterministic(self, synthetic: pl.DataFrame) -> None:
        assert FormatParser.parse_file(SYNTHETIC).equals(synthetic)


def test_daily_auto_insulin_totals_are_not_boluses(caplog: pytest.LogCaptureFixture) -> None:
    """A 780G export adds a section of daily SmartGuard totals, also labelled 'Pump' on its section line
    ("-------;<device>;Pump;<serial>;Aggregated Auto Insulin Data"). Its rows look like boluses at
    00:00:00 but are a day's automatic insulin with no time of delivery: never an INS_FAST row."""
    text = FormatParser.decode_raw_data(SYNTHETIC.read_bytes())
    lines = text.split("\n")
    sensor = next(i for i, line in enumerate(lines) if line.startswith("-------") and ";Sensor" in line)
    header = lines[sensor + 1]
    columns = MedtronicPumpColumn.get_all_columns()
    cells = {MedtronicPumpColumn.INDEX: "100,00000", MedtronicPumpColumn.DATE: "1961/04/12", MedtronicPumpColumn.TIME: "00:00:00",
             MedtronicPumpColumn.BOLUS_TYPE: "Normal", MedtronicPumpColumn.BOLUS_VOLUME_SELECTED: "20,494",
             MedtronicPumpColumn.BOLUS_VOLUME_DELIVERED: "20,494", MedtronicPumpColumn.BOLUS_SOURCE: "CLOSED_LOOP_AUTO_INSULIN"}
    row = ";".join(cells.get(c, "") for c in columns)
    lines[sensor:sensor] = ["-------;MiniMed 780G MMT-1886;Pump;XXXXXXXXXX;Aggregated Auto Insulin Data", header, row]
    with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
        df = FormatParser.parse_from_string("\n".join(lines))
    boluses = rows_of(df, UnifiedEventType.INSULIN_FAST, "insulin_fast")
    assert (at("00:00:00"), 20.494) not in boluses
    assert len(boluses) == 6  # the synthetic export's own six, unchanged
    assert any("20.494 U of automatic insulin" in r.getMessage() for r in caplog.records)


def _guardian_cells(column: MedtronicColumn) -> int:
    """Cells of one Guardian column holding a number, read with the csv module, not the parser."""
    text = FormatParser.decode_raw_data(GUARDIAN.read_bytes())
    count = 0
    header: Optional[List[str]] = None
    for row in csv.reader(StringIO(text), delimiter=";"):
        if row and row[0].strip().lstrip("﻿") == MedtronicColumn.INDEX.value:
            header = [c.strip() for c in row]
            continue
        if header is None or column.value not in header or len(row) != len(header):
            continue
        cell = row[header.index(column.value)].strip()
        if cell and not cell.startswith("-------"):
            count += 1
    return count


@pytest.mark.skipif(not GUARDIAN.exists(), reason="Guardian fixture absent")
class TestGuardianFingersticks:
    """The Guardian path used to coalesce fingersticks into the sensor series."""

    def test_egv_rows_are_sensor_readings_only(self) -> None:
        df = FormatParser.parse_file(GUARDIAN)
        assert df.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value).height == _guardian_cells(
            MedtronicColumn.SENSOR_GLUCOSE
        )

    def test_fingersticks_are_meter_rows(self) -> None:
        df = FormatParser.parse_file(GUARDIAN)
        assert df.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE_METER.value).height == _guardian_cells(
            MedtronicColumn.BG_READING
        )


REAL = real_pump_exports()


@pytest.mark.skipif(not REAL, reason="no real pump export under data/input/")
@pytest.mark.parametrize("path", REAL, ids=lambda p: p.name[-12:])
def test_real_pump_export(path: Path) -> None:
    df = FormatParser.parse_file(path)
    CGM_SCHEMA_EXTENDED.validate_dataframe(df, enforce=False)
    assert df["insulin_slow"].null_count() == df.height
    basal = df.filter(pl.col("event_type") == UnifiedEventType.BASAL_RATE.value)
    assert basal.height > 0
    assert basal["basal_rate"].min() >= 0
    # Only changes are emitted: no two consecutive basal rows carry the same rate.
    rates = basal.sort("datetime")["basal_rate"]
    assert (rates.ne_missing(rates.shift(1)))[1:].all()
    boluses = df.filter(pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value)
    assert (boluses["insulin_fast"] > 0).all()
