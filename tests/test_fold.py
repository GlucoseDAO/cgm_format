"""FormatProcessor.fold_insulin_onto_grid (RM17).

A pump frame folded onto a CGM frame's grid. Both inputs are committed synthetic exports dated
1961-04-12: the Dexcom one (`Clarity_Export_synthetic.csv`), whose first sequence runs on a grid at
00:57 + 5 min × k, and the hand-authored pump one (`medtronic_pump_synthetic.csv`, rows listed in
`scripts/make_synthetic_medtronic_pump.py`). Every expected value is worked out by hand from those
rows and the bin rule, a grid point t owning [t - 2.5 min, t + 2.5 min); none comes from running the
fold.
"""

import csv
import logging
import re
from io import StringIO
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import pytest

from cgm_format import FormatParser, FormatProcessor
from cgm_format.format_processor import ExtendedFormatProcessor, FoldSource
from cgm_format.formats.unified import CGM_SCHEMA, UnifiedEventType

DATA_DIR = Path(__file__).parent.parent / "data" / "input"
DEXCOM = DATA_DIR / "Clarity_Export_synthetic.csv"
PUMP = DATA_DIR / "medtronic_pump_synthetic.csv"

pytestmark = pytest.mark.skipif(not (DEXCOM.exists() and PUMP.exists()), reason="synthetic fixtures absent")

Bin = Dict[str, Optional[float]]


def at(hhmm: str) -> datetime:
    return datetime.strptime(f"1961-04-12 {hhmm}", "%Y-%m-%d %H:%M")


def bins(folded: pl.DataFrame, *times: str) -> Dict[str, Bin]:
    wanted = pl.Series([at(t) for t in times], dtype=pl.Datetime("ms"))
    rows = folded.filter(pl.col("datetime").is_in(wanted.implode()))
    return {
        r["datetime"].strftime("%H:%M"): {k: r[k] for k in ("carbs", "insulin_slow", "insulin_fast", "basal_rate")}
        for r in rows.iter_rows(named=True)
    }


def with_entries(raw: pl.DataFrame, entries: List[tuple]) -> pl.DataFrame:
    """The parsed Dexcom frame with extra CGM-logged entries (time, event, column, value) added."""
    rows = pl.DataFrame(
        [
            {"datetime": at(t), "original_datetime": at(t), "event_type": e.value, "quality": 0, "sequence_id": 0, col: v}
            for t, e, col, v in entries
        ]
    )
    return CGM_SCHEMA.validate_dataframe(pl.concat([raw, rows], how="diagonal_relaxed"), enforce=True)


@pytest.fixture(scope="module")
def dexcom() -> pl.DataFrame:
    return FormatParser.parse_file(DEXCOM)


@pytest.fixture(scope="module")
def cgm(dexcom: pl.DataFrame) -> pl.DataFrame:
    return FormatProcessor.synchronize_timestamps(dexcom)


@pytest.fixture(scope="module")
def pump() -> pl.DataFrame:
    return FormatParser.parse_file(PUMP)


@pytest.fixture(scope="module")
def folded(cgm: pl.DataFrame, pump: pl.DataFrame) -> pl.DataFrame:
    return FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.PUMP)


class TestBasal:
    def test_rate_is_the_bin_mean(self, folded: pl.DataFrame) -> None:
        got = bins(folded, "00:57", "01:02", "01:32", "05:02", "05:07", "15:57")
        assert {t: b["basal_rate"] for t, b in got.items()} == pytest.approx({
            "00:57": 1.0,    # 1.0 throughout
            "01:02": 0.55,   # 0.5 min at 1.0, then 4.5 min at 50 % of it
            "01:32": 0.41,   # 0.5 min at 0.5, then 4.5 min at 50 % of 0.8
            "05:02": 0.11,   # 0.5 min at 1.1, then suspended
            "05:07": 0.0,    # suspended: a measured 0, not unknown
            "15:57": 1.1,
        })

    def test_unknown_beyond_the_pump_is_null(self, folded: pl.DataFrame) -> None:
        got = bins(folded, "16:02", "17:37")
        assert got["16:02"]["basal_rate"] is None   # the pump's rows end at 16:00, mid-bin
        assert got["17:37"]["basal_rate"] is None
        assert got["17:37"]["insulin_fast"] is None  # no source covers it: not 0

    def test_not_folded_by_default(self, folded: pl.DataFrame) -> None:
        assert bins(folded, "00:57")["00:57"]["insulin_fast"] == 0.0

    def test_fold_basal_adds_the_dose(self, cgm: pl.DataFrame, pump: pl.DataFrame) -> None:
        got = bins(FormatProcessor.fold_insulin_onto_grid(cgm, pump, fold_basal=True, source=FoldSource.PUMP), "00:57", "07:02", "16:02")
        assert got["00:57"]["insulin_fast"] == pytest.approx(1.0 * 5 / 60)
        assert got["07:02"]["insulin_fast"] == pytest.approx(3.0 + 1.1 * 5 / 60)
        assert got["16:02"]["insulin_fast"] is None


class TestBoluses:
    def test_instant_and_spread(self, folded: pl.DataFrame) -> None:
        got = bins(folded, "07:02", "08:02", "08:07", "08:57", "09:02", "10:02", "10:07", "10:22", "11:02", "13:02")
        assert {t: b["insulin_fast"] for t, b in got.items()} == pytest.approx({
            "07:02": 3.0,                  # normal bolus at 07:00
            "08:02": 2.0 + 1.5 * 4.5 / 60, # dual: 2.0 at once, square 1.5 over 60 min from 08:00
            "08:07": 1.5 * 5 / 60,
            "08:57": 1.5 * 5 / 60,
            "09:02": 1.5 * 0.5 / 60,       # the square's last half-minute
            "10:02": 1.0 * 4.5 / 20,       # square cancelled at 20 min: 1.0 over 20 min
            "10:07": 1.0 * 5 / 20,
            "10:22": 1.0 * 0.5 / 20,
            "11:02": 1.5 * 4.5 / 30,
            "13:02": 4.5,
        })

    def test_every_unit_delivered_lands_once(self, folded: pl.DataFrame) -> None:
        day = folded.filter(pl.col("datetime").dt.date() == pl.date(1961, 4, 12))
        assert day["insulin_fast"].sum() == pytest.approx(3.0 + 2.0 + 1.5 + 1.0 + 1.5 + 4.5)

    def test_carbs_and_long_acting(self, folded: pl.DataFrame) -> None:
        got = bins(folded, "13:02", "06:32", "07:02")
        assert got["13:02"]["carbs"] == 45.0
        assert got["07:02"]["carbs"] is None        # no entry recorded: never 0
        assert got["06:32"]["insulin_slow"] == 27.0  # the CGM frame's long-acting dose at 06:30


class TestSources:
    def test_cgm_source_takes_the_cgm_entries(self, cgm: pl.DataFrame, pump: pl.DataFrame) -> None:
        got = bins(FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.CGM), "07:02", "07:47", "17:37", "01:02")
        assert got["07:47"]["insulin_fast"] == 5.0   # logged at 07:45
        assert got["17:37"]["insulin_fast"] == 4.0   # logged at 17:38
        assert got["07:02"]["insulin_fast"] is None  # the pump's bolus is not this source's
        assert got["01:02"]["basal_rate"] == pytest.approx(0.55)  # basal still comes from the pump

    def test_merge(self, dexcom: pl.DataFrame, pump: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
        cgm = FormatProcessor.synchronize_timestamps(with_entries(dexcom, [
            ("07:03", UnifiedEventType.INSULIN_FAST, "insulin_fast", 3.0),  # same slot as the pump's 07:00, equal
            ("13:06", UnifiedEventType.INSULIN_FAST, "insulin_fast", 5.0),  # one slot from the pump's 4.5
            ("10:01", UnifiedEventType.INSULIN_FAST, "insulin_fast", 0.0),  # a 0 against the pump's square 1.0
            ("11:10", UnifiedEventType.INSULIN_FAST, "insulin_fast", 1.5),  # two slots from the pump's 11:00
            ("13:04", UnifiedEventType.CARBOHYDRATES, "carbs", 45.0),       # the wizard's 45 g, logged again
        ]))
        with caplog.at_level(logging.WARNING, logger="cgm_format.format_processor"):
            merged = FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.MERGE)
        got = bins(merged, "07:02", "13:02", "10:02", "11:12", "13:07")
        assert got["07:02"]["insulin_fast"] == pytest.approx(3.0)           # paired: once, not 6.0
        assert got["13:02"]["insulin_fast"] == pytest.approx(4.75)          # paired on the pump's slot, the mean
        assert got["13:07"]["insulin_fast"] == pytest.approx(0.0)
        assert got["10:02"]["insulin_fast"] == pytest.approx(1.0 * 4.5 / 20)  # the 0 said nothing
        assert got["11:12"]["insulin_fast"] == pytest.approx(1.5 + 1.5 * 5 / 30)  # two slots: both kept
        assert got["13:02"]["carbs"] == 45.0                                 # paired: once
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "1 merged insulin pair(s) differ by more than 0.1 U" in messages
        assert "pump 4.5 U" in messages and "cgm 5 U" in messages

    def test_merge_pair_threshold(self, dexcom: pl.DataFrame, pump: pl.DataFrame) -> None:
        cgm = FormatProcessor.synchronize_timestamps(with_entries(dexcom, [
            ("11:10", UnifiedEventType.INSULIN_FAST, "insulin_fast", 1.5),
        ]))
        wide = FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.MERGE, merge_pair_slots=2)
        day = wide.filter(pl.col("datetime").dt.date() == pl.date(1961, 4, 12))
        # The CGM's 1.5 pairs with the pump's square 1.5 two slots away: counted once, spread as the pump delivered it.
        assert bins(wide, "11:12")["11:12"]["insulin_fast"] == pytest.approx(1.5 * 5 / 30)
        # The pump's 13.5 U once, plus the Dexcom frame's own 5.0 (07:45) and 4.0 (17:38), which pair with nothing.
        assert day["insulin_fast"].sum() == pytest.approx(13.5 + 5.0 + 4.0)


class TestContract:
    def test_merge_is_the_default(self, cgm: pl.DataFrame, pump: pl.DataFrame) -> None:
        """The person's call (2026-10-21): a meal or dose logged in only one source still reaches its row."""
        assert FormatProcessor.fold_source == FoldSource.MERGE
        assert FormatProcessor.fold_insulin_onto_grid(cgm, pump).equals(
            FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.MERGE)
        )

    def test_glucose_and_grid_are_the_cgm_frames(self, cgm: pl.DataFrame, folded: pl.DataFrame) -> None:
        readings = cgm.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value).select("datetime", "glucose")
        assert folded.filter(pl.col("glucose").is_not_null()).select("datetime", "glucose").sort("datetime").equals(
            readings.unique("datetime", keep="first", maintain_order=True).sort("datetime")
        )

    def test_inputs_untouched_and_output_deterministic(self, cgm: pl.DataFrame, pump: pl.DataFrame, folded: pl.DataFrame) -> None:
        cgm_before, pump_before = cgm.clone(), pump.clone()
        again = FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.PUMP)
        assert again.equals(folded)
        assert cgm.equals(cgm_before) and pump.equals(pump_before)

    def test_overlapping_exports_count_once(self, cgm: pl.DataFrame, pump: pl.DataFrame, folded: pl.DataFrame) -> None:
        assert FormatProcessor.fold_insulin_onto_grid(cgm, [pump, pump], source=FoldSource.PUMP).equals(folded)

    def test_order_of_grid_ops_does_not_matter(self, dexcom: pl.DataFrame, pump: pl.DataFrame) -> None:
        a = FormatProcessor.synchronize_timestamps(FormatProcessor.interpolate_gaps(dexcom))
        b = FormatProcessor.interpolate_gaps(FormatProcessor.synchronize_timestamps(dexcom))
        assert FormatProcessor.fold_insulin_onto_grid(a, pump, source=FoldSource.PUMP).equals(FormatProcessor.fold_insulin_onto_grid(b, pump, source=FoldSource.PUMP))

    def test_a_folded_frame_cannot_be_folded_again(self, folded: pl.DataFrame, pump: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="cannot be folded again"):
            FormatProcessor.fold_insulin_onto_grid(folded, pump, source=FoldSource.PUMP)

    def test_unsynchronized_cgm_is_refused(self, dexcom: pl.DataFrame, pump: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="synchronize_timestamps"):
            FormatProcessor.fold_insulin_onto_grid(FormatProcessor.detect_and_assign_sequences(dexcom), pump)

    def test_extended_processor_folds_the_same(self, cgm: pl.DataFrame, pump: pl.DataFrame, folded: pl.DataFrame) -> None:
        assert ExtendedFormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.PUMP).equals(folded)


REAL_CGM = DATA_DIR / "Clarity_Export__Patient_2026-07-17_215123.csv"


def _real_pumps() -> List[Path]:
    found = []
    for f in sorted(DATA_DIR.glob("*.csv")):
        if f == PUMP:
            continue
        head = f.read_bytes()[:4000].decode("utf-8", errors="ignore")
        if "Estimated Reservoir Volume after Fill (U)" in head:
            found.append(f)
    return found


def _raw_delivered_units(paths: List[Path]) -> float:
    """Units the pump delivered, read from the raw CSVs with the csv module, not the parser: every row
    with a delivered bolus volume, a row present in two overlapping exports counted once."""
    seen = set()
    for path in paths:
        text = FormatParser.decode_raw_data(path.read_bytes())
        header: Optional[List[str]] = None
        for row in csv.reader(StringIO(text), delimiter=";"):
            if row and row[0].strip().lstrip("\ufeff") == "Index":
                header = [c.strip() for c in row]
                continue
            if header is None or len(row) != len(header):
                continue
            cell = row[header.index("Bolus Volume Delivered (U)")].strip()
            if cell:
                fields = ("Date", "Time", "Bolus Number", "Bolus Type", "Bolus Volume Delivered (U)")
                seen.add(tuple(row[header.index(f)] for f in fields))
    return sum(float(key[-1].replace(",", ".")) for key in seen)


@pytest.mark.skipif(not REAL_CGM.exists() or not _real_pumps(), reason="real CGM and pump exports absent")
def test_real_pair_accounts_for_every_bolus_unit(caplog: pytest.LogCaptureFixture) -> None:
    """On the real overlap, every bolus unit the pumps delivered is on a row or reported outside.

    The expected total is read from the raw CSVs with the csv module (`_raw_delivered_units`),
    independent of both the parser and the fold, so a bolus split across two exports and counted
    twice would show here.
    """
    pumps = [FormatParser.parse_file(p) for p in _real_pumps()]
    cgm = FormatProcessor.synchronize_timestamps(FormatParser.parse_file(REAL_CGM))
    with caplog.at_level(logging.WARNING, logger="cgm_format.format_processor"):
        wide = FormatProcessor.fold_insulin_onto_grid(cgm, pumps, source=FoldSource.PUMP)
    message = " ".join(r.getMessage() for r in caplog.records)
    outside = re.search(r"([\d.]+) U of bolus insulin outside every glucose sequence", message)
    delivered = _raw_delivered_units(_real_pumps())
    on_rows = wide["insulin_fast"].sum()
    assert on_rows + (float(outside.group(1)) if outside else 0.0) == pytest.approx(delivered, abs=0.01)


class TestSugarOneShape:
    """to_sugarone_ready_df (RM21): the fold's output in the trained models' input columns."""

    def test_columns_are_the_models(self, folded: pl.DataFrame) -> None:
        out = FormatProcessor.to_sugarone_ready_df(folded)
        # src/common/data/columns.py in glucose_forecasting_metabonet, as SugarOne's trainer reads them.
        assert out.columns == [
            "sequence_id", "Timestamp", "Event Type", "User ID", "Glucose (mg/dL)",
            "Basal Rate (U/h)", "Bolus Insulin (U)", "Carbohydrates (g)",
            "Recommended Split", "Study Group",
        ]

    def test_basal_and_bolus_are_separate_channels(self, folded: pl.DataFrame) -> None:
        out = FormatProcessor.to_sugarone_ready_df(folded).filter(
            pl.col("Timestamp").is_in(["1961-04-12T01:02:00", "1961-04-12T07:02:00", "1961-04-12T17:37:00"])
        )
        rows = {r["Timestamp"][11:16]: (r["Basal Rate (U/h)"], r["Bolus Insulin (U)"], r["Event Type"]) for r in out.iter_rows(named=True)}
        assert rows["01:02"][0] == pytest.approx(0.55) and rows["01:02"][1] == 0.0
        assert rows["07:02"][0] == pytest.approx(1.1) and rows["07:02"][1] == 3.0
        assert rows["17:37"][:2] == (None, None)   # beyond the pump: unknown stays null, never 0
        assert {r[2] for r in rows.values()} == {"EGV"}

    def test_interpolated_rows_are_labelled(self, dexcom: pl.DataFrame, pump: pl.DataFrame) -> None:
        # Two consecutive readings removed leave a 15-minute gap, which interpolation fills with two rows.
        readings = dexcom.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value).sort("original_datetime")
        gone = readings["original_datetime"][100:102].implode()
        holed = dexcom.filter(~pl.col("original_datetime").is_in(gone))
        cgm = FormatProcessor.synchronize_timestamps(FormatProcessor.interpolate_gaps(holed))
        out = FormatProcessor.to_sugarone_ready_df(FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.PUMP))
        assert out.filter(pl.col("Event Type") == "Interpolated").height == 2

    def test_refuses_a_unified_frame(self, cgm: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="fold_insulin_onto_grid"):
            FormatProcessor.to_sugarone_ready_df(cgm)


def export_slice(first: str, last: str) -> pl.DataFrame:
    """The synthetic pump export cut to the pump rows logged from `first` to `last` (HH:MM:SS), as an
    export covering only that span would hold them."""
    text = FormatParser.decode_raw_data(PUMP.read_bytes())
    kept = []
    for line in text.split("\n"):
        cells = line.split(";")
        is_row = len(cells) > 3 and cells[1] == "1961/04/12" and cells[0][:1].isdigit()
        if not is_row or first <= cells[2] <= last:
            kept.append(line)
    return FormatParser.parse_from_string("\n".join(kept))


class TestOverlappingExports:
    """Findings of the pre-release audit: exports that overlap but do not hold the same rows."""

    def test_a_bolus_split_across_two_exports_counts_once(self, cgm: pl.DataFrame) -> None:
        # A ends at 08:30, before the dual's square part finishes (09:00); B starts at 07:01, after
        # bolus 1 was programmed (07:00) but before it finished (07:01:30).
        a, b = export_slice("00:00:00", "08:30:00"), export_slice("07:01:00", "23:59:59")
        for order in ([a, b], [b, a]):
            wide = FormatProcessor.fold_insulin_onto_grid(cgm, order, source=FoldSource.PUMP)
            day = wide.filter(pl.col("datetime").dt.date() == pl.date(1961, 4, 12))
            assert day["insulin_fast"].sum() == pytest.approx(13.5)
            assert bins(wide, "07:02")["07:02"]["insulin_fast"] == pytest.approx(3.0)
            assert bins(wide, "08:07")["08:07"]["insulin_fast"] == pytest.approx(1.5 * 5 / 60)

    def test_exports_that_disagree_give_the_earlier_ones_answer_in_any_order(self, cgm: pl.DataFrame, pump: pl.DataFrame) -> None:
        # B starts at 01:20, inside the percent temp it never saw begin: its 01:30 rate is the schedule.
        b = export_slice("01:20:00", "23:59:59")
        alone = FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.PUMP)
        assert FormatProcessor.fold_insulin_onto_grid(cgm, [pump, b], source=FoldSource.PUMP).equals(alone)
        assert FormatProcessor.fold_insulin_onto_grid(cgm, [b, pump], source=FoldSource.PUMP).equals(alone)


class TestAuditCases:
    def test_cgm_source_cannot_fold_basal(self, cgm: pl.DataFrame, pump: pl.DataFrame) -> None:
        with pytest.raises(ValueError, match="fold_basal"):
            FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.CGM, fold_basal=True)

    def test_folded_basal_is_not_also_given_apart(self, cgm: pl.DataFrame, pump: pl.DataFrame) -> None:
        wide = FormatProcessor.fold_insulin_onto_grid(cgm, pump, fold_basal=True, source=FoldSource.PUMP)
        assert wide["basal_rate"].null_count() == wide.height

    def test_a_zero_takes_no_partner(self, dexcom: pl.DataFrame, pump: pl.DataFrame) -> None:
        cgm = FormatProcessor.synchronize_timestamps(with_entries(dexcom, [
            ("12:59", UnifiedEventType.INSULIN_FAST, "insulin_fast", 0.0),
            ("13:03", UnifiedEventType.INSULIN_FAST, "insulin_fast", 4.5),  # the pump's 13:00 dose, logged again
        ]))
        got = bins(FormatProcessor.fold_insulin_onto_grid(cgm, pump, source=FoldSource.MERGE), "13:02")
        assert got["13:02"]["insulin_fast"] == pytest.approx(4.5)  # once, not 9.0

    def test_merge_reports_outside_entries_per_source(self, dexcom: pl.DataFrame, pump: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
        # The CGM's first reading is at 00:57; a dose at 00:30 logged by both sources is outside every sequence.
        early_pump = pl.concat([pump, pump.filter(pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value).head(1)
                                .with_columns(pl.lit(at("00:30")).cast(pl.Datetime("ms")).alias("datetime"),
                                              pl.lit(at("00:30")).cast(pl.Datetime("ms")).alias("original_datetime"))])
        cgm = FormatProcessor.synchronize_timestamps(with_entries(dexcom, [("00:31", UnifiedEventType.INSULIN_FAST, "insulin_fast", 3.0)]))
        with caplog.at_level(logging.WARNING, logger="cgm_format.format_processor"):
            FormatProcessor.fold_insulin_onto_grid(cgm, early_pump, source=FoldSource.MERGE)
        message = " ".join(r.getMessage() for r in caplog.records)
        assert "insulin: 3 U logged by the pump, 3 U by the CGM (one dose may be in both)" in message
        assert "U of bolus insulin outside every glucose sequence" not in message
