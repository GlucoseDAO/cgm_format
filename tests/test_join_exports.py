"""FormatParser.join_exports / parse_exports (F21): pieces of one series, joined once.

The committed synthetic Dexcom export is cut into two exports that overlap by a day, as an account
exported twice would give them. Joined, they must equal the whole file, whichever order they come in.
"""

import logging
from pathlib import Path
from typing import Callable

import polars as pl
import pytest

from cgm_format import FormatParser, FormatProcessor
from cgm_format.formats.unified import UnifiedEventType

DEXCOM = Path(__file__).parent.parent / "data" / "input" / "Clarity_Export_synthetic.csv"

pytestmark = pytest.mark.skipif(not DEXCOM.exists(), reason="synthetic Dexcom export absent")


def cut(keep: Callable[[str], bool], edit: Callable[[str], str] = lambda line: line) -> pl.DataFrame:
    """The synthetic export keeping only the data rows whose timestamp `keep` accepts."""
    lines = FormatParser.decode_raw_data(DEXCOM.read_bytes()).split("\n")
    out = []
    for line in lines:
        cells = line.split(",")
        stamp = cells[1].strip('"').replace("T", " ") if len(cells) > 1 else ""
        is_data = len(stamp) >= 10 and stamp[:4].isdigit() and stamp[4] == "-"
        if not is_data or keep(stamp):
            out.append(edit(line) if is_data else line)
    return FormatParser.parse_from_string("\n".join(out))


def sequenced(df: pl.DataFrame) -> pl.DataFrame:
    return FormatProcessor.detect_and_assign_sequences(df)


@pytest.fixture(scope="module")
def whole() -> pl.DataFrame:
    return sequenced(FormatParser.parse_file(DEXCOM))


def test_two_overlapping_exports_join_to_the_whole(whole: pl.DataFrame) -> None:
    early = cut(lambda s: s < "1961-04-16")
    late = cut(lambda s: s >= "1961-04-15")
    # The cut is real: each piece misses part of the whole, and they share a day.
    assert early.height < whole.height and late.height < whole.height
    assert early.height + late.height > whole.height
    assert sequenced(FormatParser.join_exports([early, late])).equals(whole)
    assert sequenced(FormatParser.join_exports([late, early])).equals(whole)


def test_sequence_ids_are_left_for_the_processor(whole: pl.DataFrame) -> None:
    joined = FormatParser.join_exports([cut(lambda s: s < "1961-04-16"), cut(lambda s: s >= "1961-04-15")])
    assert set(joined["sequence_id"].to_list()) == {0}


def test_a_disagreeing_overlap_is_counted_and_the_earlier_kept(whole: pl.DataFrame, caplog: pytest.LogCaptureFixture) -> None:
    early = cut(lambda s: s < "1961-04-16")
    reading = whole.filter(
        (pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
        & (pl.col("original_datetime").dt.date() == pl.date(1961, 4, 15))
    ).row(0, named=True)
    stamp = reading["original_datetime"].strftime("%Y-%m-%d %H:%M:%S")
    value = reading["original_glucose"]

    def change(line: str) -> str:
        return line.replace(f",{int(value)},", f",{int(value) + 7},", 1) if stamp in line else line

    late = cut(lambda s: s >= "1961-04-15", change)
    with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
        joined = FormatParser.join_exports([late, early])
    assert any("1 of their glucose readings are missing from it or differ in value" in r.getMessage() for r in caplog.records)
    kept = joined.filter(pl.col("original_datetime") == reading["original_datetime"])["original_glucose"].to_list()
    assert kept == [value]


def test_parse_exports_refuses_a_single_path() -> None:
    with pytest.raises(TypeError, match="sequence of paths"):
        FormatParser.parse_exports(DEXCOM)
