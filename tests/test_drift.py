"""RM22: exports of formats the library claims, in the layouts real files were seen in.

Found by the format survey (docs/FORMAT_SURVEY.md §1), each seen failing on a public sample. The
fixtures here are the committed synthetic exports reshaped at test time into that layout: never a copy
of a person's file. Each case asserts the reshaped export parses to the same frame as the original.
"""

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from cgm_format import FormatParser

DATA_DIR = Path(__file__).parent.parent / "data" / "input"
LIBRE = DATA_DIR / "FreeStyle_Libre_3_synthetic.csv"


def rewrite_libre_timestamps(text: str, to_format: str) -> str:
    """The synthetic LibreView export with every `Device Timestamp` (day-first, 24-hour) rewritten."""
    lines = text.split("\n")
    out = lines[:2]
    for line in lines[2:]:
        cells = line.split(",")
        if len(cells) > 2 and cells[2]:
            cells[2] = datetime.strptime(cells[2], "%d-%m-%Y %H:%M").strftime(to_format)
        out.append(",".join(cells))
    return "\n".join(out)


@pytest.mark.skipif(not LIBRE.exists(), reason="synthetic Libre export absent")
def test_libreview_us_12_hour_timestamps() -> None:
    """US Libre 3 exports write `MM-DD-YYYY hh:mm AM` (seen on a public LibreView export)."""
    text = FormatParser.decode_raw_data(LIBRE.read_bytes())
    us = rewrite_libre_timestamps(text, "%m-%d-%Y %I:%M %p")
    assert " PM," in us and " AM," in us  # the rewrite is real
    assert FormatParser.parse_from_string(us).equals(FormatParser.parse_from_string(text))
