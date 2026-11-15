"""`track_agreement` and the agreement warning on `track="mean"` (RM14, desk #108).

Expected values come from the CSV through the standard library's `statistics`,
never from the parser under test. The synthetic subjects that need a particular
shape (two sensors that agree, one paired reading, no overlap) are written here
from a committed fixture, so the shape is visible in the test that relies on it.
"""

from __future__ import annotations

import csv
import logging
import os
import statistics
from pathlib import Path
from typing import List, Tuple

import pytest

from cgm_format import (
    AGREEMENT_MIN_CORRELATION,
    AGREEMENT_SLOPE_TOLERANCE,
    FormatParser,
    TrackAgreement,
)
from cgm_format.formats.cgmacros import CGMACROS_MEAN_TRACK, CGMACROS_TRACKS
from cgm_format.interface.cgm_interface import ZeroValidInputError
from tests.paths import FIXTURES_DIR

FIXTURE_DIR = FIXTURES_DIR / "cgmacros_synthetic"
SUBJECTS = sorted(d.name for d in FIXTURE_DIR.glob("CGMacros-*") if d.is_dir())
LIBRE, DEXCOM = CGMACROS_TRACKS
LOGGER = "cgm_format.format_parser"


def _subject(name: str) -> Path:
    return FIXTURE_DIR / name / f"{name}.csv"


def _pairs(path: Path) -> Tuple[List[float], List[float]]:
    """(dexcom, libre) readings on the rows where both are present, read with `csv`."""
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    key = {c.strip(): c for c in rows[0]}
    dexcom: List[float] = []
    libre: List[float] = []
    for row in rows:
        d = (row[key["Dexcom GL"]] or "").strip()
        lb = (row[key["Libre GL"]] or "").strip()
        if d and lb:
            dexcom.append(float(d))
            libre.append(float(lb))
    return dexcom, libre


def _rewrite(tmp_path: Path, source: Path, libre_of_dexcom, keep_dexcom=lambda i: True) -> Path:
    """A copy of `source` under a CGMacros name, its Libre column rewritten from Dexcom.

    `libre_of_dexcom(i, dexcom)` gives row i's Libre reading (None for an empty
    cell); `keep_dexcom(i)` empties Dexcom on the rows it rejects.
    """
    with source.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    names = [h.strip() for h in header]
    li, di = names.index("Libre GL"), names.index("Dexcom GL")
    for i, row in enumerate(rows):
        dexcom = float(row[di]) if row[di].strip() else None
        if not keep_dexcom(i):
            row[di] = ""
        value = libre_of_dexcom(i, dexcom)
        row[li] = "" if value is None else f"{value:.1f}"
    out = tmp_path / "CGMacros-090" / "CGMacros-090.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    return out


def _agreeing(tmp_path: Path) -> Path:
    """Libre tracks Dexcom one-to-one with a 3 mg/dL offset and a small, fixed wobble."""
    wobble = (0.4, -0.3, 0.2, -0.5, 0.1)
    return _rewrite(
        tmp_path,
        _subject(SUBJECTS[0]),
        lambda i, d: None if d is None else d + 3.0 + wobble[i % len(wobble)],
    )


def _agreement_records(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [r.getMessage() for r in caplog.records if "agreement of" in r.getMessage()]


def _finding(message: str) -> str:
    """The shortfalls a warning names, without the advice that follows them."""
    return message.split("outside the cut-offs:")[1].split(". The mean")[0]


def _fit(path: Path) -> Tuple[float, float]:
    """(slope, r) of a subject whose overlap defines both."""
    a = FormatParser.track_agreement(path)
    assert a.slope is not None and a.correlation is not None
    return a.slope, a.correlation


class TestTheMeasure:
    """`track_agreement` reports the fit, computed independently here."""

    @pytest.mark.parametrize("name", SUBJECTS)
    def test_it_matches_a_least_squares_fit_of_the_csv(self, name: str) -> None:
        dexcom, libre = _pairs(_subject(name))
        slope, intercept = statistics.linear_regression(dexcom, libre)
        diffs = [lb - d for d, lb in zip(dexcom, libre)]

        got = FormatParser.track_agreement(_subject(name))

        assert (got.reference, got.comparand) == (DEXCOM, LIBRE)
        assert got.overlap == len(dexcom)
        assert got.slope == pytest.approx(slope, rel=1e-9)
        assert got.intercept == pytest.approx(intercept, rel=1e-9)
        assert got.correlation == pytest.approx(statistics.correlation(dexcom, libre), rel=1e-9)
        assert got.bias == pytest.approx(statistics.fmean(diffs), rel=1e-9)
        assert got.mean_abs_difference == pytest.approx(
            statistics.fmean(abs(x) for x in diffs), rel=1e-9
        )

    def test_swapping_the_tracks_refits_the_other_way(self) -> None:
        path = _subject(SUBJECTS[0])
        dexcom, libre = _pairs(path)
        forward = FormatParser.track_agreement(path)
        backward = FormatParser.track_agreement(path, reference=LIBRE, comparand=DEXCOM)

        assert backward.slope == pytest.approx(statistics.linear_regression(libre, dexcom)[0], rel=1e-9)
        assert backward.correlation == pytest.approx(forward.correlation, rel=1e-9)
        assert backward.bias == pytest.approx(-forward.bias, rel=1e-9)
        assert backward.mean_abs_difference == pytest.approx(forward.mean_abs_difference, rel=1e-9)

    def test_it_is_a_frozen_measurement_with_no_verdict(self) -> None:
        got = FormatParser.track_agreement(_subject(SUBJECTS[0]))
        with pytest.raises(AttributeError):
            got.slope = 1.0  # type: ignore[misc]
        # Desk #108 chose cut-offs as settings, not a yes/no answer on the result.
        assert not any(isinstance(getattr(got, f), bool) for f in TrackAgreement.__dataclass_fields__)

    def test_one_paired_reading_is_unknown_not_agreement(self, tmp_path: Path) -> None:
        path = _rewrite(tmp_path, _subject(SUBJECTS[0]), lambda i, d: d if i == 0 else None)
        got = FormatParser.track_agreement(path)
        assert got.overlap == 1
        assert (got.slope, got.intercept, got.correlation) == (None, None, None)

    def test_tracks_that_never_meet_are_refused(self, tmp_path: Path) -> None:
        path = _rewrite(
            tmp_path, _subject(SUBJECTS[0]),
            lambda i, d: 100.0 if i % 2 else None,
            keep_dexcom=lambda i: i % 2 == 0,
        )
        with pytest.raises(ZeroValidInputError, match="never read at the same timestamp"):
            FormatParser.track_agreement(path)

    def test_a_single_track_format_is_refused(self) -> None:
        with pytest.raises(NotImplementedError, match="single-track"):
            FormatParser.track_agreement(FIXTURES_DIR / "Clarity_Export_synthetic.csv")

    @pytest.mark.parametrize(
        "reference, comparand",
        [("dexcom", "dexcom"), ("mean", "libre"), ("dexcom", "medtronic")],
    )
    def test_track_names_are_checked(self, reference: str, comparand: str) -> None:
        with pytest.raises(ValueError):
            FormatParser.track_agreement(_subject(SUBJECTS[0]), reference, comparand)


class TestTheWarningOnMean:
    """`track="mean"` warns on disagreement, by cut-offs the caller may set, and never refuses."""

    def test_sensors_that_agree_raise_no_warning(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _agreeing(tmp_path)
        slope, r = _fit(path)
        assert abs(slope - 1) <= AGREEMENT_SLOPE_TOLERANCE
        assert r >= AGREEMENT_MIN_CORRELATION

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)
        assert _agreement_records(caplog) == []

    def test_a_stricter_slope_cut_off_flags_the_same_pair(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _agreeing(tmp_path)
        slope, _ = _fit(path)
        tolerance = abs(slope - 1) / 2

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK, slope_tolerance=tolerance)
        (message,) = _agreement_records(caplog)
        assert _finding(message).startswith(f" slope {slope:.2f} (not within 1 ± {tolerance:g})")
        assert "under" not in _finding(message)

    def test_a_low_correlation_is_named_with_its_cut_off(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # The one committed subject whose slope is near 1: only r falls short.
        path = _subject("CGMacros-005")
        slope, r = _fit(path)
        assert abs(slope - 1) <= AGREEMENT_SLOPE_TOLERANCE
        assert r < AGREEMENT_MIN_CORRELATION

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)
        (message,) = _agreement_records(caplog)
        assert f"r {r:.2f} (under {AGREEMENT_MIN_CORRELATION:g})" in _finding(message)
        assert "slope" not in _finding(message)

    def test_a_caller_cut_off_silences_it(self, caplog: pytest.LogCaptureFixture) -> None:
        path = _subject("CGMacros-005")
        _, r = _fit(path)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK, min_correlation=r - 0.01)
        assert _agreement_records(caplog) == []

    def test_the_warning_never_changes_the_frame(self) -> None:
        path = _subject("CGMacros-005")
        warned = FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)
        silent = FormatParser.parse_tracks(
            path, track=CGMACROS_MEAN_TRACK, slope_tolerance=10.0, min_correlation=-1.0
        )
        assert warned[CGMACROS_MEAN_TRACK].equals(silent[CGMACROS_MEAN_TRACK])

    def test_an_unmeasurable_overlap_is_said_as_unknown(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _rewrite(tmp_path, _subject(SUBJECTS[0]), lambda i, d: d if i == 0 else None)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)
        (message,) = _agreement_records(caplog)
        assert "not measurable: 1 paired reading" in message

    def test_no_overlap_averages_nothing_and_says_nothing(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _rewrite(
            tmp_path, _subject(SUBJECTS[0]),
            lambda i, d: 100.0 if i % 2 else None,
            keep_dexcom=lambda i: i % 2 == 0,
        )
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)
        assert _agreement_records(caplog) == []

    @pytest.mark.parametrize("track", [None, LIBRE, DEXCOM])
    def test_only_the_mean_is_checked(self, track, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_tracks(_subject("CGMacros-005"), track=track)
        assert _agreement_records(caplog) == []

    @pytest.mark.parametrize("kwargs", [{"slope_tolerance": -0.1}, {"min_correlation": 1.5}])
    def test_a_cut_off_out_of_range_is_refused(self, kwargs) -> None:
        with pytest.raises(ValueError):
            FormatParser.parse_tracks(_subject(SUBJECTS[0]), track=CGMACROS_MEAN_TRACK, **kwargs)


class TestTheCorpus:
    def test_the_corpus_says_it_once_with_a_count(self, caplog: pytest.LogCaptureFixture) -> None:
        fits = {s: _fit(_subject(s)) for s in SUBJECTS}
        flagged = [
            s for s, (slope, r) in fits.items()
            if abs(slope - 1) > AGREEMENT_SLOPE_TOLERANCE or r < AGREEMENT_MIN_CORRELATION
        ]
        assert flagged, "the committed subjects were all expected to disagree"

        with caplog.at_level(logging.WARNING, logger=LOGGER):
            FormatParser.parse_corpus(FIXTURE_DIR, track=CGMACROS_MEAN_TRACK)
        (message,) = _agreement_records(caplog)
        assert f"flagged for {len(flagged)} of {len(SUBJECTS)} subject(s)" in message
        assert all(f"{s}: " in message for s in flagged)

    def test_the_published_corpus_reproduces_s1(self) -> None:
        """S1's independent table: of 45 subjects only 033 and 035 sit inside
        slope 0.9-1.1 with r > 0.9, and 013 is the worst at slope 0.10, r 0.10."""
        configured = os.environ.get("CGM_FORMAT_CGMACROS_DIR", "").strip()
        if not configured or not Path(configured).is_dir():
            pytest.skip("CGM_FORMAT_CGMACROS_DIR is not set to the CGMacros corpus")
        root = Path(configured)
        found = {
            d.name: _fit(d / f"{d.name}.csv")
            for d in sorted(root.glob("CGMacros-*"))
            if (d / f"{d.name}.csv").is_file()
        }
        inside = sorted(
            s for s, (slope, r) in found.items()
            if abs(slope - 1) <= AGREEMENT_SLOPE_TOLERANCE and r >= AGREEMENT_MIN_CORRELATION
        )
        assert len(found) == 45
        assert inside == ["CGMacros-033", "CGMacros-035"]
        assert found["CGMacros-013"] == pytest.approx((0.10, 0.10), abs=0.005)
