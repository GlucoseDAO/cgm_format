"""BIG IDEAs: Dexcom Clarity export plus a food log, one subject per directory.

PhysioNet `big-ideas-glycemic-wearable`. The Dexcom file is a Clarity export,
so `parse_file` on it alone is DEXCOM — the corpus identity is directory
shape. Real-corpus tests locate the data through `CGM_FORMAT_BIGIDEAS_DIR`
and skip when it is unset; they must never hardcode the sibling-repo path.
Obtain the extract with `scripts/download_bigideas.py` (PhysioNet), not a
local sugar-sugar checkout.

The committed fixtures reproduce the dirt that breaks a parser written
against subject 001:

- `001` — canonical 14-col food header, plus a blank `time_begin` that still
  has `date` + `time` (the Boost row from real subject 012)
- `003` — headerless 11-col food log
- `007` — `time` renamed `time_of_day`, US date in the `date` column
"""

import csv
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import polars as pl
import pytest

from cgm_format import (
    BIGIDEAS_TRACK,
    CGM_SCHEMA_EXTENDED,
    FormatCategory,
    FormatParser,
    MalformedDataError,
    UnifiedEventType,
    UnknownFormatError,
)
from cgm_format.formats.supported import (
    FORMAT_CATEGORY,
    PATH_DETECTION_PROBES,
    SCHEMA_MAP,
    SUBJECT_PATH_PROBES,
    UNIFIED_TARGET_SCHEMA,
)
from cgm_format.interface.cgm_interface import SupportedCGMFormat

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "input" / "bigideas_synthetic"
SUBJECT_001 = FIXTURE_DIR / "001"
SUBJECT_003 = FIXTURE_DIR / "003"
SUBJECT_007 = FIXTURE_DIR / "007"


def _real_corpus() -> Path:
    configured = os.environ.get("CGM_FORMAT_BIGIDEAS_DIR", "").strip()
    if not configured:
        pytest.skip("CGM_FORMAT_BIGIDEAS_DIR is not set")
    root = Path(configured)
    if not root.is_dir():
        pytest.skip(f"BIG IDEAs corpus not found at {root}")
    return root


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")


def _frictionless(path: Path) -> Tuple[bool, str, int, int]:
    """Validate a raw file exactly the way `cgm-cli report` does.

    Returns (valid, message, residual errors, suppressed). Skips when the
    optional `frictionless` package is absent rather than reporting a clean
    run — a check that could not run is not a check that passed.
    """
    from cgm_format.cgm_cli import HAS_FRICTIONLESS, _validate_with_frictionless

    if not HAS_FRICTIONLESS:
        pytest.skip("frictionless is not installed")
    text = FormatParser.decode_raw_data(path.read_bytes())
    try:
        detected = FormatParser.detect_format(text)
    except UnknownFormatError:
        # The headerless food log (subject 003) has no header line to match,
        # so text detection cannot see it. It is still a BIG IDEAs food log —
        # that is exactly what F8 is about.
        detected = SupportedCGMFormat.BIGIDEAS
    return _validate_with_frictionless(path, detected, suppress_known=True)


class TestRegistration:
    def test_it_is_a_corpus_not_an_export(self) -> None:
        assert FORMAT_CATEGORY[SupportedCGMFormat.BIGIDEAS] is FormatCategory.CORPUS

    def test_it_targets_the_extended_schema(self) -> None:
        assert (
            UNIFIED_TARGET_SCHEMA[SupportedCGMFormat.BIGIDEAS] is CGM_SCHEMA_EXTENDED
        )

    def test_path_and_subject_probes_are_disjoint(self) -> None:
        assert PATH_DETECTION_PROBES[SupportedCGMFormat.BIGIDEAS]
        assert SUBJECT_PATH_PROBES[SupportedCGMFormat.BIGIDEAS]
        assert set(PATH_DETECTION_PROBES[SupportedCGMFormat.BIGIDEAS]) != set(
            SUBJECT_PATH_PROBES[SupportedCGMFormat.BIGIDEAS]
        )


class TestDetection:
    def test_the_corpus_root_detects_as_bigideas(self) -> None:
        _skip_if_missing(FIXTURE_DIR)
        assert (
            FormatParser.detect_path_format(FIXTURE_DIR)
            == SupportedCGMFormat.BIGIDEAS
        )

    def test_a_subject_directory_detects_as_bigideas(self) -> None:
        _skip_if_missing(SUBJECT_001)
        assert (
            FormatParser.detect_subject_format(SUBJECT_001)
            == SupportedCGMFormat.BIGIDEAS
        )

    def test_a_food_log_detects_as_bigideas_and_parse_file_refuses(self) -> None:
        """A food log is one modality; parsing it alone would drop glucose."""
        food = SUBJECT_001 / "Food_Log_001.csv"
        _skip_if_missing(food)
        text = FormatParser.decode_raw_data(food.read_bytes())
        assert FormatParser.detect_format(text) == SupportedCGMFormat.BIGIDEAS
        with pytest.raises(MalformedDataError, match="parse_bundle"):
            FormatParser.parse_file(food)

    def test_a_dexcom_file_still_detects_as_dexcom(self) -> None:
        """The Clarity export is still a Dexcom file when seen alone."""
        dexcom = SUBJECT_001 / "Dexcom_001.csv"
        _skip_if_missing(dexcom)
        text = FormatParser.decode_raw_data(dexcom.read_bytes())
        assert FormatParser.detect_format(text) == SupportedCGMFormat.DEXCOM


class TestSyntheticDirt:
    def test_the_canonical_subject_parses_glucose_and_meals(self) -> None:
        _skip_if_missing(SUBJECT_001)
        frame = FormatParser.parse_subject_directory(SUBJECT_001)
        CGM_SCHEMA_EXTENDED.validate_dataframe(frame, enforce=False)

        glucose = frame.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        raw_egv = pl.read_csv(
            SUBJECT_001 / "Dexcom_001.csv", infer_schema_length=0
        ).filter(pl.col("Event Type") == "EGV")
        raw_food = pl.read_csv(SUBJECT_001 / "Food_Log_001.csv", infer_schema_length=0)

        assert len(glucose) == len(raw_egv)
        assert len(meals) == len(raw_food)
        # Compared against the fixture's own first reading, not a copied
        # literal: a hardcoded value drifts the moment the fixture is edited.
        expected_first = float(
            raw_egv.sort("Timestamp (YYYY-MM-DDThh:mm:ss)")
            .get_column("Glucose Value (mg/dL)")
            .to_list()[0]
        )
        assert glucose.get_column("glucose").to_list()[0] == expected_first

    def test_blank_time_begin_falls_back_to_date_and_time(self) -> None:
        """The Boost row from real subject 012: empty time_begin, date+time set."""
        _skip_if_missing(SUBJECT_001)
        frame = FormatParser.parse_subject_directory(SUBJECT_001)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        times = meals.get_column("datetime").to_list()
        assert datetime(2020, 5, 11, 7, 0) in times

    def test_the_headerless_food_log_is_read(self) -> None:
        _skip_if_missing(SUBJECT_003)
        frame = FormatParser.parse_subject_directory(SUBJECT_003)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        # Every line is data — that is what "headerless" means — so the row
        # count comes from the file rather than a literal.
        raw_lines = [
            line
            for line in (SUBJECT_003 / "Food_Log_003.csv")
            .read_text(encoding="utf-8-sig")
            .splitlines()
            if line.strip()
        ]
        assert len(meals) == len(raw_lines)
        notes = [json.loads(text) for text in meals.get_column("annotations").to_list()]
        by_name = {
            note["logged_food"]: carbs
            for note, carbs in zip(notes, meals.get_column("carbs").to_list())
        }
        # The 11-column mapping is the thing under test, so the expected carb
        # value is read out of the file's own 9th field for that row.
        raw_carbs = {
            row[3]: float(row[8])
            for row in csv.reader(
                (SUBJECT_003 / "Food_Log_003.csv")
                .read_text(encoding="utf-8-sig")
                .splitlines()
            )
            if row
        }
        assert by_name == raw_carbs
        # The 11-col variant has no sugar / fat columns — keys absent, not null.
        nuggets = next(note for note in notes if note["logged_food"] == "Chicken Nuggets")
        assert "sugar" not in nuggets
        assert meals.get_column("fat").null_count() == 2

    def test_time_of_day_alias_is_absorbed(self) -> None:
        _skip_if_missing(SUBJECT_007)
        frame = FormatParser.parse_subject_directory(SUBJECT_007)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        assert len(meals) == 1
        assert meals.get_column("datetime").to_list()[0] == datetime(2020, 3, 14, 13, 44)
        assert meals.get_column("carbs").to_list()[0] == 10.0

    def test_food_items_are_not_clustered(self) -> None:
        """Two items at 18:00 stay two rows. Clustering is a consumer concern."""
        _skip_if_missing(SUBJECT_001)
        frame = FormatParser.parse_subject_directory(SUBJECT_001)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        at_six = [
            row
            for row in meals.iter_rows(named=True)
            if row["datetime"] == datetime(2020, 2, 13, 18, 0)
        ]
        assert len(at_six) == 2


class TestCorpus:
    def test_parse_corpus_keys_are_bare_subject_ids(self) -> None:
        _skip_if_missing(FIXTURE_DIR)
        frames = FormatParser.parse_corpus(FIXTURE_DIR)
        assert set(frames) == {"001", "003", "007"}
        assert "/" not in next(iter(frames))

    def test_list_subjects_ids_match_parse_corpus_keys(self) -> None:
        _skip_if_missing(FIXTURE_DIR)
        entries = FormatParser.list_subjects(FIXTURE_DIR)
        frames = FormatParser.parse_corpus(FIXTURE_DIR)
        assert [entry.subject_id for entry in entries] == sorted(frames)
        assert all(entry.tracks[0].track == BIGIDEAS_TRACK for entry in entries)

    def test_parse_bundle_on_a_subject_directory_matches_parse_subject(self) -> None:
        _skip_if_missing(SUBJECT_001)
        from_bundle = FormatParser.parse_bundle([SUBJECT_001])
        from_subject = FormatParser.parse_subject_directory(SUBJECT_001)
        assert from_bundle.equals(from_subject)

    def test_track_argument_is_refused(self) -> None:
        _skip_if_missing(FIXTURE_DIR)
        with pytest.raises(ValueError, match="single-track"):
            FormatParser.parse_corpus(FIXTURE_DIR, track="dexcom")


class TestRoundTrip:
    def test_parsed_frame_round_trips_through_unified_csv(self) -> None:
        _skip_if_missing(SUBJECT_001)
        frame = FormatParser.parse_subject_directory(SUBJECT_001)
        csv_text = FormatParser.to_csv_string(frame)
        again = FormatParser.parse_from_string(csv_text)
        assert again.equals(frame)

    def test_parse_is_idempotent(self) -> None:
        _skip_if_missing(SUBJECT_001)
        first = FormatParser.parse_subject_directory(SUBJECT_001)
        second = FormatParser.parse_subject_directory(SUBJECT_001)
        assert first.equals(second)


class TestFrictionless:
    """The fourth demonstrable property: the raw file validates.

    `docs/NEW_SCHEMA.md` step 8 — the vendor schema builds, its dialect
    matches the declared geometry, and validating the raw fixture leaves no
    error that is not a registered known issue.
    """

    def test_the_food_schema_builds_and_declares_a_plain_header(self) -> None:
        schema = SCHEMA_MAP[SupportedCGMFormat.BIGIDEAS]
        fields = schema.to_frictionless_schema()["fields"]
        assert [f["name"] for f in fields] == list(schema.get_column_names())
        # Header on line 1, no metadata rows, so no dialect is needed.
        assert schema.get_dialect() is None

    def test_a_canonical_food_log_validates_with_nothing_left_over(self) -> None:
        _skip_if_missing(SUBJECT_001)
        _, _, residual, _ = _frictionless(SUBJECT_001 / "Food_Log_001.csv")
        assert residual == 0

    def test_the_time_of_day_variant_validates_via_its_alias(self) -> None:
        """The renamed header is absorbed by alias data, not a suppression."""
        _skip_if_missing(SUBJECT_007)
        valid, _, residual, suppressed = _frictionless(SUBJECT_007 / "Food_Log_007.csv")
        assert residual == 0
        assert valid
        # The alias-driven incorrect-label suppression is what makes it clean.
        assert suppressed >= 1

    def test_the_headerless_variant_is_reported_unvalidatable_not_clean(self) -> None:
        """Subject 003 has no header, so a header-based schema cannot judge it.

        Recorded as a known limitation rather than suppressed: every column
        label and every row type is wrong against a 14-column header schema,
        and blanket-suppressing that would make a genuinely broken file look
        clean. `CLAUDE.md` §5 — a check that could not run is not a check that
        passed. See F8 in `docs/dogfooding.md`.
        """
        _skip_if_missing(SUBJECT_003)
        valid, _, residual, _ = _frictionless(SUBJECT_003 / "Food_Log_003.csv")
        assert not valid
        assert residual > 0


class TestPipeline:
    def test_extended_processor_runs_on_a_parsed_subject(self) -> None:
        from cgm_format import ExtendedFormatProcessor

        _skip_if_missing(SUBJECT_001)
        frame = FormatParser.parse_subject_directory(SUBJECT_001)
        sequenced = ExtendedFormatProcessor.detect_and_assign_sequences(frame)
        interpolated = ExtendedFormatProcessor.interpolate_gaps(sequenced)
        assert interpolated.height >= frame.height

    def test_meals_join_the_glucose_sequence_they_sit_inside(self) -> None:
        """The property `interpolated.height >= frame.height` cannot see.

        Meals carry the macronutrients this corpus was added for. A meal left
        at `sequence_id == 0` is still in the frame but invisible to every
        sequence-scoped consumer, so assert the relationship rather than a
        row count.
        """
        from cgm_format import ExtendedFormatProcessor

        _skip_if_missing(SUBJECT_001)
        frame = FormatParser.parse_subject_directory(SUBJECT_001)
        sequenced = ExtendedFormatProcessor.detect_and_assign_sequences(frame)
        glucose = sequenced.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        meals = sequenced.filter(
            (pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value)
            & (pl.col("datetime") >= glucose.get_column("datetime").min())
            & (pl.col("datetime") <= glucose.get_column("datetime").max())
        )
        assert meals.height > 0, "fixture has no meal inside the glucose span"
        assert meals.filter(pl.col("sequence_id") == 0).height == 0


class TestRealCorpus:
    def test_the_dexcom_residual_is_only_the_two_known_conditions(self) -> None:
        """Pins F7 and F9 so neither can quietly grow a third condition.

        Two errors survive suppression on these exports, both understood:

        - `Transmitter ID` is absent from the header. Frictionless reports a
          `missing-label`, which carries no `fieldName`, so no bounded rule can
          name the column (F7).
        - The blank-timestamp `constraint-error` rule caps at 1, and 4 of the
          16 exports carry two such rows, so the second is not suppressed (F9).

        The expected count is therefore derived from each file's own blank-row
        count rather than assumed, and anything beyond it is new drift.
        """
        root = _real_corpus()
        checked = 0
        for subject_dir in sorted(p for p in root.glob("0*") if p.is_dir()):
            dexcom = next(iter(sorted(subject_dir.glob("Dexcom_*.csv"))), None)
            if dexcom is None:
                continue
            raw = pl.read_csv(
                dexcom, skip_rows_after_header=10, truncate_ragged_lines=True,
                infer_schema_length=None,
            )
            raw = raw.rename({c: c.strip() for c in raw.columns})
            stamp = pl.col("Timestamp (YYYY-MM-DDThh:mm:ss)")
            blanks = raw.filter(
                stamp.is_null() | (stamp.cast(pl.Utf8).str.strip_chars() == "")
            ).height
            expected = 1 + max(0, blanks - 1)

            _, _, residual, suppressed = _frictionless(dexcom)
            assert residual == expected, (
                f"{subject_dir.name}: expected {expected} residual error(s) "
                f"(missing Transmitter ID label, plus {max(0, blanks - 1)} "
                f"uncapped blank timestamp), got {residual}"
            )
            assert suppressed > 0
            checked += 1
        assert checked > 0, "no Dexcom files found in the configured extract"

    def test_every_published_subject_parses(self) -> None:
        root = _real_corpus()
        frames = FormatParser.parse_corpus(root)
        assert frames, "published corpus produced no frames"
        for subject_id, frame in frames.items():
            CGM_SCHEMA_EXTENDED.validate_dataframe(frame, enforce=False)
            glucose = frame.filter(
                pl.col("event_type") == UnifiedEventType.GLUCOSE.value
            )
            assert len(glucose) > 0, f"{subject_id} has no EGV rows"

    def test_headerless_subject_003_has_meals(self) -> None:
        root = _real_corpus()
        subject = root / "003"
        if not subject.is_dir():
            pytest.skip("subject 003 is not in the configured extract")
        frame = FormatParser.parse_subject_directory(subject)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        raw_lines = (subject / "Food_Log_003.csv").read_text(
            encoding="utf-8"
        ).splitlines()
        assert len(meals) == len(raw_lines)
