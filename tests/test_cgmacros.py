"""CGMacros: a multi-track, many-subject research corpus.

Exercises the two things CGMacros is here to prove — that faceted output works
(one file in, two frames out; many subjects in, a keyed mapping out) and that
the header drift across 45 real subjects is absorbed declaratively rather than
by a branch per variant.

Real-corpus tests locate the data through `CGM_FORMAT_CGMACROS_DIR` and skip
when it is unset or absent, following `tests/test_libre_eu.py`. They must never
hardcode the sibling-repo path where the corpus currently happens to live. The
synthetic fixtures reproduce the *dirt* rather than the happy path, so CI
exercises the drift on every run even without the licensed corpus.
"""

import csv
import os
from pathlib import Path

import polars as pl
import pytest

from cgm_format import (
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
    FormatCategory,
    FormatParser,
    Quality,
    UnifiedEventType,
)
from cgm_format.formats.cgmacros import (
    CGMACROS_MEAN_TRACK,
    CGMACROS_METS_SCALE,
    CGMACROS_TRACKS,
    CGMacrosColumn,
)
from cgm_format.formats.supported import FORMAT_CATEGORY, PATH_DETECTION_PROBES
from cgm_format.interface.cgm_interface import (
    MultiTrackSourceError,
    SupportedCGMFormat,
)

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "input" / "cgmacros_synthetic"


def _corpus_root() -> Path:
    """The real corpus, by configuration — never a hardcoded sibling path."""
    configured = os.environ.get("CGM_FORMAT_CGMACROS_DIR", "").strip()
    if not configured:
        pytest.skip("CGM_FORMAT_CGMACROS_DIR is not set")
    root = Path(configured)
    if not root.is_dir():
        pytest.skip(f"CGMacros corpus not found at {root}")
    return root


def _first_real_subject() -> Path:
    root = _corpus_root()
    subjects = sorted(d for d in root.glob("CGMacros-*") if d.is_dir())
    if not subjects:
        pytest.skip(f"No subject directories under {root}")
    return subjects[0] / f"{subjects[0].name}.csv"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")


class TestRegistration:
    """CGMacros is the first non-EXPORT format, and the registries must say so."""

    def test_it_is_a_corpus_not_an_export(self) -> None:
        assert FORMAT_CATEGORY[SupportedCGMFormat.CGMACROS] is FormatCategory.CORPUS

    def test_it_targets_the_extended_schema(self) -> None:
        """Macronutrients and heart rate have no core home."""
        from cgm_format.formats.supported import UNIFIED_TARGET_SCHEMA

        assert (
            UNIFIED_TARGET_SCHEMA[SupportedCGMFormat.CGMACROS] is CGM_SCHEMA_EXTENDED
        )

    def test_it_registers_path_probes(self) -> None:
        """A corpus is identified by directory shape, not by a text prefix."""
        assert PATH_DETECTION_PROBES[SupportedCGMFormat.CGMACROS]


class TestSyntheticDirt:
    """The committed fixtures reproduce the variants that break a naive parser."""

    def test_the_canonical_variant_parses(self) -> None:
        _skip_if_missing(FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv")

        tracks = FormatParser.parse_tracks(
            FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        )

        assert set(tracks) == set(CGMACROS_TRACKS)

    def test_the_intensity_variant_is_absorbed_by_an_alias(self) -> None:
        """11 of 45 real subjects spell `METs` as `Intensity`.

        Asserting the *values land in `mets`* rather than that parsing merely
        succeeded: an alias that renamed the column but dropped its data would
        pass a smoke test.
        """
        path = FIXTURE_DIR / "CGMacros-002" / "CGMacros-002.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]

        assert frame["mets"].null_count() < len(frame)
        # Header says Intensity; the unified frame has no such column.
        assert "Intensity" not in frame.columns

    def test_a_subject_without_amount_consumed_still_parses(self) -> None:
        """2 of 45 real subjects lack the column entirely.

        This is the `ColumnNotFound` gotcha: a `.select(pl.col(X))` is
        evaluated even when an upstream filter leaves zero rows, so an absent
        column raises regardless of whether any row would have used it.
        """
        path = FIXTURE_DIR / "CGMacros-003" / "CGMacros-003.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]

        meals = frame.filter(pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value)
        assert len(meals) > 0
        # The annotation records the absence honestly rather than inventing 0.
        assert all("null" in a or "amount_consumed" in a for a in meals["annotations"])

    def test_a_trailing_space_in_a_header_is_stripped_then_aliased(self) -> None:
        """`Amount Consumed ` — one real subject. Strip must precede aliasing."""
        path = FIXTURE_DIR / "CGMacros-004" / "CGMacros-004.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]
        meals = frame.filter(pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value)

        assert len(meals) > 0
        assert any('"amount_consumed":' in a for a in meals["annotations"])

    def test_index_and_unknown_columns_are_dropped_not_smuggled(self) -> None:
        """`Unnamed: 0`, `RecordIndex` and `Sugar` have no unified home.

        Dropping is deliberate: a row index means nothing outside its own file,
        and `Sugar` is a macronutrient the extended schema does not declare.
        Hiding them in `annotations` would be worse than dropping them.
        """
        path = FIXTURE_DIR / "CGMacros-005" / "CGMacros-005.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]

        assert frame.columns == CGM_SCHEMA_EXTENDED.get_column_names()
        annotations = " ".join(a for a in frame["annotations"] if a)
        assert "Sugar" not in annotations
        assert "RecordIndex" not in annotations


class TestMultiTrackRefusal:
    """`parse_file` must not silently pick a sensor (design decision D5)."""

    def test_parse_file_refuses_and_names_both_tracks(self) -> None:
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        with pytest.raises(MultiTrackSourceError) as excinfo:
            FormatParser.parse_file(path)

        message = str(excinfo.value)
        for track in CGMACROS_TRACKS:
            assert track in message
        assert "parse_tracks" in message

    def test_single_track_formats_are_unaffected(self) -> None:
        """The refusal must not leak into the export path."""
        dexcom = Path(__file__).parent.parent / "data" / "input" / "Clarity_Export_synthetic.csv"
        _skip_if_missing(dexcom)

        frame = FormatParser.parse_file(dexcom)

        CGM_SCHEMA.validate_dataframe(frame, enforce=False)

    def test_parse_tracks_refuses_a_single_track_format(self) -> None:
        """A one-entry mapping would blur the distinction the categories draw."""
        dexcom = Path(__file__).parent.parent / "data" / "input" / "Clarity_Export_synthetic.csv"
        _skip_if_missing(dexcom)

        with pytest.raises(NotImplementedError, match="single-track"):
            FormatParser.parse_tracks(dexcom)


class TestTracksAreAlternativesNotShards:
    """The design's most likely misuse, quantified rather than only documented."""

    def test_concatenating_tracks_exactly_doubles_the_meals(self) -> None:
        """Both totals computed from the frames at runtime, not hardcoded.

        A consumer who reasonably assumes tracks are shards and stacks them
        gets carbohydrate totals that are exactly twice reality, with nothing
        raised anywhere. That is why the docstrings say pick one or compare
        them, never add them.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        tracks = FormatParser.parse_tracks(path)
        libre, dexcom = tracks[CGMACROS_TRACKS[0]], tracks[CGMACROS_TRACKS[1]]

        meals_of = lambda df: df.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        libre_meals, dexcom_meals = meals_of(libre), meals_of(dexcom)

        # Replicated, not sharded: each track carries every meal.
        assert len(libre_meals) == len(dexcom_meals)
        assert libre_meals["carbs"].sum() == dexcom_meals["carbs"].sum()

        stacked = meals_of(pl.concat([libre, dexcom], how="diagonal"))
        assert len(stacked) == 2 * len(libre_meals)
        assert stacked["carbs"].sum() == 2 * libre_meals["carbs"].sum()

    def test_annotation_rows_are_replicated_into_both_tracks(self) -> None:
        """Photo-only rows belong to neither sensor, so both tracks carry them.

        Filtered on a populated `annotations` cell rather than on `OTHEREVT`
        alone: wearable-only rows share that event type and carry no
        annotation, so an event-type filter would compare two different kinds
        of row.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        tracks = FormatParser.parse_tracks(path)
        photo_annotations_of = lambda df: set(
            df.filter(
                (pl.col("event_type") == UnifiedEventType.OTHER.value)
                & pl.col("annotations").is_not_null()
            )["annotations"].to_list()
        )

        assert photo_annotations_of(tracks[CGMACROS_TRACKS[0]]) == (
            photo_annotations_of(tracks[CGMACROS_TRACKS[1]])
        )

    def test_the_wearable_stream_is_replicated_into_both_tracks(self) -> None:
        """The wrist wearable is not a glucose sensor.

        Heart rate, METs and activity calories belong to the subject, not to
        Libre or Dexcom, and D4 lists them among the rows replicated into every
        track. Before this was fixed, a timestamp where *this* track's sensor
        was null contributed no row at all and its wearable sample vanished —
        about 8% of the stream on the dexcom track, since `Dexcom GL` is
        populated on roughly 92% of rows, with nothing raised or logged.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        tracks = FormatParser.parse_tracks(path)
        wearable_moments_of = lambda df: set(
            df.filter(pl.col("heart_rate").is_not_null())["datetime"].to_list()
        )

        libre = wearable_moments_of(tracks[CGMACROS_TRACKS[0]])
        dexcom = wearable_moments_of(tracks[CGMACROS_TRACKS[1]])

        assert libre == dexcom, (
            "wearable samples differ between tracks: "
            f"{len(libre ^ dexcom)} timestamp(s) present in one but not the other"
        )

        # And every wearable sample in the source survived into both.
        with open(path, encoding="utf-8-sig") as fh:
            source_moments = {
                r[CGMacrosColumn.TIMESTAMP.value]
                for r in csv.DictReader(fh)
                if (r.get(CGMacrosColumn.HEART_RATE.value) or "").strip()
            }
        assert {m.strftime("%Y-%m-%d %H:%M:%S") for m in libre} == source_moments


class TestMeanTrackIsAuditable:
    """The one place this branch synthesizes a glucose value (D5a)."""

    def test_the_mean_track_is_opt_in_only(self) -> None:
        """It is a derived view, never a member of the corpus."""
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        assert CGMACROS_MEAN_TRACK not in FormatParser.parse_tracks(path)
        assert CGMACROS_MEAN_TRACK in FormatParser.parse_tracks(
            path, track=CGMACROS_MEAN_TRACK
        )

    def test_every_flagged_row_lies_between_its_two_sources(self) -> None:
        """Asserts the relationship, not a count.

        Expected values come from the raw CSV read independently — never from
        running the parser under test.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        mean_frame = FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)[
            CGMACROS_MEAN_TRACK
        ]
        glucose = mean_frame.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        flagged = glucose.filter(
            (pl.col("quality") & Quality.TRACK_MERGE.value) != 0
        )
        unflagged = glucose.filter(
            (pl.col("quality") & Quality.TRACK_MERGE.value) == 0
        )

        with open(path, encoding="utf-8-sig") as fh:
            raw = list(csv.DictReader(fh))
        both = [
            r
            for r in raw
            if r[CGMacrosColumn.LIBRE_GLUCOSE.value].strip()
            and r[CGMacrosColumn.DEXCOM_GLUCOSE.value].strip()
        ]
        exactly_one = [
            r
            for r in raw
            if bool(r[CGMacrosColumn.LIBRE_GLUCOSE.value].strip())
            ^ bool(r[CGMacrosColumn.DEXCOM_GLUCOSE.value].strip())
        ]

        # Flagged rows are exactly the two-sensor rows; single-sensor rows are
        # that sensor's real reading and must NOT be flagged.
        assert len(flagged) == len(both)
        assert len(unflagged) == len(exactly_one)

        # Paired by timestamp, never by sorting both sides independently:
        # sorting would compare a row's value against some other row's bounds
        # and pass or fail for reasons unrelated to the merge.
        bounds = {
            r[CGMacrosColumn.TIMESTAMP.value]: (
                min(
                    float(r[CGMacrosColumn.LIBRE_GLUCOSE.value]),
                    float(r[CGMacrosColumn.DEXCOM_GLUCOSE.value]),
                ),
                max(
                    float(r[CGMacrosColumn.LIBRE_GLUCOSE.value]),
                    float(r[CGMacrosColumn.DEXCOM_GLUCOSE.value]),
                ),
            )
            for r in both
        }
        for moment, value in zip(
            flagged["datetime"].to_list(), flagged["glucose"].to_list()
        ):
            low, high = bounds[moment.strftime("%Y-%m-%d %H:%M:%S")]
            assert low <= value <= high, moment

    def test_the_merge_flag_survives_a_round_trip(self) -> None:
        """A synthesized value must stay distinguishable after serialization."""
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        original = FormatParser.parse_tracks(path, track=CGMACROS_MEAN_TRACK)[
            CGMACROS_MEAN_TRACK
        ]
        reparsed = FormatParser.parse_from_string(
            FormatParser.to_csv_string(original)
        )

        flagged = lambda df: len(
            df.filter((pl.col("quality") & Quality.TRACK_MERGE.value) != 0)
        )
        assert flagged(reparsed) == flagged(original)
        assert flagged(original) > 0

    def test_an_unknown_track_is_refused(self) -> None:
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        with pytest.raises(ValueError, match="Unknown track"):
            FormatParser.parse_tracks(path, track="fitbit")


class TestUnitsAndEventTypes:
    """Conventions that would silently corrupt values if wrong."""

    def test_mets_is_divided_by_ten(self) -> None:
        """Stored x10 per the data dictionary; a physiological MET is ~1-13.

        The expected value is read from the raw CSV, not from the parser.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]

        with open(path, encoding="utf-8-sig") as fh:
            raw_mets = [
                float(r[CGMacrosColumn.METS.value])
                for r in csv.DictReader(fh)
                if r.get(CGMacrosColumn.METS.value, "").strip()
            ]
        pytest.importorskip("statistics")
        assert raw_mets, "fixture carries no METs values"

        parsed = [v for v in frame["mets"].to_list() if v is not None]
        assert max(parsed) == pytest.approx(max(raw_mets) / CGMACROS_METS_SCALE)

    def test_annotation_only_rows_use_otherevt(self) -> None:
        """The meal-END photo: a real row with no meal and no glucose event.

        Across the real corpus these are the *majority* of photo rows, which is
        why annotations cannot simply hang off a CARBS_IN event.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]
        annotation_rows = frame.filter(
            pl.col("event_type") == UnifiedEventType.OTHER.value
        )

        assert len(annotation_rows) > 0
        # An annotation-only row carries no meal and no reading — it survives
        # postprocessing on the strength of its annotation alone.
        assert annotation_rows["carbs"].null_count() == len(annotation_rows)
        assert annotation_rows["glucose"].null_count() == len(annotation_rows)
        assert annotation_rows["annotations"].null_count() == 0

    def test_meal_labels_are_normalized_but_the_raw_string_is_kept(self) -> None:
        """Ten raw spellings for four meals; normalization stays inspectable."""
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )

        assert len(meals) > 0
        for annotation in meals["annotations"]:
            assert '"meal_type_raw":' in annotation
            assert '"meal_type":' in annotation

    def test_the_parser_does_not_resample(self) -> None:
        """Native cadence is 1 minute; regridding is the processor's job.

        Every source timestamp must survive, and none may be invented.
        """
        path = FIXTURE_DIR / "CGMacros-001" / "CGMacros-001.csv"
        _skip_if_missing(path)

        frame = FormatParser.parse_tracks(path)[CGMACROS_TRACKS[0]]

        with open(path, encoding="utf-8-sig") as fh:
            raw_timestamps = {
                r[CGMacrosColumn.TIMESTAMP.value]
                for r in csv.DictReader(fh)
                if r[CGMacrosColumn.TIMESTAMP.value].strip()
            }

        parsed = {
            d.strftime("%Y-%m-%d %H:%M:%S") for d in frame["datetime"].to_list()
        }
        assert parsed <= raw_timestamps, "parser invented timestamps"


class TestRealCorpus:
    """Against all 45 subjects, when the licensed corpus is available."""

    def test_the_corpus_root_is_detected_by_directory_shape(self) -> None:
        root = _corpus_root()

        assert (
            FormatParser.detect_path_format(root) == SupportedCGMFormat.CGMACROS
        )

    def test_every_subject_directory_yields_a_frame_per_track(self) -> None:
        """Subjects are enumerated from the filesystem, never a numeric range.

        The real corpus runs 001-049 with gaps at 024, 025, 037 and 040, so a
        range would both miss subjects and look for ones that do not exist.
        """
        root = _corpus_root()
        expected_subjects = {
            d.name for d in root.glob("CGMacros-*") if (d / f"{d.name}.csv").exists()
        }

        corpus = FormatParser.parse_corpus(root)

        assert {key.split("/")[0] for key in corpus} == expected_subjects
        assert len(corpus) == len(expected_subjects) * len(CGMACROS_TRACKS)

    def test_every_real_header_variant_parses(self) -> None:
        """9 distinct headers across 45 subjects — drift within one release.

        The variants are discovered from the files rather than listed here, so
        this keeps testing whatever the corpus actually contains.
        """
        root = _corpus_root()
        by_header: dict[str, Path] = {}
        for subject in sorted(root.glob("CGMacros-*")):
            csv_path = subject / f"{subject.name}.csv"
            if not csv_path.exists():
                continue
            with open(csv_path, encoding="utf-8-sig") as fh:
                by_header.setdefault(fh.readline().strip(), csv_path)

        assert len(by_header) > 1, "corpus shows no header drift; check the fixture"

        for header, csv_path in by_header.items():
            tracks = FormatParser.parse_tracks(csv_path)
            assert set(tracks) == set(CGMACROS_TRACKS), header
            for frame in tracks.values():
                assert frame.columns == CGM_SCHEMA_EXTENDED.get_column_names(), header

    def test_corpus_keys_use_the_documented_separator(self) -> None:
        """`/` is public contract; a subject id may contain `_` but never `/`."""
        root = _corpus_root()

        corpus = FormatParser.parse_corpus(root)

        for key in corpus:
            subject, _, track = key.partition("/")
            assert track in CGMACROS_TRACKS
            assert "/" not in subject
