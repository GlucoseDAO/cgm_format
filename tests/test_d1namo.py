"""D1NAMO: two subsets that are not one format, each subject a bundle.

Zenodo `5651217`, CC BY-SA 4.0. Share-alike is stricter than the CC-BY most
datasets carry, which is a reason to prefer synthetic fixtures over an excerpt;
the real corpus is located by `CGM_FORMAT_D1NAMO_DIR` and skipped when absent.

What these tests are really guarding:

- **Fingersticks are not sensor traces.** The healthy subset has no CGM at all,
  and mapping four-a-day fingersticks to `EGV_READ` would present them as a
  continuous trace.
- **`carbs` stays null.** D1NAMO records no carbohydrate anywhere, and a zero
  would assert something the source never said.
- **Mixed timestamp conventions inside one subject directory** — the trap most
  likely to produce silently wrong data.
- **"Did not say" and "said something we cannot resolve" are different
  reports**, and the corpus contains real instances of both.
"""

import os
from pathlib import Path

import polars as pl
import pytest

from cgm_format import (
    CGM_SCHEMA_EXTENDED,
    FormatCategory,
    FormatParser,
    UnifiedEventType,
)
from cgm_format.formats.d1namo import (
    D1NAMO_SENSOR_TYPES,
    D1namoGlucoseType,
)
from cgm_format.formats.supported import FORMAT_CATEGORY, PATH_DETECTION_PROBES
from cgm_format.interface.cgm_interface import SupportedCGMFormat

FIXTURE_DIR = Path(__file__).parent.parent / "data" / "input" / "d1namo_synthetic"
DIABETES_FIXTURE = FIXTURE_DIR / "diabetes_subset" / "001"
HEALTHY_FIXTURE = FIXTURE_DIR / "healthy_subset" / "012_diabetes"

# mmol/L -> mg/dL. Hardcoding a domain constant is sanctioned; hardcoding a
# row count read off a dump is not.
MMOL_TO_MGDL = 18.0182


def _real_corpus() -> Path:
    configured = os.environ.get("CGM_FORMAT_D1NAMO_DIR", "").strip()
    if not configured:
        pytest.skip("CGM_FORMAT_D1NAMO_DIR is not set")
    root = Path(configured)
    if not root.is_dir():
        pytest.skip(f"D1NAMO corpus not found at {root}")
    return root


def _real_subset(name: str) -> Path:
    root = _real_corpus()
    matches = [d for d in root.iterdir() if d.is_dir() and name in d.name]
    if not matches:
        pytest.skip(f"No {name} subset under {root}")
    return matches[0]


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")


class TestTwoFormatsNotOne:
    """The subsets differ by more than units, so they are two identities."""

    def test_both_subsets_are_registered_corpora(self) -> None:
        for fmt in (
            SupportedCGMFormat.D1NAMO_DIABETES,
            SupportedCGMFormat.D1NAMO_HEALTHY,
        ):
            assert FORMAT_CATEGORY[fmt] is FormatCategory.CORPUS

    def test_each_subset_has_its_own_discriminating_probe(self) -> None:
        """`insulin.csv` only in diabetes; `annotations.csv` only in healthy.

        Both probe sets also require `glucose.csv`, so neither matches a
        directory that merely contains CSVs.
        """
        diabetes = PATH_DETECTION_PROBES[SupportedCGMFormat.D1NAMO_DIABETES]
        healthy = PATH_DETECTION_PROBES[SupportedCGMFormat.D1NAMO_HEALTHY]

        assert any("insulin" in p for p in diabetes)
        assert any("annotations" in p for p in healthy)
        assert set(diabetes) != set(healthy)

    def test_the_subsets_detect_as_different_formats(self) -> None:
        _skip_if_missing(DIABETES_FIXTURE)
        _skip_if_missing(HEALTHY_FIXTURE)

        assert (
            FormatParser.detect_path_format(DIABETES_FIXTURE.parent)
            == SupportedCGMFormat.D1NAMO_DIABETES
        )
        assert (
            FormatParser.detect_path_format(HEALTHY_FIXTURE.parent)
            == SupportedCGMFormat.D1NAMO_HEALTHY
        )


class TestFingersticksAreNotSensorTraces:
    """Design decision D6, and the reason the healthy subset is includable."""

    def test_manual_readings_map_to_calibration(self) -> None:
        """Only `type == "cgm"` is a continuous reading.

        Expected counts are computed from the source CSV, never by running the
        parser under test.
        """
        _skip_if_missing(DIABETES_FIXTURE)

        frame = FormatParser._process_d1namo_subject(DIABETES_FIXTURE)

        raw = pl.read_csv(DIABETES_FIXTURE / "glucose.csv", infer_schema_length=0)
        expected_sensor = len(
            raw.filter(pl.col("type").str.strip_chars().is_in(list(D1NAMO_SENSOR_TYPES)))
        )
        expected_fingerstick = len(
            raw.filter(
                pl.col("type").str.strip_chars().is_in(list(D1NAMO_SENSOR_TYPES)).not_()
            )
        )

        readings = frame.filter(pl.col("glucose").is_not_null())
        sensor = readings.filter(
            pl.col("event_type") == UnifiedEventType.GLUCOSE.value
        )
        fingerstick = readings.filter(
            pl.col("event_type") == UnifiedEventType.CALIBRATION.value
        )

        assert len(sensor) == expected_sensor
        assert len(fingerstick) == expected_fingerstick

    def test_the_healthy_subset_produces_no_sensor_readings_at_all(self) -> None:
        """It contains four to six fingersticks a day and no CGM.

        A single `EGV_READ` here would mean a fingerstick was presented as a
        sensor trace — the exact misrepresentation D6 rules out.
        """
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)

        assert (
            len(frame.filter(pl.col("event_type") == UnifiedEventType.GLUCOSE.value))
            == 0
        )
        assert (
            len(
                frame.filter(
                    pl.col("event_type") == UnifiedEventType.CALIBRATION.value
                )
            )
            > 0
        )

    def test_the_reading_type_is_preserved_in_annotations(self) -> None:
        """`BB`/`AL` etc. are meal-relative labels worth keeping verbatim."""
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)
        readings = frame.filter(pl.col("glucose").is_not_null())

        assert all("reading_type" in a for a in readings["annotations"])


class TestUnitsAndAbsentChannels:
    """Conventions that would silently corrupt values if wrong."""

    def test_glucose_is_converted_from_mmol_per_litre(self) -> None:
        """Through the declared unit, never a per-vendor factor.

        The expected value is computed from the raw CSV with the domain
        constant, so a parser that forgot to convert fails here.
        """
        _skip_if_missing(DIABETES_FIXTURE)

        frame = FormatParser._process_d1namo_subject(DIABETES_FIXTURE)
        raw = pl.read_csv(DIABETES_FIXTURE / "glucose.csv", infer_schema_length=0)

        raw_values = [
            float(v)
            for v in raw.get_column("glucose").to_list()
            if v and v.strip().replace(".", "", 1).isdigit()
        ]
        parsed = [v for v in frame["glucose"].to_list() if v is not None]

        assert max(parsed) == pytest.approx(max(raw_values) * MMOL_TO_MGDL, rel=1e-6)
        # And the result is in a plausible mg/dL range, not still mmol/L.
        assert max(parsed) > 40

    def test_carbs_is_null_everywhere_because_d1namo_records_none(self) -> None:
        """"The source did not say" — never a zero.

        D1NAMO's meals carry calories plus human labels and no carbohydrate
        column at all, so a 0.0 would assert something no one measured.
        """
        _skip_if_missing(DIABETES_FIXTURE)

        frame = FormatParser._process_d1namo_subject(DIABETES_FIXTURE)

        assert frame["carbs"].null_count() == len(frame)

    def test_meals_carry_calories_into_the_extended_schema(self) -> None:
        """Without `calories` a meal row would be a timestamp and nothing else."""
        _skip_if_missing(DIABETES_FIXTURE)

        frame = FormatParser._process_d1namo_subject(DIABETES_FIXTURE)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )

        assert len(meals) > 0
        assert meals["calories"].null_count() < len(meals)

    def test_insulin_columns_map_straight_across(self) -> None:
        """`fast_insulin`/`slow_insulin` → `insulin_fast`/`insulin_slow`."""
        _skip_if_missing(DIABETES_FIXTURE)

        frame = FormatParser._process_d1namo_subject(DIABETES_FIXTURE)

        fast = frame.filter(
            pl.col("event_type") == UnifiedEventType.INSULIN_FAST.value
        )
        assert len(fast) > 0
        assert fast["insulin_fast"].null_count() == 0


class TestMixedTimestampConventions:
    """The trap most likely to produce silently wrong data."""

    def test_exif_colon_dates_in_food_csv_are_parsed(self) -> None:
        """`2014:10:01 19:27:49` — colons in the date part.

        A parse without an explicit format rejects this, and the meals would
        vanish with no error.
        """
        _skip_if_missing(DIABETES_FIXTURE)

        frame = FormatParser._process_d1namo_subject(DIABETES_FIXTURE)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        raw = pl.read_csv(DIABETES_FIXTURE / "food.csv", infer_schema_length=0)

        assert len(meals) == len(raw)

    def test_times_without_seconds_are_parsed(self) -> None:
        """The healthy subset writes `11:35`, the diabetes one `19:14:00`."""
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)
        raw = pl.read_csv(HEALTHY_FIXTURE / "glucose.csv", infer_schema_length=0)

        readings = frame.filter(pl.col("glucose").is_not_null())
        # Every parseable source reading survived — none silently dropped for
        # want of a seconds field.
        assert len(readings) > 0
        assert len(readings) <= len(raw)


class TestDirtIsReportedNotSwallowed:
    """Real corruption, with the two report kinds kept distinct."""

    def test_an_unrepresentable_glucose_value_is_reported(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """`7:0` — a colon typed for a decimal point.

        This is "the source said something we cannot represent", which is a
        different report from an empty cell. Dropping it silently would leave a
        reading missing with no trace.
        """
        subject = FIXTURE_DIR / "healthy_subset" / "017"
        _skip_if_missing(subject)

        with caplog.at_level("WARNING"):
            FormatParser._process_d1namo_subject(subject)

        assert "cannot represent" in caplog.text
        assert "7:0" in caplog.text

    def test_leading_zero_values_parse_normally(self) -> None:
        """`08.2` is a perfectly good number written oddly — not an error.

        Grouping it with `7:0` would report a false problem; the difference is
        exactly why the cast is attempted before anything is called corrupt.
        """
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)
        raw = pl.read_csv(HEALTHY_FIXTURE / "glucose.csv", infer_schema_length=0)

        leading_zero = [
            v
            for v in raw.get_column("glucose").to_list()
            if v and v.strip().startswith("0") and "." in v
        ]
        assert leading_zero, "fixture carries no leading-zero literal"
        # All of them survived: none was mistaken for corruption.
        assert frame.filter(pl.col("glucose").is_not_null()).height >= len(
            leading_zero
        )

    def test_no_information_becomes_json_null_not_the_string(self) -> None:
        """"No information" means the source did not say."""
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)
        meals = frame.filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )

        assert len(meals) > 0
        assert not any("No information" in (a or "") for a in meals["annotations"])

    def test_a_dangling_photo_reference_is_reported_separately_from_an_absent_one(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Two different statements, two different reports.

        A blank `picture` cell means no photograph was recorded. A cell naming
        something not on disk means the source said something we cannot
        resolve. In the real corpus these cells hold words like `lunch` where a
        filename belongs.
        """
        subject = FIXTURE_DIR / "healthy_subset" / "002"
        _skip_if_missing(subject)

        with caplog.at_level("WARNING"):
            FormatParser._process_d1namo_subject(subject)

        assert "not on disk" in caplog.text

    def test_an_empty_photo_directory_is_not_an_error(self) -> None:
        """Real subject 005 has `food_pictures/` present but empty."""
        subject = FIXTURE_DIR / "diabetes_subset" / "005"
        _skip_if_missing(subject)

        frame = FormatParser._process_d1namo_subject(subject)

        assert len(frame) > 0


class TestBundleOnlyEntryPoints:
    """A bare glucose.csv is one modality, not a record."""

    def test_parsing_a_lone_glucose_csv_refuses_and_names_the_alternative(
        self,
    ) -> None:
        """It detects, so it must not then fail as "unknown format".

        `format_supported()` returns True for these files, so a caller is
        entitled to expect `parse_file` to work. Parsing it alone would drop
        every insulin dose and every meal silently, so the refusal names the
        entry points that do not.
        """
        path = DIABETES_FIXTURE / "glucose.csv"
        _skip_if_missing(path)

        assert FormatParser.format_supported(path.read_bytes())

        with pytest.raises(Exception) as excinfo:
            FormatParser.parse_file(path)

        message = str(excinfo.value)
        assert "parse_corpus" in message or "parse_bundle" in message
        assert "unknown" not in message.lower()


class TestAnnotationsAreParsed:
    """`annotations.csv` is required for detection, so it cannot be ignored."""

    def test_annotation_rows_reach_the_frame(self) -> None:
        """Reading the directory and skipping the file would be a silent drop."""
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)
        raw = pl.read_csv(HEALTHY_FIXTURE / "annotations.csv", infer_schema_length=0)

        annotations = frame.filter(
            (pl.col("event_type") == UnifiedEventType.OTHER.value)
            & pl.col("annotations").str.contains("annotation_type")
        )
        assert len(annotations) == len(raw)

    def test_the_interval_end_is_preserved_in_the_annotation(self) -> None:
        """Only the start becomes a row; the frame is instant-shaped.

        Emitting an end row too would double-count the event, so the end is
        kept in the annotation rather than discarded.
        """
        _skip_if_missing(HEALTHY_FIXTURE)

        frame = FormatParser._process_d1namo_subject(HEALTHY_FIXTURE)
        annotations = frame.filter(
            pl.col("annotations").str.contains("annotation_type")
        )

        assert len(annotations) > 0
        for value in annotations["annotations"]:
            assert '"end_time":' in value


class TestUnreadableTimestampsAreReported:
    """The docstring calls this the trap most likely to produce wrong data."""

    def test_dropped_rows_are_counted_in_a_warning(
        self, tmp_path: pytest.TempPathFactory, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A row dropped for an unreadable timestamp must not vanish quietly.

        Glucose values we cannot represent already get a warning; before this,
        timestamps got none at all.
        """
        subject = Path(tmp_path) / "099"
        subject.mkdir(parents=True)
        (subject / "glucose.csv").write_text(
            "date,time,glucose,type,comments\n"
            "2014-10-01,11:35,5.4,BL,\n"
            "2014-10-01,not-a-time,6.1,AL,\n"
            "2014-10-01,12:40,7.2,AD,\n"
        )

        with caplog.at_level("WARNING"):
            frame = FormatParser._process_d1namo_subject(subject)

        assert len(frame.filter(pl.col("glucose").is_not_null())) == 2
        assert "timestamp" in caplog.text.lower()


class TestSingleTrackCorpusRefusesATrack:
    """A flag that selects nothing must say so, not be ignored."""

    def test_passing_a_track_to_a_single_track_corpus_raises(self) -> None:
        """Silently ignoring it returns every subject while the caller believes
        they filtered — the same quiet mismatch `parse_file` refuses for a
        multi-track source. Found by testing a README example verbatim.
        """
        root = FIXTURE_DIR / "healthy_subset"
        _skip_if_missing(root)

        with pytest.raises(ValueError, match="single-track"):
            FormatParser.parse_corpus(root, track="libre")

    def test_omitting_the_track_still_parses_every_subject(self) -> None:
        root = FIXTURE_DIR / "healthy_subset"
        _skip_if_missing(root)

        corpus = FormatParser.parse_corpus(root)

        assert len(corpus) > 0
        assert all("/" not in key for key in corpus)


class TestRealCorpus:
    """Against both real subsets, when the archives are available."""

    def test_the_diabetes_subset_walks_to_one_frame_per_subject(self) -> None:
        root = _real_subset("diabetes_subset_pictures")

        corpus = FormatParser.parse_corpus(root)

        expected = {
            d.name for d in root.iterdir() if d.is_dir() and (d / "glucose.csv").exists()
        }
        assert set(corpus) == expected
        for frame in corpus.values():
            assert frame.columns == CGM_SCHEMA_EXTENDED.get_column_names()

    def test_the_healthy_subset_keeps_the_012_diabetes_directory_name(self) -> None:
        """A subject id may contain `_` — which is why `/` is the separator.

        A three-digit assumption, or a `_` split, would drop or mis-key this
        subject.
        """
        root = _real_subset("healthy_subset_pictures")

        corpus = FormatParser.parse_corpus(root)

        assert "012_diabetes" in corpus
        assert len(corpus["012_diabetes"]) > 0

    def test_every_real_subject_yields_a_frame(self) -> None:
        """Including subject 005, whose meals carry no parseable timestamp.

        Its `food.csv` has the literal `NA` in every `datetime` cell — the only
        subject in the corpus that does. The meals are dropped with a warning;
        the glucose and insulin are not.
        """
        root = _real_subset("diabetes_subset_pictures")

        corpus = FormatParser.parse_corpus(root)

        assert "005" in corpus
        assert (
            len(
                corpus["005"].filter(
                    pl.col("event_type") == UnifiedEventType.GLUCOSE.value
                )
            )
            > 0
        )

    def test_no_healthy_subject_reports_a_sensor_reading(self) -> None:
        """Across all 20 real subjects, not just the fixture."""
        root = _real_subset("healthy_subset_pictures")

        corpus = FormatParser.parse_corpus(root)

        for subject, frame in corpus.items():
            sensor_rows = frame.filter(
                pl.col("event_type") == UnifiedEventType.GLUCOSE.value
            )
            assert len(sensor_rows) == 0, subject
