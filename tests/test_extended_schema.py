"""The extended unified schema: its ordering, its serializer, its round-trip.

`CGM_SCHEMA_EXTENDED` widens `CGM_SCHEMA` with food, wearable and analyte
channels plus an `annotations` escape hatch. Widening a schema here is not a
local change: `get_stable_sort_keys()` returns *every* column
(`interface/schema.py:224-242`), so appending a column silently widens the
total ordering that the byte-level round-trip and idempotency guarantees rest
on. `CLAUDE.md` §2.1 requires every new ordering get a test; this file is it.

The `annotations` column is what makes determinism load-bearing rather than
cosmetic. It is a *data* column, so it joins `primary_key` and the sort keys —
which is what keeps two annotation-only rows at the same timestamp
distinguishable instead of collapsing into an arbitrary order. If two runs
serialized the same mapping to different bytes, rows would sort differently
run to run and the guarantees would become *flaky* rather than false, which is
harder to notice and worse to debug.
"""

from datetime import datetime

import polars as pl
import pytest

from cgm_format import (
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
    FormatParser,
    Quality,
    UnifiedEventType,
    annotations_to_json,
)
from cgm_format.interface.cgm_interface import SupportedCGMFormat

# Channels the extended schema adds, in the order the design fixes them.
# Hardcoding a vocabulary is sanctioned (`CLAUDE.md` §6); hardcoding a count
# read off a data dump is not, and this is the former.
_EXPECTED_APPENDED = (
    "calories",
    "protein",
    "fat",
    "fiber",
    "heart_rate",
    "breathing_rate",
    "acceleration",
    "mets",
    "activity_calories",
    "steps",
    "ketones",
    "annotations",
)


class TestExtendedSchemaShape:
    """The extended schema widens the core one without disturbing it."""

    def test_core_schema_is_a_prefix_of_the_extended_one(self) -> None:
        """Additive means additive: every core column keeps its position.

        This is the property that lets an extended frame be narrowed to a core
        one by projection alone, and it is what makes the release *minor*
        rather than major — an existing consumer reading positionally sees an
        unchanged frame.
        """
        core = CGM_SCHEMA.get_column_names()
        extended = CGM_SCHEMA_EXTENDED.get_column_names()

        assert extended[: len(core)] == core
        assert extended[len(core) :] == list(_EXPECTED_APPENDED)

    def test_core_schema_was_not_mutated_by_deriving_from_it(self) -> None:
        """`derive_schema` returns a new frozen schema; the base is untouched."""
        assert set(_EXPECTED_APPENDED).isdisjoint(CGM_SCHEMA.get_column_names())
        assert CGM_SCHEMA.get_column_names() == [
            "sequence_id",
            "original_datetime",
            "quality",
            "event_type",
            "datetime",
            "glucose",
            "carbs",
            "insulin_slow",
            "insulin_fast",
            "exercise",
        ]

    def test_annotations_is_a_data_column_not_a_service_column(self) -> None:
        """Its placement is the whole reason two annotation-only rows survive.

        A service column is stripped by `to_data_only_df`; more importantly a
        column outside `get_polars_schema(data_only=False)` would not join the
        sort keys, and two annotation-only rows sharing a timestamp would then
        have no tie-break at all.
        """
        service = [c["name"] for c in CGM_SCHEMA_EXTENDED.service_columns]
        data = [c["name"] for c in CGM_SCHEMA_EXTENDED.data_columns]

        assert "annotations" not in service
        assert "annotations" in data
        assert "annotations" in CGM_SCHEMA_EXTENDED.get_column_names(data_only=True)

    def test_ketones_carries_its_own_unit_and_is_not_a_glucose_column(self) -> None:
        """Clinical ketones are mmol/L already (design decision D8).

        Routing them through the glucose convention would apply one analyte's
        rule to another and silently scale them by 18.
        """
        assert CGM_SCHEMA_EXTENDED.get_unit("ketones") == "mmol/L"
        assert CGM_SCHEMA_EXTENDED.get_unit("glucose") == "mg/dL"

    def test_exercise_declares_a_unit_the_conversion_table_can_reach(self) -> None:
        """`exercise` declares `"s"`, the spelling `UNIT_CONVERSIONS` is keyed on.

        The table converts minutes and hours *to* seconds, so a duration column
        declaring `"seconds"` could never be a conversion target: a parser
        calling `_to_canonical_unit` for it would find no entry and pass the
        value through unscaled. Nothing reads `exercise`'s unit today — this
        pins the declaration against the table so a future duration parser has
        a key that resolves rather than a spelling that silently misses.
        """
        from cgm_format.formats.unified import UNIT_CONVERSIONS

        exercise_unit = CGM_SCHEMA.get_unit("exercise")

        # The specific lookups a duration parser would perform, named rather
        # than a membership test that `"mg/dL"` would also satisfy.
        assert ("min", exercise_unit) in UNIT_CONVERSIONS
        assert ("h", exercise_unit) in UNIT_CONVERSIONS


class TestWidenedTotalOrdering:
    """`CLAUDE.md` §2.1: every new ordering gets a real test."""

    def test_sort_keys_are_every_column_in_schema_order(self) -> None:
        """Derived from the schema, not restated beside it.

        Asserting a hardcoded key list here would drift from the schema and the
        drift would only show on the input where the two disagree.
        """
        assert (
            CGM_SCHEMA_EXTENDED.get_stable_sort_keys()
            == CGM_SCHEMA_EXTENDED.get_column_names()
        )

    def test_widening_extends_the_ordering_rather_than_reordering_it(self) -> None:
        """The core keys keep their precedence; the new ones only break ties."""
        core_keys = CGM_SCHEMA.get_stable_sort_keys()
        extended_keys = CGM_SCHEMA_EXTENDED.get_stable_sort_keys()

        assert extended_keys[: len(core_keys)] == core_keys

    def test_annotations_participates_in_the_primary_key(self) -> None:
        """Two annotation-only rows at one timestamp must stay distinguishable."""
        assert CGM_SCHEMA_EXTENDED.primary_key is not None
        assert "annotations" in CGM_SCHEMA_EXTENDED.primary_key

    def test_two_annotation_only_rows_at_one_timestamp_sort_deterministically(
        self,
    ) -> None:
        """The real case from CGMacros: 1,553 photo rows carry no meal.

        Same timestamp, same (null) measurements, differing only in the
        annotation. Without `annotations` in the sort keys their relative order
        would be whatever the previous operation happened to leave, and the
        CSV bytes would differ run to run.
        """
        moment = datetime(2024, 3, 1, 12, 30)
        frame = pl.DataFrame(
            {
                "sequence_id": [0, 0],
                "original_datetime": [moment, moment],
                "quality": [Quality(0).value, Quality(0).value],
                "event_type": [UnifiedEventType.OTHER.value] * 2,
                "datetime": [moment, moment],
                "glucose": [None, None],
                "carbs": [None, None],
                "insulin_slow": [None, None],
                "insulin_fast": [None, None],
                "exercise": [None, None],
                "calories": [None, None],
                "protein": [None, None],
                "fat": [None, None],
                "fiber": [None, None],
                "heart_rate": [None, None],
                "breathing_rate": [None, None],
                "acceleration": [None, None],
                "mets": [None, None],
                "activity_calories": [None, None],
                "steps": [None, None],
                "ketones": [None, None],
                "annotations": [
                    annotations_to_json({"image_path": "b.jpg"}),
                    annotations_to_json({"image_path": "a.jpg"}),
                ],
            }
        )
        enforced = CGM_SCHEMA_EXTENDED.validate_dataframe(frame, enforce=True)

        # Sorting is what disambiguates them, and it puts "a" before "b"
        # regardless of the order they arrived in.
        assert enforced["annotations"].to_list() == sorted(
            enforced["annotations"].to_list()
        )
        # Re-sorting an already-sorted frame changes nothing (idempotent).
        resorted = CGM_SCHEMA_EXTENDED.stable_sort_dataframe(enforced)
        assert resorted.equals(enforced)


class TestAnnotationSerialization:
    """`annotations` bytes must be a pure function of the mapping."""

    def test_key_order_in_the_input_does_not_change_the_output(self) -> None:
        """Sorted keys: two mappings that differ only in insertion order agree."""
        first = annotations_to_json({"meal_type": "lunch", "image_path": "a.jpg"})
        second = annotations_to_json({"image_path": "a.jpg", "meal_type": "lunch"})

        assert first == second

    def test_repeated_serialization_is_byte_identical(self) -> None:
        """The property the round-trip guarantee actually rests on."""
        mapping = {"meal_type": "lunch", "amount_consumed": 0.5, "photo": True}

        assert annotations_to_json(mapping) == annotations_to_json(mapping)

    def test_nothing_to_record_is_null_not_an_empty_object(self) -> None:
        """`None`, never `""` and never `"{}"`.

        Polars writes a null as an empty CSV field and reads an empty field
        back as null, so an empty-string cell would not survive a round-trip
        unchanged. An empty mapping means the caller had nothing to record.
        """
        assert annotations_to_json(None) is None
        assert annotations_to_json({}) is None

    def test_a_key_with_a_null_value_is_kept(self) -> None:
        """"The source named this field and gave us nothing" is not "absent".

        Collapsing the two would be exactly the three-valued-logic mistake
        `CLAUDE.md` §5 forbids.
        """
        serialized = annotations_to_json({"balance": None})

        assert serialized is not None
        assert "balance" in serialized

    def test_a_non_finite_value_raises_rather_than_emitting_invalid_json(
        self,
    ) -> None:
        """Bare `NaN` is not valid JSON and poisons every downstream reader."""
        with pytest.raises(ValueError):
            annotations_to_json({"mets": float("nan")})


class TestExtendedRoundTrip:
    """CSV → parse → CSV, byte-identically, through the registry."""

    @staticmethod
    def _extended_frame() -> pl.DataFrame:
        moments = pl.datetime_range(
            datetime(2024, 5, 2, 7, 0),
            datetime(2024, 5, 2, 7, 15),
            interval="5m",
            eager=True,
        ).cast(pl.Datetime("ms"))
        frame = pl.DataFrame(
            {
                "sequence_id": [1, 1, 1, 1],
                "original_datetime": moments,
                "quality": [Quality(0).value] * 4,
                "event_type": [
                    UnifiedEventType.GLUCOSE.value,
                    UnifiedEventType.GLUCOSE.value,
                    UnifiedEventType.CARBOHYDRATES.value,
                    UnifiedEventType.GLUCOSE.value,
                ],
                "datetime": moments,
                "glucose": [96.0, 103.0, None, 117.0],
                "carbs": [None, None, 52.0, None],
                "insulin_slow": [None, None, None, None],
                "insulin_fast": [None, None, 6.0, None],
                "exercise": [None, None, None, None],
                "calories": [None, None, 730.0, None],
                "protein": [None, None, 31.0, None],
                "fat": [None, None, 24.0, None],
                "fiber": [None, None, 7.0, None],
                "heart_rate": [68.0, 71.0, None, 74.0],
                "breathing_rate": [14.0, None, None, None],
                "acceleration": [None, None, None, None],
                "mets": [1.2, None, None, None],
                "activity_calories": [None, None, None, None],
                "steps": [None, None, None, None],
                "ketones": [None, None, None, None],
                "annotations": [
                    None,
                    None,
                    annotations_to_json({"meal_type": "breakfast"}),
                    None,
                ],
            }
        )
        return CGM_SCHEMA_EXTENDED.validate_dataframe(frame, enforce=True)

    def test_an_extended_csv_detects_as_extended_not_as_core_unified(self) -> None:
        """The registry ordering is what this actually proves.

        An extended round-trip CSV also carries `sequence_id`/`event_type`/
        `quality`, so it matches the generic unified patterns too. Detection
        returns on first match, so `UNIFIED_EXTENDED` has to sit ahead of
        `UNIFIED_CGM` — if it ever slips behind, this test fails and the
        round-trip below silently narrows instead.
        """
        csv_text = FormatParser.to_csv_string(self._extended_frame())

        assert FormatParser.detect_format(csv_text) == (
            SupportedCGMFormat.UNIFIED_EXTENDED
        )

    def test_extended_frame_survives_csv_round_trip_byte_identically(self) -> None:
        """Losslessness at the byte level, which is the guarantee we ship."""
        original = self._extended_frame()

        first_csv = FormatParser.to_csv_string(original)
        reparsed = FormatParser.parse_from_string(first_csv)
        second_csv = FormatParser.to_csv_string(reparsed)

        assert second_csv == first_csv
        assert reparsed.columns == original.columns
        assert reparsed.equals(original)

    def test_the_extended_channels_survive_with_their_values(self) -> None:
        """Column names surviving is not the same as values surviving."""
        original = self._extended_frame()
        reparsed = FormatParser.parse_from_string(
            FormatParser.to_csv_string(original)
        )

        for column in _EXPECTED_APPENDED:
            assert reparsed[column].to_list() == original[column].to_list(), column

    def test_parsing_is_idempotent(self) -> None:
        """Re-parsing a serialized frame is a bit-level no-op."""
        once = FormatParser.parse_from_string(
            FormatParser.to_csv_string(self._extended_frame())
        )
        twice = FormatParser.parse_from_string(FormatParser.to_csv_string(once))

        assert twice.equals(once)


class TestCoreDetectionIsUnaffected:
    """Putting `UNIFIED_EXTENDED` first must not hijack anything."""

    def test_a_core_unified_csv_still_detects_as_core_unified(self) -> None:
        """The discriminating pattern is `annotations`, absent from core CSVs."""
        moments = pl.datetime_range(
            datetime(2024, 5, 2, 7, 0),
            datetime(2024, 5, 2, 7, 10),
            interval="5m",
            eager=True,
        ).cast(pl.Datetime("ms"))
        core = CGM_SCHEMA.validate_dataframe(
            pl.DataFrame(
                {
                    "sequence_id": [1, 1, 1],
                    "original_datetime": moments,
                    "quality": [Quality(0).value] * 3,
                    "event_type": [UnifiedEventType.GLUCOSE.value] * 3,
                    "datetime": moments,
                    "glucose": [96.0, 103.0, 117.0],
                    "carbs": [None, None, None],
                    "insulin_slow": [None, None, None],
                    "insulin_fast": [None, None, None],
                    "exercise": [None, None, None],
                }
            ),
            enforce=True,
        )
        csv_text = FormatParser.to_csv_string(core)

        assert FormatParser.detect_format(csv_text) == SupportedCGMFormat.UNIFIED_CGM
