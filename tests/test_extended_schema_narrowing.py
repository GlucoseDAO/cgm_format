"""The two ways an extended frame used to lose its extra columns.

`FormatProcessor` referred to `CGM_SCHEMA` at 24 sites. 17 are gated on
`validation_mode` and fail loudly; **7 are ungated structural reads** that
enforce unconditionally, and enforcement drops extra columns
(`interface/schema.py:410`, `dataframe.select(expected_columns.keys())`). A
frame carrying macronutrients or annotations handed to the processor therefore
came back silently narrowed, mid-pipeline, with no warning and no error.

That is two distinct failure modes, not one, and they need separate tests
because they surface under *different* validation modes:

1. **Loud** — under the shipped default (`ValidationMethod.INPUT`, non-forced)
   an extended frame trips a gated check before it reaches an ungated one.
   This behaviour was always correct; the test pins it so the seam does not
   accidentally start accepting-and-narrowing instead.
2. **Silent** — under `NO_VALIDATION` (and the `*_FORCED` variants) the gated
   checks step aside and the ungated sites narrow the frame. This is the real
   defect the schema seam fixes.

Both tests deliberately build their extended schema with `derive_schema` and
subclass `FormatProcessor` **locally**, rather than importing
`CGM_SCHEMA_EXTENDED` / `ExtendedFormatProcessor`. That is not incidental: it
is what lets this file run unchanged against the code from *before* the seam
existed, where a `schema` class attribute is simply an attribute nobody reads.
There it fails on the assertion — which is the demonstration `CLAUDE.md` §2
requires — rather than dying at import, which would prove nothing about
narrowing.
"""

from datetime import datetime

import polars as pl
import pytest

from cgm_format import CGM_SCHEMA, FormatProcessor, Quality, UnifiedEventType
from cgm_format.interface.cgm_interface import (
    NO_VALIDATION,
    MalformedDataError,
    ValidationMethod,
)
from cgm_format.interface.schema import ColumnSchema, derive_schema

# The two channels an extended frame carries that the core schema has no home
# for: a typed measurement and the stringly-typed escape hatch. One of each is
# enough to prove narrowing — the failure is structural, not per-column.
_APPENDED: tuple[ColumnSchema, ...] = (
    {
        "name": "calories",
        "dtype": pl.Float64,
        "description": "Energy content of a recorded meal",
        "unit": "kcal",
        "constraints": {"minimum": 0},
    },
    {
        "name": "annotations",
        "dtype": pl.Utf8,
        "description": "JSON object for anything with no typed home",
    },
)

_EXTENDED_SCHEMA = derive_schema(CGM_SCHEMA, append_data_columns=_APPENDED)
_EXTRA_COLUMNS = ("calories", "annotations")


class _LocallyExtendedProcessor(FormatProcessor):
    """A processor pointed at the extended schema.

    Before the seam existed this attribute was inert — every method read the
    module-level `CGM_SCHEMA` directly — so this subclass behaves exactly like
    the core processor and narrows the frame. That inertness is what the
    silent-narrowing test detects.
    """

    schema = _EXTENDED_SCHEMA


def _extended_frame() -> pl.DataFrame:
    """Build a small extended frame with values a device could have produced.

    Five minutes apart so the rows land on the default grid and sequence
    detection keeps them in one sequence — the point of the test is the column
    set, so nothing here should provoke a split.

    **The frame must carry a non-glucose event.** `detect_and_assign_sequences`
    narrows in its `else` branch, reached only when
    `event_type != "EGV_READ"` rows exist: an all-glucose frame short-circuits
    to `result_df = glucose_events` and never reaches the enforcement call at
    all. An all-`EGV_READ` fixture therefore passes against pre-fix code and
    proves nothing — which is exactly what a first draft of this file did. The
    `CARBS_IN` row below is what routes execution through the ungated site, and
    it is also the row carrying the macronutrients, which is what makes it a
    realistic case rather than a contrivance.
    """
    # `original_datetime` equals `datetime` the way parsing leaves it — the
    # idempotency anchor every downstream stage computes from.
    moments = pl.datetime_range(
        datetime(2024, 3, 1, 8, 0),
        datetime(2024, 3, 1, 8, 15),
        interval="5m",
        eager=True,
    ).cast(pl.Datetime("ms"))

    frame = pl.DataFrame(
        {
            "sequence_id": [0, 0, 0, 0],
            "original_datetime": moments,
            "quality": [Quality(0).value] * 4,
            "event_type": [
                UnifiedEventType.GLUCOSE.value,
                UnifiedEventType.GLUCOSE.value,
                # The non-glucose row that routes execution through the
                # narrowing branch, and the row the macronutrients hang off.
                UnifiedEventType.CARBOHYDRATES.value,
                UnifiedEventType.GLUCOSE.value,
            ],
            "datetime": moments,
            # Null on the carb row: the sensor did not report a reading there,
            # and a null is "did not say", never a zero.
            "glucose": [104.0, 111.0, None, 98.0],
            "carbs": [None, None, 45.0, None],
            "insulin_slow": [None, None, None, None],
            "insulin_fast": [None, None, 4.0, None],
            "exercise": [None, None, None, None],
            # The channels the core schema cannot hold.
            "calories": [None, None, 612.0, None],
            "annotations": [None, None, '{"meal_type":"lunch"}', None],
        }
    )
    return _EXTENDED_SCHEMA.validate_dataframe(frame, enforce=True)


def test_extended_frame_is_well_formed_against_the_extended_schema() -> None:
    """The fixture itself is valid — so a later failure is the processor's."""
    frame = _extended_frame()
    # No exception: the frame conforms without enforcement having to fix it.
    _EXTENDED_SCHEMA.validate_dataframe(frame, enforce=False)
    assert set(frame.columns) == set(_EXTENDED_SCHEMA.get_column_names())
    assert set(_EXTRA_COLUMNS) <= set(frame.columns)
    assert set(_EXTRA_COLUMNS).isdisjoint(CGM_SCHEMA.get_column_names())


def test_core_processor_rejects_an_extended_frame_loudly_under_default_mode() -> None:
    """Failure mode 1: the gated checks catch it before anything narrows.

    Under `ValidationMethod.INPUT` the column-count check
    (`interface/schema.py:388-390`) fires first and raises `MalformedDataError`.
    `ExtraColumnError` is a subclass of it, so catching the base covers whichever
    of the two gates the frame happens to reach first — asserting on one
    specific subclass would pin an implementation detail rather than the
    behaviour.
    """
    frame = _extended_frame()

    with pytest.raises(MalformedDataError):
        FormatProcessor.detect_and_assign_sequences(
            frame, validation_mode=ValidationMethod.INPUT
        )


def test_extended_processor_preserves_extra_columns_through_ungated_sites() -> None:
    """Failure mode 2: the silent one, and the reason the seam exists.

    `NO_VALIDATION` stands the gated checks down, so the frame reaches the
    ungated structural reads — `detect_and_assign_sequences` ends in an
    unconditional `validate_dataframe(..., enforce=True)`. Against the core
    schema that call *drops* `calories` and `annotations` and returns happily.

    Against pre-seam code this assertion fails with the extra columns missing
    from the output. That failure is the point of the test.
    """
    frame = _extended_frame()

    result = _LocallyExtendedProcessor.detect_and_assign_sequences(
        frame, validation_mode=NO_VALIDATION
    )

    assert set(result.columns) == set(frame.columns), (
        "extended columns were dropped by an ungated structural read: "
        f"missing {sorted(set(frame.columns) - set(result.columns))}"
    )
    # Losslessness: the values survived, not just the column names.
    assert result["calories"].to_list() == frame["calories"].to_list()
    assert result["annotations"].to_list() == frame["annotations"].to_list()


def test_extended_processor_preserves_extra_columns_through_interpolation() -> None:
    """The same silent narrowing, reached by a different ungated site.

    `interpolate_gaps` narrows at `_join_and_interpolate_values` and sorts on
    `get_stable_sort_keys()`, both unconditional. Covering a second entry point
    matters because the 7 ungated sites are spread across four methods — a fix
    that threaded the schema through only one of them would still pass the test
    above.
    """
    frame = _extended_frame()

    result = _LocallyExtendedProcessor.interpolate_gaps(
        frame, validation_mode=NO_VALIDATION
    )

    assert set(result.columns) == set(frame.columns), (
        "extended columns were dropped during interpolation: "
        f"missing {sorted(set(frame.columns) - set(result.columns))}"
    )
