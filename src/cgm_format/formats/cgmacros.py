"""CGMacros research corpus — 45 subjects, two concurrent CGM sensors, meals.

PhysioNet `cgmacros/1.0.0`, CC BY-NC-SA. Each subject is a directory holding one
CSV and a `photos/` folder. Two sensors are worn over the same ten days, so a
subject file carries **two independent glucose series** — `Libre GL` and
`Dexcom GL` are not two spellings of one column, they are two measurements of
the same quantity that disagree by design. That is what makes CGMacros a
multi-*track* source: one file in, two frames out.

Alongside glucose it carries channels the core six data columns have no home
for — macronutrients, heart rate, activity — which is why it targets
`CGM_SCHEMA_EXTENDED`.

Three facts about the data that shaped this module, all read off the 45 real
subject files rather than the published data dictionary:

**The header is not stable across subjects — there are 9 distinct variants.**
This is drift *within a single release*, and it is what breaks a parser written
against subject 001 and tested only on subject 001. Every variant is absorbed
declaratively (`aliases` + `normalize_headers`), never by a branch:

===========================  ==========  ====================================
Variant                      Subjects    Handling
===========================  ==========  ====================================
`METs` vs `Intensity`        11          alias
`Unnamed: 0` index column    9           dropped as an extra
`Amount Consumed ` (space)   1           header strip, then alias
`Amount Consumed` absent     2           typed null
`Sugar`                      1           dropped (no core or extended home)
`Steps` + `RecordIndex`      1           `Steps` is an extended column
`Libre GL`/`Dexcom GL` swap  1           harmless; Polars is name-keyed
===========================  ==========  ====================================

The two subjects missing `Amount Consumed` are the reason the parser adds typed
nulls for absent optional columns up front rather than guarding each `select`: a
`.select(pl.col(X))` is evaluated even when an upstream `.filter()` leaves zero
rows, so an absent column raises `ColumnNotFound` regardless of whether any row
would have used it.

**`METs` is stored multiplied by 10** (per the data dictionary, and confirmed by
the observed 10–126 range against a physiological 1.0–12.6).

**Native cadence is one minute** — the authors interpolated both series onto a
common 1-minute grid, so neither is at its device's native rate (Libre Pro
samples every 15 minutes, Dexcom G6 Pro every 5). **This parser does not
resample.** Regridding is the processor's job and is lossy, so callers pass
`expected_interval_minutes=1`. Note that `small_gap_max_minutes` defaults to 15,
which against a 1-minute interval is 15× rather than the intended 3×; pass it
explicitly when processing a CGMacros frame.
"""

from typing import Dict, List, Tuple

import polars as pl

from cgm_format.interface.schema import (
    ColumnSchema,
    EnumLiteral,
    CGMSchemaDefinition,
    regenerate_schema_json as _regenerate,
)

# One header row, data immediately after. No metadata rows.
CGMACROS_HEADER_LINE = 1
CGMACROS_DATA_START_LINE = 2
CGMACROS_METADATA_LINES: Tuple[int, ...] = ()

# Observed in every one of the 45 subject files. The published data dictionary
# describes `Month/Day/Year HH:MM` instead, so probe rather than trust either.
CGMACROS_TIMESTAMP_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
)

# `detect_format` is DISJUNCTIVE — it returns on the first pattern that appears
# anywhere in the scanned lines — so each entry here must be specific enough to
# identify CGMacros on its own. "Libre GL" and "Dexcom GL" both are: no other
# supported export uses either spelling. (Contrast `detect_path_format`, whose
# probes are conjunctive; the two mechanisms differ deliberately, because a
# directory match sends a whole tree to one parser.)
CGMACROS_DETECTION_PATTERNS = [
    "Libre GL",
    "Dexcom GL",
]

# Directory shape, for detect_path_format. Conjunctive: a subject directory
# holds a CSV named after itself, which distinguishes a corpus root from any
# folder that merely contains CSVs.
CGMACROS_PATH_PROBES: Tuple[str, ...] = (
    "CGMacros-*/CGMacros-*.csv",
)

# The same shape one level down, for `detect_subject_format`: a CGMacros
# *subject* directory holds its own CSV directly rather than in a child. The
# two probe sets are disjoint by construction — a corpus root has no
# `CGMacros-*.csv` beside it and a subject directory has no matching
# grandchild — which is what keeps a root from being mistaken for a subject.
CGMACROS_SUBJECT_PROBES: Tuple[str, ...] = (
    "CGMacros-*.csv",
)

# The two sensor series, as track names. Order is the key order of
# `parse_tracks`, and `libre` leads because it is populated on essentially
# every row while `Dexcom GL` is populated on about 92%.
CGMACROS_TRACKS: Tuple[str, ...] = ("libre", "dexcom")

#: Opt-in synthetic track: the per-timestamp mean of the two sensors. Never
#: produced by `parse_tracks` — it is a derived view, not a member of the
#: corpus — and every row it synthesizes from two readings is flagged
#: `Quality.TRACK_MERGE`.
CGMACROS_MEAN_TRACK = "mean"


class CGMacrosColumn(EnumLiteral):
    """Column names in a CGMacros subject CSV."""

    TIMESTAMP = "Timestamp"
    LIBRE_GLUCOSE = "Libre GL"
    DEXCOM_GLUCOSE = "Dexcom GL"
    HEART_RATE = "HR"
    ACTIVITY_CALORIES = "Calories (Activity)"
    METS = "METs"
    MEAL_TYPE = "Meal Type"
    CALORIES = "Calories"
    CARBS = "Carbs"
    PROTEIN = "Protein"
    FAT = "Fat"
    FIBER = "Fiber"
    AMOUNT_CONSUMED = "Amount Consumed"
    IMAGE_PATH = "Image path"
    STEPS = "Steps"


#: Which raw column each track name reads. One mapping, so the parser and
#: `list_subjects`' coverage reader cannot drift into disagreeing about what a
#: track name means — a drift that would be invisible, because both would go on
#: returning plausible numbers for the wrong sensor. Key order is
#: `CGMACROS_TRACKS` order.
CGMACROS_TRACK_COLUMNS: Dict[str, str] = {
    CGMACROS_TRACKS[0]: CGMacrosColumn.LIBRE_GLUCOSE.value,
    CGMACROS_TRACKS[1]: CGMacrosColumn.DEXCOM_GLUCOSE.value,
}


class CGMacrosMealType(EnumLiteral):
    """Normalized meal vocabulary.

    The raw column carries ten spellings for four meals — `Breakfast`/
    `breakfast`, `Lunch`/`lunch`, `Dinner`/`dinner`, and
    `Snack`/`Snacks`/`snack`/`snack 1`. Case and plural are normalized to these
    members; the raw string is preserved in `annotations` so the normalization
    stays inspectable rather than being a lossy rewrite.
    """

    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


#: Raw `Meal Type` spelling → normalized member. Data, not a regex: the
#: vocabulary is closed and observed in full across all 45 subjects, so an
#: unrecognized spelling should surface as a warning rather than be silently
#: lowercased into a member it may not belong to.
CGMACROS_MEAL_TYPE_NORMALIZATION: Dict[str, str] = {
    "breakfast": CGMacrosMealType.BREAKFAST.value,
    "lunch": CGMacrosMealType.LUNCH.value,
    "dinner": CGMacrosMealType.DINNER.value,
    "snack": CGMacrosMealType.SNACK.value,
    "snacks": CGMacrosMealType.SNACK.value,
    "snack 1": CGMacrosMealType.SNACK.value,
}

#: `METs` is stored ×10. Divided on parse so the emitted value is the real
#: metabolic equivalent, matching the extended schema's declared unit.
CGMACROS_METS_SCALE = 10.0

CGMACROS_SCHEMA = CGMSchemaDefinition(
    # Timestamp is the row's identity, matching every other vendor schema:
    # Dexcom puts `Timestamp` here, Libre `Device Timestamp`, Nightscout
    # `Date`/`Time`. Service vs data is "what identifies the row" vs "what was
    # measured", and CGMacros identifies a row by time alone.
    service_columns=(
        {
            "name": CGMacrosColumn.TIMESTAMP.value,
            "dtype": pl.Utf8,
            "description": "Reading timestamp on the authors' common 1-minute grid",
            "constraints": {"required": True},
        },
    ),
    data_columns=(
        {
            "name": CGMacrosColumn.LIBRE_GLUCOSE.value,
            "dtype": pl.Float64,
            "description": "FreeStyle Libre Pro glucose reading",
            "unit": "mg/dL",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.DEXCOM_GLUCOSE.value,
            "dtype": pl.Float64,
            "description": "Dexcom G6 Pro glucose reading",
            "unit": "mg/dL",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.HEART_RATE.value,
            "dtype": pl.Float64,
            "description": "Heart rate from the wearable",
            "unit": "bpm",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.ACTIVITY_CALORIES.value,
            "dtype": pl.Float64,
            "description": "Energy expenditure attributed to activity",
            "unit": "kcal",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.METS.value,
            "dtype": pl.Float64,
            "description": "Metabolic equivalent of task, stored multiplied by 10",
            # 11 subjects spell this column "Intensity" instead.
            "aliases": ("Intensity",),
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.MEAL_TYPE.value,
            "dtype": pl.Utf8,
            "description": "Meal label; ten raw spellings for four meals",
        },
        {
            "name": CGMacrosColumn.CALORIES.value,
            "dtype": pl.Float64,
            "description": "Energy content of the recorded meal",
            "unit": "kcal",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.CARBS.value,
            "dtype": pl.Float64,
            "description": "Carbohydrate content of the recorded meal",
            "unit": "g",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.PROTEIN.value,
            "dtype": pl.Float64,
            "description": "Protein content of the recorded meal",
            "unit": "g",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.FAT.value,
            "dtype": pl.Float64,
            "description": "Fat content of the recorded meal",
            "unit": "g",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.FIBER.value,
            "dtype": pl.Float64,
            "description": "Fiber content of the recorded meal",
            "unit": "g",
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.AMOUNT_CONSUMED.value,
            "dtype": pl.Float64,
            "description": (
                "Percentage of the recorded meal actually eaten. Absent "
                "entirely from 2 of 45 subjects; one more spells it with a "
                "trailing space."
            ),
            "aliases": ("Amount Consumed ",),
            "constraints": {"minimum": 0},
        },
        {
            "name": CGMacrosColumn.IMAGE_PATH.value,
            "dtype": pl.Utf8,
            "description": "Meal photograph, relative to the subject directory",
        },
        {
            "name": CGMacrosColumn.STEPS.value,
            "dtype": pl.Float64,
            "description": "Step count; present for 1 of 45 subjects",
            "constraints": {"minimum": 0},
        },
    ),
    header_line=CGMACROS_HEADER_LINE,
    data_start_line=CGMACROS_DATA_START_LINE,
    metadata_lines=CGMACROS_METADATA_LINES,
)

#: Columns the schema declares that a given subject file may simply not have.
#: Added as typed nulls before any `select`, because a `.select(pl.col(X))` is
#: evaluated even when an upstream `.filter()` leaves zero rows — the
#: `ColumnNotFound` gotcha that used to crash the LibreView insulin sub-frame.
CGMACROS_OPTIONAL_COLUMNS: Tuple[str, ...] = (
    CGMacrosColumn.AMOUNT_CONSUMED.value,
    CGMacrosColumn.STEPS.value,
    CGMacrosColumn.ACTIVITY_CALORIES.value,
    CGMacrosColumn.METS.value,
)

#: File-local row indices. Dropped without comment: they carry no measurement
#: and mean nothing outside the file they came from.
CGMACROS_INDEX_COLUMNS: Tuple[str, ...] = (
    "Unnamed: 0",
    "RecordIndex",
)

#: Real measurements the schema cannot hold. Dropped, but **reported** — "the
#: source said something we cannot represent" is a different statement from
#: "the source did not say", and silently discarding a macronutrient someone
#: measured is the half of that distinction it would be easy to get wrong.
#: `Sugar` appears on 1 of 45 subjects; declaring it is a schema decision, not
#: something to settle inside a parser.
CGMACROS_UNREPRESENTABLE_COLUMNS: Tuple[str, ...] = ("Sugar",)

#: Everything dropped, for the single `drop` call.
CGMACROS_IGNORED_COLUMNS: Tuple[str, ...] = (
    CGMACROS_INDEX_COLUMNS + CGMACROS_UNREPRESENTABLE_COLUMNS
)


def regenerate_schema_json() -> None:
    """Regenerate the committed Frictionless schema JSON for this format."""
    _regenerate(CGMACROS_SCHEMA, __file__)
