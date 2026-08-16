"""BIG IDEAs research corpus — Dexcom Clarity export plus a food log.

PhysioNet `big-ideas-glycemic-wearable/1.1.3`, Bent et al., *npj Digital
Medicine*, 2021. Each subject is a directory holding `Dexcom_NNN.csv` and
`Food_Log_NNN.csv`, which makes BIG IDEAs a bundle-per-subject corpus: many
files in, one frame out per person. There is one sensor, so keys from
`parse_corpus` are bare subject ids with no `/track` suffix.

The Dexcom file **is** a Clarity export — `parse_file` on it alone correctly
detects as DEXCOM and returns glucose without meals. That is why this format
is identified by **directory shape**, not by sniffing the Dexcom header.
A bare `Food_Log_*.csv` is the text-detectable half of the bundle, and
`parse_to_unified` refuses it: parsing the food log alone would silently
return meals with no glucose.

Ground truth, read off all 16 published subjects rather than the data
dictionary:

===========================  ==========  ====================================
Variant                      Subjects    Handling
===========================  ==========  ====================================
Canonical 14-col food header 11          the schema
`time` renamed `time_of_day` 4           alias (`007`, `013`, `015`, `016`)
Headerless 11-col food log   1           `003` — first row is data, and
                                         `time_end` / `sugar` / `total_fat`
                                         are absent
Empty `time_begin`           1 row       fall back to `date` + `time`
                                         (`012`, Boost at 07:00)
Dexcom missing Transmitter   16          already tolerated by `_process_dexcom`
 ID
Metadata row count drift     12 of 16    no `PatientIdentifier` row; the
                                         Dexcom parser drops extra
                                         blank-timestamp rows
===========================  ==========  ====================================

There are no meal photographs. Food items stay one row each — clustering
items into a sitting is a consumer concern, not a parser one. Macronutrients
land on the extended schema; the food name, amount, unit and sugar have no
typed home and go in `annotations`.

`Demographics.csv` at the corpus root is per-subject attributes, not a time
series, and has no home in a frame keyed by timestamp.
"""

from typing import Tuple

import polars as pl

from cgm_format.interface.schema import (
    CGMSchemaDefinition,
    EnumLiteral,
    regenerate_schema_json as _regenerate,
)

BIGIDEAS_HEADER_LINE = 1
BIGIDEAS_DATA_START_LINE = 2
BIGIDEAS_METADATA_LINES: Tuple[int, ...] = ()

# `time_begin` is ISO-like with a space. The `date` column is ISO on most
# subjects and US `MM/DD/YYYY` on the `time_of_day` variant, so both are
# probed. Seconds are omitted on that variant's clock column.
BIGIDEAS_FOOD_TIMESTAMP_FORMATS: Tuple[str, ...] = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M:%S",
)
BIGIDEAS_DATE_FORMATS: Tuple[str, ...] = ("%Y-%m-%d", "%m/%d/%Y")
BIGIDEAS_TIME_FORMATS: Tuple[str, ...] = ("%H:%M:%S", "%H:%M")

# Text detection for a bare Food_Log_*.csv. Each pattern must identify the
# format on its own (`detect_format` is disjunctive). None of these strings
# appear in a Dexcom, Libre, CGMacros or D1NAMO header.
BIGIDEAS_DETECTION_PATTERNS = [
    "logged_food",
    "searched_food",
    "time_begin,time_end",
]

# Directory shape. Conjunctive: a corpus root holds per-subject Dexcom *and*
# food-log files. Dexcom_*.csv alone would also match a folder of renamed
# Clarity exports; the food log is what makes this BIG IDEAs.
BIGIDEAS_PATH_PROBES: Tuple[str, ...] = (
    "*/Dexcom_*.csv",
    "*/Food_Log_*.csv",
)

# One level down, for `detect_subject_format`. Dropping the `*/` keeps the
# two probe sets disjoint: a corpus root has no Dexcom_*.csv beside it, and
# a subject directory has no matching grandchild.
BIGIDEAS_SUBJECT_PROBES: Tuple[str, ...] = (
    "Dexcom_*.csv",
    "Food_Log_*.csv",
)

#: Single track name. Named after the device, matching CGMacros' `"dexcom"`.
BIGIDEAS_TRACK: str = "dexcom"

#: Subject `003` ships a food log with no header row and three columns
#: dropped (`time_end`, `sugar`, `total_fat`). Field count is 11; the
#: mapping was read off the file against known foods (chicken nuggets:
#: 393 kcal / 19 g carb / 0.1 g fiber / 20 g protein).
BIGIDEAS_FOOD_HEADERLESS_11: Tuple[str, ...] = (
    "date",
    "time",
    "time_begin",
    "logged_food",
    "amount",
    "unit",
    "searched_food",
    "calorie",
    "total_carb",
    "dietary_fiber",
    "protein",
)


class BigIdeasFoodColumn(EnumLiteral):
    """Column names in a BIG IDEAs `Food_Log_*.csv`."""

    DATE = "date"
    TIME = "time"
    TIME_BEGIN = "time_begin"
    TIME_END = "time_end"
    LOGGED_FOOD = "logged_food"
    AMOUNT = "amount"
    UNIT = "unit"
    SEARCHED_FOOD = "searched_food"
    CALORIE = "calorie"
    TOTAL_CARB = "total_carb"
    DIETARY_FIBER = "dietary_fiber"
    SUGAR = "sugar"
    PROTEIN = "protein"
    TOTAL_FAT = "total_fat"


BIGIDEAS_FOOD_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": BigIdeasFoodColumn.DATE.value,
            "dtype": pl.Utf8,
            "description": "Meal date; ISO on most subjects, US MM/DD/YYYY on the time_of_day variant",
        },
        {
            "name": BigIdeasFoodColumn.TIME.value,
            "dtype": pl.Utf8,
            "description": "Clock time; aliased from time_of_day on 4 subjects",
            "aliases": ("time_of_day",),
        },
        {
            "name": BigIdeasFoodColumn.TIME_BEGIN.value,
            "dtype": pl.Utf8,
            "description": "Preferred meal timestamp (YYYY-MM-DD HH:MM:SS)",
        },
        {
            "name": BigIdeasFoodColumn.TIME_END.value,
            "dtype": pl.Utf8,
            "description": "Optional meal-end clock; absent on the headerless 11-col variant",
        },
    ),
    data_columns=(
        {
            "name": BigIdeasFoodColumn.LOGGED_FOOD.value,
            "dtype": pl.Utf8,
            "description": "Food name as the participant typed it",
        },
        {
            "name": BigIdeasFoodColumn.AMOUNT.value,
            "dtype": pl.Utf8,
            "description": "Amount, kept as text (mix of numbers and words such as 'quarter')",
        },
        {
            "name": BigIdeasFoodColumn.UNIT.value,
            "dtype": pl.Utf8,
            "description": "Unit of amount (fluid ounce, cup, gram, …)",
        },
        {
            "name": BigIdeasFoodColumn.SEARCHED_FOOD.value,
            "dtype": pl.Utf8,
            "description": "Matched database name, when the log found one",
        },
        {
            "name": BigIdeasFoodColumn.CALORIE.value,
            "dtype": pl.Utf8,
            "description": "Calories (kcal)",
            "unit": "kcal",
        },
        {
            "name": BigIdeasFoodColumn.TOTAL_CARB.value,
            "dtype": pl.Utf8,
            "description": "Carbohydrates in grams",
            "unit": "g",
            "aliases": ("total_carb ",),
        },
        {
            "name": BigIdeasFoodColumn.DIETARY_FIBER.value,
            "dtype": pl.Utf8,
            "description": "Fiber in grams",
            "unit": "g",
        },
        {
            "name": BigIdeasFoodColumn.SUGAR.value,
            "dtype": pl.Utf8,
            "description": "Sugar in grams; no unified column, recorded in annotations",
            "unit": "g",
        },
        {
            "name": BigIdeasFoodColumn.PROTEIN.value,
            "dtype": pl.Utf8,
            "description": "Protein in grams",
            "unit": "g",
        },
        {
            "name": BigIdeasFoodColumn.TOTAL_FAT.value,
            "dtype": pl.Utf8,
            "description": "Fat in grams",
            "unit": "g",
        },
    ),
    header_line=BIGIDEAS_HEADER_LINE,
    data_start_line=BIGIDEAS_DATA_START_LINE,
    metadata_lines=BIGIDEAS_METADATA_LINES,
)


def regenerate_schema_json() -> None:
    """Regenerate the committed Frictionless schema JSON for this format."""
    _regenerate(BIGIDEAS_FOOD_SCHEMA, __file__)
