"""D1NAMO research corpus — two subsets that are not one format.

Zenodo record `5651217` v1.2.0, CC BY-SA 4.0. Each subject is a *directory* of
files, one per modality — glucose here, insulin there, meals in a third — which
makes D1NAMO the motivating case for the bundle category: many files in, one
frame out per subject.

**Two registered formats, not one with a flag.** The subsets differ by more
than units, and `derive_schema` expresses renames and units, not a different
column set. Verified by reading both archives:

=======================  ==========================================  ==========================================
                         diabetes (9 subjects)                       healthy (20 subjects)
=======================  ==========================================  ==========================================
`food.csv` header        `picture,description,calories,balance,      `date,time,picture,description,calories,
                         quality,datetime`                           balance,quality`
`insulin.csv`            present                                     absent
`annotations.csv`        absent                                      present
`glucose.csv` `type`     `cgm` / `manual`                            `BB,AB,BL,AL,BD,AD` and empty
photo numbering          1-based (`001.jpg`)                         0-based (`000.jpg`)
=======================  ==========================================  ==========================================

`glucose.csv` shares one header — `date,time,glucose,type,comments` — and is
**mmol/L in both**, converted through the declared unit rather than a
per-vendor factor.

**The healthy subset has no CGM at all** — four to six fingersticks a day.
Those are calibration-style readings, and mapping them to `EGV_READ` would
present a fingerstick as a sensor trace. They map to `CALIBRAT`, exactly as
Libre strip readings already do (design decision D6).

**There is no carbohydrate column anywhere in D1NAMO.** Meals carry `calories`
plus human-assigned `balance` and `quality` labels, so `carbs` stays null — a
genuine "the source did not say", never a zero. That makes the food data
entirely dependent on the extended schema: without `calories` a meal row would
carry nothing but a timestamp and an annotation.

**Timestamp conventions are mixed inside one subject directory**, which is the
trap most likely to produce silently wrong data. All four were verified against
the archives:

===============================  ==============================  ==========================
File                             Literal                         Format
===============================  ==============================  ==========================
`glucose.csv`, `insulin.csv`     `2014-10-01` / `19:14:00`       `%Y-%m-%d` / `%H:%M:%S`
`glucose.csv` (healthy)          `2014-10-01` / `11:35`          `%H:%M` — **no seconds**
`food.csv` (diabetes)            `2014:10:01 19:27:49`           `%Y:%m:%d %H:%M:%S` — EXIF
`annotations.csv` (healthy)      `2014-10-01` / `11:35`          `%Y-%m-%d` / `%H:%M`
===============================  ==============================  ==========================

`annotations.csv` (healthy subset) **is** parsed: only the interval start becomes a row, because
the unified frame is instant-shaped and emitting an end row would double-count the event. The end is
preserved inside the annotation rather than discarded.

The Zephyr physiological streams are **deliberately not parsed** (D7). ECG is
250 Hz — roughly 86 million samples per subject over four days — and this is a
library whose unified schema is event- and reading-shaped. Declining loudly is
a decision; omitting quietly is a gap someone re-opens. Only the two annotation
archives are needed, not the full 11.2 GB corpus.
"""

from typing import Dict, Tuple

import polars as pl

from cgm_format.interface.schema import (
    CGMSchemaDefinition,
    EnumLiteral,
    regenerate_schema_json as _regenerate,
)

D1NAMO_HEADER_LINE = 1
D1NAMO_DATA_START_LINE = 2
D1NAMO_METADATA_LINES: Tuple[int, ...] = ()

#: Glucose is mmol/L in both subsets. Declared as a unit so conversion goes
#: through `_glucose_to_canonical` and `UNIT_CONVERSIONS`, never a local factor.
D1NAMO_GLUCOSE_UNIT = "mmol/L"

# `glucose.csv` (both subsets) and `insulin.csv` (diabetes only). The healthy
# subset omits seconds, so the shorter form must be probed too.
D1NAMO_DATE_FORMATS: Tuple[str, ...] = ("%Y-%m-%d", "%d/%m/%Y")
D1NAMO_TIME_FORMATS: Tuple[str, ...] = ("%H:%M:%S", "%H:%M")

#: `food.csv` in the diabetes subset carries an EXIF-style datetime with colons
#: in the date part. A naive parse silently rejects it.
D1NAMO_FOOD_DATETIME_FORMATS: Tuple[str, ...] = (
    "%Y:%m:%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S",
)

# Directory shape. Conjunctive, and the discriminating file is what separates
# the two subsets: insulin.csv exists only in diabetes, annotations.csv only in
# healthy. Registration order puts the more specific probe first.
D1NAMO_DIABETES_PATH_PROBES: Tuple[str, ...] = (
    "*/glucose.csv",
    "*/insulin.csv",
)
D1NAMO_HEALTHY_PATH_PROBES: Tuple[str, ...] = (
    "*/glucose.csv",
    "*/annotations.csv",
)

# Text detection for a bare glucose.csv. Both subsets share this header, so a
# single file cannot be told apart from its contents alone — which is exactly
# why the corpus is identified by directory shape instead.
D1NAMO_DETECTION_PATTERNS = [
    "date,time,glucose,type,comments",
]


class D1namoGlucoseColumn(EnumLiteral):
    """`glucose.csv` — identical header in both subsets."""

    DATE = "date"
    TIME = "time"
    GLUCOSE = "glucose"
    TYPE = "type"
    COMMENTS = "comments"


class D1namoInsulinColumn(EnumLiteral):
    """`insulin.csv` — diabetes subset only.

    The header was never recorded in the design doc's ground truth; read off
    the archive directly. `fast_insulin` and `slow_insulin` map straight onto
    the unified `insulin_fast` / `insulin_slow` columns.
    """

    DATE = "date"
    TIME = "time"
    FAST_INSULIN = "fast_insulin"
    SLOW_INSULIN = "slow_insulin"
    COMMENT = "comment"


class D1namoFoodColumn(EnumLiteral):
    """`food.csv` — diabetes subset (EXIF-style `datetime` column)."""

    PICTURE = "picture"
    DESCRIPTION = "description"
    CALORIES = "calories"
    BALANCE = "balance"
    QUALITY = "quality"
    DATETIME = "datetime"


class D1namoHealthyFoodColumn(EnumLiteral):
    """`food.csv` — healthy subset. A genuinely different column set.

    Split `date` + `time` instead of one `datetime`, and the columns arrive in
    a different order. This is why the subsets are two registered formats
    rather than one with a flag: `derive_schema` patches names and units, not
    a different set of columns.
    """

    DATE = "date"
    TIME = "time"
    PICTURE = "picture"
    DESCRIPTION = "description"
    CALORIES = "calories"
    BALANCE = "balance"
    QUALITY = "quality"


class D1namoAnnotationColumn(EnumLiteral):
    """`annotations.csv` — healthy subset only. Interval-shaped events."""

    START_DATE = "start_date"
    START_TIME = "start_time"
    END_DATE = "end_date"
    END_TIME = "end_time"
    TYPE = "type"
    DESCRIPTION = "description"


class D1namoGlucoseType(EnumLiteral):
    """The `type` vocabulary, disjoint between the subsets.

    Diabetes: `cgm` is a sensor trace, `manual` a fingerstick.
    Healthy: meal-relative fingerstick labels — before/after breakfast, lunch,
    dinner — and an empty string on 13 rows, which means the reading was taken
    without a recorded relation to a meal, not that it is invalid.
    """

    CGM = "cgm"
    MANUAL = "manual"
    BEFORE_BREAKFAST = "BB"
    AFTER_BREAKFAST = "AB"
    BEFORE_LUNCH = "BL"
    AFTER_LUNCH = "AL"
    BEFORE_DINNER = "BD"
    AFTER_DINNER = "AD"


#: Only `cgm` is a continuous sensor reading. Everything else in the vocabulary
#: is a fingerstick and maps to CALIBRAT (D6) — presenting one as a sensor
#: trace would misrepresent what the device did.
D1NAMO_SENSOR_TYPES: Tuple[str, ...] = (D1namoGlucoseType.CGM.value,)

#: Values that appear where a number or a label belongs and mean "no
#: information", not zero. Observed in the healthy subset's food.csv.
D1NAMO_NULL_LITERALS: Tuple[str, ...] = ("No information", "")

D1NAMO_GLUCOSE_SCHEMA = CGMSchemaDefinition(
    service_columns=(
        {
            "name": D1namoGlucoseColumn.DATE.value,
            "dtype": pl.Utf8,
            "description": "Reading date, ISO year-first",
            "constraints": {"required": True},
        },
        {
            "name": D1namoGlucoseColumn.TIME.value,
            "dtype": pl.Utf8,
            "description": "Reading time; seconds omitted in the healthy subset",
            "constraints": {"required": True},
        },
        {
            "name": D1namoGlucoseColumn.TYPE.value,
            "dtype": pl.Utf8,
            "description": "cgm/manual (diabetes) or meal-relative label (healthy)",
        },
    ),
    data_columns=(
        {
            "name": D1namoGlucoseColumn.GLUCOSE.value,
            "dtype": pl.Utf8,
            "description": (
                "Glucose reading in mmol/L. Read as text because the healthy "
                "subset contains literals a numeric reader would silently drop "
                "(a colon typed for a decimal point, leading zeros)."
            ),
            "unit": D1NAMO_GLUCOSE_UNIT,
        },
        {
            "name": D1namoGlucoseColumn.COMMENTS.value,
            "dtype": pl.Utf8,
            "description": "Free-text comment",
        },
    ),
    header_line=D1NAMO_HEADER_LINE,
    data_start_line=D1NAMO_DATA_START_LINE,
    metadata_lines=D1NAMO_METADATA_LINES,
)


def regenerate_schema_json() -> None:
    """Regenerate the committed Frictionless schema JSON for this format."""
    _regenerate(D1NAMO_GLUCOSE_SCHEMA, __file__)
