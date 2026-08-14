
from typing import Dict, List, Optional, Tuple
from cgm_format.interface.cgm_interface import FormatCategory, SupportedCGMFormat
from cgm_format.interface.schema import CGMSchemaDefinition
from cgm_format.formats.unified import (
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
    UNIFIED_DETECTION_PATTERNS,
    UNIFIED_DATA_START_LINE,
    UNIFIED_EXTENDED_DETECTION_PATTERNS,
    UNIFIED_EXTENDED_DATA_START_LINE,
)
from cgm_format.formats.dexcom import DEXCOM_SCHEMA, DEXCOM_DETECTION_PATTERNS, DEXCOM_DATA_START_LINE
from cgm_format.formats.dexcom_eu import DEXCOM_EU_SCHEMA, DEXCOM_EU_DETECTION_PATTERNS, DEXCOM_EU_DATA_START_LINE
from cgm_format.formats.libre import LIBRE_SCHEMA, LIBRE_DETECTION_PATTERNS, LIBRE_DATA_START_LINE
from cgm_format.formats.libre_eu import LIBRE_EU_SCHEMA, LIBRE_EU_DETECTION_PATTERNS, LIBRE_EU_DATA_START_LINE
from cgm_format.formats.medtronic import MEDTRONIC_SCHEMA, MEDTRONIC_DETECTION_PATTERNS, MEDTRONIC_DATA_START_LINE
from cgm_format.formats.nightscout import NIGHTSCOUT_ENTRIES_SCHEMA, NIGHTSCOUT_DETECTION_PATTERNS, NIGHTSCOUT_DATA_START_LINE



# Schema map for validation
SCHEMA_MAP: Dict[SupportedCGMFormat, CGMSchemaDefinition] = {
    SupportedCGMFormat.UNIFIED_EXTENDED: CGM_SCHEMA_EXTENDED,
    SupportedCGMFormat.UNIFIED_CGM: CGM_SCHEMA,
    SupportedCGMFormat.DEXCOM: DEXCOM_SCHEMA,
    SupportedCGMFormat.DEXCOM_EU: DEXCOM_EU_SCHEMA,
    SupportedCGMFormat.LIBRE: LIBRE_SCHEMA,
    SupportedCGMFormat.LIBRE_EU: LIBRE_EU_SCHEMA,
    SupportedCGMFormat.MEDTRONIC: MEDTRONIC_SCHEMA,
    SupportedCGMFormat.NIGHTSCOUT: NIGHTSCOUT_ENTRIES_SCHEMA,
}

# Insertion order IS detection priority: detect_format iterates this dict and
# returns on the first format whose pattern appears. Register the more specific
# identity first, always.
#
# UNIFIED_EXTENDED must precede UNIFIED_CGM: an extended round-trip CSV carries
# every core unified pattern (sequence_id, event_type, quality) too, so the
# generic entry would capture it. Its own ",annotations" pattern is what the
# extended file has and the core one does not.
# DEXCOM_EU must precede DEXCOM: the EU header also matches generic Dexcom
# patterns (e.g. "Timestamp (YYYY-MM-DDThh:mm:ss)"), so the more specific
# mmol/L check must win first. Same for LIBRE_EU before LIBRE: the mmol/L
# export also matches generic Libre patterns ("Glucose Data,Generated",
# "FreeStyle Libre"), so "Historic Glucose mmol/L" must win first.
FORMAT_DETECTION_PATTERNS: Dict[SupportedCGMFormat, List[str]] = {
    SupportedCGMFormat.UNIFIED_EXTENDED: UNIFIED_EXTENDED_DETECTION_PATTERNS,
    SupportedCGMFormat.UNIFIED_CGM: UNIFIED_DETECTION_PATTERNS,
    SupportedCGMFormat.DEXCOM_EU: DEXCOM_EU_DETECTION_PATTERNS,
    SupportedCGMFormat.DEXCOM: DEXCOM_DETECTION_PATTERNS,
    SupportedCGMFormat.LIBRE_EU: LIBRE_EU_DETECTION_PATTERNS,
    SupportedCGMFormat.LIBRE: LIBRE_DETECTION_PATTERNS,
    SupportedCGMFormat.MEDTRONIC: MEDTRONIC_DETECTION_PATTERNS,
    SupportedCGMFormat.NIGHTSCOUT: NIGHTSCOUT_DETECTION_PATTERNS,
}

FORMAT_DETECTION_LINE_COUNT: Dict[SupportedCGMFormat, int] = {
    SupportedCGMFormat.UNIFIED_EXTENDED: UNIFIED_EXTENDED_DATA_START_LINE,
    SupportedCGMFormat.UNIFIED_CGM: UNIFIED_DATA_START_LINE,
    SupportedCGMFormat.DEXCOM: DEXCOM_DATA_START_LINE,
    SupportedCGMFormat.DEXCOM_EU: DEXCOM_EU_DATA_START_LINE,
    SupportedCGMFormat.LIBRE: LIBRE_DATA_START_LINE,
    SupportedCGMFormat.LIBRE_EU: LIBRE_EU_DATA_START_LINE,
    SupportedCGMFormat.MEDTRONIC: MEDTRONIC_DATA_START_LINE,
    SupportedCGMFormat.NIGHTSCOUT: NIGHTSCOUT_DATA_START_LINE,
}

DETECTION_LINE_COUNT: int = max(FORMAT_DETECTION_LINE_COUNT.values())*2

# The *unified* schema each vendor format is parsed INTO — distinct from
# SCHEMA_MAP above, which describes the raw vendor file. The parser knows the
# vendor, so it looks its target up here rather than carrying a ClassVar (D2 of
# docs/PLAN_0.10.0.md): adding a format stays "add registry entries", which is
# the property the charter's registry section is protecting.
#
# Every SupportedCGMFormat member must appear. A source whose signal does not
# fit the core six data columns (macronutrients, wearable streams, free-form
# annotations) maps to CGM_SCHEMA_EXTENDED; everything a device exports today
# fits the core schema and maps to CGM_SCHEMA.
UNIFIED_TARGET_SCHEMA: Dict[SupportedCGMFormat, CGMSchemaDefinition] = {
    SupportedCGMFormat.UNIFIED_EXTENDED: CGM_SCHEMA_EXTENDED,
    SupportedCGMFormat.UNIFIED_CGM: CGM_SCHEMA,
    SupportedCGMFormat.DEXCOM: CGM_SCHEMA,
    SupportedCGMFormat.DEXCOM_EU: CGM_SCHEMA,
    SupportedCGMFormat.LIBRE: CGM_SCHEMA,
    SupportedCGMFormat.LIBRE_EU: CGM_SCHEMA,
    SupportedCGMFormat.MEDTRONIC: CGM_SCHEMA,
    SupportedCGMFormat.NIGHTSCOUT: CGM_SCHEMA,
}

# The source shape each format arrives as. A sidecar dict rather than a field
# on SupportedCGMFormat: the enum is public API and its members are compared
# and serialized by consumers, so widening its shape to carry metadata would
# change something a consumer reads for no gain the registry does not give.
#
# Every member must appear — `test_supported.py` asserts exhaustiveness, so a
# format added without a category fails the suite rather than silently
# defaulting to EXPORT.
#
# Everything here is EXPORT today, and that is the honest state rather than an
# oversight: BUNDLE and CORPUS describe entry points (parse_bundle,
# parse_corpus) that no *registered format* uses yet. Nightscout is the case
# worth explaining — `from_nightscout_exports` genuinely takes several files
# and is the bundle shape, but SupportedCGMFormat.NIGHTSCOUT identifies the
# single-file exporter CSV that `detect_format` recognizes. The bundle-ness
# lives in the entry point, not in the format identity.
FORMAT_CATEGORY: Dict[SupportedCGMFormat, FormatCategory] = {
    SupportedCGMFormat.UNIFIED_EXTENDED: FormatCategory.EXPORT,
    SupportedCGMFormat.UNIFIED_CGM: FormatCategory.EXPORT,
    SupportedCGMFormat.DEXCOM: FormatCategory.EXPORT,
    SupportedCGMFormat.DEXCOM_EU: FormatCategory.EXPORT,
    SupportedCGMFormat.LIBRE: FormatCategory.EXPORT,
    SupportedCGMFormat.LIBRE_EU: FormatCategory.EXPORT,
    SupportedCGMFormat.MEDTRONIC: FormatCategory.EXPORT,
    SupportedCGMFormat.NIGHTSCOUT: FormatCategory.EXPORT,
}

# Path-shaped detection, a second mechanism beside the text-prefix one.
#
# `detect_format` matches patterns against the first N lines of decoded text. A
# bundle or a corpus has no single text to sniff: what identifies it is
# *directory shape* — whether `CGMacros-001/CGMacros-001.csv` exists, whether
# there is a `diabetes_subset/`. So this is a separate registry rather than
# more patterns fed to the existing loop.
#
# Glob patterns only, never callables. `docs/NEW_SCHEMA.md` is explicit that
# schemas and registries stay pure data; a predicate here would move detection
# logic out of the parser and into the registry, which is the split the
# charter's registry section exists to prevent.
#
# Insertion order is priority, exactly as in FORMAT_DETECTION_PATTERNS: the
# first format whose every probe matches wins. Register the more specific
# identity first.
#
# Deliberately empty until Waves 4-5 register CGMacros and D1NAMO. Speculative
# entries for formats that do not exist would be fabricated values, and
# `CLAUDE.md` §2 forbids those; `detect_path_format` is exercised against
# synthetic trees instead.
PATH_DETECTION_PROBES: Dict[SupportedCGMFormat, Tuple[str, ...]] = {}

# Known issues to suppress per format (can't fix vendor CSV format issues)
KNOWN_ISSUES_TO_SUPPRESS = {
    SupportedCGMFormat.DEXCOM: [
        # Dexcom exports have variable-length rows - non-EGV events don't include
        # trailing Transmitter Time/ID columns (missing cells, not just empty values)
        ('missing-cell', 'Transmitter ID', None),
        ('missing-cell', 'Transmitter Time (Long Integer)', None),
        # Dexcom uses "Low" (<50 mg/dL) and "High" (>400 mg/dL) text markers 
        # instead of numeric values for out-of-range glucose readings
        ('type-error', 'Glucose Value (mg/dL)', 'Low'),
        ('type-error', 'Glucose Value (mg/dL)', 'High'),
        # Some Dexcom exports include UTF-8 BOM marker in header
        ('incorrect-label', 'Index', None),
        # Newer Clarity exports add an extra metadata row (e.g. "Sensor") beyond
        # the static comment-row count, so exactly ONE blank-timestamp
        # metadata/alert row leaks past the fixed dialect skip. The parser drops
        # it dynamically; here we tolerate the resulting required-constraint
        # error on that single blank cell. The optional 4th element caps how many
        # times the rule may suppress per file — a second blank timestamp would
        # be a real data issue and must still fail.
        ('constraint-error', 'Timestamp (YYYY-MM-DDThh:mm:ss)', None, 1),
    ],
    SupportedCGMFormat.DEXCOM_EU: [
        ('missing-cell', 'Transmitter ID', None),
        ('missing-cell', 'Transmitter Time (Long Integer)', None),
        ('type-error', 'Glucose Value (mmol/L)', 'Low'),
        ('type-error', 'Glucose Value (mmol/L)', 'High'),
        ('incorrect-label', 'Index', None),
        # Same single-row metadata-drift tolerance as standard Dexcom (see above):
        # the EU export may likewise gain/lose a metadata row across versions.
        ('constraint-error', 'Timestamp (YYYY-MM-DDThh:mm:ss)', None, 1),
    ],
    SupportedCGMFormat.UNIFIED_CGM: [], #this is ours, none should be suppressed
    SupportedCGMFormat.UNIFIED_EXTENDED: [], #also ours, same reason
    SupportedCGMFormat.LIBRE: [],
    SupportedCGMFormat.LIBRE_EU: [],
    SupportedCGMFormat.MEDTRONIC: [
        # Medtronic CareLink exports contain "-------" placeholders in numeric columns,
        # repeated header rows mid-file, and European decimal format (comma separator)
        ('type-error', 'Sensor Glucose (mg/dL)', None),
        ('type-error', 'BG Reading (mg/dL)', None),
        ('type-error', 'Bolus Volume Delivered (U)', None),
        ('type-error', 'BWZ Carb Input (grams)', None),
        # BOM marker in header
        ('incorrect-label', 'Index', None),
    ],
    SupportedCGMFormat.NIGHTSCOUT: [
        # nightscout-exporter CSV uses blank lines and "# TREATMENTS ..." comment
        # lines as section separators — frictionless sees these as blank/missing rows
        ('blank-row', None, None),
        ('missing-cell', 'Type', None),
        ('missing-cell', 'Device', None),
        ('missing-cell', 'Trend', None),
        ('missing-cell', 'ID', None),
    ],
}
