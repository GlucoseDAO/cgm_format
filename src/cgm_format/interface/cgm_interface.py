"""Abstract Base Class interface for CGM data processing pipeline.

Separated into two concerns:
- CGMParser: Vendor-specific parsing to unified format (Stages 1-3)
- CGMProcessor: Vendor-agnostic unified format processing (Stages 4-5)
"""

from datetime import datetime
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Flag, auto
from typing import Union, Tuple, List, Dict, Sequence, ClassVar, TYPE_CHECKING
from enum import Enum
from pathlib import Path
import polars as pl
from base64 import b64decode

if TYPE_CHECKING:
    from cgm_format.interface.schema import CGMSchemaDefinition

# Check pandas availability
try:
    import pyarrow as pa
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

# from schema import EventType, Quality

CALIBRATION_GAP_THRESHOLD = 2*60*60+45*60  # 2 hours and 45 minutes
CALIBRATION_PERIOD_HOURS = 24

EXPECTED_INTERVAL_MINUTES = 5
TOLERANCE_INTERVAL_MINUTES = 1.2*EXPECTED_INTERVAL_MINUTES
# Threshold for small (fillable) gaps: up to 3 intervals (15 min).
# Must be a multiple of EXPECTED_INTERVAL_MINUTES so that grid-aligned gap
# measurement (used for commutativity between interpolate_gaps and
# synchronize_timestamps) produces stable fill/skip decisions.
# Matches glucose_data_processing's default small_gap_max_minutes=15.
SMALL_GAP_MAX_MINUTES = EXPECTED_INTERVAL_MINUTES * 3

# Expected sequence (- = 1 minute, | = registered value, every 5 minutes):
#|-----|-----|-----|-----|-----|
# Fillable gap schema (x = missing value, : = synchronized value, | = registered value):
#|---|--:-----x-----:--|---|

MINIMUM_DURATION_MINUTES = 60 # minimum expected duration of a sequence for inference
MAXIMUM_WANTED_DURATION_MINUTES = 480 # maximum duration of a sequence to be included in the inference

# Type alias to highlight that this is the unified format
# (No way to add additional constraints on DF shape in type hints)
UnifiedFormat = pl.DataFrame

class SupportedCGMFormat(Enum):
    """Supported CGM vendor formats."""
    DEXCOM = "dexcom"
    DEXCOM_EU = "dexcom_eu"  # European Dexcom exports with mmol/L units
    LIBRE = "libre"
    LIBRE_EU = "libre_eu"  # European Libre exports with mmol/L units
    MEDTRONIC = "medtronic"
    NIGHTSCOUT = "nightscout"
    UNIFIED_CGM = "unified"  # Format that this library provides
    UNIFIED_EXTENDED = "unified_extended"  # Unified format widened with macros/wearables/annotations
    CGMACROS = "cgmacros"  # PhysioNet research corpus: 45 subjects, two concurrent sensors
    # Two registered formats, not one with a flag: the subsets differ in
    # food.csv's header, in which modality files exist, and in a disjoint
    # glucose type vocabulary. derive_schema patches names and units, not a
    # different column set.
    D1NAMO_DIABETES = "d1namo_diabetes"  # glucose + insulin + meals, CGM present
    D1NAMO_HEALTHY = "d1namo_healthy"  # glucose + meals + annotations, fingersticks only
    BIGIDEAS = "bigideas"  # PhysioNet BIG IDEAs: Dexcom Clarity + food log, one subject per directory

class FormatCategory(Enum):
    """How many files a source arrives as, and how many subjects are in them.

    Every format the library supported before 0.10.0 was one shape — one file,
    one subject, one device — and that assumption was never written down
    because nothing violated it. It is baked into `parse_file(path) ->
    UnifiedFormat` and into `detect_format`, which reads a text prefix of a
    single file. Research corpora break it along two *orthogonal* axes: how
    many files come in, and how many subjects are in them. Two axes give three
    categories rather than a spectrum.

    The categories **compose**: a corpus's member is a bundle or an export, so
    `parse_corpus` is built out of `parse_bundle` rather than implemented a
    third time. That composition is most of the value of naming them.

    A plain `Enum`, not an `EnumLiteral`: this vocabulary describes a source's
    shape and is never written into a frame or a CSV, so it needs no
    string-comparison behaviour. `EnumLiteral` exists for values that must
    survive a round-trip through a data column.
    """

    #: One file, one subject. `parse_file(path) -> UnifiedFormat`.
    EXPORT = "export"

    #: Several files, ONE subject, each file a different modality —
    #: glucose here, insulin there, meals in a third.
    #: `parse_bundle(paths) -> UnifiedFormat`, merging on the modality axis.
    BUNDLE = "bundle"

    #: Many subjects. `parse_corpus(root) -> dict[str, UnifiedFormat]`,
    #: one frame per subject, identity living in the key and never in a column.
    CORPUS = "corpus"


@dataclass(frozen=True)
class TrackCoverage:
    """How much glucose one track of one subject actually carries.

    Reported by `CGMParser.list_subjects` so a caller can size and filter a
    corpus **before** parsing it. Counted from the subject's raw glucose
    source, not from a parsed frame: the two are independently authored, which
    is what makes comparing them a real cross-check rather than a restatement.

    `values` counts cells that hold **something**, and that is deliberately not
    the same as readings the schema can represent. D1NAMO ships a glucose cell
    reading `7:0` — a colon typed for a decimal point — which is a value the
    source did say and the parser then drops with a warning. Counting it here
    keeps the two numbers honestly different: `values` is what the source
    offered, and a smaller parsed count means the parser rejected something,
    which is a finding rather than a discrepancy to reconcile away.

    `rows` is every row of the track's source, so `values / rows` is the
    fraction of the period the track speaks for. Both are three-valued at the
    boundary: a track that could not be read at all is absent from
    `SubjectEntry.tracks` rather than present with zeros, because "nothing was
    recorded" and "we could not look" are different answers (`CLAUDE.md` §5).
    """

    #: Track name, matching `parse_tracks` keys — `"libre"`, `"dexcom"`. A
    #: single-track source reports one entry named `"glucose"`, which names the
    #: *column* rather than a device: D1NAMO's healthy subset has no CGM at
    #: all, and calling its fingersticks a sensor track would assert a sensor
    #: trace that does not exist.
    track: str

    #: Rows whose glucose cell holds something — the source's own offer,
    #: before the parser decides what it can represent.
    values: int

    #: Total rows in the track's source, so `values / rows` is coverage.
    rows: int

    #: Earliest and latest timestamp carrying a glucose value, or `None` when
    #: the track has no values at all. `None` means "no reading to date", never
    #: a sentinel epoch.
    first: Union[datetime, None] = None
    last: Union[datetime, None] = None


@dataclass(frozen=True)
class SubjectEntry:
    """One member of a corpus, described without parsing it.

    `CGMParser.list_subjects` returns these so the id passed to
    `parse_corpus(root, subjects=[...])` is one the caller *read off the
    corpus* rather than guessed, and so the choice of which subjects to parse
    can be made on evidence — how much glucose each carries, which modalities
    it has — instead of after a full parse.

    A frozen dataclass for the same reason `CGMSchemaDefinition` is one: it
    describes something fixed at the moment it was read, and a mutable record
    of a directory's shape invites being edited into disagreement with the
    directory.
    """

    #: Subject id, exactly as `parse_corpus` keys it. For a multi-track corpus
    #: the corpus key is `f"{subject_id}/{track}"`; the `/` is public contract
    #: and a subject id never contains one.
    subject_id: str

    #: Which registered format this subject belongs to.
    format: "SupportedCGMFormat"

    #: The subject's directory.
    path: Path

    #: Modality **CSV** file names present, sorted — `("food.csv",
    #: "glucose.csv", "insulin.csv")`. Names only, as the vendor spells them:
    #: translating them here would invent a vocabulary. CSVs only, so a
    #: `food_pictures/` directory is a modality this does not enumerate — and
    #: so a stray `.DS_Store` is not reported as one either.
    modalities: Tuple[str, ...] = ()

    #: One entry per glucose track that could be read, in `parse_tracks` order.
    #: A track the reader could not open is **absent** rather than zeroed.
    tracks: Tuple[TrackCoverage, ...] = ()


class ValidationMethod(Flag):
    """Validation method for validating the input and output dataframes."""
    INPUT = auto()
    OUTPUT = auto()
    INPUT_FORCED = auto()
    OUTPUT_FORCED = auto()

NO_VALIDATION = ValidationMethod(0)
class ProcessingWarning(Flag):
    """Warnings that can occur during additional transformations.
    
    These are flags that can be combined using bitwise OR operations.
    Example: warnings = ProcessingWarning.TOO_SHORT | ProcessingWarning.QUALITY
    """
    TOO_SHORT = auto()  # Minimum duration requirement not met
    CALIBRATION = auto()  # Output sequence contains calibration events or 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD
    OUT_OF_RANGE = auto()  # Contains out-of-range values
    IMPUTATION = auto()  # Contains imputed gaps
    TIME_DUPLICATES = auto()  # Sequence contains non-unique time entries
    SYNCHRONIZATION = auto()  # Sequence undergone synchronization corrections
    QUALITY = auto()  # Other quality issues

NO_WARNING = ProcessingWarning(0)

class WarningDescription(Enum):
    """Descriptions of warnings."""
    TOO_SHORT = "Minimum duration requirement not met"
    CALIBRATION = "Sequence contains calibration events or 24hr period after gap ≥ CALIBRATION_GAP_THRESHOLD"
    OUT_OF_RANGE = "Contains out-of-range values"
    IMPUTATION = "Contains imputed gaps"
    TIME_DUPLICATES = "Sequence contains non-unique time entries"
    SYNCHRONIZATION = "Sequence undergone synchronization corrections"
    QUALITY = "Other quality issues"

# Simple tuple return types
ValidationResult = Tuple[pl.DataFrame, int, int]  # (dataframe, bad_rows, valid_rows)
InferenceResult = Tuple[UnifiedFormat, ProcessingWarning]  # (dataframe, warnings)



class MalformedDataError(ValueError):
    """Raised when data cannot be parsed or converted properly."""
    pass

class MissingColumnError(MalformedDataError):
    """Raised when a required column is missing from the dataframe."""
    pass

class ExtraColumnError(MalformedDataError):
    """Raised when an extra column is present in the dataframe."""
    pass

class ColumnOrderError(MalformedDataError):
    """Raised when the column order is not correct."""
    pass

class ColumnTypeError(MalformedDataError):
    """Raised when the column type is not correct."""
    pass

class UnknownFormatError(ValueError):
    """Raised when format cannot be determined."""
    pass

class ZeroValidInputError(ValueError):
    """Raised when there are no valid data points in the sequence."""
    pass

class MultiTrackSourceError(ValueError):
    """Raised when a multi-track source is parsed as if it were a single frame.

    A multi-track source carries more than one independent measurement of the
    same quantity — CGMacros wears a Libre and a Dexcom over the same ten days.
    There is no honest way for `parse_file` to return one frame from that: it
    would have to pick a sensor silently, and the caller could not see which.
    So it refuses and names both the tracks and the entry point that returns
    them.
    """
    pass

# Maximum length for error messages to prevent huge CSV dumps in logs
MAX_ERROR_MESSAGE_LENGTH = 8192

def truncate_error_message(message: str, max_length: int = MAX_ERROR_MESSAGE_LENGTH) -> str:
    """Truncate error message to prevent huge data dumps in logs.
    
    Args:
        message: Original error message
        max_length: Maximum length in bytes (default 8192)
        
    Returns:
        Truncated error message with indicator if truncated
    """
    if len(message) <= max_length:
        return message
    
    truncated = message[:max_length]
    return f"{truncated}... [ERROR MESSAGE TRUNCATED - original length: {len(message)} bytes]"

class CGMParser(ABC):
    """Abstract base class for vendor-specific CGM data parsing (Stages 1-3).
    
    This interface handles:
    - Stage 1: Preprocessing raw data (BOM removal, encoding fixes)
    - Stage 2: Format detection (identifying vendor)
    - Stage 3: Vendor-specific parsing to unified format
    
    After stage 3, data is in UnifiedFormat and can be serialized or passed to CGMProcessor.

    """

    #: Unified schemas a merged frame may conform to, widest first. Used only
    #: to give `merge_bundle_frames` a canonical column ordering. Declared here
    #: and populated by the concrete parser because `formats.unified` imports
    #: `interface.schema`, which imports this module — naming a schema at this
    #: level would close an import cycle. Empty is safe: the canonical ordering
    #: then degrades to alphabetical, which is still a pure function of the
    #: column set and so still deterministic.
    unified_schemas: ClassVar[Tuple["CGMSchemaDefinition", ...]] = ()

    # ===== STAGE 1: Preprocess Raw Data =====
    
    @classmethod
    @abstractmethod
    def decode_raw_data(cls, raw_data: Union[bytes, str]) -> str:
        """Remove BOM marks, encoding artifacts, and other junk from raw input.
        
        Args:
            raw_data: Raw file contents (bytes or string)
            
        Returns:
            Cleaned string data ready for format detection
        """
        pass
    
    # ===== STAGE 2: Format Detection  =====
    
    @classmethod
    @abstractmethod
    def detect_format(cls, text_data: str) -> SupportedCGMFormat:
        """Guess the vendor format based on header patterns in raw CSV string.
        
        This determines which vendor-specific processor to use.
        Works on string data before parsing to avoid vendor-specific CSV quirks.
        
        Args:
            text_data: Preprocessed string data
            
        Returns:
            SupportedCGMFormat enum value 
            
        Raises:
            UnknownFormatError: If format cannot be determined
        """
        pass

    @classmethod
    @abstractmethod
    def format_supported(cls, raw_data: Union[bytes, str]) -> bool:
        """Check if the library can parse the given data format.
        
        Uses the detector to determine if the format is supported without parsing the data.
        
        Args:
            raw_data: Raw file contents (bytes or string)
            
        Returns:
            True if format is supported and can be parsed, False otherwise
        """
        pass

    # ===== STAGE 3: Device-Specific Parsing to Unified Format =====
    
    @classmethod
    @abstractmethod
    def parse_to_unified(cls, text_data: str, format_type: SupportedCGMFormat) -> UnifiedFormat:
        """Parse vendor-specific CSV to unified format (device-specific parsing).
        
        This stage combines:
        - CSV validation and sanity checks
        - Vendor-specific quirk handling (High/Low values, timezone fixes, etc.)
        - Column mapping to unified schema
        - Populating service fields (event_type, quality)
        - Sequence detection and assignment (sequence_id)
        
        After this stage, processing flow converges to UnifiedFormat with sequence_id assigned.
        
        Args:
            text_data: Preprocessed string data
            
        Returns:
            DataFrame in unified format matching CGM_SCHEMA with sequence_id assigned
            
        Raises:
            MalformedDataError: If CSV is unparseable, zero valid rows, or conversion fails
        """
        pass
    
    # ===== Serialization (Roundtrip Support) =====
    
    @staticmethod
    def to_csv_string(dataframe: UnifiedFormat) -> str:
        """Serialize unified format DataFrame to CSV string.
        
        Args:
            dataframe: DataFrame in unified format
            
        Returns:
            CSV string representation of the unified format
        """
        return dataframe.write_csv(separator=",")
    
    @staticmethod
    def to_csv_file(dataframe: UnifiedFormat, file_path: str) -> None:
        """Save unified format DataFrame to CSV file.
        
        Args:
            dataframe: DataFrame in unified format
            file_path: Path where to save the CSV file
        """
        dataframe.write_csv(file_path)
    
    # ===== Convenience Methods =====
    
    @classmethod
    def parse_from_bytes(cls, raw_data: bytes) -> UnifiedFormat:
        """Convenience method to parse raw bytes directly to unified format.
        
        This method chains all stages together:
        1. Decode raw data
        2. Detect format
        3. Parse to unified format
        
        Args:
            raw_data: Raw file contents as bytes
            
        Returns:
            DataFrame in unified format with sequence_id assigned
            
        Raises:
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        text_data = cls.decode_raw_data(raw_data)
        format_type = cls.detect_format(text_data)
        return cls.parse_to_unified(text_data, format_type)
    
    @classmethod
    def parse_from_string(cls, text_data: str) -> UnifiedFormat:
        """Convenience method to parse cleaned string directly to unified format.
        
        This method assumes data is already decoded and chains:
        1. Detect format
        2. Parse to unified format
        
        Args:
            text_data: Cleaned CSV string
            
        Returns:
            DataFrame in unified format with sequence_id assigned
            
        Raises:
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        format_type = cls.detect_format(text_data)
        return cls.parse_to_unified(text_data, format_type)
    
    @classmethod
    def parse_file(cls, file_path: Union[str, Path]) -> UnifiedFormat:
        """Parse CGM data from file path.
        
        Convenience method that reads file and parses to unified format.
        Automatically detects format and handles encoding.
        
        Args:
            file_path: Path to CGM data file (CSV format)
            
        Returns:
            DataFrame in unified format
            
        Raises:
            FileNotFoundError: If file doesn't exist
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        
        file_path = Path(file_path)
        with open(file_path, 'rb') as f:
            raw_data = f.read()
        
        return cls.parse_from_bytes(raw_data)
    
    @classmethod
    def parse_bundle(cls, paths: Sequence[Union[str, Path]]) -> UnifiedFormat:
        """Parse several files describing ONE subject into a single frame.

        A **bundle** is the second source category: several files, one subject,
        each file a different *modality* — glucose in one, insulin in another,
        meals in a third. They merge to one frame because they are different
        views of the same record, which is a diagonal concat.

        **A member may be a subject directory rather than a file.** Passing
        `[root / "001"]` parses that whole subject, because the directory is
        the bundle — its files are the modalities, and which files are present
        is what identifies the format (`detect_subject_format`). Naming the
        files individually does not work for every corpus and is not meant to:
        D1NAMO's `parse_file` refuses a bare `glucose.csv` precisely because
        one modality is not a record. `list_subjects(root)` enumerates the
        directories worth passing here.

        This category already existed unrecognized: `from_nightscout_exports`
        takes entries + treatments describing one person and merges exactly
        this way. It was written as a Nightscout special case and is in fact
        the general shape, which is why an app-API pull (several endpoints, one
        user) is a bundle by any other name.

        **Merging modalities is not merging subjects, and the library cannot
        tell them apart.** Two files from *different people* concatenate just
        as cleanly as two modalities from one, and nothing downstream raises:
        `_postprocess_unified` sorts by `datetime`, so two subjects interleave,
        and `detect_and_assign_sequences` then splices them into shared
        sequences. Interpolation will happily invent rows bridging one person's
        Tuesday to another's Thursday. No check here can catch that — a
        subject's identity is not in the data — so **the caller owns the
        guarantee that these paths describe one subject.** For many subjects,
        use `parse_corpus`, which does exactly this: one frame per subject in a
        mapping keyed by subject id, identity in that key and never in a
        column.

        Each file is detected and parsed independently, so a bundle may mix
        formats. Rows are concatenated diagonally, so a column absent from one
        member is null there rather than an error — "this modality did not say"
        rather than a shape mismatch.

        **Members must be disjoint modalities, not two views of the same one.**
        Bundling two *glucose* sources for one subject — two sensors worn
        concurrently, or an export overlapping a Nightscout pull — yields two
        readings per timestamp, which then splice into shared sequences and get
        interpolated across, exactly as two subjects would. That case is a
        *track*, not a bundle: keep each sensor's series as its own frame and
        compare them rather than stacking them. `parse_tracks` serves it,
        returning one frame per sensor and flagging any synthesized value with
        `Quality.TRACK_MERGE`. Nothing on **this** path sets that flag, because
        nothing here merges readings.

        The result is **not** revalidated against a schema. A bundle of a core
        and an extended member is legitimately the wider shape, and narrowing
        it here would discard the extended member's channels; conversely
        enforcing the wider schema would invent columns for a core-only bundle.
        Validate against the schema you expect, or narrow with
        `FormatProcessor.to_core_df`.

        Args:
            paths: Files — or subject directories — describing one subject, as
                a sequence. Order does not matter; the result is
                deterministically ordered either way. A bare `str` or `Path` is
                rejected rather than iterated.

        Returns:
            One DataFrame in unified format.

        Raises:
            TypeError: If `paths` is a single `str` or `Path` rather than a
                sequence of them. A `str` is itself a sequence of `str`, so it
                would otherwise be walked character by character and fail on a
                one-letter filename.
            ValueError: If `paths` is empty. An empty bundle has no
                meaningful frame to return, and returning an empty one would
                be a silent substitute for "you gave me nothing".
            FileNotFoundError: If any path does not exist.
            UnknownFormatError: If a member's format cannot be determined, or a
                directory member matches no registered subject shape.
            MalformedDataError: If any member cannot be parsed.
        """
        if isinstance(paths, (str, Path)):
            raise TypeError(
                "parse_bundle takes a sequence of paths, not a single path — "
                f"got {type(paths).__name__}. A str is itself a sequence of "
                "str, so iterating it would walk the filename character by "
                f"character. Pass [{paths!r}], or use parse_file for one file."
            )

        resolved = [Path(p) for p in paths]
        if not resolved:
            raise ValueError(
                "parse_bundle requires at least one path; got an empty sequence"
            )

        # A directory member is a whole subject; a file member is one modality.
        # Dispatching here rather than inside parse_file keeps parse_file's
        # contract — one file in, one frame out — and keeps the refusal it
        # raises on a bare D1NAMO glucose.csv meaningful.
        frames = [
            cls.parse_subject_directory(path) if path.is_dir()
            else cls.parse_file(path)
            for path in resolved
        ]

        # Every bundle goes through the merge, including a one-member one: the
        # merge is the documented subclass extension point, and skipping it for
        # a single file would make an override's behaviour depend on how many
        # files the caller happened to pass.
        return cls.merge_bundle_frames(frames)

    @classmethod
    def parse_subject_directory(cls, subject_dir: Union[str, Path]) -> UnifiedFormat:
        """Parse one subject directory — the bundle-shaped member of a corpus.

        Where `parse_file` takes a file and `parse_corpus` takes a corpus root,
        this takes the thing between them: one subject's folder, whose files
        are its modalities. The format is identified by which files are present
        (`detect_subject_format`), because a corpus member's *contents* usually
        look like a plain vendor export and sniffing one would mis-route the
        whole subject.

        Reached through `parse_bundle([subject_dir])` in normal use; named
        separately so the dispatch has somewhere to live and so a subclass can
        override the per-subject step without reimplementing the merge.

        Args:
            subject_dir: One subject's directory.

        Returns:
            One DataFrame in unified format, all modalities merged.

        Raises:
            UnknownFormatError: If the directory matches no registered subject
                shape.
            MalformedDataError: If the subject cannot be parsed.
        """
        raise NotImplementedError(
            "parse_subject_directory is implemented by the concrete parser"
        )

    @classmethod
    def merge_bundle_frames(cls, frames: Sequence[UnifiedFormat]) -> UnifiedFormat:
        """Merge already-parsed bundle members into one frame.

        Split out from `parse_bundle` so a caller that already holds frames —
        `from_nightscout_url` pulling several API endpoints, or a corpus
        walker that has parsed a subject directory — reuses the same merge
        rather than reimplementing the concat. Subclasses that need
        vendor-specific merge behaviour override this one method.

        Ordering is taken from the merged frame's own columns rather than from
        a named schema, because a diagonal concat of a core member and an
        extended one yields the wider shape and the core key list would leave
        the extended columns outside the total ordering.

        But the concat's *own* column order is not usable as a sort key
        either: it is union-by-first-appearance, so two members carrying
        **disjoint** extra columns produce a different key list depending on
        which was passed first (`[…, calories, heart_rate]` one way,
        `[…, heart_rate, calories]` the other). Sorting by that would make row
        order depend on argument order — the precise nondeterminism the
        stable-sort invariant exists to remove. So the keys are canonicalized:
        columns present in the schema keep their schema order, and any
        remainder follows in a stable alphabetical tail.

        This method does **not** validate: a bundle may legitimately be wider
        than the core schema, and both narrowing and enforcing would lose
        information. Its only job is a defined row order.

        Args:
            frames: Parsed unified frames belonging to one subject.

        Returns:
            One DataFrame in unified format, diagonally concatenated.

        Raises:
            ValueError: If `frames` is empty.
        """
        if not frames:
            raise ValueError(
                "merge_bundle_frames requires at least one frame; got none"
            )

        merged = pl.concat(frames, how="diagonal")
        keys = cls._canonical_sort_keys(merged.columns)
        # Canonicalize the column *layout* as well as the row order. `sort`
        # only permutes rows, so without the `select` a frame's columns would
        # still sit in concat order and two argument orders would produce
        # frames that differ by layout alone — CSV bytes included.
        return merged.select(keys).sort(keys)

    @classmethod
    def _canonical_sort_keys(cls, columns: Sequence[str]) -> List[str]:
        """Order a merged frame's columns independently of concat order.

        Known columns first, in the order the widest registered unified schema
        declares them — that is the total ordering the determinism invariant is
        defined against. Anything no schema declares follows in a sorted tail.
        Either way the result is a pure function of the column *set*, never of
        the order the members arrived in.

        The schemas come from `unified_schemas`, a ClassVar the concrete parser
        populates, rather than from an import: `formats.unified` imports
        `interface.schema`, which imports this module, so naming a schema here
        would close a cycle.
        """
        present = set(columns)
        known: List[str] = []
        for schema in cls.unified_schemas:
            for name in schema.get_column_names():
                if name in present and name not in known:
                    known.append(name)
        return known + sorted(present - set(known))

    @classmethod
    def parse_tracks(cls, file_path: Union[str, Path]) -> "Dict[str, UnifiedFormat]":
        """Parse a multi-track source into one frame per track.

        A **track** is one of several independent measurements of the same
        quantity in one source — CGMacros wears a Libre and a Dexcom over the
        same ten days, and the two disagree by design.

        **Tracks are alternative views, never shards.** Rows belonging to
        neither device — meals, macronutrients, heart rate, photo annotations —
        are *replicated into every track*, so each frame is a complete
        self-contained view of the period as seen through one sensor. The
        consequence matters more than the rule: **concatenating two track
        frames double-counts every meal**, giving carbohydrate totals exactly
        twice reality with nothing raised anywhere. Pick one, or compare them.
        Never add them.

        Args:
            file_path: A multi-track source file.

        Returns:
            Track name → frame. Keys are the source's real sensors; a synthetic
            track (a per-timestamp mean, say) is never among them, because it
            is a derived view rather than a member of the corpus.

        Raises:
            UnknownFormatError: If the format cannot be determined.
            NotImplementedError: If the format is single-track — `parse_file`
                is the entry point for those, and returning a one-entry mapping
                would blur a distinction the categories exist to draw.
        """
        raise NotImplementedError(
            "parse_tracks is implemented by the concrete parser"
        )

    @classmethod
    def parse_corpus(cls, root: Union[str, Path]) -> "Dict[str, UnifiedFormat]":
        """Parse a many-subject corpus into one frame per subject (per track).

        The third source category. Built out of `parse_file` / `parse_tracks` /
        `parse_subject_directory` rather than implemented a third time — that
        composition is most of the value of naming the categories.

        **Identity lives in the key and never in a column.** The reason is
        mechanical, not aesthetic: parsing sorts by `datetime`, so many
        subjects in one frame interleave; `detect_and_assign_sequences` then
        splices them into shared sequences with nothing raised; and
        interpolation invents rows bridging one person's Tuesday to another's
        Thursday. A `dict[str, UnifiedFormat]` holds exactly the same
        information, and every frame in it is independently valid.

        Keys are flat composite strings for a multi-track corpus —
        `"CGMacros-001/libre"` — which keeps the return type a plain mapping
        rather than a nested one. **The `/` separator is part of the public
        contract.** A subject id may contain `_` but never `/`; D1NAMO's
        `012_diabetes` directory is exactly why the separator cannot be `_`.

        Args:
            root: Corpus root directory.

        Returns:
            Subject id (or `subject/track`) → frame, one entry per subject per
            track.

        Raises:
            UnknownFormatError: If `root` is not a recognized corpus.
        """
        raise NotImplementedError(
            "parse_corpus is implemented by the concrete parser"
        )

    @classmethod
    def list_subjects(cls, root: Union[str, Path]) -> "Tuple[SubjectEntry, ...]":
        """Enumerate a corpus's subjects without parsing it.

        The companion to `parse_corpus(root, subjects=[...])`: it supplies the
        ids that filter accepts, so a caller selects from what is on disk
        rather than guessing a naming convention. D1NAMO's healthy subset
        contains a subject directory literally named `012_diabetes` and
        CGMacros runs `001`–`049` with four numbers missing — neither is
        derivable, and both are obvious the moment they are listed.

        It also answers *which* subjects are worth parsing. Each entry carries
        the modality files present and, per glucose track, how many rows carry
        a value — enough to skip a subject whose sensor barely reported before
        paying to parse it.

        **Reads glucose, does not parse it.** The counts come from the raw
        glucose source, so they are cheap relative to a parse and they are an
        *independent* measurement of the same thing the parser produces. A
        parsed frame legitimately holds fewer readings — the parser drops
        values the schema cannot represent, and says so — and that difference
        is information rather than an inconsistency.

        Args:
            root: Corpus root directory.

        Returns:
            One entry per subject, ordered by subject id, so the sequence and
            everything derived from it is deterministic.

        Raises:
            UnknownFormatError: If `root` is not a recognized corpus.
        """
        raise NotImplementedError(
            "list_subjects is implemented by the concrete parser"
        )

    @classmethod
    def parse_base64(cls, base64_data: str) -> UnifiedFormat:
        """Parse CGM data from base64 encoded string.
        
        Useful for web API endpoints that receive base64 encoded CSV data.
        Automatically decodes base64, detects format, and parses to unified format.
        
        Args:
            base64_data: Base64 encoded CSV data string
            
        Returns:
            DataFrame in unified format
            
        Raises:
            ValueError: If base64 decoding fails
            UnknownFormatError: If format cannot be determined
            MalformedDataError: If data cannot be parsed
        """
        try:
            raw_data = b64decode(base64_data)
        except Exception as e:
            raise ValueError(f"Failed to decode base64 data: {e}")
        
        return cls.parse_from_bytes(raw_data)



class CGMProcessor(ABC):
    """Abstract base class for unified CGM data processing (Stages 4-5).
    
    This interface handles vendor-agnostic operations on UnifiedFormat data:
    - Stage 4: Postprocessing (synchronization, interpolation)
    - Stage 5: Inference preparation (truncation, validation, warnings)
    
    This class operates only on data already in UnifiedFormat with sequence_id assigned.
    Data should come from CGMParser (which assigns sequences automatically) or have
    sequences assigned via detect_and_assign_sequences() before processing.
    
    **Important**: All methods expect sequence_id to exist in the input dataframe.
    If parsing with FormatParser, sequences are assigned automatically.
    
    All methods are classmethods - no need to instantiate.
    """
    
    # ===== Quality Flag Management =====
    
    @classmethod
    @abstractmethod
    def mark_time_duplicates(
        cls,
        df: UnifiedFormat,
        **kwargs
    ) -> UnifiedFormat:
        """Mark events with duplicate timestamps (keeping first occurrence).
        
        Args:
            df: DataFrame in unified format
            **kwargs: Implementation-specific parameters (e.g., validation_mode)
            
        Returns:
            DataFrame with TIME_DUPLICATE flag added to quality column
        """
        pass
    
    @classmethod
    @abstractmethod
    def mark_calibration_periods(
        cls,
        dataframe: UnifiedFormat,
        **kwargs
    ) -> UnifiedFormat:
        """Mark periods after calibration gaps with SENSOR_CALIBRATION quality flag.
        
        Args:
            dataframe: DataFrame with sequences and original_datetime column
            **kwargs: Implementation-specific parameters (e.g., validation_mode)
            
        Returns:
            DataFrame with quality flags updated for calibration periods
        """
        pass
    
    # ===== STAGE 4: Postprocessing (Unified Operations) =====
    
    @classmethod
    @abstractmethod
    def detect_and_assign_sequences(
        cls,
        dataframe: UnifiedFormat,
        **kwargs
    ) -> UnifiedFormat:
        """Detect large gaps and assign sequence_id column (lossless annotation).
        
        This is a final parsing step that splits data into continuous sequences
        based on time gaps. It's a lossless operation that only adds metadata.
        
        **Separation of Concerns:**
        - This method is called automatically at the end of parse_to_unified()
        - Can also be called standalone for re-detecting sequences on existing data
        - Ensures sequence_id is always present in parsed data
        
        Large gaps (> large_gap_threshold_minutes) create new sequences.
        This method is idempotent - if sequence_id already exists, it validates
        and potentially splits sequences with internal large gaps.
        
        Args:
            dataframe: DataFrame in unified format (may or may not have sequence_id)
            expected_interval_minutes: Expected data collection interval (default: 5)
            large_gap_threshold_minutes: Threshold for creating new sequences (default: 15)
            
        Returns:
            DataFrame with sequence_id column assigned
        """
        pass

    @classmethod
    @abstractmethod
    def synchronize_timestamps(
        cls,
        dataframe: UnifiedFormat,
        **kwargs
    ) -> UnifiedFormat:
        """Align timestamps to minute boundaries.
        
        Args:
            dataframe: DataFrame in unified format
            **kwargs: Implementation-specific parameters (e.g., expected_interval_minutes, validation_mode)
            
        Returns:
            DataFrame with synchronized timestamps
        """
        pass
    
    @classmethod
    @abstractmethod
    def interpolate_gaps(
        cls,
        dataframe: UnifiedFormat,
        **kwargs
    ) -> UnifiedFormat:
        """Fill gaps in continuous data with imputed values.
        
        Adds rows with Quality.IMPUTATION flag for missing data points.
        
        Args:
            dataframe: DataFrame with potential gaps
            **kwargs: Implementation-specific parameters (e.g., expected_interval_minutes, small_gap_max_minutes, snap_to_grid, validation_mode)
            
        Returns:
            DataFrame with interpolated values
        """
        pass
    
    @classmethod
    @abstractmethod
    def get_sequence_grid_start(
        cls,
        seq_data: UnifiedFormat,
        **kwargs
    ) -> datetime:
        """Determine the grid start time for a sequence.
        
        Args:
            seq_data: Sequence data
            **kwargs: Implementation-specific parameters (e.g., expected_interval_minutes)
            
        Returns:
            Grid start timestamp (rounded to nearest minute)
        """
        pass
    
    @classmethod
    @abstractmethod
    def calculate_grid_point(
        cls,
        timestamp: datetime,
        grid_start: datetime,
        **kwargs
    ) -> datetime:
        """Calculate the nearest grid point for a given timestamp.
        
        Args:
            timestamp: Timestamp to align to grid
            grid_start: Start of the grid
            **kwargs: Implementation-specific parameters (e.g., expected_interval_minutes, round_direction)
            
        Returns:
            Timestamp aligned to grid
        """
        pass
    
    # ===== STAGE 5: Inference Preprocessing =====
    
    @classmethod
    @abstractmethod
    def prepare_for_inference(
        cls,
        dataframe: UnifiedFormat,
        minimum_duration_minutes: int = MINIMUM_DURATION_MINUTES,
        maximum_wanted_duration: int = MAXIMUM_WANTED_DURATION_MINUTES,
        **kwargs
    ) -> InferenceResult:
        """Prepare data for inference with full UnifiedFormat and warning flags.
        
        Operations performed:
        - Keep only the last (latest) sequence based on most recent timestamps
        - Truncate sequences exceeding maximum_wanted_duration
        - Collect warnings based on data quality:
          - TOO_SHORT: sequence duration < minimum_duration_minutes
          - CALIBRATION: contains calibration events
          - OUT_OF_RANGE: contains OUT_OF_RANGE quality flags
          - IMPUTATION: contains imputed data
          - TIME_DUPLICATES: contains non-unique time entries
        
        Returns full UnifiedFormat with all columns (sequence_id, event_type, quality, etc).
        Use to_data_only_df() to strip service columns if needed for ML models.
        
        Args:
            dataframe: Fully processed DataFrame in unified format
            minimum_duration_minutes: Minimum required sequence duration
            maximum_wanted_duration: Maximum desired sequence duration (truncates if exceeded)
            **kwargs: Implementation-specific parameters (e.g., validation_mode)
            
        Returns:
            Tuple of (unified_format_dataframe, warnings)
            
        Raises:
            ZeroValidInputError: If there are no valid data points
        """
        pass
    
    # ===== Data Transformation Utilities =====
    
    @classmethod
    @abstractmethod
    def to_data_only_df(
        cls,
        unified_df: UnifiedFormat,
        drop_service_columns: bool = True,
        drop_duplicates: bool = False,
        glucose_only: bool = False,
        **kwargs
    ) -> pl.DataFrame:
        """Strip service columns from UnifiedFormat, keeping only data columns.
        
        Args:
            unified_df: DataFrame in UnifiedFormat with all columns
            drop_service_columns: If True, drop service columns (sequence_id, event_type, quality)
            drop_duplicates: If True, drop duplicate timestamps
            glucose_only: If True, drop non-EGV events
            **kwargs: Implementation-specific parameters (e.g., validation_mode)
            
        Returns:
            DataFrame with only data columns (no service/metadata columns)
        """
        pass
    
    @classmethod
    @abstractmethod
    def split_glucose_events(
        cls,
        unified_df: UnifiedFormat,
        **kwargs
    ) -> Tuple[UnifiedFormat, UnifiedFormat]:
        """Split UnifiedFormat DataFrame into glucose readings and other events.
        
        Args:
            unified_df: DataFrame in UnifiedFormat with mixed event types
            **kwargs: Implementation-specific parameters (e.g., validation_mode)
            
        Returns:
            Tuple of (glucose_df, events_df)
        """
        pass
    
    
# ============================================================================
# Compatibility Layer: Output Adapters
# ============================================================================

def to_pandas(df: pl.DataFrame) -> "pd.DataFrame":
    """Convert polars DataFrame to pandas.
    
    Raises:
        ImportError: If pandas and pyarrow are not installed
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas and pyarrow are required for this function. "
        )
    return df.to_pandas()

def to_polars(df: "pd.DataFrame") -> pl.DataFrame:
    """Convert pandas DataFrame to polars.
    
    Raises:
        ImportError: If arrow and pandas are not installed
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "pandas and pyarrow are required for this function. "
        )
    return pl.from_pandas(df)

