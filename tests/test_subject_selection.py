"""Reaching one subject of a corpus: `list_subjects`, `subjects=`, directory bundles.

0.10.0 named BUNDLE as a source category with D1NAMO as its motivating case —
each subject a directory of modality files — and then left no public way to
parse one. `parse_bundle` took files, and `parse_file` refuses a bare
`glucose.csv` by design; `parse_corpus` took a root and returned every subject.
The only working route was a private method.

This file pins the three pieces that close that:

- `detect_subject_format` — a second path-shaped detector, one directory level
  below `detect_path_format`, with its own registry.
- `parse_bundle([subject_dir])` — a directory member is a whole subject.
- `list_subjects(root)` / `parse_corpus(root, subjects=[...])` — enumerate,
  then select, without parsing what was not selected.

The two properties worth more than any individual assertion, and both are here:
the two routes to one subject return the **same frame**, and the ids
`list_subjects` reports are exactly the ids `parse_corpus` keys by. Anything
else can drift; those two cannot without a test going red.
"""

import csv
import logging
import shutil
from pathlib import Path
from typing import Callable, List

import polars as pl
import pytest

from cgm_format import (
    CGMACROS_TRACKS,
    D1NAMO_TRACK,
    ExtendedFormatProcessor,
    FormatParser,
    SubjectEntry,
    TrackCoverage,
    UnifiedEventType,
)
from cgm_format.formats.supported import (
    PATH_DETECTION_PROBES,
    SUBJECT_PATH_PROBES,
)
from cgm_format.interface.cgm_interface import (
    MalformedDataError,
    MultiTrackSourceError,
    SupportedCGMFormat,
    UnknownFormatError,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "input"
CGMACROS_ROOT = DATA_DIR / "cgmacros_synthetic"
D1NAMO_DIABETES_ROOT = DATA_DIR / "d1namo_synthetic" / "diabetes_subset"
D1NAMO_HEALTHY_ROOT = DATA_DIR / "d1namo_synthetic" / "healthy_subset"
BIGIDEAS_ROOT = DATA_DIR / "bigideas_synthetic"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")


def _subject_dirs(root: Path) -> List[Path]:
    """Subject directories, read off the filesystem rather than hardcoded.

    A hardcoded list would drift with the fixture; `CLAUDE.md` §6 allows
    hardcoding a domain constant and forbids hardcoding a count read off a
    dump, and which subjects a fixture ships is the second kind.
    """
    return sorted((d for d in root.iterdir() if d.is_dir()), key=lambda d: d.name)


class TestSubjectProbeRegistry:
    """Two registries, one level apart, and they must not collide."""

    def test_subject_probes_are_glob_patterns_not_callables(self) -> None:
        """`docs/NEW_SCHEMA.md`: registries hold data, never predicates.

        A callable here would be detection logic that escaped the parser, and
        it would be invisible to every consumer reading the registry.
        """
        for fmt, probes in SUBJECT_PATH_PROBES.items():
            assert isinstance(probes, tuple), f"{fmt.value} probes are not a tuple"
            assert probes, f"{fmt.value} declares no probes"
            for probe in probes:
                assert isinstance(probe, str), f"{fmt.value} probe is not a string"

    def test_subject_probes_cover_exactly_the_directory_shaped_formats(self) -> None:
        """The same formats have corpus probes and subject probes.

        Not exhaustive over `SupportedCGMFormat` and deliberately so — an
        EXPORT is a file, and a file has no directory shape — but a format with
        one probe set and not the other would be reachable by one detector and
        invisible to the other.
        """
        assert set(SUBJECT_PATH_PROBES) == set(PATH_DETECTION_PROBES)

    def test_a_corpus_root_never_detects_as_one_of_its_subjects(self) -> None:
        """Cross-detection, the direction that would send a whole tree astray.

        The shapes differ by exactly one directory level, so this is the
        collision to guard: a root answering "yes, I am a subject" would parse
        the entire corpus as one person.
        """
        for root in (
            CGMACROS_ROOT,
            D1NAMO_DIABETES_ROOT,
            D1NAMO_HEALTHY_ROOT,
            BIGIDEAS_ROOT,
        ):
            _skip_if_missing(root)
            with pytest.raises(UnknownFormatError) as excinfo:
                FormatParser.detect_subject_format(root)
            # And the refusal must say what it *is*, not merely what it is not.
            assert "corpus root" in str(excinfo.value)
            assert "parse_corpus" in str(excinfo.value)

    def test_a_subject_directory_never_detects_as_a_corpus_root(self) -> None:
        """The mirror direction: one subject must not look like a whole corpus."""
        for root in (
            CGMACROS_ROOT,
            D1NAMO_DIABETES_ROOT,
            D1NAMO_HEALTHY_ROOT,
            BIGIDEAS_ROOT,
        ):
            _skip_if_missing(root)
            for subject_dir in _subject_dirs(root):
                with pytest.raises(UnknownFormatError):
                    FormatParser.detect_path_format(subject_dir)

    def test_the_two_d1namo_subsets_are_told_apart_at_subject_level(self) -> None:
        """`insulin.csv` and `annotations.csv` separate them one level down too.

        Asserted per subject rather than per subset: the discriminator has to
        hold on every member, not on the first one the walker happens to hit.
        """
        _skip_if_missing(D1NAMO_DIABETES_ROOT)
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        for subject_dir in _subject_dirs(D1NAMO_DIABETES_ROOT):
            assert (
                FormatParser.detect_subject_format(subject_dir)
                == SupportedCGMFormat.D1NAMO_DIABETES
            ), f"{subject_dir.name} misdetected"
        for subject_dir in _subject_dirs(D1NAMO_HEALTHY_ROOT):
            assert (
                FormatParser.detect_subject_format(subject_dir)
                == SupportedCGMFormat.D1NAMO_HEALTHY
            ), f"{subject_dir.name} misdetected"

    def test_a_d1namo_subject_is_not_a_cgmacros_subject(self) -> None:
        """Across corpora, not only within one."""
        _skip_if_missing(D1NAMO_DIABETES_ROOT)
        _skip_if_missing(CGMACROS_ROOT)
        d1namo = {
            FormatParser.detect_subject_format(d)
            for d in _subject_dirs(D1NAMO_DIABETES_ROOT)
        }
        cgmacros = {
            FormatParser.detect_subject_format(d)
            for d in _subject_dirs(CGMACROS_ROOT)
        }
        assert d1namo.isdisjoint(cgmacros)


class TestParsedCorpusInvariants:
    """The unified contract, asserted on every corpus the library walks.

    Generic on purpose. A bundle parser assembles a frame from several
    sources, and the way that goes wrong is per-source rather than
    per-format: one modality arrives already postprocessed, the rest join it
    afterwards, and the write-once guards in `_postprocess_unified` no-op on
    the second pass. BIG IDEAs shipped exactly that bug and its own tests all
    passed, so the guard belongs here, where it covers every corpus at once.
    """

    def _corpora(self) -> List[Path]:
        return [
            CGMACROS_ROOT,
            D1NAMO_DIABETES_ROOT,
            D1NAMO_HEALTHY_ROOT,
            BIGIDEAS_ROOT,
        ]

    def test_every_parsed_row_has_an_anchor(self) -> None:
        """`original_datetime` is created during parsing, for every row.

        `docs/PHILOSOPHY.md`: "Created during parsing, never overwritten."
        Every downstream operation computes from it, so a null anchor is a row
        that no grid alignment, gap detection or sequence assignment can see.
        """
        for root in self._corpora():
            _skip_if_missing(root)
            for key, frame in FormatParser.parse_corpus(root).items():
                assert frame.get_column("original_datetime").null_count() == 0, (
                    f"{root.name}/{key} has rows with no original_datetime; "
                    "they will never be assigned to a sequence"
                )

    def test_every_parsed_frame_is_in_time_order(self) -> None:
        """Rows come out chronological, not grouped by which file they came from.

        The same defect shows here first: the schema's total ordering leads
        with `original_datetime`, so null anchors sort a whole modality to the
        head of the frame instead of interleaving it with the readings.
        """
        for root in self._corpora():
            _skip_if_missing(root)
            for key, frame in FormatParser.parse_corpus(root).items():
                assert frame.get_column("datetime").is_sorted(), (
                    f"{root.name}/{key} is not in datetime order"
                )

    def test_non_glucose_events_are_assigned_to_a_sequence(self) -> None:
        """A meal inside a glucose sequence must land in that sequence.

        `sequence_id == 0` means unassigned. An event the readings cover but
        which stays at 0 is dropped by every sequence-scoped consumer,
        `prepare_for_inference` included — silently, since the row is still
        present in the frame.
        """
        for root in self._corpora():
            _skip_if_missing(root)
            for key, frame in FormatParser.parse_corpus(root).items():
                sequenced = ExtendedFormatProcessor.detect_and_assign_sequences(frame)
                glucose = sequenced.filter(
                    (pl.col("event_type") == UnifiedEventType.GLUCOSE.value)
                    & (pl.col("sequence_id") > 0)
                )
                if glucose.height == 0:
                    continue
                covered = sequenced.filter(
                    (pl.col("event_type") != UnifiedEventType.GLUCOSE.value)
                    & (pl.col("datetime") >= glucose.get_column("datetime").min())
                    & (pl.col("datetime") <= glucose.get_column("datetime").max())
                )
                stranded = covered.filter(pl.col("sequence_id") == 0)
                assert stranded.height == 0, (
                    f"{root.name}/{key}: {stranded.height} event(s) inside the "
                    "glucose span were left unassigned"
                )


class TestDirectoryBundles:
    """`parse_bundle([subject_dir])` — the route the refusal message names."""

    def test_the_refusal_on_a_bare_modality_names_a_route_that_works(self) -> None:
        """The message is only useful if following it succeeds.

        `parse_file` on a D1NAMO `glucose.csv` refuses and names
        `parse_bundle([subject_dir])`. This runs that advice rather than
        reading it — a message naming an API that errors is worse than no
        message, because it costs the reader a round trip to find out.
        """
        subject_dir = D1NAMO_DIABETES_ROOT / "001"
        _skip_if_missing(subject_dir / "glucose.csv")

        with pytest.raises(MalformedDataError) as excinfo:
            FormatParser.parse_file(subject_dir / "glucose.csv")
        assert "parse_bundle" in str(excinfo.value)

        frame = FormatParser.parse_bundle([subject_dir])
        assert len(frame) > 0

    def test_a_directory_member_yields_the_same_frame_as_the_corpus_walk(self) -> None:
        """The two public routes to one subject must not disagree.

        Frame equality, not shape equality: two paths that produce the same
        number of rows in a different order, or with a column cast
        differently, are still two answers to one question.
        """
        subject_dir = D1NAMO_DIABETES_ROOT / "001"
        _skip_if_missing(subject_dir / "glucose.csv")

        via_bundle = FormatParser.parse_bundle([subject_dir])
        via_corpus = FormatParser.parse_corpus(
            D1NAMO_DIABETES_ROOT, subjects=[subject_dir.name]
        )

        assert list(via_corpus) == [subject_dir.name]
        assert via_bundle.equals(via_corpus[subject_dir.name])

    def test_every_committed_d1namo_subject_parses_as_a_bundle(self) -> None:
        """Both subsets, every subject — not just the one the walker starts on."""
        for root in (D1NAMO_DIABETES_ROOT, D1NAMO_HEALTHY_ROOT):
            _skip_if_missing(root)
            for subject_dir in _subject_dirs(root):
                frame = FormatParser.parse_bundle([subject_dir])
                assert len(frame) > 0, f"{subject_dir.name} produced no rows"

    def test_a_corpus_root_passed_as_a_bundle_is_refused_by_name(self) -> None:
        """A root reaching a per-subject entry point is the likeliest mistake.

        An unqualified "unknown directory" would hide the answer the caller
        needs, which is that they are one level too high.
        """
        _skip_if_missing(D1NAMO_DIABETES_ROOT)
        with pytest.raises(UnknownFormatError) as excinfo:
            FormatParser.parse_bundle([D1NAMO_DIABETES_ROOT])
        message = str(excinfo.value)
        assert "corpus root" in message
        assert "parse_corpus" in message
        assert str(D1NAMO_DIABETES_ROOT) in message

    def test_a_multi_track_subject_is_refused_and_names_the_file(self) -> None:
        """A CGMacros subject is not a bundle — and the caller holds a directory.

        Naming only `parse_tracks` would leave them guessing which CSV inside
        the folder to hand it, which is the shell-user problem the release
        already fixed once for `MultiTrackSourceError`.
        """
        subject_dir = CGMACROS_ROOT / "CGMacros-001"
        _skip_if_missing(subject_dir)

        with pytest.raises(MultiTrackSourceError) as excinfo:
            FormatParser.parse_bundle([subject_dir])
        message = str(excinfo.value)
        assert "parse_tracks" in message
        for track in CGMACROS_TRACKS:
            assert track in message
        # The path it names must be the file that actually exists. The
        # message quotes it with !r, so Windows backslashes are doubled.
        named_csv = subject_dir / f"{subject_dir.name}.csv"
        assert named_csv.name in message
        assert str(named_csv) in message or repr(str(named_csv)) in message
        assert named_csv.exists()

    def test_a_single_path_is_still_rejected(self) -> None:
        """Directory support must not weaken the str-is-a-sequence guard."""
        _skip_if_missing(D1NAMO_DIABETES_ROOT)
        with pytest.raises(TypeError):
            FormatParser.parse_bundle(D1NAMO_DIABETES_ROOT / "001")


class TestListSubjects:
    """Enumerating a corpus without parsing it."""

    def test_the_ids_are_exactly_the_ids_parse_corpus_keys_by(self) -> None:
        """The filter's vocabulary and the walker's vocabulary are one list.

        Set equality against the corpus keys' subject halves, computed from the
        parse rather than restated: a helper that lists ids `parse_corpus` does
        not produce would send every caller to a `ValueError`.

        Equality holds when every listed subject parses, which is the case for
        every committed corpus. It is *not* an identity — a subject on disk
        that cannot be parsed is listed and not keyed, deliberately, and
        `test_an_unparseable_subject_is_listed_and_reported` pins that.
        """
        for root in (
            CGMACROS_ROOT,
            D1NAMO_DIABETES_ROOT,
            D1NAMO_HEALTHY_ROOT,
            BIGIDEAS_ROOT,
        ):
            _skip_if_missing(root)
            listed = {e.subject_id for e in FormatParser.list_subjects(root)}
            keyed = {
                key.split("/")[0] for key in FormatParser.parse_corpus(root)
            }
            assert listed == keyed, f"{root.name}: listed {listed}, keyed {keyed}"

    def test_an_unparseable_subject_is_listed_and_reported(
        self, tmp_path, caplog
    ) -> None:
        """A subject the corpus offers but we cannot parse must not vanish.

        BIG IDEAs enumerates subjects from the union of both modalities, so a
        directory holding only a food log reaches the parser and fails there
        with a typed error naming the missing file. That is the whole point of
        the union: listing by glucose alone would make such a subject
        *invisible* rather than reported.

        The cost is that `list_subjects` and `parse_corpus` no longer return
        the same ids for that corpus, which is why the equality above is
        conditional. Both surfaces say what happened, so nothing is silent:
        the entry carries no coverage and both walkers warn.
        """
        _skip_if_missing(BIGIDEAS_ROOT)
        root = tmp_path / "corpus"
        complete, partial = root / "001", root / "002"
        complete.mkdir(parents=True)
        partial.mkdir(parents=True)
        for name in ("Dexcom_001.csv", "Food_Log_001.csv"):
            shutil.copy(BIGIDEAS_ROOT / "001" / name, complete)
        shutil.copy(
            BIGIDEAS_ROOT / "003" / "Food_Log_003.csv",
            partial / "Food_Log_002.csv",
        )

        with caplog.at_level(logging.WARNING, logger="cgm_format.format_parser"):
            listed = [e for e in FormatParser.list_subjects(root)]
            keyed = set(FormatParser.parse_corpus(root))

        assert {e.subject_id for e in listed} == {"001", "002"}
        assert keyed == {"001"}
        # Listed with no coverage, not with a zero that would read as "no
        # readings recorded" — "we could not look" is a different answer.
        assert next(e for e in listed if e.subject_id == "002").tracks == ()
        assert "Dexcom_*.csv" in caplog.text
        assert "yielded no frame" in caplog.text

    def test_entries_are_ordered_by_subject_id(self) -> None:
        """Deterministic order, so anything derived from it is deterministic."""
        _skip_if_missing(CGMACROS_ROOT)
        ids = [e.subject_id for e in FormatParser.list_subjects(CGMACROS_ROOT)]
        assert ids == sorted(ids)

    def test_entries_are_typed_and_frozen(self) -> None:
        """A record of a directory's shape must not be editable into disagreeing."""
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        entry = FormatParser.list_subjects(D1NAMO_HEALTHY_ROOT)[0]
        assert isinstance(entry, SubjectEntry)
        assert all(isinstance(t, TrackCoverage) for t in entry.tracks)
        with pytest.raises(Exception):
            entry.subject_id = "mutated"  # type: ignore[misc]

    def test_modalities_are_the_csv_files_actually_present(self) -> None:
        """Read off the directory, compared against the directory."""
        _skip_if_missing(D1NAMO_DIABETES_ROOT)
        for entry in FormatParser.list_subjects(D1NAMO_DIABETES_ROOT):
            on_disk = {
                f.name for f in entry.path.iterdir()
                if f.is_file() and f.suffix.lower() == ".csv"
            }
            assert set(entry.modalities) == on_disk

    def test_a_multi_track_corpus_reports_one_entry_per_track(self) -> None:
        """Track names match `parse_tracks`, in the same order."""
        _skip_if_missing(CGMACROS_ROOT)
        for entry in FormatParser.list_subjects(CGMACROS_ROOT):
            assert tuple(t.track for t in entry.tracks) == CGMACROS_TRACKS

    def test_a_single_track_corpus_reports_one_entry_named_for_the_column(self) -> None:
        """`glucose`, not `sensor`.

        The healthy subset has no CGM at all — every value is a fingerstick —
        so a track named after a device would assert a continuous trace that
        does not exist.
        """
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        for entry in FormatParser.list_subjects(D1NAMO_HEALTHY_ROOT):
            assert tuple(t.track for t in entry.tracks) == (D1NAMO_TRACK,)

    def test_coverage_counts_what_the_source_filled(self) -> None:
        """Cross-checked against the raw CSV, read independently of the library.

        Counted here with the stdlib rather than with the parser, so the two
        numbers come from genuinely different code — a check that ran the
        parser to compute its own expectation would compare a convention
        against itself.
        """
        subject_dir = D1NAMO_DIABETES_ROOT / "001"
        _skip_if_missing(subject_dir / "glucose.csv")

        with open(subject_dir / "glucose.csv", encoding="utf-8-sig") as fh:
            rows = list(csv.DictReader(fh))
        expected_rows = len(rows)
        expected_values = sum(
            1 for r in rows
            if (r.get("glucose") or "").strip() not in ("", "No information")
        )

        entry = next(
            e for e in FormatParser.list_subjects(D1NAMO_DIABETES_ROOT)
            if e.subject_id == subject_dir.name
        )
        assert entry.tracks[0].rows == expected_rows
        assert entry.tracks[0].values == expected_values

    def test_a_value_the_schema_cannot_hold_is_counted_but_not_parsed(self) -> None:
        """The number the source offered and the number the schema kept differ.

        The healthy fixture carries a glucose cell reading `7:0` — a colon
        typed for a decimal point. The source did say something, so coverage
        counts it; the parser cannot represent it, drops it and warns. Both are
        correct, and collapsing them into one number would hide the defect.
        """
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)

        entries = {e.subject_id: e for e in FormatParser.list_subjects(D1NAMO_HEALTHY_ROOT)}
        frames = FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT)

        discrepancies = {
            sid: (
                entry.tracks[0].values,
                len(frames[sid].filter(pl.col("glucose").is_not_null())),
            )
            for sid, entry in entries.items()
            if sid in frames
        }
        # Coverage is never *below* what the parser kept: the parser cannot
        # invent a reading from a cell the source left empty.
        for sid, (offered, kept) in discrepancies.items():
            assert offered >= kept, f"{sid}: parser kept {kept} of {offered} offered"

        if not any(offered > kept for offered, kept in discrepancies.values()):
            pytest.skip("No unrepresentable glucose value in this fixture")

    def test_first_and_last_are_none_rather_than_a_sentinel(self) -> None:
        """A track with no values reports `None`, never epoch zero.

        Built from a real subject with its glucose column emptied, so the
        frame reaching the reader is one the reader could genuinely receive.
        """
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        entry = FormatParser.list_subjects(D1NAMO_HEALTHY_ROOT)[0]
        raw = pl.read_csv(entry.path / "glucose.csv", infer_schema_length=0)
        blanked = raw.with_columns(pl.lit("").alias("glucose"))

        coverage = FormatParser._coverage_from(
            D1NAMO_TRACK,
            blanked.with_columns(pl.lit(None, dtype=pl.Datetime("ms")).alias("_ts")),
            "glucose",
            len(blanked),
        )
        assert coverage.values == 0
        assert coverage.first is None
        assert coverage.last is None

    def test_a_corpus_root_is_required(self) -> None:
        """A subject directory is not a corpus, and saying so beats guessing."""
        _skip_if_missing(D1NAMO_DIABETES_ROOT)
        with pytest.raises(UnknownFormatError):
            FormatParser.list_subjects(D1NAMO_DIABETES_ROOT / "001")


class TestSubjectFilter:
    """`parse_corpus(root, subjects=[...])` — select, do not post-filter."""

    def test_it_returns_exactly_the_requested_subjects(self) -> None:
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        every = FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT)
        wanted = sorted(every)[:2]
        some = FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT, subjects=wanted)
        assert set(some) == set(wanted)

    def test_a_filtered_frame_is_identical_to_its_unfiltered_twin(self) -> None:
        """Filtering selects; it must not change what a subject parses to."""
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        every = FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT)
        subject_id = sorted(every)[0]
        one = FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT, subjects=[subject_id])
        assert one[subject_id].equals(every[subject_id])

    def test_the_result_is_ordered_by_subject_id_whatever_order_was_asked(self) -> None:
        """Determinism does not depend on how the caller happened to type it."""
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        every = sorted(FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT))
        if len(every) < 2:
            pytest.skip("Need at least two subjects to test ordering")
        forwards = FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT, subjects=every)
        backwards = FormatParser.parse_corpus(
            D1NAMO_HEALTHY_ROOT, subjects=list(reversed(every))
        )
        assert list(forwards) == list(backwards)

    def test_a_repeated_id_yields_one_entry(self) -> None:
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        subject_id = sorted(FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT))[0]
        result = FormatParser.parse_corpus(
            D1NAMO_HEALTHY_ROOT, subjects=[subject_id, subject_id]
        )
        assert list(result) == [subject_id]

    def test_an_unknown_id_raises_rather_than_selecting_nothing(self) -> None:
        """The lesson `track=` already taught, one parameter over.

        A filter that silently selects nothing hands back a result the caller
        believes is filtered when it is really just missing a subject, and a
        typo is then indistinguishable from a subject with no data.
        """
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        with pytest.raises(ValueError) as excinfo:
            FormatParser.parse_corpus(
                D1NAMO_HEALTHY_ROOT, subjects=["definitely-not-a-subject"]
            )
        message = str(excinfo.value)
        assert "definitely-not-a-subject" in message
        # And it must say what *is* available rather than only what is not.
        for subject_id in FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT):
            assert subject_id in message

    def test_an_empty_selection_raises(self) -> None:
        """`[]` is "you selected nothing", which is never what was meant."""
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        with pytest.raises(ValueError):
            FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT, subjects=[])

    def test_it_composes_with_the_track_filter(self) -> None:
        """Both filters at once, on the corpus that has both axes."""
        _skip_if_missing(CGMACROS_ROOT)
        subject_id = sorted(
            e.subject_id for e in FormatParser.list_subjects(CGMACROS_ROOT)
        )[0]
        track = CGMACROS_TRACKS[0]
        result = FormatParser.parse_corpus(
            CGMACROS_ROOT, track=track, subjects=[subject_id]
        )
        assert list(result) == [f"{subject_id}/{track}"]

    def test_it_prunes_before_parsing(self) -> None:
        """One id costs one subject's work, not the whole corpus's then a filter.

        Counted by how many times the per-subject parser is called, because
        wall-clock on a small fixture proves nothing. A post-filter
        implementation parses every subject and passes every other assertion
        in this class.
        """
        _skip_if_missing(D1NAMO_HEALTHY_ROOT)
        every = sorted(FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT))
        if len(every) < 2:
            pytest.skip("Need at least two subjects to tell pruning from filtering")

        calls: List[str] = []
        original: Callable[..., pl.DataFrame] = (
            FormatParser._process_d1namo_subject.__func__
        )

        def counting(cls, subject_dir: Path) -> pl.DataFrame:
            calls.append(Path(subject_dir).name)
            return original(cls, subject_dir)

        FormatParser._process_d1namo_subject = classmethod(counting)
        try:
            FormatParser.parse_corpus(D1NAMO_HEALTHY_ROOT, subjects=[every[0]])
        finally:
            FormatParser._process_d1namo_subject = classmethod(original)

        assert calls == [every[0]], (
            f"parsed {calls} — pruning happens after parsing, not before"
        )
