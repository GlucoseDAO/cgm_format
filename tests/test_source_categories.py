"""Source categories: export, bundle, corpus.

Everything the library supported before 0.10.0 was one shape — one file, one
subject, one device — and nothing wrote that assumption down because nothing
violated it. `FormatCategory` names it, `FORMAT_CATEGORY` records which shape
each registered format arrives as, `detect_path_format` identifies a source by
directory shape where there is no single text to sniff, and `parse_bundle`
generalizes the several-files-one-subject merge that
`from_nightscout_exports` was already doing under a Nightscout-specific name.

The load-bearing distinction under test: **a bundle merges modalities, never
subjects.** Merging two people the way you merge glucose-plus-insulin produces
a frame that interleaves them, splices them into shared sequences, and raises
nothing at all.
"""

import json
from pathlib import Path

import polars as pl
import pytest

from cgm_format import (
    CGM_SCHEMA,
    CGM_SCHEMA_EXTENDED,
    FormatCategory,
    FormatParser,
)
from cgm_format.formats.supported import (
    FORMAT_CATEGORY,
    PATH_DETECTION_PROBES,
    SCHEMA_MAP,
)
from cgm_format.interface.cgm_interface import (
    SupportedCGMFormat,
    UnknownFormatError,
)

DATA_DIR = Path(__file__).parent.parent / "data" / "input"
NIGHTSCOUT_ENTRIES = DATA_DIR / "nightscout_entries.json"
NIGHTSCOUT_TREATMENTS = DATA_DIR / "nightscout_treatments.json"
NIGHTSCOUT_PROFILE = DATA_DIR / "nightscout_profile.json"
DEXCOM_FIXTURE = DATA_DIR / "Clarity_Export_synthetic.csv"
LIBRE_FIXTURE = DATA_DIR / "FreeStyle_Libre_3_synthetic.csv"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        pytest.skip(f"Fixture not found: {path}")


class TestFormatCategoryRegistry:
    """The sidecar registry, and why it is a sidecar."""

    def test_every_registered_format_has_a_category(self) -> None:
        """Exhaustive over `SupportedCGMFormat`, checked rather than assumed.

        This is the test that fires when Waves 4-5 add CGMacros and D1NAMO
        without categorising them — the alternative is a `.get(fmt, EXPORT)`
        somewhere that quietly calls a corpus an export.
        """
        assert set(FORMAT_CATEGORY) == set(SupportedCGMFormat), (
            "FORMAT_CATEGORY is not exhaustive; missing: "
            f"{sorted(f.value for f in set(SupportedCGMFormat) - set(FORMAT_CATEGORY))}"
        )

    def test_category_registry_agrees_with_the_schema_registry(self) -> None:
        """A format that can be parsed must also declare its shape."""
        assert set(FORMAT_CATEGORY) == set(SCHEMA_MAP)

    def test_every_value_is_a_format_category(self) -> None:
        assert all(
            isinstance(category, FormatCategory) for category in FORMAT_CATEGORY.values()
        )

    def test_category_is_not_a_field_on_the_public_enum(self) -> None:
        """`SupportedCGMFormat` keeps its shape — it is public API.

        Consumers compare and serialize its members; widening it to carry
        metadata would change something they read, and the sidecar dict holds
        the same information at no cost.
        """
        assert not hasattr(SupportedCGMFormat.DEXCOM, "category")
        assert SupportedCGMFormat.DEXCOM.value == "dexcom"


class TestPathDetection:
    """Detection by directory shape, beside the text-prefix mechanism."""

    def test_probes_registry_is_pure_data(self) -> None:
        """Glob strings, never callables (`docs/NEW_SCHEMA.md`).

        A predicate here would move detection logic out of the parser and into
        the registry, which is the split the charter's registry section exists
        to protect.
        """
        for fmt, probes in PATH_DETECTION_PROBES.items():
            assert isinstance(probes, tuple), fmt
            assert all(isinstance(p, str) for p in probes), fmt

    def test_a_directory_matching_no_probe_raises(self, tmp_path: Path) -> None:
        """Mirrors `detect_format`: unknown is an error, never a guess."""
        (tmp_path / "something.csv").write_text("a,b\n1,2\n")

        with pytest.raises(UnknownFormatError):
            FormatParser.detect_path_format(tmp_path)

    def test_a_file_is_not_a_directory_shaped_source(self, tmp_path: Path) -> None:
        """Passing a file is a caller error, reported as one.

        Globbing a non-directory silently yields nothing, which would surface
        as "unknown format" and send the caller hunting for a missing probe
        rather than at their own argument.
        """
        target = tmp_path / "export.csv"
        target.write_text("a,b\n1,2\n")

        with pytest.raises(UnknownFormatError, match="directory"):
            FormatParser.detect_path_format(target)

    def test_probes_are_conjunctive_and_ordered(self, tmp_path: Path) -> None:
        """Every probe must match, and the first satisfied format wins.

        Exercised against a synthetic registry rather than shipped entries:
        no corpus format is registered until Waves 4-5, and inventing probe
        entries for formats that do not exist would be a fabricated value.
        """
        subject = tmp_path / "SUBJ-001"
        subject.mkdir()
        (subject / "glucose.csv").write_text("date,time,glucose\n")

        class _ProbeParser(FormatParser):
            # Two candidates: the specific one needs both files and is
            # registered first; the loose one needs only the directory.
            path_detection_probes = {
                SupportedCGMFormat.MEDTRONIC: ("SUBJ-*/glucose.csv", "SUBJ-*/insulin.csv"),
                SupportedCGMFormat.DEXCOM: ("SUBJ-*/glucose.csv",),
            }

        # insulin.csv is absent, so the first (conjunctive) format must NOT win.
        assert _ProbeParser.detect_path_format(tmp_path) == SupportedCGMFormat.DEXCOM

        # Once it is present, the earlier-registered, more specific one wins.
        (subject / "insulin.csv").write_text("date,time,fast_insulin\n")
        assert _ProbeParser.detect_path_format(tmp_path) == SupportedCGMFormat.MEDTRONIC

    def test_an_empty_probe_tuple_never_matches(self, tmp_path: Path) -> None:
        """A format with no probes must not swallow every directory.

        `all(...)` over an empty tuple is vacuously True, so without the guard
        an unprobed format would match the first directory it was offered.
        """
        (tmp_path / "anything.csv").write_text("a\n1\n")

        class _EmptyProbeParser(FormatParser):
            path_detection_probes = {SupportedCGMFormat.DEXCOM: ()}

        with pytest.raises(UnknownFormatError):
            _EmptyProbeParser.detect_path_format(tmp_path)


class TestParseBundle:
    """Several files, one subject, merged on the modality axis."""

    def test_an_empty_bundle_refuses_rather_than_returning_an_empty_frame(
        self,
    ) -> None:
        """"You gave me nothing" is not the same as "there is no data"."""
        with pytest.raises(ValueError, match="at least one path"):
            FormatParser.parse_bundle([])

    def test_a_single_member_bundle_equals_parse_file(self) -> None:
        """The degenerate bundle is exactly the export case."""
        _skip_if_missing(DEXCOM_FIXTURE)

        assert FormatParser.parse_bundle([DEXCOM_FIXTURE]).equals(
            FormatParser.parse_file(DEXCOM_FIXTURE)
        )

    def test_bundling_two_formats_preserves_every_source_row(self) -> None:
        """Losslessness across the merge, asserted as a relationship.

        Two different vendors stand in for two modalities here: the property
        under test is that a diagonal concat keeps every row and invents none,
        which does not depend on what the members are.
        """
        _skip_if_missing(DEXCOM_FIXTURE)
        _skip_if_missing(LIBRE_FIXTURE)

        dexcom = FormatParser.parse_file(DEXCOM_FIXTURE)
        libre = FormatParser.parse_file(LIBRE_FIXTURE)
        bundled = FormatParser.parse_bundle([DEXCOM_FIXTURE, LIBRE_FIXTURE])

        assert len(bundled) == len(dexcom) + len(libre)
        # Every source timestamp survives, and nothing new appears.
        assert set(bundled["datetime"].to_list()) == (
            set(dexcom["datetime"].to_list()) | set(libre["datetime"].to_list())
        )

    def test_bundle_order_does_not_change_the_result(self) -> None:
        """Deterministic row order regardless of the order files arrive in.

        A merge that sorted on a fixed key list while the frames disagreed on
        width would leave row order depending on concat order — the exact
        nondeterminism the stable-sort invariant exists to remove.
        """
        _skip_if_missing(DEXCOM_FIXTURE)
        _skip_if_missing(LIBRE_FIXTURE)

        forward = FormatParser.parse_bundle([DEXCOM_FIXTURE, LIBRE_FIXTURE])
        reverse = FormatParser.parse_bundle([LIBRE_FIXTURE, DEXCOM_FIXTURE])

        assert forward.equals(reverse)

    def test_a_missing_member_raises_rather_than_being_skipped(
        self, tmp_path: Path
    ) -> None:
        """A bundle silently short one modality is a wrong frame, not a small one."""
        _skip_if_missing(DEXCOM_FIXTURE)

        with pytest.raises(FileNotFoundError):
            FormatParser.parse_bundle([DEXCOM_FIXTURE, tmp_path / "absent.csv"])

    @pytest.mark.parametrize("as_type", [str, Path])
    def test_a_single_path_is_refused_not_iterated(self, as_type: type) -> None:
        """A `str` is itself a `Sequence[str]`, so it would be walked per-character.

        `parse_file` next door takes a plain path, so passing one here is the
        natural mistake. Left alone it surfaces as
        `FileNotFoundError: No such file or directory: 'd'` — an error pointing
        nowhere near the cause.
        """
        _skip_if_missing(DEXCOM_FIXTURE)

        with pytest.raises(TypeError, match="sequence of paths"):
            FormatParser.parse_bundle(as_type(DEXCOM_FIXTURE))

    def test_merge_of_disjoint_extra_columns_is_order_independent(self) -> None:
        """The nondeterminism a naive `merged.sort(merged.columns)` reintroduces.

        A diagonal concat orders columns by first appearance, so two members
        with *disjoint* extras yield different key lists depending on argument
        order. Today's registered schemas cannot produce this (core's columns
        are a prefix of extended's), but `merge_bundle_frames` is public and is
        offered to corpus walkers, so the property is pinned directly.
        """
        left = pl.DataFrame(
            {"datetime": [1, 2], "glucose": [100.0, 101.0], "calories": [None, 500.0]}
        )
        right = pl.DataFrame(
            {"datetime": [3], "glucose": [99.0], "heart_rate": [70.0]}
        )

        forward = FormatParser.merge_bundle_frames([left, right])
        reverse = FormatParser.merge_bundle_frames([right, left])

        assert forward.columns == reverse.columns
        assert forward.equals(reverse)

    def test_canonical_keys_follow_schema_order_then_sorted_remainder(self) -> None:
        """Known columns in schema order; anything else in a stable tail."""
        keys = FormatParser._canonical_sort_keys(
            ["zzz_custom", "heart_rate", "glucose", "datetime", "aaa_custom"]
        )

        schema_part = [k for k in keys if k in CGM_SCHEMA_EXTENDED.get_column_names()]
        assert schema_part == ["datetime", "glucose", "heart_rate"]
        assert keys[-2:] == ["aaa_custom", "zzz_custom"]

    def test_merging_no_frames_refuses_with_a_named_error(self) -> None:
        """Public API: a clear refusal, not a raw polars error."""
        with pytest.raises(ValueError, match="at least one frame"):
            FormatParser.merge_bundle_frames([])

    def test_a_one_member_bundle_still_goes_through_the_merge(self) -> None:
        """The merge is the documented subclass extension point.

        Short-circuiting a single member would make an override's behaviour
        depend on how many files the caller happened to pass — present for two,
        absent for one.
        """
        _skip_if_missing(DEXCOM_FIXTURE)
        seen: list[int] = []

        class _CountingParser(FormatParser):
            @classmethod
            def merge_bundle_frames(cls, frames):
                seen.append(len(frames))
                return super().merge_bundle_frames(frames)

        _CountingParser.parse_bundle([DEXCOM_FIXTURE])

        assert seen == [1], "one-member bundle bypassed merge_bundle_frames"


class TestBundlesMergeModalitiesNotSubjects:
    """The design's most dangerous misuse, made explicit.

    The library cannot detect it — a subject's identity is not in the data —
    so the contract is the caller's, and these tests pin the *consequence*
    rather than pretending to a check that cannot exist.
    """

    def test_two_subjects_bundled_are_silently_interleaved(self) -> None:
        """Demonstrates the corruption, so the docstring warning is not theory.

        Bundling one file with itself stands in for two subjects whose records
        overlap in time: every timestamp then appears twice, the rows
        interleave on sort, and nothing anywhere raises. That is precisely why
        `parse_corpus` keeps identity in the mapping key instead.
        """
        _skip_if_missing(DEXCOM_FIXTURE)

        single = FormatParser.parse_file(DEXCOM_FIXTURE)
        doubled = FormatParser.parse_bundle([DEXCOM_FIXTURE, DEXCOM_FIXTURE])

        # No error was raised, and the frame is exactly twice the size.
        assert len(doubled) == 2 * len(single)
        # Every timestamp now appears twice as often as it truly occurred.
        assert doubled["datetime"].n_unique() == single["datetime"].n_unique()

    def test_the_docstring_warns_about_it(self) -> None:
        """The warning is the only guard available, so it must be present.

        `CLAUDE.md` forbids overpromising; it equally forbids shipping a
        footgun with no sign on it.
        """
        doc = FormatParser.parse_bundle.__doc__ or ""

        assert "subject" in doc.lower()
        # The consequence must be named, not just the rule. Deliberately not
        # asserting on `parse_corpus`: it arrives in Wave 3, and pinning a
        # not-yet-existing name would make the docstring's forward reference
        # load-bearing on a test.
        assert "modalit" in doc.lower()


class TestNightscoutBundleCompatibility:
    """`from_nightscout_exports` keeps working, unchanged, and says what it ignores."""

    def test_entries_and_treatments_still_merge(self) -> None:
        _skip_if_missing(NIGHTSCOUT_ENTRIES)
        _skip_if_missing(NIGHTSCOUT_TREATMENTS)

        merged = FormatParser.from_nightscout_exports(
            NIGHTSCOUT_ENTRIES, NIGHTSCOUT_TREATMENTS
        )

        CGM_SCHEMA.validate_dataframe(merged, enforce=False)
        entries_only = FormatParser.from_nightscout_exports(NIGHTSCOUT_ENTRIES)
        # Treatments add rows; they never replace entries.
        assert len(merged) > len(entries_only)

    def test_profile_path_is_ignored_loudly_not_silently(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A parameter that does nothing must say so.

        The Nightscout profile holds settings, not readings — there is no row
        for it to contribute. Accepting it and discarding it without a word is
        the silent substitution the charter forbids, so it warns and names
        itself.
        """
        _skip_if_missing(NIGHTSCOUT_ENTRIES)
        _skip_if_missing(NIGHTSCOUT_PROFILE)

        with caplog.at_level("WARNING"):
            with_profile = FormatParser.from_nightscout_exports(
                NIGHTSCOUT_ENTRIES, profile_path=NIGHTSCOUT_PROFILE
            )

        assert "profile" in caplog.text.lower()
        # And it genuinely changed nothing about the result.
        assert with_profile.equals(
            FormatParser.from_nightscout_exports(NIGHTSCOUT_ENTRIES)
        )

    def test_no_warning_when_profile_is_not_passed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """The warning is about the caller's argument, not about Nightscout."""
        _skip_if_missing(NIGHTSCOUT_ENTRIES)

        with caplog.at_level("WARNING"):
            FormatParser.from_nightscout_exports(NIGHTSCOUT_ENTRIES)

        assert "profile" not in caplog.text.lower()
