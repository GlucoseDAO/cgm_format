"""BIG IDEAs downloader talks to PhysioNet, then the importer reads that extract.

People using this library will not have a sibling sugar-sugar checkout. The
script therefore fetches the published Dexcom + food-log files from PhysioNet
(`files.physionet.org`; the open S3 mirror stops at 1.1.2 and cannot serve
the pinned 1.1.3). Empatica streams are never requested.

This file hits the live URLs. It does not mock the HTTP layer and it does
not read a local sugar-sugar `data/bigideas/` tree.
"""

from __future__ import annotations

import importlib.util
import urllib.error
from pathlib import Path
from types import ModuleType

import polars as pl
import pytest

from cgm_format import CGM_SCHEMA_EXTENDED, FormatParser, UnifiedEventType

SCRIPT = Path(__file__).parent.parent / "scripts" / "download_bigideas.py"


def _load_downloader() -> ModuleType:
    spec = importlib.util.spec_from_file_location("download_bigideas", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


DOWNLOADER = _load_downloader()


class TestPublicUrls:
    def test_the_script_points_at_physionet_not_a_local_path(self) -> None:
        assert DOWNLOADER.PHYSIONET_PAGE.startswith("https://physionet.org/content/")
        assert DOWNLOADER.FILES_BASE.startswith("https://physionet.org/files/")
        assert "sugar-sugar" not in DOWNLOADER.FILES_BASE

    def test_file_urls_are_https_paths_under_the_pinned_release(self) -> None:
        rel = "001/Dexcom_001.csv"
        assert DOWNLOADER.files_url(rel) == f"{DOWNLOADER.FILES_BASE}/{rel}"
        # The page and the file base must name the same release, or the script
        # documents one version and downloads another.
        assert DOWNLOADER.PHYSIONET_PAGE.rstrip("/").endswith(
            DOWNLOADER.FILES_BASE.rsplit("/", 1)[-1]
        )

    def test_there_is_exactly_one_source(self) -> None:
        """The S3 mirror was removed, not merely deprioritized.

        It carries this dataset only through 1.1.2, so a tier pinned to 1.1.3
        404s on every file and doubles the request count for nothing.
        """
        assert not hasattr(DOWNLOADER, "S3_BASE")
        assert not hasattr(DOWNLOADER, "s3_url")


class TestOnlineDownloadAndImport:
    def test_two_published_subjects_download_and_parse(self, tmp_path: Path) -> None:
        """Subject 001 is the canonical food header; 003 is the headerless log."""
        try:
            dest = DOWNLOADER.fetch_bigideas(
                tmp_path / "bigideas",
                force=True,
                subjects=(1, 3),
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            pytest.skip(f"PhysioNet is unreachable: {exc}")

        dexcom_001 = dest / "001" / "Dexcom_001.csv"
        food_001 = dest / "001" / "Food_Log_001.csv"
        food_003 = dest / "003" / "Food_Log_003.csv"
        assert dexcom_001.is_file()
        assert food_001.is_file()
        assert food_003.is_file()
        assert not any(dest.rglob("ACC_*.csv"))
        assert not any(dest.rglob("BVP_*.csv"))

        frames = FormatParser.parse_corpus(dest)
        assert set(frames) == {"001", "003"}

        for subject_id, frame in frames.items():
            CGM_SCHEMA_EXTENDED.validate_dataframe(frame, enforce=False)
            glucose = frame.filter(
                pl.col("event_type") == UnifiedEventType.GLUCOSE.value
            )
            raw_egv = pl.read_csv(
                dest / subject_id / f"Dexcom_{subject_id}.csv",
                infer_schema_length=0,
            ).filter(pl.col("Event Type") == "EGV")
            assert len(glucose) == len(raw_egv), f"{subject_id} EGV count drifted"

        meals_003 = frames["003"].filter(
            pl.col("event_type") == UnifiedEventType.CARBOHYDRATES.value
        )
        raw_003_lines = food_003.read_text(encoding="utf-8").splitlines()
        assert len(meals_003) == len(raw_003_lines)
