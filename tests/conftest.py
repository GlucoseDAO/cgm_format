"""Shared test configuration and fixtures.

Loads environment variables from the project root ``.env`` file and provides
session-scoped fixtures for downloading Nightscout test data.

The Nightscout download runs **during collection** (not as a lazy fixture) so
that ``test_nightscout.py`` can discover the downloaded files for
parameterisation.  Pass ``--nightscout-redownload`` to force a fresh download;
this flag **requires** ``NIGHTSCOUT_URL`` to be set and will fail the session
immediately if it is missing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

from tests.paths import LOCAL_INPUT_DIR, NIGHTSCOUT_JSON, PROJECT_ROOT, nightscout_json_dir

load_dotenv(PROJECT_ROOT / ".env")

NIGHTSCOUT_EXPECTED_FILES = [LOCAL_INPUT_DIR / name for name in NIGHTSCOUT_JSON]


def _nightscout_download(force: bool) -> Path:
    """Download Nightscout JSON data to data/input/ (local, git-ignored) if needed.

    Called once during collection so files are available for test
    parameterisation.
    """
    LOCAL_INPUT_DIR.mkdir(parents=True, exist_ok=True)

    nightscout_url = os.environ.get("NIGHTSCOUT_URL")
    nightscout_token = os.environ.get("NIGHTSCOUT_TOKEN")
    nightscout_secret = os.environ.get("NIGHTSCOUT_API_SECRET")

    if not nightscout_url:
        if force:
            raise pytest.UsageError(
                "--nightscout-redownload requires NIGHTSCOUT_URL to be set "
                "(in .env or environment)"
            )
        return LOCAL_INPUT_DIR

    all_exist = all(f.exists() and f.stat().st_size > 0 for f in NIGHTSCOUT_EXPECTED_FILES)
    if all_exist and not force:
        return LOCAL_INPUT_DIR

    from cgm_format.nightscout_downloader import download_nightscout

    download_nightscout(
        base_url=nightscout_url,
        output_dir=LOCAL_INPUT_DIR,
        token=nightscout_token,
        api_secret=nightscout_secret,
    )

    return LOCAL_INPUT_DIR


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--nightscout-redownload",
        action="store_true",
        default=False,
        help="Force re-download of Nightscout test data even if files exist",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Run Nightscout download at collection time so files are available for
    parameterised test discovery."""
    force = config.getoption("--nightscout-redownload", default=False)
    _nightscout_download(force)


@pytest.fixture(scope="session")
def nightscout_data_dir() -> Path:
    """The directory the Nightscout JSON is read from: a live pull in data/input/ when there is one
    (downloaded during ``pytest_configure``), else the committed fixtures."""
    return nightscout_json_dir()
