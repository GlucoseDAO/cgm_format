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

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "input"

load_dotenv(PROJECT_ROOT / ".env")

NIGHTSCOUT_EXPECTED_JSON = [
    DATA_DIR / "nightscout_entries.json",
    DATA_DIR / "nightscout_treatments.json",
    DATA_DIR / "nightscout_profile.json",
]
NIGHTSCOUT_EXPECTED_CSV = [
    DATA_DIR / "nightscout_entries.csv",
    DATA_DIR / "nightscout_treatments.csv",
]


def _nightscout_download(force: bool) -> Path:
    """Download Nightscout data to data/input/ if needed.

    Called once during collection so files are available for test
    parameterisation.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    nightscout_url = os.environ.get("NIGHTSCOUT_URL")
    nightscout_token = os.environ.get("NIGHTSCOUT_TOKEN")

    if not nightscout_url:
        if force:
            raise pytest.UsageError(
                "--nightscout-redownload requires NIGHTSCOUT_URL to be set "
                "(in .env or environment)"
            )
        return DATA_DIR

    all_expected = NIGHTSCOUT_EXPECTED_JSON + NIGHTSCOUT_EXPECTED_CSV
    all_exist = all(f.exists() and f.stat().st_size > 0 for f in all_expected)
    if all_exist and not force:
        return DATA_DIR

    from cgm_format.nightscout_downloader import download_nightscout

    download_nightscout(
        base_url=nightscout_url,
        output_dir=DATA_DIR,
        token=nightscout_token,
        api_format="json",
    )

    download_nightscout(
        base_url=nightscout_url,
        output_dir=DATA_DIR,
        token=nightscout_token,
        api_format="csv",
    )

    return DATA_DIR


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
    """Return the data/input/ directory (download already happened during
    ``pytest_configure``)."""
    return DATA_DIR
