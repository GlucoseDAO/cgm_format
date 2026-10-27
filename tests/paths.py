"""Where the tests read their inputs and write their outputs (RM3).

Committed fixtures live in `assets/fixtures/`. A person's own exports are never committed: they live in
`data/input/`, git-ignored and possibly absent, and a test that needs one skips without it. Generated
output goes to `data/output/`, git-ignored too.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = PROJECT_ROOT / "assets" / "fixtures"
LOCAL_INPUT_DIR = PROJECT_ROOT / "data" / "input"
OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

NIGHTSCOUT_JSON = ("nightscout_entries.json", "nightscout_treatments.json", "nightscout_profile.json")


def input_csv_files() -> list[Path]:
    """The top-level CSV files of the committed fixtures, then those of the local exports when present."""
    return sorted(FIXTURES_DIR.glob("*.csv")) + sorted(LOCAL_INPUT_DIR.glob("*.csv"))


def nightscout_json_dir() -> Path:
    """`data/input/` when it holds a live Nightscout pull (`tests/conftest.py` downloads one when
    `NIGHTSCOUT_URL` is set), else the committed fixtures."""
    pulled = LOCAL_INPUT_DIR / NIGHTSCOUT_JSON[0]
    return LOCAL_INPUT_DIR if pulled.is_file() and pulled.stat().st_size > 0 else FIXTURES_DIR
