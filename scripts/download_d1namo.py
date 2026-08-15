#!/usr/bin/env python3
"""Fetch the two D1NAMO archives the parser needs.

A developer setup chore, not a runtime feature: this lives in `scripts/` behind
the `dev` extra and is **never shipped in the package**. Fetching a static
published archive once is not something a consuming app should be able to
trigger by accident, and the full corpus is 11.2 GB.

Only two of the six archives are downloaded. The Zephyr physiological streams
(ECG at 250 Hz, accelerometer at 100 Hz) are deliberately out of scope — see
`formats/d1namo.py` — so the ~9.4 GB of waveform data is not fetched.

Zenodo record 5651217, CC BY-SA 4.0, anonymous access. Share-alike: a derived
excerpt carries obligations onward, which is why the repo ships synthetic
fixtures instead of corpus samples.

Usage:
    uv run python scripts/download_d1namo.py --dest data/input/d1namo
"""

import argparse
import logging
import sys
import urllib.request
import zipfile
from pathlib import Path

logger = logging.getLogger("download_d1namo")

ZENODO_RECORD = "5651217"
BASE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD}/files"

# The annotation archives only. Names must match the Zenodo record exactly.
WANTED_ARCHIVES = (
    "diabetes_subset_pictures-glucose-food-insulin.zip",
    "healthy_subset_pictures-glucose-food.zip",
)


def download(dest: Path, extract: bool) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    for name in WANTED_ARCHIVES:
        target = dest / name
        if target.exists() and target.stat().st_size > 0:
            logger.info("%s already present, skipping", name)
        else:
            url = f"{BASE_URL}/{name}/content"
            logger.info("fetching %s", url)
            urllib.request.urlretrieve(url, target)
            logger.info("wrote %s (%.1f MB)", target, target.stat().st_size / 1e6)

        if extract:
            with zipfile.ZipFile(target) as archive:
                archive.extractall(dest)
            logger.info("extracted %s", name)

    logger.info(
        "Done. Point CGM_FORMAT_D1NAMO_DIR at %s so the tests can find it.", dest
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/input/d1namo"),
        help="Directory to download into (default: data/input/d1namo)",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Download the zips without extracting them",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    return download(args.dest, extract=not args.no_extract)


if __name__ == "__main__":
    sys.exit(main())
