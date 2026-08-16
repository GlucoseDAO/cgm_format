#!/usr/bin/env python3
"""Fetch the public BIG IDEAs Dexcom + food-log subset from PhysioNet.

A developer setup chore, not a runtime feature: this lives in `scripts/`
behind the `dev` extra and is **never shipped in the package**. People
without a local copy get the published files from PhysioNet; this script
does not look at a sibling checkout.

The full PhysioNet zip is ~4.7 GB because it includes Empatica ACC/BVP
streams. This downloader fetches only Demographics.csv, Dexcom_NNN.csv,
and Food_Log_NNN.csv (~3 MB). Empatica files are never downloaded.

Source: https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/
Paper: Bent et al., npj Digital Medicine, 2021.
License: ODC-By 1.0.

Usage:
    uv run python scripts/download_bigideas.py --dest data/input/bigideas
"""

from __future__ import annotations

import argparse
import logging
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, Optional, Sequence

logger = logging.getLogger("download_bigideas")

PHYSIONET_PAGE: str = "https://physionet.org/content/big-ideas-glycemic-wearable/1.1.3/"
# Single source, deliberately. The `physionet-open` S3 mirror carries this
# dataset only up to 1.1.2 (listing the bucket shows 1.0.0, 1.1.0, 1.1.1,
# 1.1.2 and no 1.1.3), so an S3-first tier pinned to 1.1.3 404s on every file
# and merely doubles the request count before falling back here. 1.1.3 is the
# version the ground truth in `formats/bigideas.py` was read against, so the
# version stays and the dead tier goes.
FILES_BASE: str = "https://physionet.org/files/big-ideas-glycemic-wearable/1.1.3"
USER_AGENT: str = "cgm-format-bigideas-downloader"
SUBJECT_IDS: tuple[int, ...] = tuple(range(1, 17))
_CHUNK_BYTES: int = 64 * 1024
_TIMEOUT_SECONDS: int = 60


def files_url(rel_path: str) -> str:
    return f"{FILES_BASE}/{rel_path}"


def dataset_is_present(dest: Path) -> bool:
    if not dest.is_dir():
        return False
    return any(dest.rglob("Dexcom_*.csv"))


def _retrieve(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_name(dest.name + ".part")
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as resp, tmp.open(
        "wb"
    ) as out:
        while True:
            chunk = resp.read(_CHUNK_BYTES)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def _download_one(rel_path: str, dest: Path) -> None:
    url = files_url(rel_path)
    logger.info("fetching %s", url)
    try:
        _retrieve(url, dest)
    except (urllib.error.URLError, TimeoutError, OSError):
        # Leave nothing half-written behind, then let the caller see the real
        # error: there is no second source to fall back to.
        dest.unlink(missing_ok=True)
        dest.with_name(dest.name + ".part").unlink(missing_ok=True)
        raise
    logger.info("wrote %s (%.1f KB)", dest, dest.stat().st_size / 1024)


def _parse_subjects(raw: Optional[str]) -> tuple[int, ...]:
    if raw is None or not raw.strip():
        return SUBJECT_IDS
    values: list[int] = []
    for part in raw.split(","):
        subject_id = int(part.strip())
        if subject_id not in SUBJECT_IDS:
            raise argparse.ArgumentTypeError(
                f"subject {subject_id} is not in the published 1–16 range"
            )
        values.append(subject_id)
    return tuple(values)


def fetch_bigideas(
    dest: Path,
    *,
    force: bool = False,
    subjects: Sequence[int] = SUBJECT_IDS,
) -> Path:
    """Download Dexcom + food logs from PhysioNet into ``dest``.

    Returns ``dest``. Empatica streams are never requested.
    """
    dest = dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    wanted: tuple[int, ...] = tuple(subjects)

    if dataset_is_present(dest) and not force and set(wanted) == set(SUBJECT_IDS):
        logger.info(
            "BIG IDEAs already present at %s (use --force to re-download)", dest
        )
        return dest

    logger.info("Downloading BIG IDEAs Dexcom + food logs from PhysioNet (%s)", PHYSIONET_PAGE)
    demo = dest / "Demographics.csv"
    if force or not demo.is_file():
        _download_one("Demographics.csv", demo)

    for subject_id in wanted:
        folder = f"{subject_id:03d}"
        subject_dir = dest / folder
        subject_dir.mkdir(parents=True, exist_ok=True)
        for name in (f"Dexcom_{folder}.csv", f"Food_Log_{folder}.csv"):
            target = subject_dir / name
            if target.is_file() and not force:
                logger.info("%s already present, skipping", target.name)
                continue
            _download_one(f"{folder}/{name}", target)

    if not dataset_is_present(dest):
        raise FileNotFoundError(
            f"BIG IDEAs Dexcom tables were not downloaded into {dest}"
        )
    logger.info(
        "Done. Point CGM_FORMAT_BIGIDEAS_DIR at %s so the tests can find it.", dest
    )
    return dest


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/input/bigideas"),
        help="Directory to download into (default: data/input/bigideas)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the files are already present",
    )
    parser.add_argument(
        "--subjects",
        default=None,
        help="Comma-separated subject numbers (default: all 1-16)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    fetch_bigideas(args.dest, force=args.force, subjects=_parse_subjects(args.subjects))
    return 0


if __name__ == "__main__":
    sys.exit(main())
