"""Nightscout REST API downloader.

Downloads CGM data from a Nightscout instance via its REST API v1 endpoints.
Requires ``httpx`` (optional dependency — install via ``uv add httpx`` or
``pip install cgm-format[cli]``).

Usage::

    from cgm_format.nightscout_downloader import download_nightscout, download_and_parse_nightscout

    # Download raw JSON files
    entries_path, treatments_path, profile_path = download_nightscout(
        "https://my-nightscout.example.com",
        output_dir=Path("data"),
        count=10000,
        days=30,
    )

    # Download and parse to unified DataFrame in one call
    unified_df = download_and_parse_nightscout(
        "https://my-nightscout.example.com",
        count=10000,
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import polars as pl


def _ensure_httpx():  # type: ignore[no-untyped-def]
    """Import and return httpx, raising a clear error if missing."""
    try:
        import httpx
        return httpx
    except ImportError:
        raise ImportError(
            "httpx is required for Nightscout downloads. "
            "Install it with: uv add httpx  (or pip install httpx)"
        )


def download_nightscout(
    base_url: str,
    output_dir: Path = Path("."),
    count: int = 10_000,
    token: Optional[str] = None,
    days: Optional[int] = None,
    timeout: float = 60.0,
) -> tuple[Path, Path, Path]:
    """Download entries, treatments, and profile JSON from a Nightscout instance.

    Args:
        base_url: Nightscout base URL (e.g. ``https://my-ns.example.com``)
        output_dir: Directory to save JSON files into (created if missing)
        count: Maximum number of entries/treatments to fetch
        token: Optional API token for authenticated instances
        days: If set, only fetch data from the last N days
        timeout: HTTP request timeout in seconds

    Returns:
        Tuple of (entries_path, treatments_path, profile_path)
    """
    httpx = _ensure_httpx()

    base = base_url.rstrip("/")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    params: dict[str, str] = {"count": str(count)}
    if token:
        params["token"] = token
    if days:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params["find[dateString][$gte]"] = since

    treatment_params: dict[str, str] = {"count": str(count)}
    if token:
        treatment_params["token"] = token
    if days:
        treatment_params["find[created_at][$gte]"] = since  # type: ignore[possibly-undefined]

    profile_params: dict[str, str] = {}
    if token:
        profile_params["token"] = token

    with httpx.Client(timeout=timeout) as client:
        entries_resp = client.get(f"{base}/api/v1/entries.json", params=params)
        entries_resp.raise_for_status()

        treatments_resp = client.get(f"{base}/api/v1/treatments.json", params=treatment_params)
        treatments_resp.raise_for_status()

        profile_resp = client.get(f"{base}/api/v1/profile.json", params=profile_params)
        profile_resp.raise_for_status()

    entries_path = output_dir / "nightscout_entries.json"
    treatments_path = output_dir / "nightscout_treatments.json"
    profile_path = output_dir / "nightscout_profile.json"

    entries_path.write_text(json.dumps(entries_resp.json(), indent=2, ensure_ascii=False))
    treatments_path.write_text(json.dumps(treatments_resp.json(), indent=2, ensure_ascii=False))
    profile_path.write_text(json.dumps(profile_resp.json(), indent=2, ensure_ascii=False))

    return entries_path, treatments_path, profile_path


def download_and_parse_nightscout(
    base_url: str,
    count: int = 10_000,
    token: Optional[str] = None,
    days: Optional[int] = None,
    timeout: float = 60.0,
    output_dir: Optional[Path] = None,
) -> pl.DataFrame:
    """Download Nightscout data and parse to unified format in one call.

    If *output_dir* is provided, raw JSON files are also saved to disk.

    Args:
        base_url: Nightscout base URL
        count: Maximum number of entries/treatments to fetch
        token: Optional API token
        days: If set, only fetch data from the last N days
        timeout: HTTP request timeout in seconds
        output_dir: Optional directory to persist raw JSON alongside parsing

    Returns:
        Unified-format Polars DataFrame
    """
    from cgm_format.format_parser import FormatParser

    httpx = _ensure_httpx()
    base = base_url.rstrip("/")

    params: dict[str, str] = {"count": str(count)}
    if token:
        params["token"] = token
    if days:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params["find[dateString][$gte]"] = since

    treatment_params: dict[str, str] = {"count": str(count)}
    if token:
        treatment_params["token"] = token
    if days:
        treatment_params["find[created_at][$gte]"] = since  # type: ignore[possibly-undefined]

    with httpx.Client(timeout=timeout) as client:
        entries_resp = client.get(f"{base}/api/v1/entries.json", params=params)
        entries_resp.raise_for_status()

        treatments_resp = client.get(f"{base}/api/v1/treatments.json", params=treatment_params)
        treatments_resp.raise_for_status()

    entries_json = json.dumps(entries_resp.json(), ensure_ascii=False)
    treatments_json = json.dumps(treatments_resp.json(), ensure_ascii=False)

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "nightscout_entries.json").write_text(
            json.dumps(entries_resp.json(), indent=2, ensure_ascii=False)
        )
        (output_dir / "nightscout_treatments.json").write_text(
            json.dumps(treatments_resp.json(), indent=2, ensure_ascii=False)
        )

    return FormatParser.parse_nightscout(entries_json, treatments_json)
