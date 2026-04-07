"""Nightscout REST API downloader.

Downloads CGM data from a Nightscout instance via its REST API v1 endpoints.
Requires ``httpx`` (optional dependency — install via ``uv add httpx`` or
``pip install cgm-format[cli]``).

The Nightscout API supports multiple output formats via endpoint extensions:

- ``.json`` — native JSON (default)
- ``.csv`` — comma-separated values

Both formats are accepted by :class:`~cgm_format.format_parser.FormatParser`
and auto-detected during parsing.

Usage::

    from cgm_format.nightscout_downloader import download_nightscout, download_and_parse_nightscout

    # Download raw JSON files (default)
    entries_path, treatments_path, profile_path = download_nightscout(
        "https://my-nightscout.example.com",
        output_dir=Path("data"),
        count=10000,
        days=30,
    )

    # Download as CSV instead
    entries_path, treatments_path, profile_path = download_nightscout(
        "https://my-nightscout.example.com",
        output_dir=Path("data"),
        api_format="csv",
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
from typing import Literal, Optional

import polars as pl

NightscoutApiFormat = Literal["json", "csv"]


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


def _build_params(
    count: int,
    token: Optional[str],
    days: Optional[int],
    date_field: str = "dateString",
) -> dict[str, str]:
    """Build common query parameters for Nightscout API requests."""
    params: dict[str, str] = {"count": str(count)}
    if token:
        params["token"] = token
    if days:
        since = (datetime.now(tz=timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        params[f"find[{date_field}][$gte]"] = since
    return params


def download_nightscout(
    base_url: str,
    output_dir: Path = Path("."),
    count: int = 10_000,
    token: Optional[str] = None,
    days: Optional[int] = None,
    timeout: float = 60.0,
    api_format: NightscoutApiFormat = "json",
) -> tuple[Path, Path, Path]:
    """Download entries, treatments, and profile from a Nightscout instance.

    Args:
        base_url: Nightscout base URL (e.g. ``https://my-ns.example.com``)
        output_dir: Directory to save files into (created if missing)
        count: Maximum number of entries/treatments to fetch
        token: Optional API token for authenticated instances
        days: If set, only fetch data from the last N days
        timeout: HTTP request timeout in seconds
        api_format: Response format to request from the API (``"json"`` or ``"csv"``)

    Returns:
        Tuple of (entries_path, treatments_path, profile_path).
        Profile is always downloaded as JSON regardless of *api_format*.
    """
    httpx = _ensure_httpx()

    base = base_url.rstrip("/")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entry_params = _build_params(count, token, days, date_field="dateString")
    treatment_params = _build_params(count, token, days, date_field="created_at")

    profile_params: dict[str, str] = {}
    if token:
        profile_params["token"] = token

    ext = api_format  # "json" or "csv"

    with httpx.Client(timeout=timeout) as client:
        entries_resp = client.get(f"{base}/api/v1/entries.{ext}", params=entry_params)
        entries_resp.raise_for_status()

        treatments_resp = client.get(f"{base}/api/v1/treatments.{ext}", params=treatment_params)
        treatments_resp.raise_for_status()

        profile_resp = client.get(f"{base}/api/v1/profile.json", params=profile_params)
        profile_resp.raise_for_status()

    entries_path = output_dir / f"nightscout_entries.{ext}"
    treatments_path = output_dir / f"nightscout_treatments.{ext}"
    profile_path = output_dir / "nightscout_profile.json"

    if api_format == "json":
        entries_path.write_text(json.dumps(entries_resp.json(), indent=2, ensure_ascii=False))
        treatments_path.write_text(json.dumps(treatments_resp.json(), indent=2, ensure_ascii=False))
    else:
        entries_path.write_text(entries_resp.text)
        treatments_path.write_text(treatments_resp.text)

    profile_path.write_text(json.dumps(profile_resp.json(), indent=2, ensure_ascii=False))

    return entries_path, treatments_path, profile_path


def download_and_parse_nightscout(
    base_url: str,
    count: int = 10_000,
    token: Optional[str] = None,
    days: Optional[int] = None,
    timeout: float = 60.0,
    output_dir: Optional[Path] = None,
    api_format: NightscoutApiFormat = "json",
) -> pl.DataFrame:
    """Download Nightscout data and parse to unified format in one call.

    If *output_dir* is provided, raw response files are also saved to disk.

    Args:
        base_url: Nightscout base URL
        count: Maximum number of entries/treatments to fetch
        token: Optional API token
        days: If set, only fetch data from the last N days
        timeout: HTTP request timeout in seconds
        output_dir: Optional directory to persist raw data alongside parsing
        api_format: Response format to request from the API (``"json"`` or ``"csv"``)

    Returns:
        Unified-format Polars DataFrame
    """
    from cgm_format.format_parser import FormatParser

    httpx = _ensure_httpx()
    base = base_url.rstrip("/")

    entry_params = _build_params(count, token, days, date_field="dateString")
    treatment_params = _build_params(count, token, days, date_field="created_at")

    ext = api_format

    with httpx.Client(timeout=timeout) as client:
        entries_resp = client.get(f"{base}/api/v1/entries.{ext}", params=entry_params)
        entries_resp.raise_for_status()

        treatments_resp = client.get(f"{base}/api/v1/treatments.{ext}", params=treatment_params)
        treatments_resp.raise_for_status()

    if api_format == "json":
        entries_text = json.dumps(entries_resp.json(), ensure_ascii=False)
        treatments_text = json.dumps(treatments_resp.json(), ensure_ascii=False)
    else:
        entries_text = entries_resp.text
        treatments_text = treatments_resp.text

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if api_format == "json":
            (output_dir / "nightscout_entries.json").write_text(
                json.dumps(entries_resp.json(), indent=2, ensure_ascii=False)
            )
            (output_dir / "nightscout_treatments.json").write_text(
                json.dumps(treatments_resp.json(), indent=2, ensure_ascii=False)
            )
        else:
            (output_dir / "nightscout_entries.csv").write_text(entries_text)
            (output_dir / "nightscout_treatments.csv").write_text(treatments_text)

    return FormatParser.parse_nightscout(entries_text, treatments_text)
