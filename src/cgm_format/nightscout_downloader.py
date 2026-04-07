"""Nightscout REST API downloader.

Downloads CGM data from a Nightscout instance via its REST API v1 endpoints.
Requires ``httpx`` (optional dependency — install via ``uv add httpx`` or
``pip install cgm-format[cli]``).

The downloader fetches three JSON files:

- **entries** (``/api/v1/entries.json``): sensor glucose values
- **treatments** (``/api/v1/treatments.json``): insulin, carbs, temp basals, etc.
- **profile** (``/api/v1/profile.json``): pump / loop profile configuration

Authentication: Nightscout supports two auth methods:

- **token** — a readable access token passed as ``?token=...`` query param
- **api_secret** — the ``API_SECRET`` value, sent as a SHA1 hash in the
  ``api-secret`` header

Usage::

    from cgm_format.nightscout_downloader import download_nightscout

    # With readable token
    download_nightscout("https://my-ns.example.com", token="mytoken-1234")

    # With API secret
    download_nightscout("https://my-ns.example.com", api_secret="my-api-secret")
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional


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


def _build_headers(api_secret: Optional[str]) -> dict[str, str]:
    """Build HTTP headers, including api-secret if provided."""
    headers: dict[str, str] = {}
    if api_secret:
        # Nightscout expects the SHA1 hash of the API_SECRET
        secret_hash = hashlib.sha1(api_secret.encode("utf-8")).hexdigest()
        headers["api-secret"] = secret_hash
    return headers


def download_nightscout(
    base_url: str,
    output_dir: Path = Path("."),
    count: int = 10_000,
    token: Optional[str] = None,
    api_secret: Optional[str] = None,
    days: Optional[int] = None,
    timeout: float = 60.0,
) -> tuple[Path, Path, Path]:
    """Download entries, treatments, and profile JSON from a Nightscout instance.

    Args:
        base_url: Nightscout base URL (e.g. ``https://my-ns.example.com``)
        output_dir: Directory to save files into (created if missing)
        count: Maximum number of entries/treatments to fetch
        token: Optional readable access token (passed as query param)
        api_secret: Optional API_SECRET (hashed and sent as ``api-secret`` header)
        days: If set, only fetch data from the last N days
        timeout: HTTP request timeout in seconds

    Returns:
        Tuple of (entries_json_path, treatments_json_path, profile_json_path)
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

    headers = _build_headers(api_secret)

    with httpx.Client(timeout=timeout, headers=headers) as client:
        entries_resp = client.get(f"{base}/api/v1/entries.json", params=entry_params)
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
