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
import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


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


_TOKEN_QUERY_RE = re.compile(r"([?&]token=)[^&\s]*")


def _raise_for_status_without_token(response: "httpx.Response") -> None:
    """``raise_for_status`` with the read token stripped from the message.

    httpx builds ``HTTPStatusError`` messages containing the full request URL, and the
    Nightscout read token travels as a ``?token=`` query parameter. So any caller that
    logs the exception -- which is the normal thing to do with a failed download --
    writes a live credential into its logs (S5).

    The exception type is preserved deliberately: callers already catching
    ``httpx.HTTPStatusError`` keep working, and this is a redaction rather than a new
    error contract.
    """
    if not response.is_error:
        return
    httpx = _ensure_httpx()
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise httpx.HTTPStatusError(
            _TOKEN_QUERY_RE.sub(r"\1<redacted>", str(exc)),
            request=exc.request,
            response=exc.response,
        ) from None


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

    # follow_redirects=True is not httpx's default (it is requests'), and every shape of
    # Nightscout host redirects: http:// -> https://, apex -> www, a PaaS hostname 301ing to
    # its canonical domain, trailing-slash normalisation. Without it those all die on the
    # raise_for_status below with "Redirect response '301 Moved Permanently'" instead of
    # working, and users type bare hostnames constantly (S3).
    with httpx.Client(timeout=timeout, headers=headers, follow_redirects=True) as client:
        entries_resp = client.get(f"{base}/api/v1/entries.json", params=entry_params)
        _raise_for_status_without_token(entries_resp)

        treatments_resp = client.get(f"{base}/api/v1/treatments.json", params=treatment_params)
        _raise_for_status_without_token(treatments_resp)

        # The profile is fetched and then discarded -- it holds settings, not readings.
        # So a permission that allows entries but denies profile must not fail the whole
        # import on an endpoint whose result is thrown away (S5).
        profile_resp = client.get(f"{base}/api/v1/profile.json", params=profile_params)
        if profile_resp.is_error:
            logger.warning(
                "Nightscout profile endpoint returned %s; continuing without it. "
                "The profile holds settings rather than readings and is not used.",
                profile_resp.status_code,
            )

    entries_path = output_dir / "nightscout_entries.json"
    treatments_path = output_dir / "nightscout_treatments.json"
    profile_path = output_dir / "nightscout_profile.json"

    entries_path.write_text(json.dumps(entries_resp.json(), indent=2, ensure_ascii=False))
    treatments_path.write_text(json.dumps(treatments_resp.json(), indent=2, ensure_ascii=False))
    profile_path.write_text(json.dumps(profile_resp.json(), indent=2, ensure_ascii=False))

    return entries_path, treatments_path, profile_path
