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
from typing import Dict, Optional, Tuple, Type, TypeVar

from cgm_format.interface.cgm_interface import (
    NightscoutError,
    NightscoutResponseError,
    NightscoutStatusError,
    NightscoutUnreachableError,
)

logger = logging.getLogger(__name__)

_Kind = TypeVar("_Kind", bound=NightscoutError)
_KIND_CLASSES: Dict[Tuple[type, type], type] = {}


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


def _redact(text: str) -> str:
    return _TOKEN_QUERY_RE.sub(r"\1<redacted>", text)


def _as_kind(kind: Type[_Kind], exc: Exception, message: Optional[str] = None) -> _Kind:
    """`exc` remade as an instance of `kind` that is still an instance of `type(exc)`.

    This is how each `NightscoutError` kind is also the exception it replaces (desk
    #109): the class derives from both, and the instance carries the original's
    attributes (`request`, `response`, a JSON error's `doc` and `pos`), so code
    catching `httpx.ConnectError` or `json.JSONDecodeError` catches it as before.
    The message is redacted of the read token whatever its source.
    """
    original = type(exc)
    cls = _KIND_CLASSES.get((kind, original))
    if cls is None:
        cls = type(
            kind.__name__,
            (kind, original),
            {
                "__module__": kind.__module__,
                "__qualname__": f"{kind.__name__}[{original.__name__}]",
                "__str__": _kind_str,
            },
        )
        _KIND_CLASSES[(kind, original)] = cls
    text = _redact(message if message is not None else str(exc))
    if isinstance(exc, UnicodeDecodeError):
        # A C-level error whose fields only its constructor sets; its args are those fields.
        made = cls(exc.encoding, exc.object, exc.start, exc.end, exc.reason)
    else:
        made = cls.__new__(cls)
        made.__dict__.update(exc.__dict__)
        made.args = (text,)
    made._nightscout_message = text
    return made  # type: ignore[return-value]


def _kind_str(self: BaseException) -> str:
    return getattr(self, "_nightscout_message", "") or BaseException.__str__(self)


def _raise_for_status_without_token(response: "httpx.Response") -> None:
    """``raise_for_status`` as a `NightscoutStatusError`, the read token stripped from the message.

    httpx builds ``HTTPStatusError`` messages containing the full request URL, and the
    Nightscout read token travels as a ``?token=`` query parameter. So any caller that
    logs the exception -- which is the normal thing to do with a failed download --
    writes a live credential into its logs (S5).

    The error is still an ``httpx.HTTPStatusError``: callers already catching that keep
    working. The unredacted original is not chained, since it carries the token.
    """
    if not response.is_error:
        return
    httpx = _ensure_httpx()
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        error = _as_kind(NightscoutStatusError, exc)
        error.status_code = response.status_code
        raise error from None


def _get(client: "httpx.Client", url: str, params: dict[str, str]) -> "httpx.Response":
    """One GET, its failure to get any reply raised as a `NightscoutError` kind."""
    httpx = _ensure_httpx()
    try:
        return client.get(url, params=params)
    except httpx.TransportError as exc:
        # Refused, unresolvable, timed out: nothing came back.
        raise _as_kind(NightscoutUnreachableError, exc) from exc
    except (httpx.TooManyRedirects, httpx.DecodingError) as exc:
        # Something came back, but not a reply that could be read.
        raise _as_kind(NightscoutResponseError, exc) from exc


def _json(response: "httpx.Response", endpoint: str) -> object:
    """The reply's JSON, or a `NightscoutResponseError` naming the endpoint."""
    try:
        return response.json()
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise _as_kind(
            NightscoutResponseError, exc, f"{endpoint}: the reply is not JSON ({exc})"
        ) from exc


def _profile(response: "httpx.Response") -> object:
    """The profile's JSON, or None with a warning: the profile holds settings rather
    than readings and is not used, so neither an error status nor a reply that is not
    JSON may fail an import over it (S5)."""
    if response.is_error:
        reason = f"returned {response.status_code}"
    else:
        try:
            return response.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            reason = "replied with something that is not JSON"
    logger.warning(
        "Nightscout profile endpoint %s; continuing without it, and nightscout_profile.json "
        "holds null. The profile holds settings rather than readings and is not used.",
        reason,
    )
    return None


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

    Raises:
        NightscoutUnreachableError: No reply (refused, unresolvable, timed out).
        NightscoutStatusError: An error status on entries or treatments. The profile's,
            or a profile that is not JSON, is a warning: the profile is not used.
        NightscoutResponseError: A reply that is not JSON, or could not be read.
        Each is also the httpx or json exception this raised before RM15.
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
        entries_resp = _get(client, f"{base}/api/v1/entries.json", entry_params)
        _raise_for_status_without_token(entries_resp)

        treatments_resp = _get(client, f"{base}/api/v1/treatments.json", treatment_params)
        _raise_for_status_without_token(treatments_resp)

        # The profile is fetched and then discarded -- it holds settings, not readings.
        # So a permission that allows entries but denies profile must not fail the whole
        # import on an endpoint whose result is thrown away (S5).
        profile_resp = _get(client, f"{base}/api/v1/profile.json", profile_params)
        entries = _json(entries_resp, "entries.json")
        treatments = _json(treatments_resp, "treatments.json")
        profile = _profile(profile_resp)

    entries_path = output_dir / "nightscout_entries.json"
    treatments_path = output_dir / "nightscout_treatments.json"
    profile_path = output_dir / "nightscout_profile.json"

    entries_path.write_text(json.dumps(entries, indent=2, ensure_ascii=False))
    treatments_path.write_text(json.dumps(treatments, indent=2, ensure_ascii=False))
    profile_path.write_text(json.dumps(profile, indent=2, ensure_ascii=False))

    return entries_path, treatments_path, profile_path
