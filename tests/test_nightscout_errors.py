"""RM15 (desk #109): a failed Nightscout download raises the library's own `NightscoutError` kinds,
each of which is also the httpx or json exception the download raised before.

Every case runs the real downloader against a real HTTP server on a local port (or a closed one),
so the code under test is not mocked: only the replies are chosen.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Dict, Iterator, Tuple

import pytest

httpx = pytest.importorskip("httpx", reason="httpx is an optional (cli/dev) dependency")

from cgm_format import (  # noqa: E402
    FormatParser,
    NightscoutError,
    NightscoutResponseError,
    NightscoutStatusError,
    NightscoutUnreachableError,
)
from cgm_format.nightscout_downloader import download_nightscout  # noqa: E402

CLOSED_PORT_URL = "http://127.0.0.1:9"
TOKEN = "READTOKEN-rm15-7f3a"
SGV = json.dumps([{"type": "sgv", "dateString": "2026-11-15T00:00:00.000Z", "sgv": 110}])

# endpoint name -> (status, content type, body)
Replies = Dict[str, Tuple[int, str, str]]


@contextlib.contextmanager
def _server(replies: Replies) -> Iterator[str]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            endpoint = self.path.split("?")[0].rsplit("/", 1)[-1]
            status, ctype, body = replies[endpoint]
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.end_headers()
            self.wfile.write(body.encode())

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()


@pytest.fixture
def healthy() -> Replies:
    return {
        "entries.json": (200, "application/json", SGV),
        "treatments.json": (200, "application/json", "[]"),
        "profile.json": (200, "application/json", "[{}]"),
    }


def _download(url: str, tmp_path: Path) -> Tuple[Path, Path, Path]:
    return download_nightscout(url, output_dir=tmp_path / "pull", token=TOKEN, timeout=10)


class TestTheThreeKinds:
    def test_no_reply_is_unreachable_and_still_an_httpx_connect_error(self, tmp_path: Path) -> None:
        with pytest.raises(NightscoutUnreachableError) as excinfo:
            _download(CLOSED_PORT_URL, tmp_path)
        error = excinfo.value
        assert isinstance(error, httpx.ConnectError)
        assert isinstance(error, NightscoutError)
        assert isinstance(error.__cause__, httpx.ConnectError)  # the original stays attached
        assert TOKEN not in str(error)

    def test_an_error_status_carries_its_code_and_is_still_http_status_error(
        self, tmp_path: Path, healthy: Replies
    ) -> None:
        healthy["entries.json"] = (502, "text/plain", "bad gateway")
        with _server(healthy) as url:
            with pytest.raises(NightscoutStatusError) as excinfo:
                _download(url, tmp_path)
        error = excinfo.value
        assert error.status_code == 502
        assert isinstance(error, httpx.HTTPStatusError)
        assert error.response.status_code == 502
        assert TOKEN not in str(error) and "<redacted>" in str(error)
        # The unredacted original carries the token in its message: it must not ride along.
        assert error.__cause__ is None and error.__suppress_context__

    def test_a_reply_that_is_not_json_names_its_endpoint(self, tmp_path: Path, healthy: Replies) -> None:
        healthy["entries.json"] = (200, "text/html", "<html>a proxy's page</html>")
        with _server(healthy) as url:
            with pytest.raises(NightscoutResponseError) as excinfo:
                _download(url, tmp_path)
        error = excinfo.value
        assert isinstance(error, json.JSONDecodeError)
        assert str(error).startswith("entries.json: the reply is not JSON")
        assert not (tmp_path / "pull" / "nightscout_entries.json").exists()

    def test_the_family_is_caught_without_naming_httpx(self, tmp_path: Path) -> None:
        caught = None
        try:
            _download(CLOSED_PORT_URL, tmp_path)
        except NightscoutError as e:
            caught = e
        assert isinstance(caught, NightscoutUnreachableError)

    def test_the_base_classes_do_not_need_httpx(self) -> None:
        for kind in (NightscoutError, NightscoutUnreachableError, NightscoutStatusError, NightscoutResponseError):
            assert all(c.__module__.split(".")[0] in {"cgm_format", "builtins"} for c in kind.__mro__)

    def test_from_nightscout_url_raises_the_same_kind(self) -> None:
        with pytest.raises(NightscoutUnreachableError):
            FormatParser.from_nightscout_url(CLOSED_PORT_URL, timeout=10)


class TestTheProfileNeverFailsAnImport:
    """The profile is fetched and thrown away (S5): its failures are warnings."""

    @pytest.mark.parametrize(
        "reply, said",
        [
            ((403, "application/json", '{"status": 403}'), "returned 403"),
            ((200, "text/html", "<html>login</html>"), "not JSON"),
        ],
    )
    def test_a_failed_profile_is_a_warning_and_null(
        self, tmp_path: Path, healthy: Replies, caplog: pytest.LogCaptureFixture,
        reply: Tuple[int, str, str], said: str,
    ) -> None:
        healthy["profile.json"] = reply
        with _server(healthy) as url:
            with caplog.at_level(logging.WARNING, logger="cgm_format.nightscout_downloader"):
                entries, _, profile = _download(url, tmp_path)
        assert json.loads(entries.read_text()) == json.loads(SGV)
        assert json.loads(profile.read_text()) is None
        assert any(said in r.getMessage() for r in caplog.records)

    def test_a_healthy_profile_is_kept(self, tmp_path: Path, healthy: Replies) -> None:
        with _server(healthy) as url:
            _, _, profile = _download(url, tmp_path)
        assert json.loads(profile.read_text()) == [{}]


def test_the_cli_says_each_kind_in_one_line(tmp_path: Path, healthy: Replies) -> None:
    healthy["entries.json"] = (401, "text/plain", "Unauthorized")
    env = {k: v for k, v in os.environ.items() if not k.startswith("NIGHTSCOUT_")}
    with _server(healthy) as url:
        result = subprocess.run(
            [sys.executable, "-m", "cgm_format.cgm_cli", "download-nightscout", url,
             "--token", TOKEN, "--output", str(tmp_path / "pull")],
            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120,
        )
    output = result.stdout + result.stderr
    assert result.returncode == 1
    assert "replied 401" in output
    assert "Traceback" not in output
    assert TOKEN not in output
