"""RM4: cgm-cli reads its Nightscout credentials from the environment and from `.env` in the current
directory, so the credentials that make the test suite reach a Nightscout instance work for the CLI.

Each test runs the CLI in a subprocess with the NIGHTSCOUT_* variables removed from its environment:
`tests/conftest.py` loads the repository's own `.env` into this process at import, and a subprocess
inherits it, which would make a missing `.env` load pass unseen. The URL points at a closed local port,
so nothing leaves the machine and the error names the URL the CLI used.
"""

import os
import subprocess
import sys
from pathlib import Path

CLOSED_PORT_URL = "http://127.0.0.1:9"


def run_cli(args: list[str], cwd: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("NIGHTSCOUT_")}
    env.update(extra_env or {})
    return subprocess.run(
        [sys.executable, "-m", "cgm_format.cgm_cli", *args],
        cwd=cwd, env=env, capture_output=True, text=True, timeout=120,
    )


def test_download_nightscout_reads_url_from_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(f"NIGHTSCOUT_URL={CLOSED_PORT_URL}\n")
    result = run_cli(["download-nightscout", "--output", str(tmp_path / "pull")], cwd=tmp_path)
    output = result.stdout + result.stderr
    assert "Missing argument" not in output
    assert "127.0.0.1:9" in output
    assert result.returncode != 0  # the port is closed: it fails, after using the URL


def test_the_environment_wins_over_dotenv(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("NIGHTSCOUT_URL=http://127.0.0.1:8\n")
    result = run_cli(["download-nightscout", "--output", str(tmp_path / "pull")], cwd=tmp_path,
                     extra_env={"NIGHTSCOUT_URL": CLOSED_PORT_URL})
    output = result.stdout + result.stderr
    assert "127.0.0.1:9" in output and "127.0.0.1:8" not in output


def test_no_url_anywhere_is_still_refused(tmp_path: Path) -> None:
    result = run_cli(["download-nightscout", "--output", str(tmp_path / "pull")], cwd=tmp_path)
    assert result.returncode != 0
    assert "NIGHTSCOUT_URL" in result.stdout + result.stderr
