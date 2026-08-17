#!/usr/bin/env python3
"""Move answered sections from the inbox document into the history file. See TRIAGE_LOOP.md.

A tool rather than a careful copy-paste, because the one property that matters is easy to break by
hand: the reporter's prose must move **byte-for-byte**. The move is verified rather than trusted — each
section's fingerprint is computed before and after, and the write is refused if any of them changed.

It verifies the *move*, not the *verdict*: an item the ledger still calls `new` archives without
complaint. Run the ledger against the history file afterwards — a well-formed archived item reads
`current` there, so a `new` or `unmarked-reply` means something was archived unanswered.

A section is moved together with its group heading and dateline when the history file is not already
under that group. A group whose items split across the two files therefore keeps its dateline in both,
which is the documented shape.

The contents line is deliberately NOT written: naming what an item was and how it ended is editorial,
and a generated line would be a worse version of the thing the index exists for.

Configuration (environment, all optional): INBOX, HISTORY, PREFIX — as for triage-state.py, and with
the same local change: the defaults are derived from this script's own location (<repo>/docs/), not
from the current working directory.

Python, and named `.py` for it: run it or pass it to `python3`, never to `bash`. See the extension
gotcha in TRIAGE_LOOP.md § 5 — under bash this docstring is executed as commands.

Usage:
    triage-archive.py S8 S10 [--dry-run]
"""

import os
import pathlib
import re
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_INBOX = HERE.parent / "docs" / "FEEDBACK.md"
PREFIX = os.environ.get("PREFIX", "S")
INBOX = pathlib.Path(os.environ.get("INBOX", DEFAULT_INBOX)).resolve()
HISTORY = pathlib.Path(
    os.environ.get("HISTORY", INBOX.with_name(f"{INBOX.stem}_HISTORY{INBOX.suffix}"))
)
LEDGER = HERE / "triage-state.py"

SECTION_RE = re.compile(rf"^## +{re.escape(PREFIX)}(\d+)\b")
GROUP_RE = re.compile(r"^# +\S")
BOUNDARY_RE = re.compile(r"^#{1,2} ")


def section_span(lines: list[str], ident: str) -> tuple[int, int]:
    """(start, end) indices of `## <ident>` and one past its last line."""
    for i, line in enumerate(lines):
        found = SECTION_RE.match(line)
        if not found or PREFIX + found.group(1) != ident:
            continue
        end = len(lines)
        for j in range(i + 1, len(lines)):
            if BOUNDARY_RE.match(lines[j]):
                end = j
                break
        return i, end
    raise SystemExit(f"{ident}: no such section in {INBOX.name}")


def group_span(lines: list[str], before: int) -> tuple[int, int] | None:
    """(start, end) of the `# ` group heading and dateline preceding index `before`.

    **A document's own title is not a group heading**, and conflating the two was a real bug that hit
    twice before anyone noticed. A section filed under no group — the normal shape once the split keeps
    the inbox empty, since someone appending a single report writes no group heading — took the inbox's
    `# <title>` *and its whole preamble* into the history file as that item's heading, because the last
    `# ` before the section is the document title and its span runs to the next `##`.

    **The fingerprint check cannot catch this**, which is why it survived: fingerprints cover the
    reporter's prose alone, so the move verifies clean while the history file grows duplicate front
    matter.

    It does **not** disturb the section above the injection, and an earlier version of this docstring
    said it did — the heading is separated by one blank line and `fingerprint()` ends in `.strip()`, so
    the preceding hash is unaffected. TRIAGE_LOOP.md § 5 had the correction; this docstring kept the
    tempting story, which is the same "establish it, then write it" failure one layer down.

    The first `# ` heading in a document is its title by convention, so a group heading is any *later*
    one. A section with no group returns None and the caller says so out loud rather than inventing a
    name — naming a group (who reported it, when) is editorial, the same reason the contents line is
    not generated either.
    """
    headings = [i for i, line in enumerate(lines) if GROUP_RE.match(line)]
    start = None
    for i in headings[1:]:  # [0] is the document title
        if i < before:
            start = i
    if start is None:
        return None
    end = len(lines)
    for j in range(start + 1, len(lines)):
        if BOUNDARY_RE.match(lines[j]):
            end = j
            break
    return start, end


def current_group(lines: list[str]) -> str | None:
    """Text of the last `# ` heading in a file, or None."""
    found = [line for line in lines if GROUP_RE.match(line)]
    return found[-1] if found else None


def fingerprints(doc: pathlib.Path) -> dict[str, str]:
    """`{id: sha}` as the ledger reports it, so before/after can be compared on its own terms."""
    env = {**os.environ, "INBOX": str(INBOX), "HISTORY": str(HISTORY), "PREFIX": PREFIX}
    out = subprocess.run(
        [sys.executable, str(LEDGER), str(doc)],
        capture_output=True,
        text=True,
        env=env,
        check=True,
    ).stdout
    pattern = rf"\b({re.escape(PREFIX)}\d+)\b.*\bsha ([0-9a-f]{{12}})"
    return {
        m.group(1): m.group(2)
        for m in (re.search(pattern, line) for line in out.splitlines())
        if m
    }


def main() -> int:
    flags = {a for a in sys.argv[1:] if a.startswith("--")}
    idents = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not idents:
        raise SystemExit(__doc__)
    if not HISTORY.is_file():
        raise SystemExit(f"no history file at {HISTORY} — create it first (see TRIAGE_LOOP.md)")

    before = fingerprints(INBOX)
    missing = [i for i in idents if i not in before]
    if missing:
        raise SystemExit(f"not in {INBOX.name}: {', '.join(missing)}")

    inbox = INBOX.read_text().splitlines()
    history = HISTORY.read_text().splitlines()

    # Sections are cut back-to-front so earlier spans stay valid, but appended in the order given.
    moved: list[tuple[str, list[str], list[str] | None]] = []
    for ident in idents:
        start, end = section_span(inbox, ident)
        group = group_span(inbox, start)
        heading = inbox[group[0] : group[1]] if group else None
        moved.append((ident, inbox[start:end], heading))

    for ident, _, _ in reversed(list(moved)):
        start, end = section_span(inbox, ident)
        del inbox[start:end]

    ungrouped: list[str] = []
    for ident, body, heading in moved:
        if heading and current_group(history) != heading[0]:
            if history and history[-1].strip():
                history.append("")
            history += heading
        if history and history[-1].strip():
            history.append("")
        history += body
        print(f"{ident}: moved ({len(body)} lines)")
        if heading is None:
            ungrouped.append(ident)

    def render(lines: list[str]) -> str:
        return "\n".join(lines).rstrip("\n") + "\n"

    if "--dry-run" in flags:
        print("\n--dry-run: nothing written")
        return 0

    INBOX.write_text(render(inbox))
    HISTORY.write_text(render(history))

    after = fingerprints(HISTORY)
    broken = [i for i in idents if after.get(i) != before[i]]
    for ident in idents:
        print(f"  {ident}: sha {before[ident]} -> {after.get(ident, 'MISSING')}")
    if broken:
        raise SystemExit(
            f"\nFINGERPRINT CHANGED for {', '.join(broken)} — the prose was not moved verbatim"
        )
    print(
        f"\n{len(idents)} section(s) archived, every fingerprint intact."
        f"\nNow add each one's line to {HISTORY.name}'s contents list."
    )
    if ungrouped:
        print(
            f"\nNo group heading travelled with {', '.join(ungrouped)} — the section sat under the "
            f"inbox's title, which is not a group. Add a `# ` heading above it in {HISTORY.name} "
            f"(who reported it, and when) so it does not read as part of the group above."
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

