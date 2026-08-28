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

import importlib.util
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

# The span logic is DERIVED from the ledger rather than restated beside it. The two tools disagreeing
# about where a section ends is exactly how a report gets cut in half: each carried its own copy of
# the boundary scan, so fixing one left the other cutting at the wrong line. The ledger is already
# the authority for fingerprints; it is the authority for spans too. importlib because of the hyphen.
_spec = importlib.util.spec_from_file_location("triage_state", LEDGER)
_ledger = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ledger)
fenced_lines = _ledger.fenced_lines
boundary_after = _ledger.boundary_after
fence_findings = _ledger.fence_findings


def section_span(lines: list[str], ident: str) -> tuple[int, int]:
    """(start, end) indices of `## <ident>` and one past its last line."""
    for i, line in enumerate(lines):
        found = SECTION_RE.match(line)
        if not found or PREFIX + found.group(1) != ident:
            continue
        return i, boundary_after(lines, i)
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
    fenced, _ = fenced_lines(lines)
    headings = [i for i, line in enumerate(lines) if i not in fenced and GROUP_RE.match(line)]
    start = None
    for i in headings[1:]:  # [0] is the document title
        if i < before:
            start = i
    if start is None:
        return None
    return start, boundary_after(lines, start, fenced)


def current_group(lines: list[str]) -> str | None:
    """Text of the last `# ` heading in a file, or None."""
    fenced, _ = fenced_lines(lines)
    found = [line for i, line in enumerate(lines) if i not in fenced and GROUP_RE.match(line)]
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

    # Refuse before touching anything. A fence problem means a span does not end where it looks like
    # it ends, and the move's own verification cannot catch that: fingerprints cover the reporter's
    # prose, so a truncated span hashes identically on both sides of a cut that lost half a report.
    complaints = [f"{doc.name}:{c}" for doc in (INBOX, HISTORY)
                  for c in fence_findings(doc.read_text().splitlines())]
    if complaints:
        raise SystemExit(
            "refusing to archive — a fenced block breaks the section boundaries:\n  "
            + "\n  ".join(complaints)
            + "\n\nA span that ends at the wrong line moves the wrong bytes, and the fingerprint\n"
              "check cannot see it. Repair the document first; do NOT edit the reporter's prose to\n"
              "suit the tool — a missing fence usually means an earlier pass already split a report."
        )

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

