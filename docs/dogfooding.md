# Dogfooding — open findings

Layer 1 is the test suite: it proves the code does what it was told. This is layer 2, and it asks a
different question — *is this usable, and what is missing?* Findings from using the shipped surface
(`FormatParser`, `FormatProcessor`, `cgm-cli`, the public API) for real work land here.

Read this before touching the parser, processor or CLI surface. The rules for producing a finding
are in `CLAUDE.md` §7; the two that get skipped most often:

- **A capability the library lacks is the result, not an obstacle to route around.** The moment you
  reach for a hand-rolled `pl.read_csv` + rename to get past something `FormatParser` cannot do, the
  exercise stops producing signal.
- **Dogfood a finding before you report it.** Build a real example against the actual code path and
  show it fails. A loss that is mechanically possible but has no realistic CGM instantiation is
  noise.

## How findings move

`F#` ids are stable and findings **move** between files, never duplicated:

| File | Holds |
|---|---|
| this file | open quirks, bugs and UX gaps |
| [previous_issues.md](previous_issues.md) | resolved **here**, each with its resolution and a code pointer |

There is no `<upstream>-pending-fixes.md` — this library sits at the root of our dependency graph and
consumes nothing of ours. A finding against Polars or Frictionless is recorded here with the upstream
issue link and whatever defensive mitigation is already in place.

Ids are computed from this file **and** `previous_issues.md`; once a finding moves out, the highest
id visible here is not the highest ever used. Never reuse one.

---

## Open findings

*None recorded. This log was established on 2026-08-13 — an empty file means no dogfooding round has
been run against the current surface, not that the surface is clean.*
