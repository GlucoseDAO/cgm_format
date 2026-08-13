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

### F1 — Libre scan readings collide with historic timestamps

Libre historic (automatic interval) and scan (user-initiated) readings both parse as `EGV_READ`.
On the JonGrove mmol/L export, 564 of 2,807 scan timestamps (about a fifth) fall on a historic
minute before synchronization; more collide after `synchronize_timestamps` snaps off-grid scans
onto the 5-minute grid. `mark_time_duplicates` flags the later row `TIME_DUPLICATE` and keeps the
first. Whether a duplicate pair should be reconciled into one reading rather than flagged is a
policy decision, not a parser bug — left open.

### F2 — `data/.gitignore` no longer allowlists `input/`

Commit `44e3bb9` dropped the `!` so `data/input/` is ignored outright. Only fixtures already in git
history stay tracked; a new vendor file cannot be committed without `git add -f`. The mmol/L
Libre fixture therefore stays local, and tests skip when it is absent. CLAUDE.md §3 records the
current ignore rule.
