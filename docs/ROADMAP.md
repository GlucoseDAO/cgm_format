# Roadmap

Active-only, forward-only. One `## RMn — name` section per **open** item. Its first line is

```
**State:** unattended | urgent | building | waits until YYYY-MM-DD | after <ID> | blocked: desk #N · **Owner:** lead | triage · **Severity:** …
```

and the lead keeps it true (`docs/seats/LEAD.md`); the lead's plan watcher raises an item with no
state, a date come, or a block with no desk question. Shipped items move to
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md) with their rationale — nothing is deleted, only relocated.

Severity orders the queue; it does not size the release. Legality does that (`CLAUDE.md` §8).

---

## RM20 — `parse_nightscout` and `_process_nightscout` raise differently on the same failure

**State:** waits until 2027-01-21: breaking, so it ships with the next major only; looked at again then · **Owner:** lead · **Severity:** low

Motivated by **S6** (and S5 before it, whose reply did not address this bullet). The two public
routes into the Nightscout parsing code have different error contracts. `_process_nightscout`,
reached through `detect_format` / `parse_to_unified`, wraps its whole body in `except Exception`
and re-raises as `MalformedDataError`. `parse_nightscout`, reached through `from_nightscout_url` and
`from_nightscout_exports`, has no wrapper, so the same broken input raises a raw polars error
there. That is why S2's defect surfaced as `ComputeError`. The reporter calls it cosmetic now.

**The charter decides the direction, and it is the opposite of the reporter's.** `PHILOSOPHY.md`,
*Error Handling*: "Let exceptions propagate … Recovery logic is reserved for cases where there is a
real alternative path." `_process_nightscout`'s blanket `except Exception` is a wrapper with no
recovery path, which is what the charter rules out; `CLAUDE.md` §2 forbids the same shape. So:

- Adding the same wrapper to `parse_nightscout` is wrong: it doubles the pattern the charter forbids,
  and it changes what an existing call raises.
- Removing the wrapper from `_process_nightscout` is the charter-consistent repair, but it too
  changes what an existing call raises (`MalformedDataError` becomes the underlying error), and a
  caller catching `MalformedDataError` around `parse_to_unified` on a Nightscout file would stop
  catching it.
- Leaving both as they are costs only the asymmetry, which no consumer is blocked on.

**Legality:** either repair changes the exception type an existing call raises, so **major**.
Nothing to build until a major is scheduled; it waits for one.

---

## RM6 — event type and parser mapping for ketone readings

**State:** waits until 2027-01-21: a real Libre export with populated ketone cells (none in the tree; narrowed by 0.10.0) · **Owner:** lead · **Severity:** medium

**0.10.0 settled two of the four decisions and shipped the column. What remains is the event type
and the parser mapping, and it is blocked on a real fixture.**

Settled and shipped:

- **Shape (part).** `ketones` is a data column on `CGM_SCHEMA_EXTENDED` (`formats/unified.py`), null
  meaning the vendor did not say. `CGM_SCHEMA` gained no column, so nothing a core consumer reads
  changed.
- **Canonical unit.** mmol/L, declared on the column and **not** routed through
  `_glucose_to_canonical` (decision D8). Clinical ketones are already mmol/L; borrowing the glucose
  convention would apply one analyte's rule to another.

Still open, and deliberately not guessed at:

1. **What `event_type` a ketone row carries.** A column with no event type leaves a ketone row
   looking like an empty glucose event — the failure this item was opened for. A new 8-char
   `UnifiedEventType` member is the obvious candidate, but it is a vocabulary a consumer reads.
2. **Historic vs scan vs the older `Ketone mmol/L` column.** Merge into one series (as scan glucose
   merged into `EGV_READ`) or keep them distinct. Merging re-opens F1 on a new analyte. 0.10.0's
   intent was one series with the raw source column recorded in `annotations`, but that is not
   implemented and should not be treated as decided.
3. **`_process_libre` never selects the columns.** They stay on the vendor schema and are dropped at
   the unified boundary.

**Blocked on a fixture, not on a decision.** The JonGrove mmol/L export (local, gitignored) carries
all three ketone headers and **zero populated cells**; every committed Libre fixture is the same.
There is no real ketone reading anywhere in the tree, and `CLAUDE.md` §2 forbids implementing
against an invented row. Skip-if-absent until a real export arrives.

**Legality:** minor — a new `UnifiedEventType` member and a parser mapping onto a column that already
exists.

---

## RM7 — lazy `scan_csv` ingest path

**State:** blocked: desk #362 (whether the lazy reader keeps the file's row order; #107 chose a `LazyFrame`, and a measurement since shows the schema's sort defeats it) · **Owner:** lead · **Severity:** medium

**Measured 2026-11-15** (Polars 1.43.2; a synthetic unsorted unified-layout CSV, 1.95 GB, 24 M rows): `scan_csv` → `sink_parquet` without a sort finishes under a 1 GB `MemoryMax` cgroup cap; with the total-order `sort` it is OOM-killed (137). Uncapped peaks: sort 3.6 GB, eager `read_csv` + sort 4.7 GB, so the sorted lazy path saves about a quarter. Peak RSS alone misleads here: `scan_csv` memory-maps the file, so a cap is the test. Polars 2.1.0 behaved the same at 484 MB. RM11 (Loop) needs a per-`PtID` split on top of whichever reader this becomes.

Every public entry point funnels through `parse_file → bytes → decode_raw_data → str →
detect_format`, which materializes the entire file as a Python string before parsing starts. Ingest
is therefore capped at whatever fits in memory, and the cap is silent — there is no size at which the
library says "use the streaming path", because there isn't one.

This is worth doing on its own merits (multi-year Dexcom exports are already the slowest thing in the
test suite), and it is a hard blocker for the Loop corpus, where a single `LOOPDeviceCGM*.txt` is
about 2.3 GB. See [RESEARCH_CORPORA.md](RESEARCH_CORPORA.md).

Shape: a `pl.scan_csv`-based entry point sitting **beside** the eager ones, not replacing them, with
`sink_parquet` for output. The eager path stays the default and stays simplest — most vendor exports
are a few megabytes and lazy evaluation buys them nothing.

**Decisions to settle before writing code:**

1. **Detection on a prefix.** `detect_format` currently takes the whole decoded text. A lazy path has
   only the first N lines, which is all detection actually reads (`DETECTION_LINE_COUNT`) — so this
   is a refactor of the *signature*, not the logic. Confirm that before assuming it's free.
2. **Where `decode_raw_data` fits.** BOM and encoding-artifact normalization only runs on `bytes`
   (`CLAUDE.md` §12). A streaming reader hands Polars a path, not bytes, so either the artifacts are
   handled by a prefix sniff or the lazy path documents that it does not fix them. Do not let it
   silently skip normalization.
3. **What the lazy path returns.** A `LazyFrame` gives the caller the streaming win but breaks the
   `UnifiedFormat = pl.DataFrame` alias every signature uses. Collecting internally is honest but
   forfeits most of the benefit for large inputs.

**Legality:** a new entry point beside the existing ones, nothing removed or retyped, so **minor**.

---

## RM11 — Loop dataset support

**State:** after RM7 · **Owner:** lead · **Severity:** low

**RM9 no longer blocks this** — 0.10.0 shipped `parse_corpus`, so the many-subjects-out half is
solved. Loop is subject-as-key-column rather than subject-per-directory, so it needs an *iterator*
form of the corpus entry point (the eager `dict` fits CGMacros' 45 and D1NAMO's 29; it does not fit
~1,000 participants across 111 million rows). That is a new sibling entry point, not a retype —
adding it later stays cheap, which is what made deferring Loop the right call.

The Loop observational study (NCT03838900, Jaeb Center for Health Research), dataset 560. Roughly a
thousand participants of automated insulin delivery data — the largest real-world AID corpus
available to us.

Lower priority than RM10 for practical reasons, not scientific ones: it is DUA-gated, needs ~20 GB on
disk before a single test runs, and is hard-blocked on RM7. Appendix B of
[RESEARCH_CORPORA.md](RESEARCH_CORPORA.md) has what we know.

**Read `DataGlossary.rtf` from inside the archive before writing any code.** The column lists we
currently hold were reconstructed from third-party parsers' `usecols` arguments, not from the
authoritative file. Do not implement against them.

Constraints already settled:

- **No zip handling in the library.** The archive is deflate64, which the standard library cannot
  open, and the workaround is a third-party package. That must not touch the Polars-only core —
  extraction is a documented user-side step, or a `scripts/` helper at most.
- **Keep `UTCDtTm` verbatim as `datetime`.** `TmZnOffset` is reportedly null for ~63% of patients, so
  reconstructing local time from `PtRoster.PtTimezoneOffset` is a substitution. Opt-in with a
  warning, never silent (`CLAUDE.md` §2).
- **Glucose is mmol/L.** Route it through the column's declared `unit` and `_glucose_to_canonical`,
  never a hardcoded 18.018 — that is what the units mechanism exists for.
- **Dedupe is a parser step, not a caller's problem.** A reported ~35% of CGM rows are exact
  duplicates. First-occurrence with deterministic sort keys, with its own test.
- **Carbs span two mutually exclusive eras** (`LOOPDeviceWizard.txt` to mid-2018,
  `LOOPDeviceFood.txt` after), overlapping in patients but not in time.

**Legality:** a new format, nothing removed, so **minor**.
