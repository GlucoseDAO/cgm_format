# Roadmap

Active-only, forward-only. One `## RMn — name` section per **open** item, each with a
severity/status/owner line. Shipped items move to [ROADMAP_HISTORY.md](ROADMAP_HISTORY.md) with their
rationale — nothing is deleted, only relocated.

Severity orders the queue; it does not size the release. Legality does that (`CLAUDE.md` §8).

---

## RM1 — schema-regen helper prints instead of logging

**Severity:** low · **Status:** open · **Owner:** unassigned

`interface/schema.py:477` does `print(f"✓ Regenerated {schema_file}")`. The house rule is
standard-library `logging` for library diagnostics; `print` belongs only to CLI output the user asked
to see. The helper is called from `scripts/regenerate_all_schemas.py`, so the honest fix is either a
`logger.info` here, or moving the message to the script that owns the user-facing output.

Legality: pure legibility, so a patch.

---

## RM2 — hardcoded version fallback in `__init__.py`

**Severity:** low · **Status:** open · **Owner:** unassigned

`src/cgm_format/__init__.py:27` carries `__version__ = "0.9.0"` as a fallback when
`importlib.metadata.version("cgm-format")` raises. That is a second source of truth for the version,
and it is the one that gets read when the package is not installed — exactly when it is most likely
to be stale.

The fix is an editable install (`uv sync`) so metadata is always present, then dropping the literal.
Worth confirming first that nothing in CI or the docs build imports the package without installing
it.

Legality: patch.

---

## RM3 — `data/` layout does not match the house convention

**Severity:** low · **Status:** open, deliberately deferred · **Owner:** unassigned

The house layout puts committed fixtures in `assets/` and reserves `data/` for git-ignored
`input/` · `interim/` · `output/`. This repo commits `data/input/` and writes generated output to
`data/parsed/`, `data/cli_test_output/` and `data/cli_examples_output/`.

Deferred on purpose, and out of scope for any task that is not this task: every integration test
path, the `.gitignore` ignore-all (F2: `input/` is not allowlisted), the sdist exclusion and `tests/conftest.py`'s
Nightscout download target are built on the current names and all work today. Doing it properly
means one change that moves the fixtures, updates every test path, rewrites the allowlist, and fixes
the docs that name these directories — planned as a unit, not smuggled into an unrelated edit.

Legality: no effect on anything a consumer imports, so a patch — but it touches `data/input/` paths
that examples and docs quote, so the docs move with it.

---

## RM4 — the CLI does not load `.env`

**Severity:** low · **Status:** open · **Owner:** unassigned

`tests/conftest.py` loads `.env` via `python-dotenv`, so `NIGHTSCOUT_URL` / `NIGHTSCOUT_TOKEN` /
`NIGHTSCOUT_API_SECRET` reach the test suite. `cgm_cli.py` does not, so the same credentials that
make tests pass do nothing for `cgm-cli`. The house rule is that every CLI loads `.env` at startup
before reading configuration, with the values documented in a `.env.template`.

Two things to settle before writing code: `python-dotenv` currently sits in `dev`, not `cli`, so
adopting this moves it (or adds it) to the `cli` extra; and the credential-neutralizing rule from
`CLAUDE.md` §6 applies — a test asserting "no credential" must `setenv(VAR, "")`, never `delenv`,
because `load_dotenv(override=False)` skips a key that is merely present.

Legality: additive (a new optional config path), so minor.

---

## RM6 — event type and parser mapping for ketone readings

**Severity:** medium · **Status:** open, **narrowed by 0.10.0** · **Owner:** unassigned

**0.10.0 settled two of the four decisions and shipped the column. What remains is the event type
and the parser mapping, and it is blocked on a real fixture.**

Settled and shipped:

- **Shape (part).** `ketones` is a data column on `CGM_SCHEMA_EXTENDED` (`formats/unified.py`), null
  meaning the vendor did not say. `CGM_SCHEMA` is untouched, so nothing a core consumer reads changed.
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

**Severity:** medium · **Status:** open · **Owner:** unassigned

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

**Severity:** low · **Status:** open, blocked on RM7 · **Owner:** unassigned

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

---
