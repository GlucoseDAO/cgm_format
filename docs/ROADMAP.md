# Roadmap

Active-only, forward-only. One `## RMn — name` section per **open** item. Its first line is

```
**State:** queued | building | waits until YYYY-MM-DD | after <ID> | blocked: desk #N · **Owner:** lead | triage · **Severity:** …
```

and the lead keeps it true (`docs/seats/LEAD.md`); the lead's plan watcher raises an item with no
state, a date come, or a block with no desk question. Shipped items move to
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md) with their rationale — nothing is deleted, only relocated.

Severity orders the queue; it does not size the release. Legality does that (`CLAUDE.md` §8).

---

## RM1 — schema-regen helper prints instead of logging

**State:** queued · **Owner:** lead · **Severity:** low

`interface/schema.py:519` does `print(f"✓ Regenerated {schema_file}")`. The house rule is
standard-library `logging` for library diagnostics; `print` belongs only to CLI output the user asked
to see. The helper is called from `scripts/regenerate_all_schemas.py`, so the honest fix is either a
`logger.info` here, or moving the message to the script that owns the user-facing output.

Legality: pure legibility, so a patch.

---

## RM2 — hardcoded version fallback in `__init__.py`

**State:** queued · **Owner:** lead · **Severity:** low

`src/cgm_format/__init__.py:33` carries `__version__ = "0.10.0"` as a fallback when
`importlib.metadata.version("cgm-format")` raises. That is a second source of truth for the version,
and it is the one that gets read when the package is not installed — exactly when it is most likely
to be stale. It already has been: the literal still said `0.9.0` at the moment the 0.10.0 bump
landed, which is finding F3's second half.

The fix is an editable install (`uv sync`) so metadata is always present, then dropping the literal.
Worth confirming first that nothing in CI or the docs build imports the package without installing
it.

Legality: patch.

---

## RM3 — `data/` layout does not match the house convention

**State:** queued, as its own task (approved in principle 2026-08-13, never inside another change) · **Owner:** lead · **Severity:** low

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

**State:** queued · **Owner:** lead · **Severity:** low

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

**State:** queued · **Owner:** lead · **Severity:** medium

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

---

## RM14 — no way to ask whether `track="mean"` is defensible for a subject

**State:** queued · **Owner:** lead · **Severity:** medium

Motivated by **S1** (`docs/FEEDBACK_HISTORY.md`), filed on behalf of `sugar-sugar` against 0.10.0.

`parse_tracks(path, track="mean")` averages a subject's two sensor series and flags every synthesized
row `Quality.TRACK_MERGE`. That flag, and the docstrings around it, describe the mean's
**composition** — which rows came from two sensors and which from one — and for that they are
correct. Neither describes its **validity**: whether the two series are estimates of the same
quantity at all, which is a per-subject property and the thing that decides whether averaging them is
honest. The call returns the same clean frame either way, and nothing in the API lets a caller ask
first.

Reproduced on 2026-08-21 against 0.12.0 through `parse_tracks` itself (not the raw CSV), fitting
`libre = slope · dexcom + intercept` per subject over the inner join of the two frames on `datetime`,
45 subjects, 629,605 overlapping samples:

| statistic | median | range |
|---|---|---|
| slope | 0.70 | 0.10 – 1.16 |
| intercept | 5.4 mg/dL | −16.6 – +110.8 |
| Pearson r | 0.82 | 0.10 – 0.97 |

31 of 45 subjects sit under slope 0.8 and 15 under r 0.7, so the compression is subject-specific and
a third of the corpus differs in shape rather than only in level. Against a slope 0.9–1.1 and r > 0.9
bar exactly two subjects qualify (`CGMacros-033`, `CGMacros-035`). On the worst of them
(`CGMacros-013`: slope 0.10, r 0.10) `track="mean"` emits **no warning** and returns 15,918 rows with
14,275 flagged `TRACK_MERGE` — composition reported, validity not.

The shape the reporter proposes, and the one to start from:

- `TrackAgreement(reference, comparand, overlap, slope, intercept, correlation, bias,
  mean_abs_difference)` as a frozen dataclass — a measurement that computes and does not decide.
- `FormatParser.track_agreement(file_path, reference=None, comparand=None)`, mirroring
  `list_subjects`; `MultiTrackSourceError` for a single-track format, `ZeroValidInputError` when the
  tracks never read at the same time.
- An `is_interchangeable` property over two module constants (`AGREEMENT_SLOPE_TOLERANCE`,
  `AGREEMENT_MIN_CORRELATION`) — constants rather than registry entries, because the registries hold
  data only and a threshold is policy.
- On `track="mean"`, a warning through the existing channel when the tracks do not agree, naming
  slope and r and pointing at `track_agreement()` — never refusing the call.

Rejected repairs, all on charter grounds rather than cost: auto-selecting a "best" track (domain
policy, and it depends on what the caller does with the frame); gating `mean` behind a flag or
exception (breaks callers a compressed trace is legitimately fine for); and a
`Quality.TRACK_DISAGREEMENT` row bit, since disagreement is a per-subject property and every row
would carry the same value — misusing the mechanism `TRACK_MERGE` established.

Open questions for whoever builds it: whether the two threshold constants belong in the library at
all rather than in the caller's policy, and what requesting `mean` should cost when it has to parse
both tracks to fit them (`check_agreement=False`, or a per-path cache).

**Legality:** additive — a new dataclass, a new classmethod, two new module constants, a new warning
on a path that already behaved this way. Nothing removed, renamed or retyped; `CGM_SCHEMA` and
`CGM_SCHEMA_EXTENDED` untouched; no existing flag changes meaning; emitted frames stay byte-identical.
So **minor**. The warning alone would be a patch, but it rides the additive API.

---

## RM15 — network failures escape as third-party exceptions

**State:** queued · **Owner:** lead · **Severity:** medium

Motivated by **S5** (`docs/FEEDBACK.md`), filed on behalf of `sugar-sugar` against 0.12.0.

`download_nightscout` and `from_nightscout_url` let every transport failure escape raw: connect
errors, timeouts, `httpx.HTTPStatusError` and `json.JSONDecodeError`. So a caller must import `httpx`
itself to catch them — which means taking a dependency the library deliberately keeps behind the
`cli` extra, in order to handle errors from a library that hides it. `sugar-sugar` does exactly this
and reports having to match on `type(exc).__name__` substrings to classify a failure, which is what
an absent contract looks like from the outside.

Everything else in the interface raises a typed error from `interface/cgm_interface.py`
(`MalformedDataError`, `UnknownFormatError`, `ZeroValidInputError`, `MultiTrackSourceError`). The
network path is the one surface that does not, and it is the surface most likely to fail in normal
use.

**Surfaced rather than repaired**, because the shape is a design decision:

- A new `NightscoutNetworkError` hierarchy is additive and would be the obvious answer, but it has to
  decide whether it wraps *transport* failures only or also HTTP status failures, and whether the
  original exception is chained or swallowed. Chaining keeps `httpx` in the traceback, which is most
  of what the wrapper was meant to spare the caller.
- Reusing `MalformedDataError` is wrong: a 502 is not malformed data, and collapsing the two loses
  the distinction a caller needs to decide whether retrying makes sense.
- A generic `CGMDownloadError` covering every future downloader is the most useful shape and the one
  hardest to get right before a second downloader exists. Nightscout is the only one today.

The token-redaction half of this shipped already (`_raise_for_status_without_token`), because a
credential in a log line is a defect rather than a design question. That fix deliberately preserves
`httpx.HTTPStatusError` as the type, so whatever this item decides is not constrained by it.

**Legality:** additive — new exception types beside the existing ones, nothing removed or retyped. So
**minor**. Changing what an existing call already raises would be major, and that is the version to
avoid.

---
