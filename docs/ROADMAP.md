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

## RM16 — MiniMed 720G pump exports parse

**State:** building, by the lead: the parser is on main, ships in 0.13.0 with RM17 and RM21 · **Owner:** lead · **Severity:** high

The proposal is [PLAN_pump.md](PLAN_pump.md) §2 and §3.1; this item is the parser half. All four real
720G CareLink exports in the tree fail to parse today (`MalformedDataError` on dtype inference), and
the shipped Medtronic path misreads two columns on any CareLink file: `Basal Rate (U/h)` becomes an
`insulin_slow` dose, and fingersticks become `EGV_READ` glucose (255 of the Guardian fixture's
22,404 "readings" are fingersticks).

Shape: a new variant format `MEDTRONIC_PUMP` (the `DEXCOM_EU` / `LIBRE_EU` pattern) targeting
`CGM_SCHEMA_EXTENDED`, because the target schema is per format. Section-aware read, every column
`Utf8`. Boluses paired by `Bolus Number`, placed at the programmed time with the delivered amount, a
square part carrying a new extended `insulin_duration`. Fingersticks become a new `BG_METER` event
type carrying the value (the person's answer to Q2), on the Guardian path too. The effective basal rate (percent temps
applied, every suspend 0 until `NORMAL_PUMPING`, ties resolved by `Index`) as `BASAL_RT` rows in a
new extended `basal_rate` (U/h). Priming and rows after a clock change left out, each counted in one
warning. A synthetic 720G fixture from a new scrub script, expected values written by
hand.

Two pump behaviours are assumptions, stated in the plan's §3.4 (a percent temp delivers schedule ×
percent; a cancelled square bolus reports its actual duration), the second watched by a warning.

**Legality:** additive (a new format, two extended columns, two event types), so **minor**.
`insulin_slow` losing the Medtronic basal rate, and fingersticks leaving `glucose`, are repairs with
their own changelog lines.

---


## RM17 — fold pump insulin onto the glucose grid

**State:** building, by the lead: the fold is on main, ships in 0.13.0 with RM16 and RM21 · **Owner:** lead · **Severity:** high

The proposal is [PLAN_pump.md](PLAN_pump.md) §3.2. `fold_insulin_onto_grid(cgm, pump)` takes the two
frames apart, never bundles or re-synchronizes them (bundling first lets pump rows move the CGM's
grid, F19), and returns one row per grid point of each CGM sequence: glucose as the CGM had it,
`insulin_fast` = instant doses in the bin + spread doses' share + the basal rate integrated over the
bin, which is `rate × I / 60` for a constant rate (`I` = `expected_interval_minutes`, no literal 12).
Basal is **not** folded into `insulin_fast`: every model trained in `glucose_forecasting_metabonet`
reads basal (U/h) and bolus (U) as separate channels, and the person ruled that real training inputs
take precedence. It travels as the bin's mean `basal_rate`; the fold is an opt-in `fold_basal=True`.
Insulin and carbs come from a `source` option, `PUMP` / `CGM` / `MERGE`. Merge pairs entries at most
`merge_pair_slots` apart (default 1, the person's choice; beyond it both are kept), one-to-one, onto
the pump's slot; it takes a 0 as "did not say", averages only two non-zeros, and warns with both
values beyond a tolerance.
Null outside the pump's coverage, 0 where the pump was suspended; insulin outside every sequence
reported with its total, never attached; overlapping monthly exports deduplicated. A terminal view:
inputs untouched, the output refuses to be folded again.

Checked on real data over the five days where a Dexcom export and the
pump record overlap; no more exports are coming.

**Legality:** additive (a new classmethod, a new enum), so **minor**, riding RM16's release.

---

## RM18 — the collapse keeps the first dose on a grid point and drops the rest

**State:** queued · **Owner:** lead · **Severity:** medium

`_coalesce_duplicate_timestamps`, behind `to_data_only_df(drop_duplicates=True)`, keeps the first
non-null value per column per grid point, so a second bolus or carb entry on the same point is lost,
for every vendor. Amounts (`carbs`, `insulin_slow`, `insulin_fast`) are summed when at least one is
non-null, else null (Polars sums an all-null group to `0.0`, and a null amount means "not recorded").
[PLAN_pump.md](PLAN_pump.md) §2.6 and §3.3.

**Legality:** a repair of a silent loss, so a **patch**, with its own changelog line.

---

## RM19 — let a caller state the exporter CSV's date order

**State:** queued · **Owner:** triage · **Severity:** medium

Motivated by **S6** (`docs/FEEDBACK_HISTORY.md`), relayed from `sugar-sugar`. Queued by the
person on 2026-10-21 for unattended work ("Add the fix to unattended to-dos"), answering the
question whether a warning is enough or callers should be able to say the date order outright.

0.12.3 reads a nightscout-exporter CSV in whichever locale reads every row. One case is left that
the text cannot settle: a file on a 12-hour clock in which no day is past the 12th reads completely
both month-first (en-US) and day-first (en-AU, en-NZ, en-IN). It is read month first, as before,
with one warning. A day-first export of that shape therefore still comes back with day and month
swapped on every row, and only the warning says it might have.

Shape: an optional `date_order` parameter, a new `EnumLiteral` (`MONTH_FIRST`, `DAY_FIRST`) whose
default `None` keeps 0.12.3's inference. Given, it restricts `_exporter_datetime_format` to the
formats of that order, so an ambiguous file is read the way the caller says, with no warning, and
rows that do not read that way are counted as now. It belongs on `parse_nightscout` and
`from_nightscout_exports`. Open for the build: how `parse_to_unified` / `parse_file`, which take no
vendor options, reach it (a dedicated exporter entry point rather than class state, since every
method is a stateless classmethod), and whether `cgm-cli parse` grows a `--date-order` flag.

Making the order **required** was the other answer, and it is rejected: it breaks every existing
caller, which is a major, and most files (any US export, and any day-first one with a day past the
12th) need no help.

**Legality:** a new optional parameter and a new enum beside what exists, so **minor**.

---

## RM20 — `parse_nightscout` and `_process_nightscout` raise differently on the same failure

**State:** queued, for the next major only · **Owner:** lead · **Severity:** low

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

## RM21 — `to_sugarone_ready_df`: pump basal in the models' input shape

**State:** building, by the lead: on main, ships in 0.13.0 with RM16 and RM17 · **Owner:** lead · **Severity:** high

`to_ml_ready_df` leaves `Basal Rate (U/h)` empty unless `insulin_slow` is copied into it, and reads a
unified frame the fold's output is not. The person: "keep this output as legacy, build a
to_sugarone_ready_df output node meeting the format the trained models expect". A new classmethod
renders RM17's output as the models in `glucose_forecasting_metabonet` read it (`sequence_id`, `User
ID`, `Timestamp`, `Event Type`, `Glucose (mg/dL)`, `Basal Rate (U/h)` from `basal_rate`, `Bolus
Insulin (U)` from `insulin_fast`, `Carbohydrates (g)`), the exact columns and order taken from that
repo's `src/common/data/columns.py` and loader at build time; nulls stay null. `to_ml_ready_df` is
unchanged. [PLAN_pump.md](PLAN_pump.md) §3.5.

**Legality:** additive (a new classmethod), so **minor**, riding RM16's release.

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
