# Agent Guidelines — cgm-format

`cgm-format` converts vendor-specific Continuous Glucose Monitoring exports (Dexcom, Libre,
Medtronic, Nightscout) into one standardized, well-typed Polars DataFrame for ML training and
inference. Published library, single package, `src/` layout (`src/cgm_format/`, hatchling build).
**Polars is the only core dependency**; everything else (`typer`, `rich`, `httpx`, `pandas`,
`pyarrow`, `frictionless`) lives behind the `cli` / `dev` extras. GitHub: `GlucoseDAO/cgm_format`.

It sits at the **root of the dependency graph** — `sugar-sugar` and `glucosedao` consume it, it
consumes nothing of ours. That is why the dependency tier is load-bearing and why this repo runs a
consumer inbox (§8).

`AGENTS.md` is a symlink to this file. If the two ever differ, that is a bug —
`ln -sf CLAUDE.md AGENTS.md`.

**This library is in active development.** No frozen constitution, no full backward-compat
obligation. Prefer additive changes, but a breaking change to the schema, CLI, or public API is fine
when it's the right call — bump the version in `pyproject.toml` and note it in `docs/CHANGELOG.md`.
What you must *not* break casually are the runtime invariants (idempotency, losslessness,
deterministic ordering): those are correctness, not compatibility, and they're covered by tests.

---

## Read these first, in this order

Obligatory. Read them yourself. **Do not delegate a document you are about to judge a design
against** — a subagent returns a summary, and a summary of a rule drops the qualifier the decision
turned on (the difference between "additive" and "non-breaking", between `None` and `False`).
Delegation is for finding things, never for deciding them.

1. **[docs/PHILOSOPHY.md](docs/PHILOSOPHY.md)** — the charter. What this project is, what it will
   never do, the invariants every release upholds. When a plan conflicts with it, the charter wins.
   It is self-contained and names no other document; navigation lives here.
2. **[docs/ROADMAP.md](docs/ROADMAP.md)** — active-only, forward-only. One `## RMn — name` per open
   item with a severity/status/owner line. Shipped items live in `docs/ROADMAP_HISTORY.md`.
3. **[docs/CHANGELOG.md](docs/CHANGELOG.md)** — what actually shipped, newest first.
4. **[docs/dogfooding.md](docs/dogfooding.md)** — open findings from using the shipped surface for
   real work. Read it before touching the parser, processor, or CLI surface.
5. **[docs/FEEDBACK.md](docs/FEEDBACK.md)** — the consumer inbox. Empty means nothing is owed.
   `docs/FEEDBACK_HISTORY.md` holds the answered items, and
   **[docs/CONSUMER_TRIAGE_LOOP.md](docs/CONSUMER_TRIAGE_LOOP.md)** is the runbook for answering them
   — read it before triaging anything, not just §8 below.
6. **Per-area reference — read the one your task touches:**
   [docs/PIPELINE.md](docs/PIPELINE.md) (stage detail),
   [docs/UNIFIED_FORMAT.md](docs/UNIFIED_FORMAT.md) (schema),
   [docs/USAGE.md](docs/USAGE.md) (public API),
   [docs/NEW_SCHEMA.md](docs/NEW_SCHEMA.md) (new-vendor checklist),
   [docs/RESEARCH_CORPORA.md](docs/RESEARCH_CORPORA.md) (multi-subject / multi-device datasets).

Everything below is self-contained: no rule here requires following a link to know what you must not
do. Links carry positive detail only.

`.claude/` holds only `settings.local.json` — no skills, agents or workflows. If you add any, name
them here, or they get discovered by accident.

## Project map

The two public entry classes:
- `FormatParser` (`format_parser.py`) — Stages 1–3: decode raw bytes, detect vendor format, parse to
  a unified Polars DataFrame.
- `FormatProcessor` (`format_processor.py`) — Stages 4–6: sequence detection, gap interpolation,
  timestamp synchronization, inference prep, data-only export. All methods are `@classmethod` (no
  mutable instance state).

Supporting modules:
- `formats/unified.py` — `UnifiedEventType`, `Quality` flags, `CGM_SCHEMA` (the canonical
  `CGMSchemaDefinition`).
- `formats/{dexcom,dexcom_eu,libre,libre_eu,medtronic,nightscout}.py` — vendor column enums, detection
  patterns, schemas.
- `formats/supported.py` — eight registries: `SCHEMA_MAP`, `FORMAT_DETECTION_PATTERNS`,
  `FORMAT_DETECTION_LINE_COUNT`, `UNIFIED_TARGET_SCHEMA`, `FORMAT_CATEGORY`,
  `KNOWN_ISSUES_TO_SUPPRESS` (all six exhaustive over `SupportedCGMFormat`), plus
  `PATH_DETECTION_PROBES` and `SUBJECT_PATH_PROBES` for directory-shaped sources only. The last two
  must stay disjoint — a corpus root that matches a subject probe parses the whole corpus as one
  person.
- `interface/cgm_interface.py` — abstract `CGMParser` / `CGMProcessor`, all exception types,
  `ProcessingWarning`, constants, the `SupportedCGMFormat` enum.
- `interface/schema.py` — `CGMSchemaDefinition`, `ColumnSchema`, `EnumLiteral`, Frictionless export
  helpers.
- `cgm_cli.py` — Typer CLI (`cgm-cli`, owned by `[project.scripts]`).
- `nightscout_downloader.py` — `download_nightscout()` over `httpx` (JSON-only; `token` + `api_secret`
  auth).

`scripts/` holds one-off utilities (schema regen, scrub scripts) — not part of the installed package.
`examples/` shows usage patterns — keep them runnable; they're documentation.

### The 6-stage pipeline

| Stage | Class · method | Description |
|-------|----------------|-------------|
| 1 | `FormatParser.decode_raw_data` | Strip BOM, fix encoding artifacts |
| 2 | `FormatParser.detect_format` | Pattern-match header to `SupportedCGMFormat` |
| 3 | `FormatParser.parse_*` | Vendor CSV → unified Polars DataFrame |
| 4 | `FormatProcessor.detect_and_assign_sequences` | Split on large gaps, assign `sequence_id` |
| 5 | `FormatProcessor.interpolate_gaps` / `synchronize_timestamps` | Fill small gaps, snap to 5-min grid |
| 6 | `FormatProcessor.prepare_for_inference` / `to_data_only_df` | Quality checks, drop service columns |

Hard boundary: the **parser** knows vendors (BOM, metadata rows, record types, timestamp formats) and
emits `UnifiedFormat`; the **processor** operates only on `UnifiedFormat` and is vendor-agnostic. A
new sensor requires zero processor changes.

### Unified format schema

Canonical output is a Polars DataFrame conforming to `CGM_SCHEMA` (`formats/unified.py`):

- **Service columns** (metadata): `sequence_id`, `original_datetime`, `quality`, `event_type`
- **Data columns** (signal): `datetime`, `glucose`, `carbs`, `insulin_slow`, `insulin_fast`, `exercise`

`get_polars_schema(data_only=True)` / `to_data_only_df()` strip service columns for ML consumption.
`quality` is a bitwise `Quality` flag (`OUT_OF_RANGE`, `SENSOR_CALIBRATION`, `IMPUTATION`,
`TIME_DUPLICATE`, `SYNCHRONIZATION`); `0` = good. `event_type` holds 8-char `UnifiedEventType` codes
(`"EGV_READ"`, `"CALIBRAT"`, `"CARBS_IN"`, …).

**Validation vs enforcement.** Validation (`enforce=False`) raises typed errors
(`MissingColumnError`, `ExtraColumnError`, `ColumnOrderError`, `ColumnTypeError`). Enforcement
(`enforce=True`) fixes the frame — adds null columns, casts, reorders, drops extras, stable-sorts.
Internal stages enforce; external-facing APIs default to validate. `ValidationMethod` has four modes
(`INPUT`, `OUTPUT`, `INPUT_FORCED`, `OUTPUT_FORCED`); forced variants raise, non-forced only warn.

---

## 1. Adopting these guidelines: ask, never infer

**When two rules conflict — this file against a sibling repo's, this file against the user's global
preferences, a rule against what the code actually does — stop and run a questionnaire. Do not pick
the one that looks better, do not synthesize a compromise, and do not silently follow the more
specific file.** A contradiction between two live rules is almost always a real difference in the
repos' natures (this library's minimal-core tier against an application's freedom to depend on
anything), and inferring which nature applies here is exactly the guess that produces a rule nobody
agreed to.

How to run one:

1. **Survey first, ask second.** Read the conflicting rules in full and find out *why* each side
   adopted its version. A question that does not carry the reason is unanswerable.
2. **One question per contradiction, batched** — do not drip-feed. Each gets two to four concrete
   options, never an open prompt.
3. **Each option states its cost**, not just its label: what breaks, what it forces on consumers,
   which existing rule it contradicts.
4. **Recommend one and say so.** A questionnaire with no recommendation offloads work the survey
   already did.
5. **Record the answer where it will be read again** — the rule into the relevant section below, the
   reasoning into §10 in the user's own words. An answered contradiction that is not written down
   gets re-asked, which is worse than having guessed.

The one exception: a conflict with `docs/PHILOSOPHY.md` is not a questionnaire. The charter wins.

---

## 2. Non-negotiables

Read the whole list before the first edit. The reason follows each one, because a rule without its
reason gets rationalised away at 2 a.m.

- **Never `uv pip install`.** Use `uv sync` / `uv add` / `uv add --dev`. `uv pip install` writes into
  the venv without touching `pyproject.toml` or the lockfile, so the next clean checkout silently
  lacks the dependency.
- **Never call bare `python` / `pytest` / `cgm-cli`.** Always `uv run …`. A bare interpreter bypasses
  the workspace environment and system Python lacks the package → `ModuleNotFoundError`.
- **Never `uv run --extra <x> …`.** Sync the extras first (`uv sync --extra dev`), then run.
- **Never add a hard dependency to the core.** Polars is the only one. Anything heavier goes behind
  the `cli` / `dev` extras plus a guarded import. This is the charter's *Minimal core*, and it is
  what lets a consumer take this library without taking our stack.
- **Never hardcode a version string.** It comes from `importlib.metadata.version("cgm-format")`,
  which reads `pyproject.toml`. Two sources of truth drift, and the one you read is the wrong one.
- **Never rename a user-facing command to dodge a stale `uv run` wrapper.** After a dependency
  upgrade uv can keep a stale generated wrapper in `.venv/bin`. Bump this package's version and
  re-run `uv sync` so uv rebuilds the entry points. `cgm-cli` is owned by `[project.scripts]`.
- **Never use a placeholder path or a fabricated example value** in committed code —
  `/my/custom/path/`, a dummy digest, an invented glucose reading. A fabricated value proves nothing
  and outlives the session that invented it.
- **Never mock the transformation under test.** See §6.
- **Never claim a test "would have caught" a bug without first running it against the buggy code and
  watching it fail.**
- **Never commit large data.** Anything over ~5 MB that genuinely must travel goes through Git LFS.
  GitHub rejects >100 MB, and a large blob in history is extremely hard to remove afterwards (§3).
- **Never run tree operations.** No `git push`, tags, releases, branch management or history
  rewriting — the user's domain. **Never `git stash drop` / `git stash clear`**, even on explicit
  request. **Never `git add -A` or `git add .`** — it sweeps in `.env` files, editor swap files and
  junk; stage explicit paths. Commit only when asked.
- **Never silently fall back when primary data is missing.** If a vendor column, a metadata row or a
  Nightscout treatments pull is absent, do not quietly serve a substitute as if it were the real
  thing. Either refuse with a typed error or warn prominently, naming the substitute. The caller
  cannot see that the source differed.
- **Never nest a `try`/`except` inside another.** It hides the real error. Let typed exceptions
  propagate (`UnknownFormatError`, `MalformedDataError`, `ZeroValidInputError`, …); wrap only where
  there is a genuine recovery path — probing the next timestamp format, or a guarded optional-dep
  import. `format_supported()` is the sanctioned soft check that returns a bool instead of raising.
- **Never fill a value from the same source that checks it.** A cross-check compares an
  independently authored value against a source; filling it *from* that source makes the check
  compare a convention against itself, and it agrees perfectly. The row moves from honestly
  unverified to apparently verified. Concretely here: never derive an expected test value by running
  the parser you are testing, and never populate a schema field from the same frame the validator
  reads.
- **Never let a tool write a checked value from a lookup result.** Lookups report, the human decides,
  the validator checks. Where a tool refuses to apply a value, preserve the refusal verbatim — the
  refusal is the feature.
- **Never treat a determinism gate as a correctness gate.** Idempotency, a byte-identical round-trip,
  a stable row order: these mean *reproducible*, not *right*. A parser that maps the wrong vendor
  column to `glucose` passes every idempotency test we have.
- **Never collapse "unknown" into a boolean or a number.** See §5.
- **Never route around a capability the library lacks while dogfooding.** See §7.
- **Never resolve a contradiction between two rules by inference.** Run §1.
- **Never edit or re-wrap a reporter's prose** in `docs/FEEDBACK.md` — not when answering it, not
  when moving it to history. It is the record of what was *observed*, not of what was decided. A
  reply is added as its own `**Status —` paragraph above it, immediately after the heading (§8).
- **Never reuse a feedback id**, not even one answered as a non-issue; the reply is part of the
  record and a recycled id collides with it. Compute the next id from the inbox **and**
  `docs/FEEDBACK_HISTORY.md` — once answered items move out, the live file's highest visible id is
  not the corpus's highest, and an empty inbox shows none at all.
- **Never treat "no reply" as "no work done."** Establish what already shipped before reproducing a
  reported item, and certainly before designing for it.
- **Never write a placeholder reply.** Leave an untriaged item unanswered — an empty verdict is
  honest, a hedged one is not — and say what was skipped.
- **Never open a preamble line with `**Status`** in the inbox. It is read as a block reply covering
  every id it names, which marks the whole backlog answered. Use a blockquote or different wording.
- **Never file into a producer's roadmap.** The roadmap holds items a maintainer has already triaged;
  the inbox is the intake. (No upstream applies to this repo today — see §8.)

### 2.1 The load-bearing invariants

These are the pipeline's correctness contract and the reason the library is worth trusting. Changing
one is a real behavior change — gate it on a test demonstrating the new behavior and update
`docs/PHILOSOPHY.md`.

- **Idempotency.** Every op yields the same result run once or ten times. `original_datetime` is
  write-once (created at parse, never overwritten); `detect_and_assign_sequences` resets to 0 then
  reassigns from scratch; quality flags are additive via `|`. Re-running is a bit-level no-op.
- **Losslessness.** `synchronize_timestamps` keeps all rows (pure timestamp transform);
  `interpolate_gaps` only *adds* rows (marked `IMPUTATION`, plus `SYNCHRONIZATION` when snapped);
  sequence detection is pure annotation. Nothing silently drops or edits original rows.
- **Commutativity of grid ops.** `interpolate_gaps` (snap-to-grid) and `synchronize_timestamps` share
  one grid calculation rooted in each sequence's first `original_datetime`, so their order doesn't
  matter — see §12.
- **Deterministic row order.** Parquet/CSV bytes depend on row order. The schema defines a total
  ordering (sequence, time, quality, event type, then data columns); apply stable sorting after any
  `concat` / `merge`. Never derive emitted rows from `set` / `dict` iteration or from polars
  `mode()` / `unique()` without a tie-break — neither gives any order guarantee, and `mode()` is
  unstable call-to-call. Prefer explicit sort keys and first-occurrence dedup. Every new ordering
  gets a test.

---

## 3. Repository layout, data and assets

```
src/cgm_format/       source (src layout; hatchling packages = ["src/cgm_format"])
tests/                pytest suite
docs/                 all markdown except this file and README.md
data/input/           committed vendor fixtures + cached Nightscout pulls
data/parsed/          generated output          ─┐
data/cli_test_output/ generated output           ├─ git-ignored, never travels
data/cli_examples_output/ generated output      ─┘
scripts/              one-off operational scripts, not importable code
examples/             runnable usage examples — documentation, keep them working
```

- **`data/` is git-ignored by ignore-all.** `data/.gitignore` is `*`, then `input/` / `input/**`
  *without* the `!` (dropped in `44e3bb9`). Generated output, local dumps and downloaded Nightscout
  pulls stay ignored. Fixtures already in git history under `data/input/` remain tracked; a new
  fixture cannot be committed without `git add -f`. That gap is F2 in `docs/dogfooding.md`. To
  commit another top-level subtree under `data/`, add explicit `!<dir>/` and `!<dir>/**` lines.
- **Deviation from the house layout, deliberate:** committed fixtures live in `data/input/`, not
  `assets/`, and the generated dirs are not named `interim` / `output`. Every integration test, the
  gitignore rule and the sdist exclusion are built on the current names and all work. The
  rename is tracked as an open item in `docs/ROADMAP.md`, not something to do opportunistically
  mid-task.
- Never hardcode a platform-specific cache path; resolve it at runtime.
- **Anything over ~5 MB that must travel goes through Git LFS**: `git lfs install` once,
  `git lfs track "<path>"`, commit the pointer, never the raw blob.

**LFS history gotcha:** a blob committed *before* `git lfs track` stays in every past commit even
after the pointer replaces it at HEAD, so the pack still ships it. Detect it, then hand the
remediation to the user — history rewriting (`git lfs migrate import`, force-push) is theirs to run:

```bash
git lfs ls-files                       # what LFS tracks at HEAD
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | awk '/^blob/ && $3 > 5000000 {print $3, $4}' | sort -rn   # large blobs anywhere in history
```

**Sibling repos** (treat as read-only unless the task explicitly targets them):
`/data/sources/glucosedao/sugar-sugar` and `/data/sources/glucosedao/glucosedao` consume this
library; `/data/sources/glucosedao/glucose_data_processing` is the sister lib whose gap conventions
we match (§12).

---

## 4. Build, run, test

`uv` is the package manager.

```bash
uv sync --extra dev                              # FIRST: install/sync deps (dev includes cli + pytest)
uv run pytest                                    # full suite  (SLOW, ~15 min — see below)
uv run pytest -vvv tests/test_format_parser.py   # one file, verbose (iterate like this)
uv run cgm-cli --help                            # explore the CLI
uv run cgm-cli detect|parse|pipeline <file>      # detect / parse-to-unified / full 6-stage run
```

- **The full suite takes ~15 min.** Give it a 600s+ timeout or run it in the background; scope to one
  file while iterating.
- `uv lock --upgrade` only updates `uv.lock`; raise the lower-bound version constraints in
  `pyproject.toml` by hand to match the newly resolved versions.
- `cgm-cli` is the only entry point. There is no server, so nothing prints a URL.
- **Timestamps: store ISO-8601, display local.** Never write a naive `YYYY-MM-DD HH:MM:SS` into JSON,
  a report or a log line — it is misparsed as local time and breaks string comparison against ISO
  values. Inside the frame, `datetime` and `original_datetime` are Polars `Datetime`; the string form
  only ever appears at an output boundary, and there it is ISO.

**Run the commands yourself** rather than telling the user to run them — except where a command
genuinely needs an interactive terminal, which is when you hand over a verbatim line instead.

**Before a PR**, print `git diff origin/main --stat HEAD` and `git log origin/main..HEAD --oneline`,
show the output, and wait for approval. Installation-specific files stay out.

---

## 5. Coding standards

- **Type hints mandatory** (Python ≥ 3.10 syntax); **`pathlib.Path`** for every path; **absolute
  imports only** (`from cgm_format.formats.unified import ...`), never relative.
- **Never `Any`.** If the value has a shape, write it (`TypedDict`, `Protocol`, a Union of the
  members). `Dict[str, Any]` is the same slack with extra words. A check that could not name the
  type has not typed the value.
- **No inline imports** — every import at module top level. The sole exception is a guarded import of
  an *optional* dependency (`httpx`, `frictionless`, `pandas`) that raises a clear `ImportError` with
  install instructions if missing. The core `FormatParser` / `FormatProcessor` must import cleanly
  with only Polars installed.
- **Dependency tiers are load-bearing.** Core may import Polars and the standard library, nothing
  else. `typer`, `rich`, `httpx`, `pandas`, `pyarrow`, `frictionless` live behind the `cli` extra;
  `pytest` and `python-dotenv` behind `dev`. Nothing is forbidden outright, but a new core dependency
  is a charter change, not a judgement call.
- **No Pydantic — a deliberate, reasoned exemption from the house rule.** The house ruleset says
  Pydantic 2 at every boundary, and the reasoning is sound: schemas and validation want a real model
  layer. It does not apply here, because Pydantic is a means, not a value in itself, and this repo
  already has the thing it would buy: `CGMSchemaDefinition` is a frozen dataclass binding column
  names to Polars dtypes, and those dtypes *are* the validated boundary — a second type language
  beside them is exactly the drift the charter forbids. Taking Pydantic into the core would also
  break the minimal-core tier for a library nothing sits beneath. If a genuine wire contract ever
  appears (a Nightscout response model behind the `cli` extra is the plausible one), that is a §1
  questionnaire, not a silent adoption.
- **Standard-library `logging` for diagnostics — never `print`.** `print` is only for CLI output the
  user asked to see. *(Tech debt: `interface/schema.py:519` prints from a schema-regen helper — RM1
  in `docs/ROADMAP.md`.)*
- **Polars idiom.** Prefer expressions (`with_columns`, `filter`, `group_by`, `join_asof`) over
  Python loops; lazyframes (`scan_*`) and streaming (`sink_*`) on large data paths — keep hot paths
  in the Rust engine. Pre-filter before joining so you never materialize more than needed. Always
  compute *from* `original_datetime`, never from a mutated `datetime` (the idempotency anchor).
- **`__all__` is curated here.** This is a published package with an API contract, so the criterion
  lands on the curated side: keep `__all__` in sync with the actual imports in `__init__.py`, and
  never rely on `import *`. Register new public enums/schemas in both the import block and `__all__`
  (§11 checklist).
- **Constrained vocabularies are enums here.** The criterion is whether the vocabulary must grow
  additively inside a major version (then `frozenset[str]` + a validator) or is bound to a fixed
  schema/dtype (then a `str`-subclassing enum). Ours are bound to Polars dtypes and to fixed-width
  codes, so they subclass `EnumLiteral` — members compare and serialize as plain strings, and
  `event_type == "EGV_READ"` works in code and in CSV round-trips. Never a bare `Literal` in
  anything a consumer reads; it makes every addition breaking.
- **Answers are three-valued: true / false / unknown, and `None` is never `False`.** A null glucose
  means "the sensor did not say", not zero; a null `carbs` means "no carb entry was recorded", not
  "zero carbs eaten". When the answer is unknown, **withhold** — never report, never negate. Never
  substitute the lowest bin, a sentinel `0`, or the previous value to silence a downstream warning;
  that is what the `IMPUTATION` quality flag exists to make visible. **A check that could not run is
  not a check that passed** — a Frictionless report skipped because the package is absent is not a
  clean report, and the CLI must say so. Combine with Kleene semantics rather than
  withhold-on-any-unknown: `unknown AND false` really is `false`, and collapsing it loses real
  answers.
- **Aggregate repeated warnings.** A warning emitted inside a per-row loop over a vendor file needs
  collapsing before it ships: group by *reason*, not by row, say which case it is once, with a count.
  A 2,000-line wall of one-per-row warnings buries every other finding a run produces. And
  distinguish **"the vendor did not say"** (an absent field — an empty cell, legitimate) from **"the
  vendor said something we cannot represent"** (a real value the schema cannot hold — a different
  report entirely). Emitting both as one message is half the mistake.
- **Derive a guard from the schema's rule, not beside it.** A validation restated next to the schema
  it guards drifts from it, and the drift only shows on the input where the two disagree. Test the
  guard against `CGM_SCHEMA` case by case rather than asserting a message string.
- **Heed terminal warnings — deprecations especially.** They signal an API moved since training.
  Treat a deprecation in code you touched as a **blocker**: find the current upstream API, fix it,
  and update the rule here so the pattern does not return.
- **Refactor internals aggressively** — no dead code, no old API kept for nostalgia. Internals are
  free to change. The exception is the *contract* — the schema shape, the CLI surface, the public
  API: breaking it is allowed, but deliberate and versioned (see the active-development note above).
- **Version comes only from `importlib.metadata.version("cgm-format")`.** *(Tech debt:
  `__init__.py:33` hardcodes a dev-fallback literal; the right fix is an editable install
  (`uv sync`) so metadata is always present, then drop it — RM2 in `docs/ROADMAP.md`.)*

### 5.1 Type system — this repo's idiom

- **Schemas are frozen dataclasses.** `CGMSchemaDefinition` / `ColumnSchema` are
  `@dataclass(frozen=True)`; `CGM_SCHEMA` is the single canonical instance and never mutates at
  runtime. Column names bind to Polars dtypes there — dtypes are the source of truth; there is no
  separate type language that can drift.
- **New vendor column vocabularies subclass `EnumLiteral`** (`interface/schema.py`).
- **`Quality` is a bitwise `Flag`** stored as `Int64` — combine with `|`, test with `&`,
  `Quality(0)` is good. It is *not* a categorical column:
  `df.filter((pl.col("quality") & Quality.IMPUTATION.value) != 0)`.

---

## 6. Testing — layer 1

- **Run via uv, always** (§4). Use `-vvv` when diagnosing.
- **Real data + ground truth.** Integration tests against real vendor files in `data/input/`.
  Exercise the actual parse/process paths — do not mock a data transformation. **Compute expected
  values at runtime from the fixture**, and not by running the code under test (§2, the
  redundancy-check rule).
- **Meaningful assertions.** Prefer relationships, aggregates and set equality over existence or
  count-only checks: `assert set(source_ids) == set(output_ids)`, not `assert len(df) > 0`. What to
  validate: counts and aggregates; joins (pre/post counts, key coverage, nulls introduced);
  transformations (round-trip survival, key preservation); data quality (ranges, duplicates,
  malformed entries).
- **Hardcoding domain constants** — vocabulary members, the 15-minute gap threshold — is fine.
  **Hardcoding a row or unique count read off a data dump is not**; it drifts with the fixture.
- **Idempotency, round-trip, and every new ordering get a real test** — extend the dedicated
  `test_idempotency.py` / `test_roundtrip_datetime.py` suites when you touch those paths.
- **Be resilient to changing data.** For optional or live data (a specific treatment type in a
  Nightscout pull), `pytest.skip()` rather than asserting the fixture contains it; skip gracefully
  when offline rather than mocking. `pytest.mark.xfail` with a clear reason for a known upstream
  data defect.
- **Avoid the AI test anti-patterns**: happy-path-only; counts derived from inspecting data; mocking
  the transformation under test; redundant checks (a `len()` beside a set equality); ignoring nulls,
  empties, boundaries and malformed input; and asserting a test "would have caught" a bug without
  demonstrating the failure first.
- **A test that means "no credential" must say so.** `NIGHTSCOUT_TOKEN=None` is indistinguishable
  from "not passed" when the reader does `token or os.environ.get(...)` — it still picks up a real
  key, so the test is green on CI and broken on the machine that owns the credential. Neutralize in
  an autouse fixture with `setenv(VAR, "")`, **not** `delenv`: `load_dotenv(override=False)` skips a
  key that is merely *present*, so an empty value survives a later reload where a deleted one is
  silently restored.
- **Suspect ordering whenever a test passes alone and fails in the suite.** `tests/conftest.py` calls
  `load_dotenv` at import time, and that mutation stays in `os.environ` for the whole session.
- `tests/conftest.py` provides a session-scoped `nightscout_data_dir` fixture that downloads
  Nightscout JSON from `NIGHTSCOUT_URL` (optional `NIGHTSCOUT_TOKEN` / `NIGHTSCOUT_API_SECRET`) into
  `data/input/`, cached; `--nightscout-redownload` forces refresh.

---

## 7. Dogfooding — layer 2

Tests prove the code does what it was told. Dogfooding asks a different question: *is this usable,
and what is missing?* Both are required; neither substitutes for the other. Findings go in
`docs/dogfooding.md`.

**Do not "verify the parser's answers" with a second independent implementation while dogfooding —
that is a test, and it belongs in the suite.** Use the library, notice the friction, write down what
was not there.

- **A capability the library LACKS is the result, not an obstacle to route around.** The moment you
  reach for a hand-rolled `pl.read_csv` + column rename to get past something `FormatParser` cannot
  do, the exercise has stopped producing signal: you proved the task is possible with *general*
  tooling, which was never in question, and learned nothing about the library. Record the gap; if it
  blocks the work, build it into the library and carry on **with the library**.
- **Run the adversarial round.** Switch role deliberately — be a beta-tester trying to show the
  library fails at something it advertises — then switch back and fix. Two rules keep it honest:
  **attack claims, not gaps** (a documented deferral is a decision; finding it proves nothing — what
  counts is where a docstring, comment or doc *promises* something the code does not do), and **use
  real data** (a real vendor export, not an invented row). A good finding is a sentence that quotes
  the code's own claim back at it.
- **Pick the probe where the design generalized from one case.** If a vendor branch was written
  against one export, run it against a second real export from the same vendor. If a convention
  states a boundary — the 15-minute threshold, the 5-minute grid — take a real case at its edge.
- **Turn the library on the work you just did.** A parser branch written in the morning is the best
  candidate for the afternoon's probe, and it will be wrong in a way its tests were not.
- **Dogfood a finding before you report it.** Build a real, sensible example against the actual code
  path and show it fails. A loss that is mechanically possible but has no realistic CGM
  instantiation is noise, not a finding — walk the data model with a domain eye first. A demonstrated
  failure on the current code beats a plausible-looking mechanistic claim.
- **Finish each probe as a committed reference example whose README names what it broke.** The
  fixture-plus-test is the regression test; the note is the evidence. A finding recorded only in a
  commit message is not reproducible. Keep the failure in the suite by demonstrating it on the *old*
  behaviour, not by asserting that it used to fail.
- **Separate "fix it" from "surface it" before writing any code, and be strict about the line.** Fix
  a false claim, a misdiagnosis, a wall of un-aggregated warnings, a guard that is never reached.
  Surface anything where the obvious repair is itself a design decision — a schema change, a
  threshold change, a new quality flag — and say *why each candidate repair is wrong*, because that
  is what makes the item actionable months later.

There is no UI in this repo, so the seeded-state dev entry point and the browser-automation rules do
not apply. If one ever appears: never drive a multi-step UI flow with an LLM browser agent.

### The finding log

Findings carry stable `F#` IDs and **move** between files — never duplicated, except where a finding
is mitigated here but still owed upstream, which legitimately appears in two.

| File | Holds |
|---|---|
| `docs/dogfooding.md` | open quirks, bugs and UX gaps found by using the shipped surface |
| `docs/previous_issues.md` | findings resolved **here**, each with its resolution and a code pointer |

There is no `docs/<upstream>-pending-fixes.md` because this library consumes nothing of ours. A
finding against Polars or Frictionless goes to `docs/dogfooding.md` with the upstream issue link and
the defensive mitigation already in place.

---

## 8. Docs and their lifecycle

- **All new markdown goes in `docs/`** — the only exceptions are this file, `README.md`, and the
  `AGENTS.md` symlink. Keep the README usable; deep detail lives in `docs/`.
- **`docs/` is the single ground truth.** This file duplicates from it only where a fact is needed to
  *orient*, and every prohibition lives here in full, because a `don't` behind a link is a `don't`
  that does not get read. `docs/` links carry positive detail only.
- **`docs/PHILOSOPHY.md` is the charter.** Self-contained, names no other document, outranks any
  plan. Update it when an invariant genuinely changes — never to match code that drifted.
- **`docs/ROADMAP.md` is active-only.** Shipped items move to `docs/ROADMAP_HISTORY.md` with their
  rationale. Nothing is deleted; it is relocated.
- **`docs/CHANGELOG.md` records what shipped**, newest first — including cross-repo integration
  changes made on our side, so agents working in `sugar-sugar` or `glucosedao` are not surprised.
- **Update this file and the affected `docs/` in the same change as the refactor**, not after. Policy
  is written first; code complies.
- **Consumer → producer intake.** If you find something that belongs in a repo we consume but do not
  own, do not edit or commit that repo beyond appending your item to its `docs/FEEDBACK.md` — its
  inbox, not its roadmap. Today nothing of ours sits upstream of this library, so this is dormant;
  a finding against a third-party package goes to its issue tracker, with the reference recorded in
  `docs/dogfooding.md`.

### The consumer inbox

`docs/FEEDBACK.md` holds **open** consumer items; `docs/FEEDBACK_HISTORY.md` holds answered ones
verbatim plus a one-line contents entry each. Same split as ROADMAP vs ROADMAP_HISTORY, and for the
same reason: an inbox only grows, and unanswered items become invisible inside answered ones. **Empty
inbox ⟹ nothing owed** — a property destroyed the moment answered items are left in place. The reply
lives beside the report; the reporter's prose is never edited or re-wrapped (§2).

Triage order: establish what already shipped → reproduce against the **code**, not the docs → decide
**legality before severity** → route → reply → archive.

- **Legality sizes the release; severity only orders the queue inside it.** A new optional column, a
  new `UnifiedEventType` member, a new vendor, a new flag beside the old one is **minor** however
  severe the finding. A removal, a promotion to required, a dtype change — **including a rename**,
  which is a removal plus an addition — is **major** however trivial. Pure legibility (a warning, a
  count, an error message, a doc) is a **patch**. And when changing a shape a consumer reads, **add
  rather than redefine**: a consumer already compensating for the old meaning breaks *silently*
  otherwise.
- **Read the compatibility rules first-hand; never delegate the legality step.** A summary of the
  charter drops the qualifier the decision turned on, and this is the step that decides whether a
  repair is legal at all.
- **A non-issue verdict is not the cheap outcome.** "Nothing is wrong here" has to be shown — what
  was probed and did not reproduce. A bare "works as intended" is not a reply.

**The runbook is [docs/CONSUMER_TRIAGE_LOOP.md](docs/CONSUMER_TRIAGE_LOOP.md)** — the full algorithm,
the routing table, the thresholds that call the user, what may be done unattended, and the gotcha
list. Read it before triaging; the section you are reading is the orientation, not the procedure. The
generalized pattern it adapts is published at
<https://gist.github.com/winternewt/54b94bda01812be937b892146d1bb254>, and a change to the *pattern*
(the algorithm, a script's contract, a gotcha in the mechanism) belongs in both. The one thing the
gist cannot supply is this repo's own compatibility rule set, which is what the legality step reads —
that is the list just above, plus `docs/PHILOSOPHY.md`.

**The scripts are installed here**, in `scripts/` (`triage-state.py`, `triage-archive.py`,
`watch-inbox.sh`), documented in `scripts/README.md`, with defaults repointed at `docs/FEEDBACK.md`
so they run from anywhere in the tree. They are stdlib-only and import nothing from the package,
which is why they are the sanctioned exception to always-`uv run` (§2) — the workspace environment is
not what they need. Run the ledger before triaging and the archiver after replying:

```sh
./scripts/triage-state.py --pending    # what is owed          (new / revised / unmarked-reply)
./scripts/triage-state.py --next       # the next id, over BOTH documents — never guess it
./scripts/triage-archive.py S1         # archive, verifying the prose moved byte-for-byte
./scripts/triage-state.py docs/FEEDBACK_HISTORY.md   # post-archive lint: all `current`, nothing `new`
```

- **Never pass a `.py` script to `bash`.** The shebang is ignored, the module docstring runs as
  commands, and `import hashlib` reaches ImageMagick's `import`, which silently writes 0-byte files
  named after each import into the working directory.
- **A reply is a `**Status —` paragraph first in the section, ending in the ledger's
  `<!-- triaged: … sha … -->` marker**, whose fingerprint covers the reporter's text and never the
  reply — that is what stops a watcher firing on your own write. Take the sha from the ledger, never
  by hand, and never restamp a `revised` section to silence it: `revised` is the only signal that
  catches a reporter editing what they reported. `--backfill` touches `unmarked-reply` alone, on
  purpose.
- **`--dry-run` is not a rehearsal** — it returns before the write and so never reaches the
  before/after fingerprint comparison. Rehearse against copies with `INBOX=` / `HISTORY=` instead.
- **The archiver verifies the move, not the verdict.** It will archive an unanswered item without
  complaint; the lint above is what catches that.
- **The contents line in `FEEDBACK_HISTORY.md` is hand-written** — the archiver deliberately does not
  generate it. One line, under 80 characters.

### Prose style

Natural, human prose. Avoid AI tells — em-dash pile-ups, filler transitions, marketing voice. Never
hallucinate documentation or overpromise an unimplemented feature. Describe the library honestly: it
*absorbs vendor chaos and emits a trustworthy DataFrame*. **It must never be described as measuring,
diagnosing, or interpreting anything clinical** — it moves numbers a device already produced.

---

## 9. Self-correction

When outdated API knowledge causes a real crash or logic failure, fix the code **and** update this
file (and the affected `docs/`) with the correct pattern, so the next agent does not repeat it.
Update the guide immediately whenever code is refactored — stale guidance is worse than none. The
same applies when the user corrects a preference: it goes in §10, in their words, with the reason.

## 10. Learned user preferences

*Append-only. One line each, in the user's terms, with the why where it is not obvious.*

- **Pydantic is good at schemas and validation but isn't a value in itself** — the house rule should
  read "Pydantic 2 at every boundary, except where an equivalent typed boundary already exists and
  the dependency tier forbids it". This repo is that exception; the exception is worth stating in the
  ruleset rather than being an unexplained local deviation (2026-08-13).
- **Layout changes are their own task.** The `data/` → `assets/` + `interim`/`output` reorg was
  approved in principle but explicitly deferred out of a docs-only change — plan it, don't smuggle it
  (2026-08-13).
- Committing is occasionally delegated but never assumed: leave changes unstaged unless a commit was
  asked for. Tree operations, pushing and releases are the user's domain.
- **Any typehints is slack.** Name the type — `TypedDict`, `Protocol`, a Union of the members — not
  `Any`. `Dict[str, Any]` is the same refusal with extra words. Why: Any turns the annotation into a
  comment the checker will never catch (2026-08-13).

## 11. Learned workspace facts

*Append-only. Environment, paths, credential layout, host quirks.*

- Source layout `src/cgm_format/` (hatchling, `tool.hatch.build.targets.wheel.packages`).
- Optional-dep groups: `cli` (typer, rich, httpx, pandas, pyarrow, frictionless), `dev` (cli + pytest
  + python-dotenv).
- `download_nightscout()` always fetches JSON (entries, treatments, profile); `token` = query param,
  `api_secret` = SHA1-hashed header. Credentials come from `.env` at the project root, read by
  `tests/conftest.py`; the CLI does **not** load `.env` today (RM4).
- Sibling repos live side by side under `/data/sources/glucosedao/`: `sugar-sugar` and `glucosedao`
  depend on `cgm-format[cli]`; `glucose_data_processing` is the sister lib for gap conventions.
- When upgrading deps (`uv lock --upgrade`), also raise the lower-bound constraints in
  `pyproject.toml` to match the newly resolved versions.
- `LIBRE_EU` is a derived variant of `LIBRE` (`formats/libre_eu.py`), registered **before** `LIBRE`
  in `FORMAT_DETECTION_PATTERNS`. Glucose conversion for both Dexcom and Libre goes through
  `FormatParser._glucose_to_canonical`, which reads the column's declared unit. `derive_schema`
  accepts `append_data_columns` for columns a variant grew (Libre EU ketones). Those ketone
  columns stay on the vendor schema; unified parse drops them (RM6).
- `data/.gitignore` ignores `input/` outright (F2); new vendor fixtures are not auto-tracked. The
  mmol/L Libre fixture stays local and `tests/test_libre_eu.py` skips when it is absent.
- The triage-loop scripts (`scripts/triage-state.py`, `scripts/triage-archive.py`,
  `scripts/watch-inbox.sh`, adopted 2026-08-17) must stay in one directory — the archiver resolves the
  ledger relative to its own path and the watcher shells out to it the same way. Their only local
  divergence from the gist is the default `INBOX` / `FILE`, derived from the script's own location
  instead of `$PWD`. Nothing arms the watcher; that is a deliberate operational choice, not an
  oversight.

---

## 12. Domain reference

Repo-specific detail that has no home in the house sections. Prohibitions here are stated in full;
`docs/PIPELINE.md` and `docs/NEW_SCHEMA.md` carry the positive detail.

### Adding a new vendor format

1. **Create `formats/<vendor>.py`**: file-layout constants, detection patterns, column enums
   (subclassing `EnumLiteral`), and a `CGMSchemaDefinition` for the raw CSV columns.
2. **Register in `supported.py`**: an entry in each of the six exhaustive registries —
   `SCHEMA_MAP`, `FORMAT_DETECTION_PATTERNS`, `FORMAT_DETECTION_LINE_COUNT`, `UNIFIED_TARGET_SCHEMA`,
   `FORMAT_CATEGORY`, `KNOWN_ISSUES_TO_SUPPRESS` (`[]` if the format has no Frictionless quirks to
   tolerate) — plus `PATH_DETECTION_PROBES` and `SUBJECT_PATH_PROBES` if the source is a directory
   rather than a file. `docs/NEW_SCHEMA.md` has the table with what each one holds.
3. **Add the enum value** to `SupportedCGMFormat` in `interface/cgm_interface.py`.
4. **Implement parsing**: a `_process_<vendor>` method + dispatch branch in `FormatParser`. It must
   map vendor→unified columns, probe/normalize timestamps, handle edge cases (out-of-range markers,
   metadata rows, variable-length records), and return a frame passing
   `CGM_SCHEMA.validate_dataframe(enforce=True)`.
5. **Export public symbols** from `__init__.py` (import block **and** `__all__`).
6. **Write real-data integration tests** in `tests/`: detection, parsing, round-trip, full pipeline.

The processor, schema validation and CLI need zero changes — they only see `UnifiedFormat`. The long
form of this checklist is `docs/NEW_SCHEMA.md`. A **variant** of an existing vendor (mmol/L columns,
an extra metadata row) is `derive_schema` plus a `european=True` arm on the existing `_process_*`,
not a new parser: `DEXCOM_EU` and `LIBRE_EU` are the pattern. Detection order is load-bearing —
register the more-specific identity first.

### Gap thresholds & grid-aligned measurement

- **`SMALL_GAP_MAX_MINUTES = EXPECTED_INTERVAL_MINUTES * 3 = 15`** separates "small" (fillable) from
  "large" (sequence-splitting) gaps. Aligned with the sister lib
  [`glucose_data_processing`](https://github.com/GlucoseDAO/glucose_data_processing)
  (`small_gap_max_minutes=15`).
- **A grid multiple matters.** With `snap_to_grid=True` (default) raw timestamps are projected onto
  the 5-min grid before measuring gaps, so effective gaps are multiples of 5. A threshold that is
  itself a grid multiple (15) gives clean, deterministic fill/skip decisions; the old `19` was not a
  grid multiple and made `interpolate_gaps` and `synchronize_timestamps` disagree on borderline gaps.
- **Commutativity** comes from `_interpolate_sequence` projecting both endpoints of each gap onto the
  grid via `calculate_grid_point()`, then measuring grid-position distance, before applying the
  `> expected_interval` / `<= SMALL_GAP_MAX_MINUTES` thresholds. Only active when `snap_to_grid=True`.
- **Operator convention** (both libs): sequence splits use `> threshold` (a gap *at* threshold stays
  in the same sequence); interpolation fill uses `<= threshold` (a gap *at* threshold IS filled).

### Known pitfalls

- **Encoding artifacts.** Dexcom/Libre files ship UTF-8 BOM (and double/triple-encoded variants);
  `decode_raw_data` normalizes them. Test new formats with raw `bytes`, not pre-decoded strings.
- **Dexcom variable-length rows.** Non-EGV rows omit trailing columns; Frictionless flags them.
  Suppressed via `KNOWN_ISSUES_TO_SUPPRESS` — not data corruption.
- **`Quality` is a bitwise Flag, not an enum.** Values are ints; test with `&` (§5.1).
- **`frictionless` is optional.** CLI `report` / `validate` use it if present and degrade gracefully
  otherwise — and a skipped report is reported as skipped, never as clean (§5, three-valued).
- **`httpx` is optional.** `nightscout_downloader` needs it (in `cli` / `dev` extras); import inside
  functions and raise a clear `ImportError` with install instructions if missing.
- **Nightscout dual-path.** (1) JSON API path — `parse_nightscout()` / `from_nightscout_exports()` /
  `from_nightscout_url()`, combining entries + treatments, `token` / `api_secret` auth. (2)
  nightscout-exporter CSV — a combined file with `# CGM ENTRIES` / `# TREATMENTS` section headers,
  auto-detected by `detect_format()`. `_process_nightscout` dispatches both. The built-in Nightscout
  API CSV endpoints are **not** supported (headerless 5-col entries; treatments returns JSON anyway)
  — `data/input/nightscout_entries.csv` is kept as a negative control. JSON files do **not** go
  through `detect_format()`.
- **Detection order.** `FORMAT_DETECTION_PATTERNS` returns on first match. `DEXCOM_EU` before
  `DEXCOM`, `LIBRE_EU` before `LIBRE` — the mmol/L exports also match the generic patterns.

### Format-drift & known-issue handling

Vendor exports mutate over time (new metadata rows, extra columns, encoding variants). When a real
export no longer fits, that's a **schema gap to widen additively**, not a data error.

- Expected Frictionless quirks are suppressed centrally in `KNOWN_ISSUES_TO_SUPPRESS`
  (`formats/supported.py`) — not scattered through parser branches. **Keep suppressions bounded and
  specific**; never blanket-suppress a whole error class.
- For tolerance thresholds, favor a **static floor + dynamic tolerance + a warning** over a hard
  reject, so a slightly-drifted-but-valid export still parses while genuinely broken data is caught.
- A compatibility fix for a new export variant is a **patch bump** — it adds nothing a consumer reads
  and removes nothing (§8, legality).
