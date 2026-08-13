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

## RM6 — parse Libre ketone readings into UnifiedFormat

**Severity:** medium · **Status:** open · **Owner:** unassigned

Libre vendor schemas already declare ketone columns: `Ketone mmol/L` on `LIBRE_SCHEMA`, plus
`Historic Ketone mmol/L` and `Scan Ketone mmol/L` appended on `LIBRE_EU_SCHEMA`. `_process_libre`
never selects them. `CGM_SCHEMA` has no ketone field, so `validate_dataframe(enforce=True)` would
drop any extra column the parser tried to smuggle through. The values are gone after Stage 3.

The JonGrove mmol/L export (local, gitignored) carries all three headers and **zero populated
cells**. The committed Libre fixtures are the same. There is no real ketone reading in the tree.
Do not implement against an invented row.

Left out of 0.9.0 on purpose: that release put the columns on the *vendor* schema so Frictionless
accepts the 21-column header. Putting them on the *unified* schema is a shape a consumer reads —
a design, not a parser follow-up.

**Decisions to settle before writing code** (do not infer):

1. **Shape.** A new optional data column (`ketones`, null = the vendor did not say) vs a new
   8-char `UnifiedEventType` vs both. A column without an event type leaves a ketone row looking
   like an empty glucose event; an event type without a column has nowhere to put the number.
   Reusing `glucose` to hold ketones redefines a column a consumer already reads — that would be
   major and silent.
2. **Canonical unit.** Clinical ketones are already mmol/L. Routing them through
   `_glucose_to_canonical` / `CANONICAL_GLUCOSE_UNIT` applies a glucose convention to a different
   analyte. Stay mmol/L, or name a ketone target in `UNIT_CONVERSIONS` — pick one, don't borrow.
3. **Historic vs scan vs the older `Ketone mmol/L` column.** Merge into one series (as scan
   glucose merged into `EGV_READ`) or keep distinct event types. Merging re-opens F1 on a new
   analyte.
4. **Fixture.** Need a real export with nonempty ketone cells. Skip-if-absent until then.

**Legality:** minor if we *add* a column and/or event type. Major if we reuse an existing unified
column.

---

## Idea book

Freeform, unsized, no commitment. An item here has not been triaged.

- A `--strict` / determinism mode for the CLI would be useful for reproducible dataset builds — with
  the caveat stated loudly in the docs that it means *reproducible*, not *correct*. A parser mapping
  the wrong vendor column to `glucose` passes every determinism check we have.
- Aggregate the per-row parse warnings by reason rather than by row before they reach a user running
  `cgm-cli parse` on a full multi-year export. Worth measuring the actual volume on the largest
  fixture first, rather than assuming it is already a problem.
- Nothing currently distinguishes "the vendor did not record a carb entry" from "the vendor recorded
  something the schema cannot hold" in the warning stream. They are different reports and a consumer
  reading the log cannot tell them apart.
