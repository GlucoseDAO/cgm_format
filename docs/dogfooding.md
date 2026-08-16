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

### F3 — a stale `cgm_format.egg-info` silently shadows the installed version

Found while verifying the 0.10.0 bump. `cgm_format.__version__` reported **0.2.2** on a checkout
whose `pyproject.toml` said `0.10.0` and whose `.venv` dist-info said `0.10.0` too.

The cause is a stale `cgm_format.egg-info/` directory in the repo root, dated November 2025, left by
some earlier non-uv build. `importlib.metadata.distributions()` finds it first because the repo root
sorts ahead of `site-packages` on `sys.path`, so `version("cgm-format")` returns whatever that
directory's `PKG-INFO` says — forever, and silently.

It is gitignored (`.gitignore:27`) and untracked, so it never travels and CI is unaffected. But any
local run — a CLI `info`, a log line, anything embedding `__version__` in output — reported a version
from a build nobody remembers making. Nothing errors, which is what makes it worth recording.

**Mitigated, not fixed.** The stale directory was moved aside. The library cannot defend against
this on its own: it is doing exactly what `CLAUDE.md` §2 prescribes (read the version from installed
metadata, never hardcode it), and the metadata it reads is genuinely present and genuinely wrong.

Two things would help, both RM2's territory:

- Drop the `except Exception` fallback once an editable install is guaranteed, so a missing
  distribution fails loudly instead of serving a literal that drifts. The literal was still `0.9.0`
  at the moment the bump landed, which is the same class of bug one level down.
- Consider whether `uv sync` should prune a root `*.egg-info/`. That is a uv question, not ours.

Check for it with:

```bash
uv run python -c "from importlib.metadata import distributions; \
  print([(d.metadata['Name'], d.version) for d in distributions() if 'cgm' in (d.metadata['Name'] or '')])"
```

Two entries means the shadow is back.

### F7 — a missing header column cannot be suppressed by field name

Found while bringing BIG IDEAs' Clarity exports to a clean Frictionless report (2026-08-16).

All 16 published BIG IDEAs Dexcom files omit the `Transmitter ID` column entirely. Frictionless
reports that as `missing-label`, and a `missing-label` error carries **no** `fieldName` — its `label`
is the empty string. `_should_suppress_error` matches on `(type, fieldName, cell)`, so no rule can
name the absent column:

```
RESIDUAL: missing-label | - | There is a missing label in the header's field "Transmitter ID" at position "14"
```

Every other error type on these files is now suppressed (500 residual → 1, or 2 on the four
subjects in F9), so this is the main thing standing between BIG IDEAs and a clean `cgm-cli report`.

**Not fixed here, because every obvious repair is a design decision:**

- `('missing-label', None, None)` blanket-suppresses *any* missing column for all Dexcom files.
  `docs/NEW_SCHEMA.md` forbids exactly this — a genuinely truncated export would then pass.
- Matching on the error *message* text is fragile across Frictionless versions.
- Matching on `fieldNumber` would need a fourth positional meaning in the rule tuple, which
  currently means "count cap".
- Giving `ColumnSchema` an `optional` flag is the principled fix — it is a schema-vocabulary change
  affecting every format, and belongs to whoever owns that decision.

Reported rather than repaired, per the "fix it vs surface it" line: the residual is one, it is
honest, and it names the real condition.

### F8 — the headerless food log cannot be Frictionless-validated at all

Same session. BIG IDEAs subject `003` ships a food log with **no header row** and three columns
dropped. The parser handles it (`BIGIDEAS_FOOD_HEADERLESS_11`, sniffed on the absence of
`logged_food` in line 1), but Frictionless validates against `BIGIDEAS_FOOD_SCHEMA`, which declares a
14-column header — so it reads row 1 as the header and reports 185 errors: every label wrong, every
row mistyped.

This is **not** a suppression candidate. Silencing 185 errors across many types would make a
genuinely broken file indistinguishable from this one. `CLAUDE.md` §5: a check that could not run is
not a check that passed.

The honest options, none free: a second `CGMSchemaDefinition` for the headerless variant plus a
dialect carrying `header: false` (Frictionless supports it; `CGMSchemaDefinition` does not model it
today), or teaching the CLI to report "not validatable — headerless variant" as a third state beside
valid/invalid. The second is more truthful and needs the three-valued reporting the charter already
argues for elsewhere.

Pinned by `tests/test_bigideas.py::TestFrictionless::test_the_headerless_variant_is_reported_unvalidatable_not_clean`,
which asserts the current state rather than a hoped-for one.

### F9 — the blank-timestamp cap of 1 is falsified by published exports

Measured on the 16 published BIG IDEAs Clarity exports, 2026-08-16.

`KNOWN_ISSUES_TO_SUPPRESS[DEXCOM]` tolerates exactly **one** blank-timestamp metadata row per file,
and its comment states the reason: "a second blank timestamp would be a real data issue and must
still fail." Counting the real corpus:

```
blank-timestamp rows per file across 16 published subjects: {1: 12, 2: 4}
```

Four published, valid exports carry two. The premise is wrong as stated — a second row is not
evidence of breakage — so those four files keep one unsuppressed `constraint-error` each and report
as invalid under `cgm-cli report`.

**Deliberately not changed.** Raising the cap to 2 is a one-character edit, and it was tried: it
weakens the guard for *every* Dexcom file, not only this corpus, and the threshold is pinned by
`tests/test_cli_integration.py::TestSuppressionCap::test_cap_enforced_across_a_file`, which asserts
the value directly. A test that exists purely to hold a number is evidence someone chose it. Editing
both the number and the test that guards it would leave nothing checking the decision, so the value
stands and the finding is recorded instead.

What would settle it: whether the cap is meant to express "how many metadata rows Clarity is known
to emit" (then it should track the observed maximum, currently 2) or "how much drift we will
tolerate before demanding a human look" (then 1 is right and these four files *should* report).
Those are different rules and the comment reads as the first while the value behaves as the second.

Parsing is unaffected either way — the parser measures the metadata block and skips exactly what the
file has, warning once per file with the signed drift (and once per corpus walk, aggregated). This is
a reporting-fidelity item only.

### F10 — `TrackCoverage.rows` cannot express coverage for a Clarity-shaped source

Found while sizing the BIG IDEAs corpus with `list_subjects`, measured on the 16 published subjects
and the three committed fixtures, 2026-08-16.

`FormatParser._bigideas_track_coverage` filters the raw Dexcom frame down to `Event Type == "EGV"`
and only then takes `len(raw)` as the `rows` figure. Every EGV row in a Clarity export carries a
glucose value, so `values == rows` by construction and the ratio can never say anything:

```
bigideas_synthetic:  001 5/5   003 3/3   007 3/3
bigideas (16 real subjects):  values == rows on every one
```

For contrast, `_cgmacros_track_coverage` counts every row of the subject CSV, so the committed
CGMacros fixture reports 55/60 on `CGMacros-001`'s dexcom track — a ratio that carries information.

`TrackCoverage`'s own docstring in `src/cgm_format/interface/cgm_interface.py` states the meaning
this violates:

> `rows` is every row of the track's source, so `values / rows` is the fraction of the period the
> track speaks for.

For BIG IDEAs that fraction is a constant 1.0, so the documented meaning does not hold.

**Surfaced rather than repaired, because every candidate denominator is a design decision and each
is wrong differently:**

- All rows of the file counts the Clarity metadata and alert rows as period the sensor did not
  speak for.
- All non-metadata rows counts insulin, carb and exercise events as glucose opportunities, so a
  subject who logged more meals would report worse sensor coverage.
- Elapsed time over the expected interval is the number a caller actually wants, but it is a
  different measurement from the row ratio the field is defined as. Changing what `rows` means
  would silently redefine a field two other corpora already populate, and `CLAUDE.md` §8 says add
  rather than redefine when changing a shape a consumer reads.

What would settle it: whether `TrackCoverage` is meant to report a row ratio, in which case a
per-format denominator convention needs writing down, or a duty cycle, in which case it needs a new
field beside `rows` rather than a redefinition of it.
