# Consumer feedback — open inbox

The intake for people and repos that **consume** `cgm-format` — today `sugar-sugar` and `glucosedao`,
plus anyone using the published package. Layer 3: not our tests, not our own dogfooding, but our
consumers hitting the library in their own work.

**An empty inbox means nothing is owed.** That property only holds if answered items are moved out,
so once an item is answered it goes to [FEEDBACK_HISTORY.md](FEEDBACK_HISTORY.md) verbatim, reply and
all.

The file is run as a **document-as-inbox triage loop**: the document is both the transcript and the
state, and each item's state is derived by fingerprinting the reporter's own text rather than kept
anywhere else. The runbook an agent triaging this file should read is
[CONSUMER_TRIAGE_LOOP.md](CONSUMER_TRIAGE_LOOP.md) — the algorithm, this repo's legality rules, the
routing table, and what may be done unattended. The generalized pattern it adapts is published at
<https://gist.github.com/winternewt/54b94bda01812be937b892146d1bb254>. Nothing below depends on the
tooling — the ledger reads what is written, so a reply typed by hand in the right shape is a
first-class one.

## The ledger

Three dependency-free scripts in `scripts/`, documented in
[`scripts/README.md`](../scripts/README.md). They are stdlib-only Python and bash, import nothing
from the package, and so are the one place in this repo that does not need `uv run`:

```sh
./scripts/triage-state.py              # every item and its verdict: new / revised / unmarked-reply / current
./scripts/triage-state.py --pending    # just the ones needing work
./scripts/triage-state.py --next       # the next unclaimed id, computed over BOTH files
./scripts/triage-archive.py S1         # move answered items to FEEDBACK_HISTORY.md, verifying the move
```

The two `.py` files are Python: run them, or hand them to `python3` — **never to `bash`**, which
executes the module docstring as commands and reaches ImageMagick's `import`.

## Filing an item

Append it at the bottom under a new `## S<n> — <what happened>` heading. **The next id is S2**, and
that pin is a convenience rather than the authority: `./scripts/triage-state.py --next` computes it
over this file **and** `FEEDBACK_HISTORY.md`. Take it from both, because once answered items move out
the highest id visible here is not the highest ever used, and an empty inbox shows none at all. Ids
are never reused, not even for an item answered as a non-issue.

Say what you observed, on which input, with which version, and what you expected instead. If you are
relaying on behalf of a downstream repo, say so and name the case that motivated it — and restate it
as a fact about **this** library rather than in your own vocabulary: which column cannot express
what, which stage reports a shape it cannot distinguish, what a downstream reader ends up seeing.
Your route names and error codes do not exist here.

## Answering an item

The reply goes first in the section, immediately after the heading, as a paragraph opening
`**Status —` and saying four things: the verdict, where it landed (a roadmap item, a doc, or a
shipped version), what was actually reproduced, and what the reporter should do now. It ends with the
ledger's marker, which carries a fingerprint of the reporter's text and nothing of the reply:

```markdown
## S<n> — <what happened>

**Status — accepted; shipped in <version>.** Reproduced: <what was run and what it did>. <Where the
rest landed, and why any candidate repair was rejected.>
<!-- triaged: <version> · sha <12 hex digits — stamp 000000000000 first, see below> -->
```

Keep the placeholders as placeholders: an id that parses (`## S9`) makes the ledger see a phantom
section and `--next` skip to 10.

Stamp `sha 000000000000` as you write the reply, then run `./scripts/triage-state.py`: with a marker
present the reply is excluded whole, so the `revised` line prints the section's true fingerprint and
that is the value to paste back. Never copy the sha the ledger prints while the reply is in and the
marker is not — the hash then covers your own paragraphs two onward. `--backfill` computes it the same
way, so it is for replies that predate the ledger and only those. Once the reply is in,
archive with `./scripts/triage-archive.py S<n>` and add the item's one-line contents entry to
`FEEDBACK_HISTORY.md` — the archiver deliberately does not write that line, because naming what an
item was and how it ended is editorial.

- **Never edit or re-wrap the reporter's prose**, not when answering and not when archiving. It is
  the record of what was *observed*, not of what was decided. The reply is its own paragraph above
  it, immediately after the heading; the prose itself is never touched. The archiver
  verifies this rather than trusting it: every fingerprint is compared before and after the move and
  the write is refused if one changed. It verifies the *move*, not the *verdict* — so run the ledger
  against `FEEDBACK_HISTORY.md` afterwards, where a well-formed archived item reads `current` and
  anything reading `new` was archived unanswered.
- **Never open a preamble line with `**Status`.** A header line beginning that way reads as a block
  reply covering every id it names, which marks the whole backlog answered. Use a blockquote or
  different wording.
- **"No reply" is not "no work done."** Establish what already shipped before reproducing, and
  certainly before designing — the fix may be in a release the reporter has not upgraded to.
- **Never write a placeholder reply.** An untriaged item stays unanswered; an empty verdict is
  honest, a hedged one is not.
- **A non-issue verdict is not the cheap outcome.** Show what was probed and did not reproduce. A
  bare "works as intended" is not a reply.

Triage order: establish what shipped → reproduce against the **code**, not the docs → decide
**legality before severity** → route → reply → archive. Legality sizes the release (additive → minor;
removal, retype or rename → major; legibility → patch); severity only orders the queue inside it. The
compatibility rules the legality step reads are in `CLAUDE.md` §8 and `docs/PHILOSOPHY.md` — read
them first-hand, never through a summary: a summary drops the qualifier the decision turned on.

## Where an answered item lands

| | verdict | lands in | must contain |
|---|---|---|---|
| **a** | real, repairable, legal | `docs/ROADMAP.md` as an `RMn` item | severity · status · owner · the motivating report |
| **b** | non-issue | the reply only | what was probed and did not reproduce — never a bare "works as intended" |
| **c** | documentation defect | the doc, fixed in the same pass | the reply naming the file changed |
| **d** | real, no acceptable repair | `docs/ROADMAP.md`, status open only | the paragraph saying *why each candidate repair is wrong* |
| **e** | not ours to fix | the third party's issue tracker, restated in their terms, referenced from `docs/dogfooding.md` | the reply naming the upstream issue, plus whatever half *is* ours |

Bucket **(e)** means a third party here — Polars, Frictionless — because nothing of ours sits
upstream of this library. Restate rather than forward: a note written in our vocabulary reads as
somebody else's problem in their tracker. An item is often **(e)** *and* **(a)**: their half filed,
our defensive half shipped, both in the one reply.

An item filed as a documentation gap usually has a code half — look for it. A reporter describes
where *they* got stuck, which is a fact about the docs; what stuck them is often a surface that could
have told them.

---

---

# sugar-sugar — filed 2026-08-28

Routed into this inbox on 2026-08-28. The report was written on 2026-08-28 into
`sugar-sugar`'s own tree (`FEEDBACK.md` at its root, commit `6ae4550`) rather than into this
file, so it never reached the ledger and this inbox read empty while four defects were owed.
The reporter's prose below is verbatim; only the issue headings carry this inbox's ids.

Issues found in [`cgm-format`](https://github.com/GlucoseDAO/cgm_format) while working on
Sugar Sugar. Filed here rather than fixed in this repo: `sugar_sugar/corpus.py` /
`sugar_sugar/data.py` are the single boundary to the library, and working around a parser
defect on this side of it is exactly the drift that boundary exists to prevent.

Version under test: **cgm-format 0.12.0** (the latest release on PyPI as of 2026-08-28;
there is nothing newer to bump to). Library paths below are relative to the installed
package. Every claim here was reproduced by running the code, not by reading it.

---

## S2 — Nightscout URL import fails outright on any real instance — `ComputeError` on treatments

**Severity: blocking.** This is the bug a player actually hit; it takes down the whole
"paste your Nightscout URL" feature. Reported against a live Nightscout 15.0.3 instance
(URL withheld — it serves a real patient's data).

### Symptom

`FormatParser.from_nightscout_url(base_url)` raises, before any app code sees a frame:

```
polars.exceptions.ComputeError: could not append value: 47 of type: i64 to the builder;
make sure that all rows have the same schema or consider increasing `infer_schema_length`
```

Traceback bottoms out at `format_parser.py:1480`, in `_parse_nightscout_treatments_json`:

```python
return pl.DataFrame(rows)
```

### Root cause

`_parse_nightscout_treatments_json` (`format_parser.py:1452-1480`) builds `rows` as a list of
dicts and hands it to `pl.DataFrame` with **no schema**, so polars infers dtypes from the
first `infer_schema_length=100` rows only.

Nightscout treatments are heterogeneous by design — a `Temp Basal` carries `rate`/`duration`
and no `carbs`, a `Meal Bolus` carries `carbs`/`insulin` and no `rate`. On the reporting
instance, the 353 treatments had this shape:

| field | None | int | float | first non-null at row |
|---|---|---|---|---|
| `insulin` | 293 | 7 | 53 | 20 |
| `carbs` | **351** | 2 | 0 | **167** |
| `rate` | 77 | 139 | 137 | 0 |
| `duration` | 17 | 336 | 0 | 0 |

`carbs` is `null` for the first **167** rows — past the 100-row inference window — so the
column is inferred as dtype **`Null`**, and appending the `i64` at row 167 fails.

Two things worth stressing, because they rule out the obvious near-miss diagnoses:

- **This is not the int/float mix.** `insulin` mixes int and float (int first appears at row
  122, float at row 20) and appends fine — polars takes the supertype. The failure is
  specifically *any* value into a `Null`-dtype builder. Coercing the JSON's numbers to float
  before parsing does **not** help; it merely changes the message to
  `could not append value: 47.0 of type: f64`.
- **It is not exotic data.** Any Nightscout user who runs a closed loop and logs carbs
  rarely — i.e. a large fraction of them — has a null run longer than 100 rows in at least
  one of these four columns. Bolus-only users hit it via `rate`, pump-only users via `carbs`.

The library already knows to do this correctly on its *other* Nightscout path:
`_parse_nightscout_entries_csv` (`format_parser.py:1613`) passes `infer_schema_length=None`.
The JSON path just lacks the equivalent.

### Minimal repro (no network, no patient data)

```python
import json
from cgm_format import FormatParser

entries = [{"type": "sgv", "dateString": f"2026-08-28T{h:02d}:00:00.000Z", "sgv": 120}
           for h in range(5)]
treatments = [{"eventType": "Temp Basal", "created_at": "2026-08-28T00:00:00.000Z",
               "rate": 0.5, "duration": 30, "carbs": None, "insulin": None}
              for _ in range(105)]
treatments.append({"eventType": "Meal Bolus", "created_at": "2026-08-28T05:00:00.000Z",
                   "carbs": 47, "insulin": 2.5})

FormatParser.parse_nightscout(json.dumps(entries), json.dumps(treatments))
# polars.exceptions.ComputeError: could not append value: 2.5 of type: f64 to the builder
```

106 records is enough. Drop the null run below 100 and it passes, which is the whole bug.

### Suggested fix

Declare the schema instead of inferring it. Every one of these columns is cast to `Float64`
downstream anyway (`_treatments_json_to_unified`, `format_parser.py:1517`, `1534`, `1553`),
so an explicit schema changes no semantics — it only stops inference from guessing:

```python
_TREATMENTS_SCHEMA = {
    "created_at": pl.String, "eventType": pl.String,
    "insulin": pl.Float64, "carbs": pl.Float64,
    "rate": pl.Float64, "duration": pl.Float64,
}

def _num(value: object) -> float | None:
    """Nightscout uploaders are inconsistent: ints, floats and numeric strings all occur."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

# ... in the row loop, wrap the four numeric fields in _num(), then:
return pl.DataFrame(rows, schema=_TREATMENTS_SCHEMA)
```

`infer_schema_length=None` alone would also clear the reported error, but it stays fragile:
it still guesses, it forces a full scan of every row, and it does not handle the numeric
**strings** some uploaders emit (`"carbs": "47"`), which infer as `String` and are then
silently dropped by the `.cast(pl.Float64, strict=False)` downstream. The explicit schema
plus `_num()` handles both.

### Verified

Applied as a monkeypatch against the reporting instance's real payload: parses cleanly,
1367 unified rows — 1152 `EGV_READ`, 61 `INS_FAST`, 152 `INS_SLOW`, 2 `CARBS_IN` over ~4
days. The numeric-string coercion accounts for 3 events (1 bolus, 2 basals) that are lost
without it.

---

## S3 — Redirects are not followed, so `http://` and apex→www instances fail

**Severity: high, one-line fix.**

`nightscout_downloader.py:115` constructs the client as:

```python
with httpx.Client(timeout=timeout, headers=headers) as client:
```

httpx differs from `requests` here — verified on the pinned httpx 0.28.1,
`httpx/_client.py:197`, `follow_redirects: bool = False`. `download_nightscout` never passes
the argument.

So any instance that redirects — `http://` → `https://`, apex → `www`, a Heroku/Fly/Netlify
hostname 301ing to its canonical domain, a trailing-slash normalisation — fails at the
`raise_for_status()` on the next line with `httpx.HTTPStatusError: Redirect response '301
Moved Permanently'`, rather than working. Users type bare hostnames constantly.

Fix: `httpx.Client(timeout=timeout, headers=headers, follow_redirects=True)`.

---

## S4 — An empty or non-JSON API response is misrouted into the CSV parser

**Severity: medium (bad diagnostics, unreachable guard).**

`parse_nightscout` picks its branch with `_is_nightscout_entries_json`
(`format_parser.py:1413-1417`):

```python
stripped = data.strip()
return stripped.startswith("[") and '"sgv"' in stripped[:2000]
```

A Nightscout account with no data returns `200` with body `[]`. That starts with `[` but
contains no `"sgv"`, so the sniff returns `False` and the JSON falls through to
`_parse_nightscout_entries_csv`. Verified:

```
"[]"                                    -> MalformedDataError: Missing required column:
                                           'Glucose (mg/dL)'. Got columns: ['[]']
'{"status":401,"message":"Unauthorized"}' -> MalformedDataError: Missing required column:
                                           'Glucose (mg/dL)'. Got columns:
                                           ['{"status":401', 'message":"Unauthorized"']
```

Two consequences:

- The purpose-built `ZeroValidInputError("Nightscout entries JSON is empty")` guard at
  `format_parser.py:1436` is **unreachable** from a real API response.
- A server's own error text survives only as a mangled polars column name, so an operator
  debugging a 401 is shown a missing-CSV-column error naming a column nobody asked for.

There is a latent variant too: the sniff reads only `stripped[:2000]`, and the downloader
writes the file with `indent=2` (~300-400 chars per entry). An entries array whose first
~5 records are `cal`/`mbg` rather than `sgv` is misrouted the same way.

Fix: decide the branch on whether the payload *parses as JSON*, not on a substring sniff,
and let the empty case reach the existing `ZeroValidInputError`.

---

## S5 — Smaller things noticed in the same code

- **`entry.get("sgv") or entry.get("glucose")`** (`format_parser.py:1443`) — an `sgv` of `0`
  is falsy, so the row falls through to `glucose`/`None` and is dropped. `0` is not a
  physiological reading, but it is a real sensor-error value, and silently dropping differs
  from what the code reads as intending. `if "sgv" in entry` is the intended test.
- **`if entry.get("type") != "sgv": continue`** (`format_parser.py:1440`) — uploaders that
  omit `type` (some xDrip+/Loop configurations) have every row skipped, producing
  `ZeroValidInputError("No SGV entries found")` on a payload that carries `sgv` and
  `dateString` on every record.
- **The temp directory is never cleaned up.** `from_nightscout_url` with the default
  `output_dir=None` does `tempfile.mkdtemp(prefix="nightscout_")` (`format_parser.py:1938-1940`)
  and never removes it. Every call leaves raw entries + treatments + profile JSON — a
  patient's full CGM record — in `/tmp` indefinitely. A `TemporaryDirectory` context manager
  when the caller did not ask for persistence would fix it.
- **`profile.json` is fetched, `raise_for_status`-ed, then discarded.** `from_nightscout_url`
  unpacks it as `_` (`format_parser.py:1944`) and `from_nightscout_exports` documents
  `profile_path` as accepted-and-ignored. An instance whose permissions allow `entries` but
  deny `profile` therefore fails the entire import on an endpoint whose result is thrown away.
- **No library exception type wraps network failures.** Connect errors, timeouts, HTTP status
  errors and `json.JSONDecodeError` all escape raw, so a caller must import `httpx` itself to
  catch them. `sugar_sugar/components/startup.py:1107-1129` does exactly that, and has to
  match on `type(exc).__name__` substrings to classify. Relatedly, `HTTPStatusError` messages
  embed the full request URL — including the `?token=` query parameter — so any caller that
  logs the exception leaks the Nightscout read token.
- **`parse_nightscout` and `_process_nightscout` have different error contracts.**
  `_process_nightscout` (`format_parser.py:1787-1791`) wraps its body in
  `except Exception -> MalformedDataError`; `parse_nightscout` (`1813-1845`), which is what
  `from_nightscout_url` actually reaches, has no such wrapper. That asymmetry is why issue 1
  surfaces as a raw polars `ComputeError`.
- **The package ships no `Project-URL` or `Home-page` metadata**, so `pip show cgm-format`
  gives no repository link. The only pointer is a CI badge in the README body.

