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
routing table, and what may be done unattended. The generic loop it adapts is the seats kit's
`triage` role, which this repo's triage seat fills ([seats/TRIAGE.md](seats/TRIAGE.md)). Nothing
below depends on the tooling — the ledger reads what is written, so a reply typed by hand in the
right shape is a first-class one.

## The ledger

Dependency-free scripts in the seats kit's `classes/triage/`, run from the kit copy this clone pins
(`$BLUEPRINTS_KIT`, exported by the kit's plugin); the runbook's §1 has the rest. They are
stdlib-only Python, import nothing from the package, and so are the one place in this repo that does
not need `uv run`:

```sh
sh "${BLUEPRINTS_KIT:?}/bin/py" "$BLUEPRINTS_KIT/classes/triage/triage-state.py"             # every item and its verdict
sh "${BLUEPRINTS_KIT:?}/bin/py" "$BLUEPRINTS_KIT/classes/triage/triage-state.py" --pending   # just the ones needing work
sh "${BLUEPRINTS_KIT:?}/bin/py" "$BLUEPRINTS_KIT/classes/triage/triage-state.py" --next      # the next id, over BOTH files
sh "${BLUEPRINTS_KIT:?}/bin/py" "$BLUEPRINTS_KIT/classes/triage/triage-archive.py" S<n>      # move an answered item, verified
```

Below, a bare `triage-state.py` or `triage-archive.py` means that command. They are Python: **never
hand them to `bash`**, which executes the module docstring as commands and reaches ImageMagick's
`import`.

## Filing an item

Append it at the bottom under a new `## S<n> — <what happened>` heading. **The next id is S6**, and
that pin is a convenience rather than the authority: `triage-state.py --next` computes it
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

Stamp `sha 000000000000` as you write the reply, then run `triage-state.py`: with a marker
present the reply is excluded whole, so the `revised` line prints the section's true fingerprint and
that is the value to paste back. Never copy the sha the ledger prints while the reply is in and the
marker is not — the hash then covers your own paragraphs two onward. `--backfill` computes it the same
way, so it is for replies that predate the ledger and only those. Once the reply is in,
archive with `triage-archive.py S<n>` and add the item's one-line contents entry to
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
