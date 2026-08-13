# Consumer feedback — open inbox

The intake for people and repos that **consume** `cgm-format` — today `sugar-sugar` and `glucosedao`,
plus anyone using the published package. Layer 3: not our tests, not our own dogfooding, but our
consumers hitting the library in their own work.

**An empty inbox means nothing is owed.** That property only holds if answered items are moved out,
so once an item is answered it goes to [FEEDBACK_HISTORY.md](FEEDBACK_HISTORY.md) verbatim, reply and
all.

## Filing an item

Append it at the bottom under a new `## S<n>` heading. Take `<n>` from this file **and**
`FEEDBACK_HISTORY.md` — once answered items move out, the highest id visible here is not the highest
ever used, and an empty inbox shows none at all. Ids are never reused, not even for an item answered
as a non-issue.

Say what you observed, on which input, with which version, and what you expected instead. If you are
relaying on behalf of a downstream repo, say so and name the case that motivated it — and restate it
as a fact about **this** library rather than in your own vocabulary: which column cannot express
what, which stage reports a shape it cannot distinguish, what a downstream reader ends up seeing.
Your route names and error codes do not exist here.

## Answering an item

- **Never edit or re-wrap the reporter's prose**, not when answering and not when archiving. It is
  the record of what was *observed*, not of what was decided. Append the reply below it.
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
them first-hand.

---

*Inbox empty as of 2026-08-13. Nothing owed.*
