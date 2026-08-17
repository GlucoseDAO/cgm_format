# Consumer feedback — answered

Items moved out of [FEEDBACK.md](FEEDBACK.md) once answered, verbatim: the reporter's prose exactly
as filed, with the reply as its own paragraph at the top of the section. Same split as ROADMAP vs
ROADMAP_HISTORY, and for the same reason — an inbox only grows, and unanswered items become
invisible inside answered ones.

The reporter's text is never edited or re-wrapped on the way here. If it needed clarifying, that
belongs in the reply.

**Ids are never reused**, including the ones archived here and including items answered as
non-issues — the reply is part of the record and a recycled id collides with it. The next id is
computed from this file and `FEEDBACK.md` together: `./scripts/triage-state.py --next`.

## How items arrive

The loop that produces them is [CONSUMER_TRIAGE_LOOP.md](CONSUMER_TRIAGE_LOOP.md).

Moved by `./scripts/triage-archive.py S<n>`, never by hand: the one property that matters — the
reporter's prose arriving byte-for-byte — is easy to break in a copy-paste and the tool verifies it,
comparing every fingerprint before and after the write and refusing the write if one moved.

Three things the tool does not do, because each is editorial or a check of a different property:

- **The contents line below is hand-written.** Naming what an item was and how it ended is a
  judgement, and a generated line would be a worse version of the thing the index exists for.
- **A section that travelled without a group heading gets told about, not given one.** The inbox's
  own title is not a group, so an item filed under no heading arrives bare; give it a `# ` heading
  here (who reported it, and when) if it would otherwise read as part of the group above.
- **It verifies the move, not the verdict.** An item the ledger still calls `new` archives without
  complaint. The lint is the ledger pointed at *this* file — `./scripts/triage-state.py
  docs/FEEDBACK_HISTORY.md` — where a well-formed archived item reads `current`, so anything reading
  `new` or `unmarked-reply` was archived unanswered or lost its marker in transit. Two seconds, and
  it is the only thing between a silent mis-archive and a permanently lost item.

Archive a whole batch in one pass: sections append in the order given, so a group archived in two
batches ends up with its heading twice.

## Contents

*One line per archived item, under 80 characters:
`- **S<n>** <what it was> — <verdict> (<roadmap item or release, if any>)`. It is an index, not a
second copy of the reply — the detail lives in the section's own reply, the one place it cannot
drift from the answer. An item missing from this list is how a tracked item becomes unfindable, and
inbound links are usually file-level, so a reader following one lands on the live inbox and needs a
pointer onward.*

---

*Empty. Established 2026-08-13; nothing has been answered and archived yet — S1 is open in the
inbox. Delete this line when the first item lands below it.*
