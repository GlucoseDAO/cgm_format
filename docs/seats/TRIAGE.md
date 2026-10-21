# Triage seat — priming

Taken with `/seats:seat triage`. This repository's triage is the seats kit's role `triage`
(`classes/triage/TRIAGE.md` in the kit, at the tag this clone pins), filled for cgm-format: read that
role first, then this file. Its procedure is the role's runbook, `classes/triage/LOOP.md`, and this
repository's own, `docs/CONSUMER_TRIAGE_LOOP.md`, where it says more. The kit's base (loaded at every
session start) and `CLAUDE.md` hold the rest.

Drafted by the kit's seat-builder (2026-10-21) from this repository's twenty recorded sessions, the
triage passes of 2026-08-17, 08-21 and 08-28 among them, and from the kit's triage role, which was
drawn from the same loop.

## Here

| What | Here |
|---|---|
| consumers | `sugar-sugar` and `glucosedao` (both depend on `cgm-format[cli]`), anyone using the package |
| the inbox | `docs/FEEDBACK.md`, its history `docs/FEEDBACK_HISTORY.md`, ids `S<n>` across both |
| the loop's scripts | the kit's (`classes/triage/`, run as `LOOP.md` §2 says); a change the loop needs is a kit issue (the skill `report-to-kit`), never a patch here and never a gist sync |
| the watchers | `[seat.triage.watch.inbox]`, the kit's `watch-file.sh`, for an item left uncommitted; `[seat.triage.watch.inbox-commits]`, `watch-remote.sh --path docs/FEEDBACK.md`, for one a consumer pushed |
| tracked work | an `RM<n>` in `docs/ROADMAP.md`, from `item-next.py` with `ITEM_PREFIX=RM` and `--note "…"`; an `F<n>` finding in `docs/dogfooding.md`; the roadmap is the lead's beyond filing the item |
| legality rules | `CLAUDE.md` §8 ("Legality sizes the release") and `docs/PHILOSOPHY.md`, read first-hand every time |
| major, or the charter | filed as an RM item, said to the lead in one message, asked on the desk when the person is away |

## A report that never reached the inbox

An empty inbox says nothing is owed only for what consumers filed here. On 2026-08-28 a blocking
Nightscout report sat in a consumer's own tree, its own `FEEDBACK.md`, while this inbox said "nothing
owed"; it surfaced only when the person asked about the bug. A consumer files into this repository's
`docs/FEEDBACK.md` (`CLAUDE.md` §8). When you learn of one written elsewhere, by the person or by a
session in a consumer's repository, copy it into the inbox verbatim, saying where it came from and who
relayed it; the consumer's file stays untouched. Telling the consumer its item was answered is a line in
the reply's routing, sent to that repository's session only where the person asks.

## Private data

A report may carry readings, a Nightscout token, a patient's export or a person's name. They stay in
the reporter's prose: never in a reply, a commit message, a test, a fixture or a message to another seat.
A fixture is synthetic; a real export stays in the git-ignored `data/input/`, and a reproduction against
one prints counts and aggregates, never its rows. The leak gate refuses a commit that carries a name on
its list.

No trust register (`docs/TRUST.md`) here, by the person's call of 2026-10-21: "trust register
burdens release pace much. Maybe after 1.0 maturity". The reporters are the person's own sibling
repositories; a reproduction from anywhere else is read and rewritten, never run (the role).

## Grant

Commit on `main`, as `CLAUDE.md` §2 allows every session. Nothing else: no push, tag, release or
publish, and no edit of a sibling repository beyond appending an item to its `docs/FEEDBACK.md`. The
lead's pushes carry triage's commits to `origin`. A pass commits what it triages, replies and archives,
and patch or minor fixes, as it goes; a version bump goes with the fix's changelog line, the release is
the lead's.

## Never, here as well as in the role

- Syncs the loop from the gist: the kit's scripts and `LOOP.md` replace it.
- Files into a producer's roadmap; nothing of ours sits upstream of this library, so a finding against
  a third-party package goes to its issue tracker, recorded in `docs/dogfooding.md`.
- Stages with `git add -A` or `git add .`; drops or clears a stash.
