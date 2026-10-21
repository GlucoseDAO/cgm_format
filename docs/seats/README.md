# Seats: standing roles a session can take in this repo

A **seat** is a fixed role a Claude Code session takes for its whole run. The seats here run on the
seats kit (`agent-seat-blueprints`): its plugin is pinned in this clone, and its base, loaded at every
session start, holds what every seat knows: messages, the desk, unattended runs, who speaks, one
writer per file. Every seat also loads `CLAUDE.md`, so the house rules are shared. What differs is the
seat's own priming file: why it exists, what it owns, how work reaches it, its grant and what it
never does.

Take a seat with `/seats:seat <seat>` in a session started in this checkout. A session that has not
taken one does the task it was given and takes no seat's work.

| Seat | Priming | Notes | Reached by | What it does |
|---|---|---|---|---|
| **triage** | [TRIAGE.md](TRIAGE.md), on the kit's `triage` role | none: its state is `docs/FEEDBACK.md` | the inbox watchers; a relayed consumer report | Answers consumer feedback: triages, replies in the document, routes, archives. Commits; never pushes or releases. Never designs a major change unasked. |
| **lead** | [LEAD.md](LEAD.md), on the kit's `lead` role | [LEAD_NOTES.md](LEAD_NOTES.md) | the person; an RM item from triage; the plan watcher | The person's own coding sessions: the roadmap, the design, the build, the releases up to a GitHub release (PyPI is the person's). Also this clone's kit pin and leak gate, there being no tooling seat. |

`.blueprints.toml` declares the two seats and who may wake each one. The team file, who is on the team
by chat account, lives on the person's desk, the repository being public. A fault in a kit tool goes to
the kit's issues (the skill `report-to-kit`): the kit's tooling seat is this repository's.

## Notes files, and when a seat has none

A seat whose working state is not already in a tracked document gets `<SEAT>_NOTES.md` beside its
priming file: what is in flight, what it waits on, what it learned and has not yet written anywhere
durable. It is rewritten to the present before a reset, not appended to, and committed by path. It
holds state, not rules; a lesson still true next week moves into the priming file or `CLAUDE.md`.

The lead has one. Triage has none, on purpose: the loop derives its state from `docs/FEEDBACK.md` by
fingerprinting the reporter's text, and a notes file would be a second copy of state that already has a
home.

## What stays true for every seat

Repeated from `CLAUDE.md` because these are the ones a seat is most likely to meet:

- Never `git add -A` or `git add .`; stage explicit paths and read `git status` first. Commit as you go;
  never push, tag, publish or release unless your grant says so (the lead's does, short of PyPI). No
  `--amend`, rebase, `reset --hard` or force. Never `git stash drop` or `git stash clear`.
- Another seat's uncommitted file is work in flight: commit your own paths by name; never stash,
  revert or commit theirs. The one exception is a consumer's item left uncommitted in the inbox, which
  triage commits as it stands.
- Never `uv pip install`; never bare `python`. The kit's scripts run through the kit's `bin/py`, never
  under `bash`.
- Never edit or commit a sibling repo beyond appending an item to its `docs/FEEDBACK.md`.
- A real export, a token or a person's name never enters a tracked file, a commit message or a message
  to another seat; fixtures are synthetic.
- Never resolve a contradiction between two rules by inference: run the questionnaire in `CLAUDE.md` §1.
