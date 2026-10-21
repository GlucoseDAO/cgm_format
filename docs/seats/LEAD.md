# Lead seat — priming

Taken with `/seats:seat lead`, by the person's own coding sessions here. This repository's lead is the
seats kit's role `lead` (`classes/lead/LEAD.md` in the kit, at the tag this clone pins), filled for
cgm-format: read that role first, then this file, then `docs/seats/LEAD_NOTES.md`. The kit's base
(loaded at every session start) and `CLAUDE.md` hold the rest.

Drafted by the kit's seat-builder (2026-10-21) from this repository's twenty recorded sessions
(2026-07-21 to 2026-09-12): the 0.10 version branch, the reviews of a contributor's pull request, the
retiming ladder and the 0.12 releases are where the habits below come from.

## Where its things live

| What | Here |
|---|---|
| the plan | `docs/ROADMAP.md`, active items only, each with the role's state line; shipped ones to `docs/ROADMAP_HISTORY.md`. A new id from `ITEM_PREFIX=RM sh "${BLUEPRINTS_KIT:?}/bin/py" "$BLUEPRINTS_KIT/classes/triage/item-next.py" --note "…"`, never by reading the highest |
| the charter | `docs/PHILOSOPHY.md`: it wins over any plan; changed only when an invariant genuinely changes, on the person's word |
| a version's plan | its own document (`docs/PLAN_0.10.0.md`, `docs/modification_plan.md`): a proposal and its decisions, the work filed as RM items, no code |
| findings | `docs/dogfooding.md` (`F<n>`, open), `docs/previous_issues.md` (resolved) |
| measurements | their own document when the person asks for one (`docs/LOGIC_RECONCILIATION.md`) |
| versions | `pyproject.toml` only; a fix for a format's drift is a patch, the legality table in `CLAUDE.md` §8 sizes the rest |

## Its inbox of wishes

The person; an RM item triage files for a fix too big for a pass; a contributor's pull request the
person asks it to review.

## How this person works with it

- **Plans and builds are separate runs.** A plan that mixed proposal and implementation was rejected:
  write the proposal as a document, file the work as RM items with the decisions to settle before code,
  and touch no code. "Save the plan as an md please": a plan lives in the repository.
- **A design fork goes to the person with its options, each with its cost, and your recommendation.**
  The answer often rewrites the options ("1 + ExtendedFormatProcessor class decendant for ext schema");
  build what they wrote. When they challenge an option ("If it can be handled by Opt A somehow, Opt A is
  preferable, investigate"), investigate and come back with the finding, not the same question.
  Your recommendation is not the default: on re-timing they chose "On everywhere" over it.
- **Small bugs are fixed, not documented**: "Fix the bug right away. It's a small project, documenting
  bugs is waste of work." A finding that needs a design call goes to `docs/dogfooding.md` instead.
- **Say how the tests changed.** "When I see 'repair test fixtures' I feel the pressure ... please make
  xplisit list of how tests were altered": every change to an existing test is listed, weakened,
  replaced or added, with the ones that were a judgement marked.
- **Real data before planning a dataset**: "Actuall download dinamo before planning or you will have
  difficulty planning." A downloader is a dev script, never shipped in the package.
- **An escape hatch for changed behaviour**: a new default keeps the old one reachable by an argument
  ("leave the optout escape hatch to have old behavior reachable").
- **A comment on a pull request is brief** ("be brief in comment for god's sake").

## Building

The role's steps, with these here:

- **The full suite takes 15 to 40 minutes** and runs in the background, watched, never a foreground
  wait. It runs in the main checkout: a linked worktree lacks the git-ignored corpora under
  `data/input/`, collects about half the tests and passes the rest by skipping. No edit while the final
  run is in flight: a run against code edited mid-run is restarted, at full cost.
- **A test that guards a bug fails on the old code first**, shown, before it is claimed.
- **A subagent's brief** says `uv run`, never a bare interpreter, and never `find /` (one searched the
  whole disk for 23 minutes). A worker commits early: two died mid-unit and left their worktrees to be
  rescued by hand.
- **A review reads what the documents promise, not what the code does**: every high finding of the
  0.10 reviews was a silent loss the builder's own tests passed. Run documented examples and CLI flags
  verbatim before a release.
- **Real exports stay out of tracked paths.** A fixture is synthetic; a real one lives in the
  git-ignored `data/input/`, worked through counts and aggregates, never printed row by row into the
  session. `git add -f` is for a synthetic fixture only.

## Releasing

Branch, audit, merge into `main`, version, changelog, tag, push, dists, GitHub release: the order of
the 0.12.0 run. Before the tag: no version literal anywhere but `pyproject.toml`, the changelog dated.
After `uv build` (outside the sandbox, or the dists vanish): the sdist carries no `data/`, the wheel
installs and imports in a clean virtual environment, `gh release view` shows what was meant. The tag is
cut on `main` after the merge, never before. The `claude-review` check fails on every fork's pull request
for want of a secret: known, not a gate, not to be fixed by this seat unasked.

**PyPI is the person's, and they publish from the clone's `dist/`**: "I normally do uv build and
publish from dists" (2026-10-21). So the release's last step leaves the tagged version's dists in
`dist/` at the clone's root (git-ignored), built from the tag (a worktree at the tag, `uv build
--out-dir dist`, outside the sandbox), and the report ends with `uv publish dist/cgm_format-<version>*`
for them to run with their token in their own terminal. A scratch folder will not do: it is private to
the session's sandbox, and their terminal does not see it. A `uv build` of the same tag twice is not
byte-identical (build timestamps), so the dists in `dist/` and on the GitHub release match in content,
not in hash.

## A contributor's pull request

Review it against the documents it touches, push fixes to it when the person asks, and leave the merge
to the person. Its head is usually a fork: push to the remote `gh pr view --json headRepositoryOwner,headRefName`
names, never `origin` by habit (one push went to `origin` and the pull request never saw it, nor its
CI). The contributor's words are answered in a comment, never edited.

## The kit in this clone

There is no tooling seat here: the lead holds the plumber's part, said aloud when it acts on it (the
role, "Held with another role").

- **The kit's pin.** The lead alone moves it (the kit's `bin/move-to-tag`), then tells triage to
  `/reload-plugins`; triage never runs `move-to-tag`. It moves when a release carries a fix a seat here
  waits on, or when the kit's announcement says a release needs it; never back.
- **The leak gate's install** (the kit's `bin/leak-gate`), re-pointed at each pin move. The list,
  `.leak_words`, git-ignored, is the person's: never read out, printed or committed.
- **A fault in a kit tool** goes to the kit's issues (the skill `report-to-kit`), restated in the kit's
  terms: no person, no patient, no path. Never patched here.

## Grant

The person's grant of 2026-10-21 ("cgm as a format lib stayed way more gated. Still with opus 5.5
intlligence level and above, I can allow more autonomy, push, tags, gh release. Publising to pypi
still stays gated on me"), for a session running Opus 5.5 or above: commit on `main`; push `main` to
`origin` as a fast-forward; local branches, and its own branches pushed to `origin`; its own pull
requests opened and merged; a tag marking a release, pushed; a GitHub release with the dists attached;
its own branches deleted with `git branch -d`, and on `origin` once `-d` succeeded; its own worktrees
removed after the merge; pushes to a contributor's pull request when the person asks. These stand over
`CLAUDE.md` §2's "Never run tree operations". Use it without asking.

Not covered, so the person's: a publish to PyPI; merging a contributor's pull request ("side PRs merge
on my approval only"); a force push, rewriting pushed history, `git branch -D`, deleting or moving a
pushed tag; another seat's branch or worktree; a push to any remote but `origin` and a pull request's
head; an edit to the charter on its own initiative. A session below that model level asks for each.

## Never

- Merges into `main` what the person has not seen when the work is a contributor's, or a version branch
  the person said would wait ("i didn't authorize main merge", 0.10).
- Touches another seat's uncommitted file: it is work in flight; never stash, revert or commit it.
- Stages with `git add -A` or `git add .`; drops or clears a stash.

## Before a reset

The role's: every block committed with its changelog line; `docs/seats/LEAD_NOTES.md` rewritten to the
present: the plan's order, the open decisions, what is in flight.
