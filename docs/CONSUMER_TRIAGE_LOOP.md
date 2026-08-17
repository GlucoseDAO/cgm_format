# The consumer triage loop

How to run [FEEDBACK.md](FEEDBACK.md) as a conversation rather than an inbox: a watcher notices when a
consumer has finished writing, an agent triages what is new, and every item gets a **maintainer reply
written back into the document itself**. The document is the transcript, and it is also the state —
there is no queue, no database and no side-car ledger.

**The pattern is not ours.** It is published as a generalized gist —
<https://gist.github.com/winternewt/54b94bda01812be937b892146d1bb254> — with the three scripts
parameterized (`INBOX` / `HISTORY` / `PREFIX`) and every repo-specific reference stripped, and it was
adopted here on 2026-08-17. This file is the local half: which files, which charter, which release
rules, what an agent may do unattended in *this* repo. If you change the **pattern** — the algorithm,
a script's contract, a gotcha in the mechanism — it belongs in the gist as well, or the next repo to
adopt it inherits a bug we already fixed. If you change something only true of cgm-format, it stays
here.

**The live document holds only what is unanswered.** An item moves to
[FEEDBACK_HISTORY.md](FEEDBACK_HISTORY.md) once its reply is written, the same split as
[ROADMAP.md](ROADMAP.md) / [ROADMAP_HISTORY.md](ROADMAP_HISTORY.md) and for the same reason: an inbox
only grows, and unanswered items become invisible inside answered ones. So an empty inbox means
nothing is owed — a property worth having, and destroyed the moment answered items are left in place.

This is **layer 3**. Layer 1 is the test suite (`CLAUDE.md` §6): does the code do what it was told.
Layer 2 is dogfooding ([dogfooding.md](dogfooding.md), `CLAUDE.md` §7): is the shipped surface usable,
and what is missing. Layer 3 is somebody else hitting the library in their own work, which is the only
layer that reports things we had no reason to look for.

---

## 1. Setup

Three scripts, all in `scripts/`, none packaged and none importable:

| | |
|---|---|
| `scripts/watch-inbox.sh` | debounced watcher: one line of stdout when the file stops changing. The only one that is really bash |
| `scripts/triage-state.py` | the ledger: which sections are new, revised, or already answered. Takes a path, so it reads the history file too; `--next` prints the next unclaimed `Sn` |
| `scripts/triage-archive.py` | moves answered sections into the history file and **verifies** each fingerprint survived the move |

They must stay in one directory: the archiver resolves the ledger relative to its own path, and the
watcher shells out to it the same way. Their only divergence from the gist is that `INBOX` / `FILE`
default to this repo's `docs/FEEDBACK.md`, derived from the script's own location rather than `$PWD`,
so a run from `src/` or `tests/` reads the real inbox instead of nothing.

**Run the two Python ones, never `bash` them** — `./scripts/triage-state.py` or
`python3 scripts/triage-state.py`. §6 has what the mismatch costs.

They are stdlib-only Python 3.11+ and bash, and import nothing from the package, so they are the
sanctioned exception to the always-`uv run` rule in `CLAUDE.md` §2: the workspace environment is not
what they need, and a `uv run` wrapper around them only adds a way to fail.

### Arming the watcher

Nothing arms it for you. Under this harness the `Monitor` tool turns each stdout line into a
notification that re-invokes the agent:

```
Monitor({
  command: '/data/sources/glucosedao/cgm_format/scripts/watch-inbox.sh',
  description: 'FEEDBACK.md settling',
  persistent: true,
})
```

`persistent: true` keeps it alive for the session; `TaskStop` cancels it. It reacts only while the
session is open. **Editing the script does not reach a running monitor** — bash reads a script
incrementally — so `TaskStop` and re-arm after changing it. Nothing needs installing: `inotify-tools`,
`entr`, `fswatch` and python `watchdog` are all absent from this machine, and `stat` polling is enough
at this cadence.

**Our watcher has no branch guard**, unlike the one in `just-dna-format`, and does not need one: this
loop does not commit (§5), so it cannot put unattended work on top of whatever the user is mid-way
through on a branch. If the commit rule here ever changes, the branch guard has to arrive in the same
change.

**Hooks cannot do this job.** Claude Code hooks fire on the agent's own lifecycle (`PreToolUse`,
`PostToolUse`, `SessionStart`, `Stop`); a consumer editing a file triggers none of them. The trigger
has to be a process watching the file.

### The cooldown is sized for an agent author

Consumers write these notes through an agent, so the shape is a burst — five edits in a minute — with
gaps wherever the agent stops to read or probe something. A 60-second timer fires in the middle of
such a run and triages half a report. `COOLDOWN` defaults to **150s**, `POLL` to 10s, so an event
lands 150–160 seconds after the last write. Consecutive saves inside the cooldown collapse into
**one** event, because each mtime bump restarts the timer. The only cost of waiting is latency; the
cost of firing early is a reply to a half-written item.

### First run

The watcher seeds from the current mtime with the dirty flag clear, so **it never fires for a change
that predates it**. A standing backlog stays quiet — run the ledger by hand at startup, and after any
context clear:

```sh
./scripts/triage-state.py            # every section and its verdict
./scripts/triage-state.py --pending  # just the ones needing work
```

---

## 2. How state is derived

**The document is the ledger.** A triaged section carries a marker inside its reply, holding a
fingerprint of the consumer's own text:

```
**Status — accepted, filed as RM14.** … <!-- triaged: <version> · sha <12 hex> -->
```

The fingerprint covers the section body with **every `**Status` paragraph and every marker removed**,
so it describes what the consumer wrote and never what we replied. Lines are right-stripped, blank
runs collapsed and a trailing horizontal rule dropped, so trailing whitespace, reflowing and the `---`
a reporter puts before their next section do not count as changes. Four verdicts follow:

| verdict | meaning | action |
|---|---|---|
| `new` | no reply, no marker | triage it |
| `revised` | marker present, fingerprint moved | the consumer edited an answered item — re-triage |
| `unmarked-reply` | answered before the ledger existed | `--backfill` stamps the marker |
| `current` | marker matches | nothing to do |

### Why not git

Two reasons. The loop **must not commit** here (§5), so a `HEAD` baseline would never advance and
every run would re-triage everything. And a consumer may well commit their own addition, at which
point `git diff HEAD` is empty and the loop sees nothing at all — that one is fatal on its own and
survives any change to the commit rule. `git diff` and `git log -p` stay useful for *reading what
changed*; correctness never depends on them.

The in-document ledger has properties no side-car state has: it works on an uncommitted tree,
survives anyone's commits, travels with the repo, is legible to a human scanning for the backlog, and
cannot drift out of sync with the replies it describes.

### Self-firing is not a loop

Writing a reply bumps the mtime, so the watcher fires again. That run finds nothing pending — the
fingerprint excludes the reply — and no-ops. Expect the second notification; it is the mechanism
working.

---

## 3. The algorithm

### Step 0 — establish what already shipped

**Before reproducing, and certainly before designing.** `new` in the ledger means *no reply in the
document*, never *no work done*. On the first full run of this loop in the repo it was built for, two
of eleven items were already fixed — one by a release, one by a code block whose comment named the
item — and a third's preferred option had shipped from a different report. Answering those as though
they were open would have designed a feature that existed.

Cheap and mechanical here: grep the item's symbols in `src/cgm_format/`, then
[CHANGELOG.md](CHANGELOG.md), [ROADMAP_HISTORY.md](ROADMAP_HISTORY.md),
[previous_issues.md](previous_issues.md) and [dogfooding.md](dogfooding.md) for its subject;
`git log -S "<a phrase from the fix>"` finds when a guard landed. Check the reporter's version against
`pyproject.toml` — the fix may be in a release they have not upgraded to.

### Step 0b — reproduce before classifying

Compare the claim against the **code**, not only against the docs, because the docs are often the
thing that is wrong. This is the only step separating a real defect from bucket **(b)**.

Reproduce with the library, on real data. `data/input/` holds committed vendor fixtures; the research
corpora are behind `CGM_FORMAT_*_DIR` env vars and `scripts/download_*.py`. If the reporter names a
corpus we do not have locally, say so in the reply rather than reasoning about it — an unreproduced
finding answered as though it reproduced is worse than an honest "not reproduced here, and why".

**Probe the behaviour, not only the sentence: the probe is where the adjacent defect turns up.** And
**scope a negative finding to what you actually looked at** — "the vendor publishes no such column"
that was true of one export and false of the vendor becomes a permanent false constraint.

**A bucket-(b) verdict is not the cheap outcome.** "Nothing is wrong here" has to be *shown*: what was
probed, on which input, and what it did instead. A reply that cannot show its work is worthless.

### Step 1 — legality, first-hand

**Read [PHILOSOPHY.md](PHILOSOPHY.md) and `CLAUDE.md` §8 yourself.** Never delegate this to a
subagent: a summary of a charter drops the qualifier the decision turned on, and this step decides
whether a repair is legal at all. It is the one part of the loop that cannot be automated away.

**Legality sizes the release; severity only orders the queue inside it.** A severe finding whose fix
is a new optional column is still a minor; a trivial one whose fix is a rename is still a major.

| change | release | why |
|---|---|---|
| a new optional column on `CGM_SCHEMA` / `CGM_SCHEMA_EXTENDED`, a new `UnifiedEventType` member, a new `Quality` flag beside the old ones, a new vendor format, a new parameter with a default | **minor** | additive; nothing a consumer already reads changes meaning, and frames they already have stay valid |
| pure legibility — a warning, a count, an error *message*, a doc | **patch** | changes no verdict on a path that already behaved that way |
| a **new** flag, parameter or alias beside the old one | **minor** | additive; the superseded name keeps working |
| removal, promotion to required, a dtype change — **including a rename** | **major** | breaks a reader, or invalidates parquet and CSV already written against the old shape |

A rename is a removal plus an addition, so the addition being legal does not make the rename legal.
And when changing a shape a consumer reads, **add rather than redefine**: a consumer already
compensating for the old meaning breaks *silently* otherwise.

**The invariants are the trap, and they are not compatibility.** Idempotency, losslessness,
commutativity of the grid ops and deterministic row order (`CLAUDE.md` §2.1) are correctness. A repair
can be perfectly legal on the table above and still be inadmissible because it breaks one, and that
routes to **(d)**, not **(a)**. The shapes to watch for:

- A repair that fills a value to silence a downstream complaint. Null glucose means *the sensor did
  not say*, not zero; the `IMPUTATION` flag exists to make a filled value visible, and a fill that
  skips it is a losslessness break wearing a bug fix's clothes.
- A repair that drops or edits an original row. `interpolate_gaps` only adds; `synchronize_timestamps`
  keeps every row. "Drop the duplicate" is a schema decision, not a patch.
- A repair that changes row order. Order is part of the emitted bytes, so it moves every parquet and
  CSV a consumer regenerates, even when every value is identical.
- A repair whose only evidence is that the pipeline is still idempotent. **A determinism gate is not a
  correctness gate**: a parser mapping the wrong vendor column to `glucose` passes every idempotency
  test we have.

### Step 2 — route

| | verdict | lands in | must contain |
|---|---|---|---|
| **a** | real, repairable, legal | [ROADMAP.md](ROADMAP.md) as `## RMn` | `**Severity:** … · **Status:** open · **Owner:** …`, the motivating `Sn`, and the legality line |
| **b** | non-issue | the reply only | **what was probed and did not reproduce.** Never a bare "works as intended" |
| **c** | documentation defect | the doc, fixed in the same pass | the reply naming the file changed |
| **d** | real, no acceptable repair | [ROADMAP.md](ROADMAP.md), **status open only** | the paragraph saying *why each candidate repair is wrong* |
| **e** | not ours to fix | the third party's issue tracker, restated in their terms, referenced from [dogfooding.md](dogfooding.md) | the reply naming the upstream issue, plus whatever half *is* ours, plus the defensive mitigation |

**Bucket (e) means a third party here** — Polars, Frictionless, a vendor's export format, a corpus
publisher. Nothing of ours sits upstream of this library, which is why there is no
`docs/<upstream>-pending-fixes.md`. Reformulate rather than forward: a note written in our vocabulary
reads as somebody else's problem in their tracker, so restate it as a fact about the thing they own.
An item is often **(e)** *and* **(a)** — their half filed, our defensive half shipped — and both
belong in the one reply.

**An item filed as a documentation gap usually has a code half — look for it.** A reporter describes
where *they* got stuck, which is a fact about the docs; what stuck them is often a surface that could
have told them: a missing error message, a warning that names no substitute, a guard that never fires.

**A wrong reporter conclusion is a place to look for our own defect.** Bucket (b) on the report with
bucket (c) underneath it is a common pairing, and the (c) half is what stops the next person filing
the same item.

**Read what the reporter already built.** A consumer who fixed their half locally has given you
evidence about the right shape, and their own argument against their first option is often the reason
the item was filed rather than patched.

Bucket **(d)** is the one that earns its keep and the one an unattended agent does worst. It is the
`fix it` / `surface it` line from `CLAUDE.md` §7: surface anything whose obvious repair is itself a
design decision — a schema change, a threshold change, a new quality flag — and say why each candidate
fails. The best entries are the ones where a candidate is not merely unwanted but *illegal* under our
own rules.

If **(a)** ships in the same pass rather than being filed: code plus a real-data test, a
[CHANGELOG.md](CHANGELOG.md) entry under the version it lands in, and the item goes straight to
[ROADMAP_HISTORY.md](ROADMAP_HISTORY.md) with its rationale rather than being filed and immediately
closed.

### Step 3 — write the reply

`**Status —**` is the house idiom; do not invent a second one. It goes **first in the section,
immediately after the heading** — the ledger reads a reply as running from that paragraph to the
marker — and says four things: the verdict, where it landed (an `RMn` link, a doc, or a shipped
version), what was actually reproduced, and what the reporter should do now.

```markdown
## S<n> — <what happened>

**Status — accepted; the parser half shipped in <version>.** Reproduced: <what was run, on which
input, and what it did>. <Where the rest landed, and why any candidate repair was rejected.>
<!-- triaged: <version> · sha <the 12 hex digits the ledger prints for this section> -->
```

Take the sha from `./scripts/triage-state.py`, never by hand. Keep example ids as placeholders — a
heading like `## S9` in a doc the ledger reads becomes a phantom section and pushes `--next` past it.

**Append, never edit.** The reporter's prose is evidence and stays byte-for-byte; the reply is added
above it, and nothing in the report is re-wrapped, corrected or trimmed, not even a factual error —
that is what the reply is for.

One reply may cover several sections. The ledger understands that shape: a `**Status` paragraph in a
`# ` preamble that names ids answers them all, and `--backfill` marks each covered section
individually, since one paragraph cannot carry several fingerprints.

### Step 4 — move the answered item to the history file

```sh
./scripts/triage-archive.py S1 S2 [--dry-run]
```

It cuts each section — heading, reporter prose, reply and marker — out of [FEEDBACK.md](FEEDBACK.md),
appends it to [FEEDBACK_HISTORY.md](FEEDBACK_HISTORY.md) under its group heading, and **verifies the
move**: every fingerprint is compared before and after, and the write is rejected if one changed.

- **Byte-for-byte, and checked rather than intended.** Reflowing one line while pasting is the whole
  failure mode, which is why the tool refuses instead of reporting.
- **`--dry-run` is not a rehearsal.** It returns before the write and so never reaches the fingerprint
  comparison, which is the thing worth rehearsing. Every path is env-driven, so rehearse for real
  against copies: `INBOX=/tmp/copy.md HISTORY=/tmp/copy_HISTORY.md ./scripts/triage-archive.py S1`.
- **Add the contents line by hand, and keep it to one line.** The tool does not generate it: naming
  what an item was and how it ended is editorial. Format
  `- **Sn** <what it was> — <status> (<RM or release, if any>)`, **under 80 characters**, because it
  is a contents list and not a second copy of the reply — the detail lives in the section's own
  `**Status —**` paragraph, the one place it cannot drift from the answer. An item missing from the
  list is how a tracked item becomes unfindable, and inbound links elsewhere are file-level, so a
  reader following one lands on the live inbox and needs a pointer onward.
- **A section filed under no group heading arrives bare, and the tool says so** rather than inventing
  a name: the inbox's own title is not a group. Add a `# ` line naming who reported it and when, or it
  reads as part of the group above.
- **Archive a batch in one pass at the end.** Sections append in the order given, so a group archived
  in two batches ends up with its heading twice. `grep -n '^# '` the history file afterwards.
- **Then lint it**: `./scripts/triage-state.py docs/FEEDBACK_HISTORY.md`. Every archived item should
  read `current`. Anything reading `new` or `unmarked-reply` was archived unanswered or lost its
  marker in transit — the archiver verifies the move, not the verdict.

### Step 5 — hygiene

- **Serial, one item at a time.** `RMn` is a shared counter and two concurrent triages both claim it.
  Read it off the two roadmap files during a long pass, never from memory.
- **Do not commit.** Leave the changes in the tree; committing, branching, tagging and publishing are
  the user's domain (§5). This differs from the same loop in `just-dna-format`, which has a standing
  commit grant — do not carry that grant across repos.
- **Say what was skipped.** Leave an untriaged item `new` rather than writing a placeholder reply. An
  empty verdict is honest; a hedged one is not. Name the skipped ids in the final report.
- **Run the suite after each code fix.** `uv sync --extra dev` once, then `uv run pytest` — the full
  run is ~15 minutes, so give it a 600s+ timeout or background it, and scope to one file while
  iterating. A batch of fixes is only safe to leave uncommitted because the suite stayed green.
- **A new item can arrive mid-pass.** Take it if the context is warm, or leave it `new` — but do not
  let it silently miss the changelog entry the rest of the batch gets.
- **Write one changelog entry for the batch**, not one per item, under the version the batch lands in.
  If the first pass here finds answered work sitting unanswered, put that fact in the entry: a future
  reader needs to know `new` did not mean untouched.

---

## 4. Thresholds — when to call the user

The loop produces roadmap items and patch-level fixes and will do so indefinitely without ever
deciding to build or release anything. Triage answers a consumer; it does not schedule the work or cut
the version. Count off the tree rather than from memory:

```sh
grep -c '^## RM' docs/ROADMAP.md          # open roadmap items
grep -m1 '^version' pyproject.toml        # versus the top heading of docs/CHANGELOG.md
```

- **A repair that needs a major** — a removal, a retype, a rename — is a release-shape decision, not a
  triage outcome. File it, reply, and ask; do not perform it inside an unattended pass however
  obviously right it looks.
- **Ten or more open roadmap items with no build scheduled → ask whether the next minor should
  start.** This is a dev-start trigger, not a scope freeze: a minor keeps taking additive items right
  up until it is cut, so filing continues either way. Re-read the set before asking — an item that
  duplicates another, or that never had a reproduced case under it, should be merged or demoted rather
  than counted.
- **Patch fixes accumulating with no release cut → ask.** The signal is `docs/CHANGELOG.md` carrying a
  version `pyproject.toml` does not. Bumping the version as part of a change that ships is ordinary
  (`CLAUDE.md`); tagging, releasing and publishing never are.
- **Fewer is fine when something is critical** — a false claim in a docstring a consumer could act on,
  a parser silently mapping the wrong column. That is a judgement call and it is the agent's to make.

These numbers are inherited defaults, picked to be roughly right in the repo the loop came from and
not yet calibrated here. They are a trigger, not a law; update them when they drift.

---

## 5. What the loop agent may do unattended

**No standing grant exists in this repo**, and the absence is deliberate rather than an oversight.
`CLAUDE.md` §2 and the user's global preferences both hold in full during a triage pass:

**May**: read anything; run the ledger, the archiver and the test suite; edit source, tests and docs;
write replies; file roadmap items; leave everything in the working tree.

**May not**: `git commit` unless the pass was explicitly asked for one, `git push`, tag, release,
publish, branch, rewrite history, or run any `git stash` operation. Never `git add -A` or `git add .`
— stage explicit paths if staging is asked for at all.

If you corner yourself, say so plainly and offer fix-forward options. A mistake left visible is
auditable; a tidied one is not.

---

## 6. Gotchas

Inherited from the published runbook — each was a real bug in the loop, found in the repo it was built
in, and every one of them is still live in the scripts here because they are the same scripts:

- **A reply ends at its marker, not at the first blank line.** A multi-paragraph reply — the normal
  size for one that says what was probed, where it landed and why a candidate was rejected — otherwise
  leaks paragraphs two onward into the fingerprint, and writing a reply reports the section `revised`
  immediately.
- **A reply can live outside the section it answers.** A block under a `# ` heading that answers three
  items by name means a naive presence test reads three answered sections as new.
- **The marker must not be hashed**, or a block-replied section reads `revised` from the instant it is
  marked.
- **A trailing `---` belongs to the previous section.** A body runs to the next heading, so the rule a
  reporter puts before their new item lands at the end of the item above it; `fingerprint` strips a
  trailing rule, and only a trailing one, so a `---` inside a reporter's own prose still counts as
  theirs.
- **Splitting a wrapped paragraph is a substantive change** and correctly reports as `revised`.
- **A document's own title is not a group heading.** Taking the last `# ` before a section as its
  group appended the inbox's entire front matter into the history file — and the fingerprint check
  could not see it, because fingerprints cover the reporter's prose alone. The general lesson is worth
  more than the fix: **a verifier that checks one property will report success while a different
  property is being broken.**
- **A marker can carry a sha that never matched its section.** Establish that the prose is unchanged
  before restamping by hand; `--backfill` deliberately refuses to, because silently restamping a
  `revised` section is how a genuine re-triage signal gets erased.
- **A preamble line beginning `**Status` is read as a block reply** and marks every id it names
  answered — `**Status:** intake for field notes — S1 and S2 are open` inverts the loop's one job by a
  line of prose nobody would look at twice. The block-reply rule is right, so the fix is on the
  writing side: a blockquote or different wording. The inbox's header was checked against this on
  adoption and is clean.
- **A Python script named `.sh` gets run as bash sooner or later, and `import` is an ImageMagick
  binary.** Bash ignores the shebang, reads the module docstring as commands, and `import hashlib`
  reaches `/usr/bin/import`, which takes its argument as an output filename: the working directory
  grows 0-byte files named `hashlib`, `pathlib`, `re`, `sys`. It is silent, and the debris reads as
  vendored modules. Both tools are named `.py` for exactly this reason.

Found while adopting the loop here, on 2026-08-17:

- **A worked example in the inbox creates a phantom item, and `--backfill` then answers it.** The
  reply template in `FEEDBACK.md`'s header was written with a `## S9` heading first. Run against that
  version, the ledger reported `unmarked-reply S9` — not `new`, because the template's own
  `**Status —` line reads as a reply — `--next` answered `S10`, and `--backfill` stamped the phantom
  `current`, writing its marker onto the example's closing code fence. An item nobody filed, recorded
  as answered, with an id burned in a corpus where ids are never reused. Keep example ids as `S<n>`,
  which `SECTION_RE` does not match, and run the ledger once after editing the header. Reproduced on
  a copy with the heading restored, not reasoned about.
- **The rehearsal and the lint both behave as documented on this repo's real files.** Archiving the
  standing item against copies moved it with its fingerprint intact, printed the no-group notice, and
  the history lint then correctly reported it `new` — archived unanswered, which it was, because the
  rehearsal deliberately archived an item nobody had answered.

---

## 7. State

Read these off the tools rather than off this sentence, which is exactly the kind that goes stale:

```sh
./scripts/triage-state.py                          # the live inbox — empty means nothing owed
./scripts/triage-state.py docs/FEEDBACK_HISTORY.md # every answered item, all `current`
./scripts/triage-state.py --next                   # the next unclaimed Sn, over BOTH files
```

As of 2026-08-17: **S1 is open and unanswered** — filed on behalf of `sugar-sugar` against 0.10.0,
about `parse_tracks(track="mean")` offering no way to learn whether averaging a subject's two sensor
series is defensible. The next consumer id is **S2** and the next roadmap item is **RM14**, both
computed rather than remembered. `FEEDBACK_HISTORY.md` is empty: no item has been answered yet, so no
pass of this loop has run here.

**An emptied inbox breaks id numbering unless the next id is pinned, and this is the one hazard the
split introduces.** Once answered items move out, the live file's highest visible id is not the
corpus's highest — with the inbox empty it shows none at all, so the obvious next id is `S1`, which
already exists and already has a reply. Two defences, and keep both: the inbox states the next id in
its header, and `--next` computes it from both files so it cannot drift from them. **Ids are never
reused**, not even for an item answered as a non-issue.
