#!/usr/bin/env bash
# Debounced watcher for the document-as-inbox triage loop. See TRIAGE_LOOP.md.
#
# Emits one line on stdout when the watched file has stopped changing for COOLDOWN seconds.
# Consecutive saves inside the cooldown collapse into a single event, because each mtime bump restarts
# the timer. Meant to be driven by whatever turns a line of stdout into a notification — an agent
# harness, `while read`, a desktop notifier — and it works piped to anything.
#
# It never fires for a change that predates it: `last` is seeded from the current mtime and `dirty`
# starts clear, so an already-settled edit stays quiet. Run the ledger once at startup to pick up
# whatever is already pending.
#
# COOLDOWN is sized for an agent author, not a human one. When reports are written through an agent the
# pattern is a burst of edits — five in a minute is normal — with gaps wherever the agent stops to read
# or probe something. A one-minute timer fires in the middle of such a run and triages half a report.
# 150s clears the pauses that show up in practice. The only cost of waiting is latency; the cost of
# firing early is a reply to a half-written item.
#
# Nothing needs installing: `stat` polling is enough at this cadence, and it works where inotify-tools,
# entr, fswatch and python watchdog are all absent.
#
# If your triage pass is permitted to commit, it should only watch while the tree is on the branch
# that permit covers: a feature branch — or a detached HEAD — is somebody's own work, and unattended
# triage commits would land on top of whatever they are mid-way through. Off that branch the watcher
# idles at BRANCH_PAUSE instead of POLL, says so once, and says so again when it resumes. It does not
# touch `last` while paused, so an edit written during the pause is still picked up on the way back
# rather than lost. Set BRANCH to your own name for it, or to the empty string to switch the whole
# behaviour off; outside a git work tree there is no branch to speak of and it never pauses. In
# cgm-format the pass does not commit (CONSUMER_TRIAGE_LOOP.md §5), so the guard is inert on `main`
# and is carried for the day that rule changes.
#
# This is the only one of the three that is really bash. The ledger is Python and is invoked through
# $PYTHON below rather than as a bare path, so neither its exec bit nor its shebang is load-bearing
# (TRIAGE_LOOP.md § 5 — the extension gotcha).
#
#   FILE=<path> COOLDOWN=<seconds> POLL=<seconds> BRANCH=<name|empty> BRANCH_PAUSE=<seconds> \
#     RUNBOOK=<path> CAP=<n> ./watch-inbox.sh
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Local to cgm-format: FILE and RUNBOOK are derived from this script's location rather than from $PWD,
# so a run from a subdirectory watches the real inbox. The runbook is the gist, not a vendored copy.
FILE=${FILE:-${INBOX:-$HERE/../docs/FEEDBACK.md}}
PYTHON=${PYTHON:-python3}
LEDGER=${LEDGER:-$HERE/triage-state.py}
RUNBOOK=${RUNBOOK:-docs/CONSUMER_TRIAGE_LOOP.md}
COOLDOWN=${COOLDOWN:-150}
POLL=${POLL:-10}
BRANCH=${BRANCH-main}          # `-`, not `:-`: BRANCH= is an explicit "never pause", not an unset
BRANCH_PAUSE=${BRANCH_PAUSE:-900}
CAP=${CAP:-8}

# The tree to ask about is the one holding the watched file, not the one holding this script — the
# three scripts may live anywhere, including outside the repo they serve.
REPO=$(cd "$(dirname "$FILE")" 2>/dev/null && pwd || echo "$PWD")
# Answered once: a directory does not become a work tree while the watcher runs, and if it is not one
# now then `branch` means nothing here and the pause must never fire.
if [ -n "$BRANCH" ] && git -C "$REPO" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    watch_branch=1
else
    watch_branch=0
fi

mtime() { stat -c %Y "$FILE" 2>/dev/null || stat -f %m "$FILE" 2>/dev/null || echo 0; }
# A detached HEAD has no symbolic ref, and is no more a place to commit unattended than a branch is.
current_branch() { git -C "$REPO" symbolic-ref --quiet --short HEAD 2>/dev/null || echo "(detached HEAD)"; }

last=$(mtime)
dirty=0
paused=""

while true; do
    if [ "$watch_branch" = 1 ]; then
        on=$(current_branch)
        if [ "$on" != "$BRANCH" ]; then
            # Announce the transition once. A pause nobody can see reads as a dead watcher, and
            # repeating it every quarter hour would be the noise the event filter exists to avoid.
            if [ "$paused" != "$on" ]; then
                paused=$on
                echo "${FILE##*/} watch paused: tree is on $on, not $BRANCH — a branch is the human's own work"
            fi
            sleep "$BRANCH_PAUSE"
            continue
        fi
        if [ -n "$paused" ]; then
            paused=""
            echo "${FILE##*/} watch resumed: back on $BRANCH"
        fi
    fi

    sleep "$POLL"
    now=$(mtime)

    if [ "$now" != "$last" ]; then
        last=$now
        dirty=1                     # still moving: restart the cooldown
        continue
    fi

    [ "$dirty" = 1 ] || continue
    [ $(($(date +%s) - last)) -ge "$COOLDOWN" ] || continue
    dirty=0

    # The event line is capped: with a 17-item backlog an uncapped one listed every single item.
    pending=$(INBOX="$FILE" "$PYTHON" "$LEDGER" "$FILE" --pending 2>/dev/null |
              awk -v cap="$CAP" 'NF { n++; if (n <= cap) { printf "%s%s(%s)", sep, $2, $1; sep = " " } }
                   END { if (n > cap) printf " +%d more", n - cap }')
    if [ -z "$pending" ]; then
        echo "${FILE##*/} settled: nothing pending"
    else
        # Naming the runbook matters: to an agent whose context was just cleared, this line is the
        # entire brief — it has to say where the algorithm is written down.
        echo "${FILE##*/} settled: $pending — triage per $RUNBOOK"
    fi
done
