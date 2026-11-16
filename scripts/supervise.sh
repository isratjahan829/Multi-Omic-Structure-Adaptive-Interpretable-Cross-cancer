#!/usr/bin/env bash
# Start the prescribed run if it is not already running, and otherwise say what
# it is doing. Safe to call as often as you like.
#
# Why this exists: the container these runs live in is reclaimed after a couple
# of hours of session inactivity, and that kills anything running detached. A
# single fit at the prescribed settings takes about eight hours, so the run has
# to be restartable, and something has to restart it. This script is the
# restart; run_prescribed.sh skips the phases that already finished, and
# Experiment.train_fold resumes the fit in progress from its last checkpoint.
#
#   bash scripts/supervise.sh <logfile>
#
# Prints RUNNING (pid, elapsed) or STARTED, then the tail of the log.
set -u

LOG="${1:?usage: supervise.sh <logfile>}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"
cd "$HERE"

pid="$(pgrep -f 'bash scripts/run_prescribed.sh' | head -1 || true)"
if [ -n "${pid:-}" ]; then
    echo "RUNNING pid=$pid elapsed=$(ps -o etimes= -p "$pid" | tr -d ' ')s"
else
    mkdir -p "$(dirname "$LOG")"
    setsid nohup bash scripts/run_prescribed.sh >> "$LOG" 2>&1 < /dev/null &
    disown
    sleep 2
    newpid="$(pgrep -f 'bash scripts/run_prescribed.sh' | head -1 || true)"
    echo "STARTED pid=${newpid:-unknown} log=$LOG"
fi

echo "--- markers ---"
ls -1 runs_markers/*.done 2>/dev/null | sed 's#.*/##; s#\.done$##' | tr '\n' ' '
echo
echo "--- progress ---"
grep -aE '^(####|  epoch|  resumed|\[data\]|\[.*\] primary fold|  SWA)' "$LOG" \
    2>/dev/null | tail -12
