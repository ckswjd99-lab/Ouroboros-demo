#!/usr/bin/env bash
# Start (or restart) the demo server in a detached tmux session so it outlives
# whatever shell launched it.  Usage:
#   server/run_demo.sh            # start/restart, then print the URLs
#   server/run_demo.sh stop       # stop it
#   tmux attach -t ouroboros-demo # watch the per-frame log (detach: Ctrl-b d)
#
# GPU: pinned to this container's MIG slice explicitly -- tmux/background jobs do
# not reliably inherit the login shell's CUDA_VISIBLE_DEVICES.
set -euo pipefail
SESSION=ouroboros-demo
MIG=${OUROBOROS_MIG:-MIG-bb8cda48-86b2-549c-84c2-32299742a69b}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${OUROBOROS_PY:-$HOME/Ouroboros/.venv/bin/python}
LOG=$HERE/../build/demo_server.log

tmux kill-session -t "$SESSION" 2>/dev/null || true
[ "${1:-}" = stop ] && { echo "stopped"; exit 0; }

mkdir -p "$(dirname "$LOG")"
tmux new-session -d -s "$SESSION" -n server \
  "cd '$HERE' && CUDA_VISIBLE_DEVICES=$MIG '$PY' demo_server.py 2>&1 | tee '$LOG'"

echo -n "loading model"
for _ in $(seq 1 120); do
  grep -q '^ready:' "$LOG" 2>/dev/null && break
  grep -q 'Traceback' "$LOG" 2>/dev/null && { echo; tail -20 "$LOG"; exit 1; }
  echo -n .; sleep 1
done
echo; grep '^ready:' "$LOG"
HOST=${DEMO_PUBLIC_HOST:-147.46.66.242}
echo "dashboard : http://$HOST:65432/"
echo "phones    : app host field = $HOST  (naive 65433 / ouroboros 65434 / on-device 65435)"
echo "log       : tmux attach -t $SESSION   or   $LOG"
