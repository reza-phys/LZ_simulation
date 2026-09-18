#!/usr/bin/env bash
# Launch the LZ simulation in Claude Code with internet access disabled.
#
#   bash run.sh             start the simulation (sends the kick-off instruction automatically)
#   bash run.sh resume      reopen the most recent session in this folder (prints a resume message to paste)
#   bash run.sh shell       open a new session with the same restrictions and no kick-off message
#   bash run.sh --dry-run   print the claude command without running it
#
# Model: claude-fable-5-1 (Fable 5.1) by default; override with  LZ_SIM_MODEL=<model-id> bash run.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
SETTINGS="$ROOT/environment/claude-settings.generated.json"
MODEL="${LZ_SIM_MODEL:-claude-fable-5-1}"
MIN_CLAUDE_VERSION="2.1.219"

KICKOFF="Read PROMPT.md in full and carry it out autonomously, from the environment check and Phase 1 through the final synthesis. Work only inside this directory and write all deliverables under output/."
RESUME="Continue carrying out PROMPT.md. First inspect output/ (the frozen plan, results_ledger, provenance/process_log.md and data_requests/) to find exactly where you stopped, then resume from there without redoing completed work."

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

[ -f "$ROOT/environment/.setup_complete" ] && [ -f "$SETTINGS" ] || fail "Setup has not been completed. Run: bash environment/setup.sh"
grep -qF "$ROOT/environment/hooks/log_tool_call.py" "$SETTINGS" || fail "This folder has moved since setup. Re-run: bash environment/setup.sh"
command -v claude >/dev/null 2>&1 || fail "Claude Code ('claude') is not installed or not on PATH."

CLAUDE_VERSION="$(claude --version 2>/dev/null | awk '{print $1}')"
"$ROOT/.venv/bin/python" -I -S -c '
import sys
v = lambda s: tuple(int(x) for x in s.split(".")[:3])
sys.exit(0 if v(sys.argv[1]) >= v(sys.argv[2]) else 1)' "$CLAUDE_VERSION" "$MIN_CLAUDE_VERSION" \
  || fail "Claude Code $CLAUDE_VERSION is too old; version $MIN_CLAUDE_VERSION or newer is required (run: claude update)."

FLAGS=(--model "$MODEL" --settings "$SETTINGS" --permission-mode acceptEdits --disallowedTools WebFetch WebSearch "mcp__*")

MODE="${1:-start}"
case "$MODE" in
  start)
    if [ -d "$ROOT/output" ] && [ -n "$(ls -A "$ROOT/output" 2>/dev/null)" ]; then
      fail "output/ already contains results. To continue that run use: bash run.sh resume  (to start over, move output/ and sim_logs/ elsewhere first)"
    fi
    CMD=(claude "$KICKOFF" "${FLAGS[@]}") ;;
  resume)
    printf '\nWhen the session opens, paste this message:\n\n%s\n\n' "$RESUME"
    CMD=(claude --continue "${FLAGS[@]}") ;;
  shell)
    CMD=(claude "${FLAGS[@]}") ;;
  --dry-run)
    printf '%q ' claude "$KICKOFF" "${FLAGS[@]}"; echo; exit 0 ;;
  *)
    sed -n '2,10p' "$0"; exit 2 ;;
esac

echo "Starting Claude Code ($CLAUDE_VERSION, model $MODEL) in $ROOT with network access disabled."
exec "${CMD[@]}"
