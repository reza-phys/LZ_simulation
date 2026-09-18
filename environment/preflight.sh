#!/usr/bin/env bash
# Preflight: verify the lockdown before the real run, using a short, cheap headless session.
# Checks that (1) curl is blocked, (2) Python cannot reach the internet from the sandbox,
# (3) protected folders cannot be written, (4) the research environment works, and
# (5) the tool-call logging hook is recording.
#
#   bash environment/preflight.sh            (model: claude-haiku-4-5 by default)
#   LZ_PREFLIGHT_MODEL=<model-id> bash environment/preflight.sh
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SETTINGS="$ROOT/environment/claude-settings.generated.json"
MODEL="${LZ_PREFLIGHT_MODEL:-claude-haiku-4-5-20251001}"
LOG="$ROOT/sim_logs/tool_calls.jsonl"

[ -f "$ROOT/environment/.setup_complete" ] && [ -f "$SETTINGS" ] || { echo "Run bash environment/setup.sh first."; exit 1; }
command -v claude >/dev/null 2>&1 || { echo "Claude Code ('claude') not found."; exit 1; }
mkdir -p "$ROOT/sim_logs"
rm -f "$ROOT/inputs/.preflight_write_test"
BEFORE=$( [ -f "$LOG" ] && wc -l < "$LOG" | tr -d ' ' || echo 0 )

# (read -d '' rather than $(cat <<EOF) so that macOS's bash 3.2 parses the quotes correctly)
IFS= read -r -d '' PROMPT <<'EOF' || true
This is an automated configuration test. Use only the Bash tool. Run each of the four commands below exactly once, as four separate Bash calls, in order. Do not retry anything, do not work around failures or blocks, and do not use any other tool. Then reply with exactly four lines, "T1: ...", "T2: ...", "T3: ...", "T4: ...", each containing the output of the command or the error or block message.
T1: curl -sS -m 10 -o /dev/null -w "%{http_code}" https://example.com
T2: .venv/bin/python -c 'import urllib.request as u; print("HTTP", u.urlopen("https://example.com", timeout=10).status)'
T3: touch inputs/.preflight_write_test && echo WROTE
T4: .venv/bin/python environment/check_env.py | tail -n 1
EOF

echo "Running preflight session (model $MODEL)..."
REPLY="$(claude -p "$PROMPT" --model "$MODEL" --settings "$SETTINGS" --permission-mode acceptEdits --disallowedTools WebFetch WebSearch "mcp__*" 2>&1)"
echo "----- session reply -----"; echo "$REPLY"; echo "-------------------------"

STATUS=0
pass() { echo "PASS  $*"; }
bad()  { echo "FAIL  $*"; STATUS=1; }

echo "$REPLY" | grep -E '^T1:' | grep -qE '\b200\b' && bad "curl reached the internet" || pass "curl did not reach the internet"
echo "$REPLY" | grep -E '^T2:' | grep -qE 'HTTP 200' && bad "Python reached the internet from the sandbox" || pass "Python could not reach the internet"
if [ -e "$ROOT/inputs/.preflight_write_test" ]; then bad "inputs/ is writable (write protection not active)"; rm -f "$ROOT/inputs/.preflight_write_test"; else pass "inputs/ is write-protected"; fi
echo "$REPLY" | grep -E '^T4:' | grep -q 'All checks passed' && pass "research environment check passed" || bad "environment check did not report 'All checks passed' (see reply above)"
AFTER=$( [ -f "$LOG" ] && wc -l < "$LOG" | tr -d ' ' || echo 0 )
if [ "$AFTER" -gt "$BEFORE" ]; then
  pass "tool-call logging hook recorded $((AFTER - BEFORE)) entries"
  # keep preflight entries out of the simulation's log
  tail -n +"$((BEFORE + 1))" "$LOG" >> "$ROOT/sim_logs/preflight_tool_calls.jsonl"
  if [ "$BEFORE" -eq 0 ]; then rm -f "$LOG"; else head -n "$BEFORE" "$LOG" > "$LOG.tmp" && mv "$LOG.tmp" "$LOG"; fi
else
  bad "tool-call logging hook did not record anything"
fi
rm -rf "$ROOT/.cache"

echo
if [ "$STATUS" -eq 0 ]; then
  echo "Preflight passed. Start the simulation with: bash run.sh"
else
  echo "Preflight FAILED. Do not start the simulation until the failures above are resolved (see README.md, Troubleshooting)."
fi
exit "$STATUS"
