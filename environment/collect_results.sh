#!/usr/bin/env bash
# After the run: audit the session and package everything to send back.
#   bash environment/collect_results.sh
# Produces results_<timestamp>.zip in the simulation root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "Run bash environment/setup.sh first."; exit 1; }

"$PY" environment/audit.py

STAMP="$(date -u +%Y%m%d-%H%MZ)"
OUT="$ROOT/results_${STAMP}.zip"
"$PY" -I -S - "$ROOT" "$OUT" <<'PYEOF'
import os, sys, zipfile
root, out = sys.argv[1], sys.argv[2]
items = ["output", "sim_logs", "PROMPT.md",
         "environment/ENVIRONMENT_versions.txt", "environment/check_env_output.txt",
         "environment/claude-settings.generated.json"]
n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
    for item in items:
        path = os.path.join(root, item)
        if os.path.isfile(path):
            z.write(path, item); n += 1
        elif os.path.isdir(path):
            for d, _, files in os.walk(path):
                for f in files:
                    full = os.path.join(d, f)
                    z.write(full, os.path.relpath(full, root)); n += 1
print(f"Packaged {n} files into {out}")
PYEOF
echo "Send this file back: $OUT"
