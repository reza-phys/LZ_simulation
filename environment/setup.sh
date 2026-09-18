#!/usr/bin/env bash
# One-time setup for the LZ simulation. Needs internet access (the simulation itself does not).
#
#   bash environment/setup.sh                            # standard setup
#   bash environment/setup.sh --install-latex-packages   # also add missing LaTeX packages (user mode)
#
# Re-run it if you move or copy this folder: it writes absolute paths into the Claude settings.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVDIR="$ROOT/environment"
cd "$ROOT"

INSTALL_LATEX=0
for arg in "$@"; do
  case "$arg" in
    --install-latex-packages) INSTALL_LATEX=1 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "Unknown option: $arg" >&2; exit 2 ;;
  esac
done

say()  { printf '\n==> %s\n' "$*"; }
warn() { printf 'WARNING: %s\n' "$*" >&2; }
fail() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }

WIMPYDD_URL="https://wimpydd.hepforge.org/code/master_code/WimPyDD_2.0.4_without_response_functions.zip"
WIMPYDD_SHA256="30129efa94bade741f14b6528e0cceedec9c56ad45f7e5e31454abb16f0eb002"
LATEX_PKGS="revtex siunitx mhchem chemgreek cleveref physics tikz-feynman lipsum import"
LATEX_FILES="revtex4-2.cls siunitx.sty mhchem.sty cleveref.sty physics.sty tikz-feynman.sty"

# ---------------------------------------------------------------------------
say "Checking prerequisites"
OS="$(uname -s)"
[ "$OS" = "Darwin" ] || [ "$OS" = "Linux" ] || fail "macOS or Linux is required (on Windows, use WSL2)."
case "$ROOT" in
  *Dropbox*|*OneDrive*|*"Google Drive"*|*iCloud*|*CloudStorage*)
    warn "This folder is inside a cloud-synced directory. Setup creates ~1 GB of files; move the folder to a local path first (e.g. ~/LZ_simulation) and re-run." ;;
esac
command -v uv    >/dev/null 2>&1 || fail "uv is not installed. Install it (https://docs.astral.sh/uv/): 'curl -LsSf https://astral.sh/uv/install.sh | sh' or 'brew install uv', then re-run."
command -v curl  >/dev/null 2>&1 || fail "curl is required."
command -v unzip >/dev/null 2>&1 || fail "unzip is required."
if command -v claude >/dev/null 2>&1; then
  echo "Claude Code: $(claude --version 2>/dev/null | head -n 1)"
else
  warn "Claude Code ('claude') is not on PATH. Install it before running run.sh."
fi
if [ "$OS" = "Linux" ]; then
  for bin in bwrap socat; do
    command -v "$bin" >/dev/null 2>&1 || warn "'$bin' not found. Claude Code's sandbox needs bubblewrap and socat on Linux (e.g. 'sudo apt install bubblewrap socat'); run.sh will not start without the sandbox."
  done
fi

# ---------------------------------------------------------------------------
say "Creating the Python 3.12 environment (.venv)"
if [ -x "$ROOT/.venv/bin/python" ]; then
  echo "Reusing existing .venv"
else
  uv venv "$ROOT/.venv" --python 3.12 --python-preference only-managed
fi
PY="$ROOT/.venv/bin/python"

say "Installing Python packages"
REQ_USED="requirements.lock.txt (exact pins)"
if ! uv pip install --python "$PY" -r "$ENVDIR/requirements.lock.txt"; then
  warn "Exact pinned versions could not be installed on this platform; falling back to requirements.txt (latest compatible versions)."
  uv pip install --python "$PY" -r "$ENVDIR/requirements.txt"
  REQ_USED="requirements.txt (fallback, unpinned)"
fi

say "Installing offline defaults into the environment"
SITE="$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
cp "$ENVDIR/python/sitecustomize.py" "$SITE/sitecustomize.py"
printf '%s\n' "$ROOT" > "$SITE/lz_simulation_root.pth"   # makes WimPyDD/ importable from any script

# ---------------------------------------------------------------------------
say "Installing WimPyDD 2.0.4 (HepForge)"
mkdir -p "$ENVDIR/downloads"
ZIP="$ENVDIR/downloads/WimPyDD_2.0.4_without_response_functions.zip"
[ -f "$ZIP" ] || { echo "downloading (~40 MB)..."; curl -fsSL --retry 3 -o "$ZIP" "$WIMPYDD_URL"; }
if command -v shasum >/dev/null 2>&1; then GOT="$(shasum -a 256 "$ZIP" | awk '{print $1}')"; else GOT="$(sha256sum "$ZIP" | awk '{print $1}')"; fi
[ "$GOT" = "$WIMPYDD_SHA256" ] || { rm -f "$ZIP"; fail "WimPyDD checksum mismatch (got $GOT). The download was removed; re-run setup."; }
if [ -f "$ROOT/WimPyDD/__init__.py" ]; then
  echo "WimPyDD/ already present; keeping it"
else
  unzip -q "$ZIP" -d "$ROOT"
  [ -f "$ROOT/WimPyDD/__init__.py" ] || fail "Unexpected WimPyDD archive layout."
fi

# ---------------------------------------------------------------------------
say "Checking LaTeX (optional)"
if command -v pdflatex >/dev/null 2>&1 && command -v kpsewhich >/dev/null 2>&1; then
  MISSING=""
  for f in $LATEX_FILES; do kpsewhich "$f" >/dev/null 2>&1 || MISSING="$MISSING $f"; done
  if [ -n "$MISSING" ] && [ "$INSTALL_LATEX" = "1" ] && command -v tlmgr >/dev/null 2>&1; then
    tlmgr init-usertree >/dev/null 2>&1 || true
    tlmgr --usermode install $LATEX_PKGS || warn "tlmgr could not install some packages; LaTeX may be partially available."
  elif [ -n "$MISSING" ]; then
    warn "LaTeX packages missing:$MISSING  (optional; re-run with --install-latex-packages to add them)"
  else
    echo "pdflatex and the required LaTeX packages are available"
  fi
else
  echo "pdflatex not found: LaTeX will be recorded as unavailable (optional; the papers are Markdown)."
fi

# ---------------------------------------------------------------------------
say "Writing Claude Code settings for this folder"
"$PY" -I -S - "$ROOT" "$ENVDIR/claude-settings.template.json" "$ENVDIR/claude-settings.generated.json" <<'PYEOF'
import json, sys
root, template, out = sys.argv[1], sys.argv[2], sys.argv[3]
text = open(template, encoding="utf-8").read().replace("@ROOT@", json.dumps(root)[1:-1])
settings = json.loads(text)  # validates the result
with open(out, "w", encoding="utf-8") as fh:
    json.dump(settings, fh, indent=2)
    fh.write("\n")
print("wrote", out)
PYEOF

# ---------------------------------------------------------------------------
say "Recording exact versions (environment/ENVIRONMENT_versions.txt)"
{
  echo "# LZ simulation research environment: exact versions"
  echo "# generated $(date -u +%Y-%m-%dT%H:%MZ) by environment/setup.sh"
  echo
  echo "## System"
  uname -a
  [ "$OS" = "Darwin" ] && echo "macOS $(sw_vers -productVersion)"
  echo
  echo "## Python interpreter"
  "$PY" --version
  echo "path: .venv/bin/python   (installed from: $REQ_USED)"
  echo
  echo "## Python packages (uv pip freeze)"
  uv pip freeze --python "$PY" 2>/dev/null
  echo
  echo "## Source-installed Python package"
  echo "WimPyDD==2.0.4  source: $WIMPYDD_URL  sha256: $WIMPYDD_SHA256"
  echo "  location: WimPyDD/ in the simulation root; run with the working directory = simulation root"
  echo
  echo "## LaTeX"
  if command -v pdflatex >/dev/null 2>&1; then
    pdflatex --version | head -n 1
    command -v bibtex >/dev/null 2>&1 && bibtex --version | head -n 1
    for f in $LATEX_FILES feynmp.sty booktabs.sty natbib.sty; do printf '  %-18s %s\n' "$f" "$(kpsewhich "$f" >/dev/null 2>&1 && echo available || echo MISSING)"; done
  else
    echo "pdflatex: NOT AVAILABLE"
  fi
  echo
  echo "## Tooling used to build and run (not for use inside the simulation)"
  echo "uv $(uv --version | awk '{print $2}')"
  command -v claude >/dev/null 2>&1 && echo "Claude Code $(claude --version 2>/dev/null | head -n 1)"
  echo
  echo "## Deliberately not provided"
  echo "network access; Fortran/C++ physics codes (micrOMEGAs, MadGraph, CLASS, DarkSUSY); ROOT; package installation"
} > "$ENVDIR/ENVIRONMENT_versions.txt" 2>&1

# ---------------------------------------------------------------------------
say "Running the environment check"
if "$PY" "$ENVDIR/check_env.py" > "$ENVDIR/check_env_output.txt" 2>&1; then
  tail -n 20 "$ENVDIR/check_env_output.txt"
else
  cat "$ENVDIR/check_env_output.txt"
  fail "Environment check failed (full output: environment/check_env_output.txt)."
fi
rm -rf "$ROOT/.cache"
mkdir -p "$ROOT/sim_logs"
date -u +%Y-%m-%dT%H:%MZ > "$ENVDIR/.setup_complete"

say "Setup complete"
cat <<EOF
Next steps (from $ROOT):
  1. Optional but recommended, checks that the internet is really blocked (a few cents of usage):
       bash environment/preflight.sh
  2. Start the simulation:
       bash run.sh
EOF
