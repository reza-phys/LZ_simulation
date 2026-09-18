# LZ simulation: operator guide

This folder runs a self-contained research simulation in **Claude Code**. Claude reads the LZ collaboration's
paper *"Search for dark matter particle interactions in an extended nuclear recoil energy window with the
LUX-ZEPLIN (LZ) experiment"* (arXiv:2609.02823, 2 September 2026). It then acts as the research community in
the two weeks after the announcement, and writes 100 one-page papers backed by detailed work files and
complete tool records. The full task is in `PROMPT.md`.

The run is **offline by design**. Claude works only from the paper, which is included, from its own training
knowledge, and from a fixed, pre-installed research environment. As the operator, you set it up, start it,
decide on any data requests, and send the results back.

---

## What's in the folder

| Path | What it is |
|---|---|
| `PROMPT.md` | The task given to Claude. Do not edit. |
| `inputs/` | The LZ paper: full arXiv source (`.tex`, full-resolution figure PDFs, bibliography), a flattened single-file copy, and PNG figures. |
| `run.sh` | Starts (or resumes) Claude Code with the no-internet settings. **Always start the simulation through this script.** |
| `environment/setup.sh` | One-time setup (needs internet): Python environment, WimPyDD, settings, checks. |
| `environment/preflight.sh` | Optional short test that confirms the internet is blocked before the real run. |
| `environment/collect_results.sh` | After the run: audits the session and zips everything to send back. |
| `environment/…` | Pinned package list, environment check, offline defaults, the settings template, the logging hook, the audit script. |
| `.claude/settings.json` | A fallback that blocks web tools if someone starts `claude` here without `run.sh`. |

Created by setup or during the run: `.venv/` (Python environment, ~1 GB), `WimPyDD/` (dark-matter rate code),
`environment/claude-settings.generated.json`, `environment/ENVIRONMENT_versions.txt`,
`environment/check_env_output.txt`, `sim_logs/` (independent tool-call log), `output/` (everything Claude
produces), `.cache/`.

---

## Requirements

- **macOS** (sandbox built in) or **Linux** / **WSL2**. On Linux, install the sandbox dependencies first, e.g.
  `sudo apt install bubblewrap socat`.
- **Claude Code ≥ 2.1.219**, installed and logged in (`claude --version`; update with `claude update`). Use the
  **terminal CLI**, not the desktop app: the desktop app adds its own browser tools.
- **uv**, the Python package manager: `curl -LsSf https://astral.sh/uv/install.sh | sh` or `brew install uv`.
- `curl` and `unzip`; about 1.5 GB of free disk; **internet access during setup only**.
- Optional: a TeX installation (`pdflatex`). The papers are Markdown, so LaTeX is not required.
- A Claude plan with enough usage for a long run. Expect many hours and substantial token use.

---

## Quick start

```bash
# 0. Put the folder on a local disk (not Dropbox/iCloud/OneDrive) and enter it
cd ~/LZ_simulation

# 1. One-time setup (internet needed; a few minutes)
bash environment/setup.sh

# 2. Recommended: confirm the lockdown works (short, cheap test session)
bash environment/preflight.sh

# 3. Start the simulation
bash run.sh

# 4. When it has finished: audit and package the results
bash environment/collect_results.sh
```

---

## Step by step

### 1. Setup: `bash environment/setup.sh`

This script:
- creates `.venv/` with Python 3.12 and installs the exact pinned package versions. If a pin is not available
  for your platform, it falls back to compatible versions and records that.
- installs offline defaults. Caches stay in this folder, and astropy is not allowed to download anything.
- downloads **WimPyDD 2.0.4** from HepForge and verifies its SHA-256 checksum.
- checks LaTeX. Add `--install-latex-packages` to install missing LaTeX packages into your user TeX tree.
- writes `environment/claude-settings.generated.json` with this folder's absolute paths.
- records every version in `environment/ENVIRONMENT_versions.txt`, and runs `environment/check_env.py`. It must
  end with *All checks passed*.

**If you move or copy the folder, run setup again.** The settings contain absolute paths, and `run.sh` refuses
to start if they don't match.

### 2. Preflight: `bash environment/preflight.sh`

This runs a short headless Claude session (Haiku by default) that tries to reach the internet with `curl` and
with Python, and tries to write into `inputs/`. It then runs the environment check. Every line must say
**PASS**:
- curl did not reach the internet
- Python could not reach the internet
- `inputs/` is write-protected
- the research environment check passed
- the tool-call logging hook recorded entries

Do not start the real run if anything fails.

### 3. Run: `bash run.sh`

This opens Claude Code in this folder with the lockdown settings and sends the kick-off instruction ("Read
PROMPT.md … carry it out autonomously"). On first launch, Claude Code may ask whether you trust the folder:
accept. Then leave it running.

- Default model: **Fable 5.1** (`claude-fable-5-1`). To use another model: `LZ_SIM_MODEL=<model-id> bash run.sh`.
- The session works without asking for permission, so it can run unattended: the ordinary working tools (Bash,
  Read, Write, Edit, Glob, Grep) are explicitly allowed. That is safe because deny rules always beat allow
  rules, and the sandbox still enforces no network access and the read-only folders at the OS level. A blocked
  action shows up in the session as a sandbox violation, not as a question to you.
- If Claude ever does ask for permission, the answer is **no**, unless it is plainly ordinary work inside this
  folder. Note what it asked for when you report back.
- **Do not help it.** Do not paste information, answer physics questions, add MCP servers or connectors, or
  change settings mid-run. Your only inputs are the resume message and data-request decisions (below).
- If the session ends or crashes before finishing, run **`bash run.sh resume`**, then paste the message it
  prints. Claude finds its place from the saved plan, ledger and logs.
- `bash run.sh --dry-run` shows the exact command without running it.

**Monitoring progress** (from another terminal):
- Papers written: `ls output/papers | wc -l`
- Results table: `output/results_ledger.csv`
- Data requests: `output/data_requests/index.csv`
- Independent tool log: `sim_logs/tool_calls.jsonl`

### 4. Data requests (optional, your decision)

Papers may request public datasets, for example a HEPData record or telescope data. Claude **never** fetches
data itself: it files `output/data_requests/DR-###.md` with what is needed and where to find it, and continues
with a fallback.

For each request, decide whether to supply it:
- **Only supply data that was public on or before 2 September 2026.** Never supply papers or commentary about
  the LZ event.
- To supply it, download the files yourself and put them in `inputs/provided_data/DR-###/` (create the folder).
  Then, in the running session, or after `bash run.sh resume`, send:

  > Data for DR-### has been provided in inputs/provided_data/DR-###/. Revise the affected papers as described in §4 of PROMPT.md.

- To decline, you can simply leave it. If you want it recorded, send "DR-### is declined."

Keep a note of what you provided, and when.

### 5. Finish: `bash environment/collect_results.sh`

This script:
- writes `sim_logs/audit_report.md`, covering tool usage counts, every shell command, any network or installer
  attempts, any other Python used, permission denials, and network references in the session's scripts.
- copies this folder's Claude Code transcripts into `sim_logs/transcripts/`.
- creates **`results_<timestamp>.zip`** containing `output/`, `sim_logs/`, the environment records and
  `PROMPT.md`.

Send that zip back, along with your notes on any data you provided.

---

## How internet access is blocked

| Layer | What it does |
|---|---|
| Permission deny rules | Remove WebFetch, WebSearch and all MCP tools; block `curl`, `wget`, `pip`, `uv`, `brew`, `tlmgr`, `conda`. |
| `disableClaudeAiConnectors` | No claude.ai connectors are loaded. |
| Bash sandbox (OS-level) | Every shell command and its child processes run with **no allowed network domains** (strict allowlist). Unsandboxed fallback is disabled, and Claude Code refuses to start if the sandbox is unavailable. |
| Offline Python defaults | astropy internet access and IERS downloads are off; pip and uv are set to offline. |
| Filesystem protection | `PROMPT.md`, `README.md`, `run.sh`, `inputs/`, `environment/`, `.venv/` and `sim_logs/` are read-only for the session. Other projects' Claude transcripts cannot be read. |
| Independent logging | A hook records every tool call (attempted, completed, failed, denied) to `sim_logs/tool_calls.jsonl`, which the session cannot modify. |

Claude Code itself still talks to Anthropic's API, which it has to. That connection does not give the model
a way to browse.

---

## Troubleshooting

- **Setup: pinned install fails.** Setup falls back to `requirements.txt` automatically. The versions actually
  installed are recorded in `ENVIRONMENT_versions.txt`.
- **Setup: WimPyDD checksum mismatch or download failure.** Re-run setup. If HepForge is unreachable, place
  `WimPyDD_2.0.4_without_response_functions.zip`, with the SHA-256 listed in `setup.sh`, in
  `environment/downloads/` and re-run.
- **"Sandbox unavailable" or Claude Code will not start (Linux).** Install `bubblewrap` and `socat`. Inside
  Docker, the sandbox may need extra privileges; running directly on the host or in WSL2 is simpler.
- **Preflight: logging hook recorded nothing.** Make sure `.venv/bin/python` exists (re-run setup), and that you
  started from this folder. Hook errors are written to `sim_logs/hook_errors.log`.
- **`run.sh`: "This folder has moved since setup".** Run `bash environment/setup.sh` again.
- **`run.sh start` refuses because `output/` exists.** Use `bash run.sh resume` to continue. To start over, move
  `output/` and `sim_logs/` elsewhere first.
