"""Post-run audit of the LZ simulation.

Run from the simulation root (collect_results.sh does this for you):
    .venv/bin/python environment/audit.py

It reads the hook log (sim_logs/tool_calls.jsonl), copies the Claude Code transcripts of
this folder into sim_logs/transcripts/, scans the logs, the transcripts and the session's
scripts for anything that looks like network access or tool use outside the provided
environment, and writes sim_logs/audit_report.md. Flags are items to review, not verdicts.
"""
import collections
import datetime
import json
import pathlib
import re
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOGS = ROOT / "sim_logs"
HOOK_LOG = LOGS / "tool_calls.jsonl"
REPORT = LOGS / "audit_report.md"

NETWORK_TOOLS = re.compile(r"^(WebFetch|WebSearch|mcp__.*)$")
SUSPECT_COMMAND = re.compile(
    r"(\bcurl\b|\bwget\b|\bnc\b|\bssh\b|\bscp\b|\brsync\b|\bgit\s+(clone|fetch|pull|push)\b|"
    r"\bpip3?\b|\buv\b|\bbrew\b|\btlmgr\b|\bconda\b|https?://|\burllib\b|\brequests\b|\bsocket\b|http\.client)"
)
SUSPECT_CODE = re.compile(
    r"(^\s*import\s+(requests|urllib|socket|http\.client)|^\s*from\s+(urllib|requests|http|socket)\b|urlopen|https?://)",
    re.MULTILINE,
)
OTHER_INTERPRETER = re.compile(r"(^|[\s;&|(])(python3?(\.\d+)?|/usr/bin/python3?|/usr/local/bin/python3?)\b")


def load_jsonl(path):
    records = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return records


def transcript_dir():
    name = re.sub(r"[^A-Za-z0-9]", "-", str(ROOT))
    return pathlib.Path.home() / ".claude" / "projects" / name


def command_text(record):
    tool_input = record.get("tool_input") or {}
    cmd = tool_input.get("command", "") if isinstance(tool_input, dict) else ""
    if isinstance(cmd, dict):  # truncated long command
        cmd = cmd.get("head", "")
    return cmd or ""


def main():
    LOGS.mkdir(exist_ok=True)
    lines = [f"# Audit report", f"generated {datetime.datetime.now().isoformat(timespec='seconds')} for {ROOT}", ""]
    flags = []

    # --- hook log -----------------------------------------------------------
    records = load_jsonl(HOOK_LOG)
    lines.append("## 1. Hook log (sim_logs/tool_calls.jsonl)")
    if not records:
        lines.append("No hook records found. If the simulation ran, the logging hook was not active; rely on the transcripts below.")
        flags.append("hook log empty")
    else:
        sessions = sorted({r.get("session_id") for r in records if r.get("session_id")})
        lines.append(f"{len(records)} records, {len(sessions)} session(s), from {records[0].get('ts')} to {records[-1].get('ts')}.")
        counts = collections.Counter((r.get("event"), r.get("tool_name")) for r in records)
        lines += ["", "| event | tool | count |", "|---|---|---|"]
        lines += [f"| {e} | {t} | {n} |" for (e, t), n in sorted(counts.items(), key=lambda kv: (str(kv[0][0]), -kv[1]))]

        attempts = [r for r in records if r.get("event") == "PreToolUse"]
        net_tools = [r for r in attempts if NETWORK_TOOLS.match(r.get("tool_name") or "")]
        bash = [r for r in attempts if r.get("tool_name") == "Bash"]
        suspect_cmds = [r for r in bash if SUSPECT_COMMAND.search(command_text(r))]
        other_py = [r for r in bash if OTHER_INTERPRETER.search(command_text(r)) and ".venv/bin/python" not in command_text(r)]
        denied = [r for r in records if r.get("event") == "PermissionDenied"]
        failed = [r for r in records if r.get("event") == "PostToolUseFailure"]
        subagents = collections.Counter(r.get("agent_type") for r in attempts if r.get("agent_id"))

        def listing(title, rows, fmt):
            lines.extend(["", f"### {title}: {len(rows)}"])
            lines.extend(f"- {fmt(r)}" for r in rows[:500])

        listing("Network-capable tool attempts (WebFetch/WebSearch/MCP)", net_tools, lambda r: f"{r.get('ts')} {r.get('tool_name')} {json.dumps(r.get('tool_input'))[:300]}")
        listing("Bash commands matching network/installer patterns", suspect_cmds, lambda r: f"{r.get('ts')} `{command_text(r)[:300]}`")
        listing("Bash commands using a Python other than .venv/bin/python", other_py, lambda r: f"{r.get('ts')} `{command_text(r)[:300]}`")
        listing("Permission denials", denied, lambda r: f"{r.get('ts')} {r.get('tool_name')} {json.dumps(r.get('tool_input'))[:300]}")
        listing("Failed tool calls", failed, lambda r: f"{r.get('ts')} {r.get('tool_name')} {str(r.get('error'))[:300]}")
        lines.extend(["", f"### Subagent tool calls by agent type: {dict(subagents) or 'none'}"])
        lines.extend(["", f"### All Bash commands ({len(bash)})"])
        lines.extend(f"{i + 1}. `{command_text(r)[:400]}`" for i, r in enumerate(bash))

        if net_tools:
            flags.append(f"{len(net_tools)} network-capable tool attempt(s)")
        if suspect_cmds:
            flags.append(f"{len(suspect_cmds)} Bash command(s) matching network/installer patterns")
        if other_py:
            flags.append(f"{len(other_py)} Bash command(s) using another Python")

    # --- transcripts --------------------------------------------------------
    lines.extend(["", "## 2. Claude Code transcripts"])
    tdir = transcript_dir()
    dest = LOGS / "transcripts"
    transcript_files = sorted(tdir.rglob("*.jsonl")) if tdir.exists() else []
    tool_uses = collections.Counter()
    for f in transcript_files:
        rel = f.relative_to(tdir)
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(f, dest / rel)
        for rec in load_jsonl(f):
            content = (rec.get("message") or {}).get("content")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "tool_use":
                        tool_uses[item.get("name")] += 1
    lines.append(f"Source: {tdir} ({len(transcript_files)} file(s), copied to sim_logs/transcripts/).")
    if tool_uses:
        lines += ["", "| tool (from transcripts) | uses |", "|---|---|"]
        lines += [f"| {name} | {n} |" for name, n in tool_uses.most_common()]
        net_in_transcripts = {n: c for n, c in tool_uses.items() if NETWORK_TOOLS.match(n or "")}
        if net_in_transcripts:
            flags.append(f"network-capable tools requested in transcripts: {net_in_transcripts}")

    # --- scripts written by the session -------------------------------------
    lines.extend(["", "## 3. Session scripts (output/**/*.py) referencing network access"])
    hits = []
    for f in sorted((ROOT / "output").rglob("*.py")) if (ROOT / "output").exists() else []:
        text = f.read_text(encoding="utf-8", errors="replace")
        found = sorted({m.group(0).strip() for m in SUSPECT_CODE.finditer(text)})
        if found:
            hits.append((f.relative_to(ROOT), found))
    lines.extend(f"- {path}: {', '.join(found)[:300]}" for path, found in hits)
    if not hits:
        lines.append("none")
    else:
        flags.append(f"{len(hits)} script(s) reference network access")

    # --- summary ------------------------------------------------------------
    lines[3:3] = ["## Summary", ("No flags." if not flags else "Items to review: " + "; ".join(flags)), ""]
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {REPORT.relative_to(ROOT)}")
    print("No flags." if not flags else "Items to review: " + "; ".join(flags))
    return 0


if __name__ == "__main__":
    sys.exit(main())
