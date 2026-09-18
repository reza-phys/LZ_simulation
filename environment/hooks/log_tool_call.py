"""Claude Code hook: append every tool call to sim_logs/tool_calls.jsonl.

Registered for PreToolUse, PostToolUse, PostToolUseFailure and PermissionDenied in
environment/claude-settings.generated.json. It runs outside the Bash sandbox, uses only
the Python standard library, never blocks a tool call, and always exits 0. This log is
an independent record of tool use that the session cannot edit.
"""
import datetime
import hashlib
import json
import os
import sys

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LOG_DIR = os.path.join(ROOT, "sim_logs")


def shrink(value, limit=4000):
    """Keep log lines manageable: long strings become a digest with head and tail."""
    if isinstance(value, str):
        if len(value) <= limit:
            return value
        return {
            "truncated": True,
            "length": len(value),
            "sha256": hashlib.sha256(value.encode("utf-8", "replace")).hexdigest(),
            "head": value[:1000],
            "tail": value[-500:],
        }
    if isinstance(value, dict):
        return {k: shrink(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        if len(value) > 50:
            return {"truncated_list": True, "length": len(value), "head": [shrink(v, limit) for v in value[:20]]}
        return [shrink(v, limit) for v in value]
    return value


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="milliseconds")


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    try:
        data = json.loads(sys.stdin.read())
        record = {
            "ts": now(),
            "event": data.get("hook_event_name"),
            "session_id": data.get("session_id"),
            "agent_id": data.get("agent_id"),
            "agent_type": data.get("agent_type"),
            "tool_name": data.get("tool_name"),
            "tool_use_id": data.get("tool_use_id"),
            "permission_mode": data.get("permission_mode"),
            "tool_input": shrink(data.get("tool_input")),
        }
        for key in ("tool_response", "error", "reason", "is_interrupt"):
            if key in data:
                record[key] = shrink(data[key], 2000)
        line = json.dumps(record, ensure_ascii=False)
        with open(os.path.join(LOG_DIR, "tool_calls.jsonl"), "a", encoding="utf-8") as fh:
            if fcntl:
                fcntl.flock(fh, fcntl.LOCK_EX)
            fh.write(line + "\n")
    except Exception as exc:  # noqa: BLE001  (a logging failure must never block the session)
        with open(os.path.join(LOG_DIR, "hook_errors.log"), "a", encoding="utf-8") as fh:
            fh.write(f"{now()} {type(exc).__name__}: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
