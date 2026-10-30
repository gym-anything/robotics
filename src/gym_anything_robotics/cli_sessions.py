"""Operator session selection and live summaries; no implicit mutations."""
from concurrent.futures import ThreadPoolExecutor
import math
import time
import json
from pathlib import Path

from . import sessions
from .launch import describe as environment
from .remote import RemoteEnv
from .session_handoff import _process_alive


def records():
    result = []
    for path in sorted(sessions.session_root().glob("*/session.json")):
        try:
            manifest = json.loads(path.read_text())
            if not isinstance(manifest, dict) or "launch" not in manifest or "session_id" not in manifest:
                raise ValueError("Missing session identity or launch")
            result.append((path.parent, manifest))
        except (OSError, ValueError) as error:
            result.append((path.parent, {"session_id": path.parent.name, "status": "invalid",
                                         "error": str(error), "launch": {}}))
    return result


def task_name(directory, manifest):
    if (directory / "task.json").is_file():
        try:
            return json.loads((directory / "task.json").read_text())["task_id"]
        except (OSError, ValueError, KeyError):
            return "task metadata unreadable"
    options = manifest.get("launch", {}).get("spec", {}).get("backend_options", {})
    if "garment" in options.get("config", {}):
        return options["config"]["garment"]
    for key in ("task_kind", "variation_id", "episode_index"):
        if key in options:
            return f"{key}={options[key]}"
    return "default"


def probe(record):
    directory, manifest = record
    launch = manifest.get("launch", {})
    row = {"session_id": manifest["session_id"], "directory": str(directory),
           "environment": launch.get("spec", {}).get("environment_id", "unknown"),
           "task": task_name(directory, manifest), "status": manifest["status"],
           "url": manifest.get("url"), "created_at": manifest.get("created_at"),
           "active": manifest["status"] in ("running", "starting")}
    if not row["active"]:
        if manifest.get("error"):
            row["error"] = manifest["error"]
        return row
    if manifest["status"] == "running":
        try:
            keys = json.loads(Path(launch["credential_file"]).read_text())
            with RemoteEnv(manifest["url"], api_key=keys["privileged"], timeout_s=1) as remote:
                live = remote.session()
            if live["session_id"] != manifest["session_id"]:
                row.update(status="stale", active=False, error="A different session owns this URL")
                return row
            row["status"] = ("stopping" if live["stopping"] else "faulted" if not live["health"]["ready"]
                             else "paused" if live["paused"] else "ready")
            return row
        except (OSError, ValueError, RuntimeError, KeyError) as error:
            row["status"], row["error"] = "unreachable", str(error)
    if not _process_alive(manifest.get("pid")):
        row.update(status="exited", active=False)
    return row


def inventory(*, include_all=False):
    entries = records()
    with ThreadPoolExecutor(max_workers=8) as workers:
        rows = list(workers.map(probe, entries))
    # Unknown/corrupt records are not silently discarded.
    warnings = [f"Unreadable session {r['session_id']}: {r.get('error', 'invalid record')}"
                for r in rows if r["status"] == "invalid"]
    return {"sessions": [r for r in rows if include_all or r["active"]],
            "all": include_all, "warnings": warnings}


def choose(value):
    if value is None:
        raise ValueError("Choose a session: use an environment name or short ID from 'gar sessions'.")
    value = str(value)
    direct = Path(value).expanduser()
    if (direct / "session.json").is_file():
        return direct.resolve()
    entries = records()
    exact = [p for p, m in entries if m["session_id"] == value]
    if exact:
        return exact[0]
    if len(value) >= 4 and all(c in "0123456789abcdef" for c in value):
        matches = [(p, m) for p, m in entries if m["session_id"].startswith(value)]
    else:
        try:
            canonical = environment(value)["environment_id"]
        except ValueError:
            canonical = value
        matches = [(Path(r["directory"]), r) for r in inventory()["sessions"]
                   if r["environment"] == canonical]
    if len(matches) == 1:
        return matches[0][0]
    if len(matches) > 1:
        ids = ", ".join(m["session_id"][:12] for _, m in matches)
        raise ValueError(f"Ambiguous session '{value}': {ids}. Use a longer ID from 'gar sessions'.")
    raise ValueError(f"No matching active session '{value}'. Run 'gar sessions' or 'gar sessions --all' for history.")


def stop(directory, *, timeout_s=30):
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("Shutdown timeout must be finite and positive")
    directory = choose(directory)
    with sessions.client(directory, timeout_s=timeout_s) as remote:
        response = remote.stop()
    deadline = time.monotonic() + timeout_s
    while True:
        manifest = json.loads((directory / "session.json").read_text())
        if manifest["status"] == "stopped":
            return {"operation": "stop", **response, "stopped": True}
        if manifest["status"] == "failed":
            raise RuntimeError(f"Server stopped with an error: {manifest.get('close_error') or manifest.get('error')}. See 'gar logs {manifest['session_id'][:12]}'.")
        if not _process_alive(manifest.get("pid")):
            return {"operation": "stop", **response, "stopped": True}
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Shutdown still in progress after {timeout_s:g}s. No force-kill performed. Check 'gar status {manifest['session_id'][:12]}'.")
        time.sleep(min(.2, max(0, deadline - time.monotonic())))


def logs(directory, *, tail=40):
    if tail < 1:
        raise ValueError("--tail must be positive")
    directory = choose(directory)
    manifest = json.loads((directory / "session.json").read_text())
    path = Path(manifest["log"])
    # Read backwards in bounded chunks; honor the requested number of lines.
    with path.open("rb") as stream:
        stream.seek(0, 2)
        position, chunks, lines = stream.tell(), [], 0
        while position and lines <= tail:
            size = min(position, 16384)
            position -= size
            stream.seek(position)
            chunk = stream.read(size)
            chunks.append(chunk)
            lines += chunk.count(b"\n")
    text = b"".join(reversed(chunks)).decode(errors="replace")
    keys = json.loads(Path(manifest["launch"]["credential_file"]).read_text())
    for token in keys.values():
        if token:
            text = text.replace(token, "<redacted>")
    return {"session_id": manifest["session_id"], "path": str(path),
            "lines": text.splitlines()[-tail:]}
