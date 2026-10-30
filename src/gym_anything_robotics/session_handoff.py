"""Read-only startup readiness and owner connection details for the local CLI.

Keys are returned to the invoking owner, never written to logs or manifests.
None of these checks advances, resets, stops, or restarts an environment.
"""

import json
import math
import os
from pathlib import Path
import time

from . import sessions
from .remote import RemoteEnv, RemoteError


def validate_timeout(value):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError("Startup timeout must be finite and positive")
    return value


def _log_tail(path):
    try:
        with Path(path).open("rb") as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell() - 16384))
            return stream.read().decode(errors="replace")
    except OSError:
        return ""


def _warnings(log):
    result, active = [], False
    for line in log.splitlines():
        if "warning" in line.lower():
            result.append(line)
            active = True
        elif active and (line.startswith((" ", "\t")) or line.startswith("Consider ")):
            result[-1] += "\n" + line
        else:
            active = False
    return result


def _process_alive(pid):
    if pid is None or os.name == "nt":
        return True  # Unknown is not evidence of process death.
    # os.kill(pid, 0) is a POSIX probe, not a safe Windows liveness check.
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    # An exited, not-yet-reaped Linux child still has a PID.
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def describe(directory, *, timeout_s=3):
    directory = sessions.resolve(str(directory))
    manifest = json.loads((directory / "session.json").read_text())
    launch = manifest["launch"]
    keys = json.loads(Path(launch["credential_file"]).read_text())
    url = manifest.get("url")
    report = {
        "ready": False, "status": manifest["status"],
        "session_id": manifest["session_id"],
        "environment": launch["spec"]["environment_id"],
        "url": url, "dashboard_url": url + "/dashboard" if url else None,
        "help_url": url + "/v1/help" if url else None,
        "listen_host": launch["host"],
        "url_scope": "Server host; use your tunnel/preview address from another computer.",
        "api_keys": keys, "credential_file": launch["credential_file"],
        "log": manifest["log"],
    }
    for field in ("error", "close_error"):
        if field in manifest:
            report[field] = manifest[field]
    log = _log_tail(manifest["log"])
    # Defend against a backend accidentally logging a credential.
    for token in keys.values():
        if token:
            log = log.replace(token, "<redacted>")
    warnings = _warnings(log)
    if warnings:
        report["startup_warnings"] = warnings
    if manifest["status"] == "failed":
        report["log_tail"] = "\n".join(log.splitlines()[-30:])
    if manifest["status"] != "running":
        return report
    try:
        with RemoteEnv(url, api_key=keys["privileged"], timeout_s=timeout_s) as owner:
            live = owner.session()
            if live["session_id"] != manifest["session_id"]:
                raise ValueError("Endpoint identity differs from manifest")
            report["health"] = live["health"]
            report["paused"] = live["paused"]
            if not live["health"]["ready"] or live["stopping"]:
                report["status"] = "not_ready"
                return report
            if owner.request("/v1/help").get("role") != "privileged":
                raise ValueError("Privileged credential did not authenticate as privileged")
            try:
                report["task"] = owner.request("/v1/task")
            except RemoteError as error:
                if error.status not in (400, 404, 501):
                    raise
                report["task"] = {"available": False, "error": str(error)}
        if keys["policy"]:
            with RemoteEnv(url, api_key=keys["policy"], timeout_s=timeout_s) as policy:
                if policy.request("/v1/help").get("role") != "policy":
                    raise ValueError("Policy credential did not authenticate as policy")
        report.update(ready=True, status="ready", authentication_verified=True)
    except RemoteError as error:
        report.update(status="authentication_failed" if error.status in (401, 403) else "unreachable",
                      error=str(error))
    except (OSError, ValueError, RuntimeError) as error:
        report.update(status="unreachable", error=str(error))
    return report


def wait(directory, *, timeout_s=600, pid=None, progress=None):
    """Wait only for readiness. Timeout/interruption leaves the child intact."""
    validate_timeout(timeout_s)
    started = time.monotonic()
    deadline, last_progress = started + timeout_s, started
    while True:
        remaining = deadline - time.monotonic()
        report = describe(directory, timeout_s=max(.001, min(3, remaining / 4)))
        if report["ready"] or report["status"] in {"failed", "stopped", "not_ready", "authentication_failed"}:
            return report
        if not _process_alive(pid):
            report.update(status="failed", error="Server process exited before readiness")
            log = _log_tail(report["log"])
            for token in report["api_keys"].values():
                if token:
                    log = log.replace(token, "<redacted>")
            report["log_tail"] = "\n".join(log.splitlines()[-30:])
            return report
        now = time.monotonic()
        if now >= deadline:
            report.update(status="timeout", error=f"Server not ready after {timeout_s:g}s; process was not stopped. Check gar sessions info {report['session_id']} before launching again.")
            return report
        if progress and now - last_progress >= 15:
            progress(f"Still waiting ({now - started:.0f}s): {report['status']}; log: {report['log']}")
            last_progress = now
        time.sleep(min(.5, deadline - now))
