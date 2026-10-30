"""Recover only explicit Claude temporary-session quota errors; never generic failures."""
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import re
import time


def temporary_reset(event, observed_at):
    """Parse the provider's UTC reset relative to the receipt, not a later restart."""
    if not isinstance(event, dict) or event.get("is_error") is not True:
        return None
    message = event.get("result", "")
    if not isinstance(message, str):
        return None
    match = re.fullmatch(
        r"You've hit your session limit\s*[·\-–—]?\s*resets "
        r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)\s*\(UTC\)\s*", message, re.I)
    if not match:
        return None
    hour, minute = int(match[1]), int(match[2] or 0)
    if not 1 <= hour <= 12 or not 0 <= minute <= 59:
        return None
    hour = hour % 12 + (12 if match[3].lower() == "pm" else 0)
    seen = datetime.fromtimestamp(observed_at, timezone.utc)
    reset = seen.replace(hour=hour, minute=minute, second=0, microsecond=0)
    if reset <= seen:
        reset += timedelta(days=1)
    # A session reset is hours away. Do not turn an ambiguous/stale date into a day wait.
    if (reset - seen).total_seconds() > 6 * 3600:
        return None
    return reset.timestamp() + 60  # clock/propagation grace, recorded explicitly


def read_event(log):
    try:
        return json.loads(Path(log).read_text())
    except (OSError, ValueError):
        return None


def recovery_model_evidence(event, model, session_id, since, *, config_dir=None):
    """Aggregate usage includes secondary CLI calls; verify the actual primary turn.

    Never erase those secondary identities or accept a primary-model fallback.
    Evaluation policy identity checks remain separate and unchanged.
    """
    identities = set((event or {}).get("modelUsage", {})) - {"<synthetic>"}
    evidence = {"aggregate_models": sorted(identities)}
    if not identities - {model}:
        return evidence
    config = Path(config_dir or os.environ.get("CLAUDE_CONFIG_DIR", Path.home()/".claude"))
    primary = set()
    count = 0
    for path in (config/"projects").glob("*/" + session_id + ".jsonl"):
        with path.open() as stream:
            for line in stream:
                try:
                    entry = json.loads(line)
                    at = datetime.fromisoformat(entry.get("timestamp", "").replace("Z", "+00:00")).timestamp()
                except (ValueError, TypeError):
                    continue
                if (at < since or entry.get("type") != "assistant"
                        or entry.get("isSidechain") or entry.get("sessionId") != session_id):
                    continue
                identity = entry.get("message", {}).get("model")
                if identity and identity != "<synthetic>":
                    primary.add(identity)
                    count += 1
    if primary != {model}:
        return None
    evidence.update(primary_models=sorted(primary), primary_messages=count,
                    verified_since=since, secondary_usage_retained=True)
    return evidence


def wait_until(reset_at, deadline, *, clock=time.time, sleep=time.sleep):
    """Keep the original deadline. Returns False if quota outlasts that budget."""
    while clock() < reset_at:
        remaining = deadline - clock()
        if remaining <= 0:
            return False
        sleep(min(30, reset_at - clock(), remaining))
    return clock() < deadline


def invoke_with_recovery(invoke, backend, binary, prompt, *, session_id, resume,
                         workspace, timeout, log, model):
    """Native creator/auditor continuation; failed logs remain byte-for-byte intact."""
    from uuid import uuid4
    log = Path(log)
    current_log = log
    deadline = time.time() + timeout
    attempts = []
    sidecar = log.with_suffix(log.suffix + ".recovery.json")

    def record(status, **extra):
        state = dict(status=status, session_id=session_id, model=model,
                     original_log=str(log), attempts=attempts, **extra)
        temp = sidecar.with_name(sidecar.name + "." + uuid4().hex + ".tmp")
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(sidecar)

    while True:
        invocation_started = time.time()
        try:
            result = invoke(backend, binary, prompt, session_id=session_id,
                            resume=resume, workspace=workspace,
                            timeout=max(1, deadline-time.time()), log=current_log, model=model)
            if attempts:
                record("completed", final_log=str(current_log), completed_at=time.time())
            return result
        except RuntimeError:
            event = read_event(current_log)
            reset = temporary_reset(event, current_log.stat().st_mtime) if current_log.exists() else None
            identity = recovery_model_evidence(event, model, session_id, invocation_started)
            if (backend != "cc" or reset is None or identity is None
                    or event.get("session_id") != session_id):
                if attempts:
                    record("blocked", final_log=str(current_log), stopped_at=time.time())
                raise
            attempts.append(dict(log=str(current_log), reset_at=reset,
                                 model_evidence=identity,
                                 received_at=current_log.stat().st_mtime,
                                 reason=event["result"]))
            record("waiting_session_reset", reset_at=reset)
            print(f"Temporary session limit; preserving {session_id}; resume at "
                  f"{datetime.fromtimestamp(reset, timezone.utc).isoformat()}; {sidecar}", flush=True)
            if not wait_until(reset, deadline):
                record("blocked", reason="Invocation deadline expired during provider wait")
                raise RuntimeError(f"Invocation deadline expired during session-limit wait; {sidecar}")
            resume = True
            prompt = "Continue the interrupted request in this session from your saved work."
            current_log = log.with_name(log.stem + f".quota-resume-{len(attempts)}-" + uuid4().hex[:8] + log.suffix)
            record("resuming", final_log=str(current_log), resumed_at=time.time())
