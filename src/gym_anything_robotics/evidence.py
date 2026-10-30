"""Read-only export of retained owner history; never reset or invent lost data."""
import hashlib
import json
from pathlib import Path
from urllib.parse import quote

from .benchmark import save, utc
from . import sessions


def export(session, output, *, include_traces=False):
    if type(include_traces) is not bool:
        raise ValueError("include_traces must be boolean")
    directory = sessions.resolve(session)
    manifest = json.loads((directory / "session.json").read_text())
    with sessions.client(directory) as remote:
        initial = remote.session()
        if not initial["paused"]:
            raise ValueError("Pause the session explicitly before exporting a stable history")
        before = remote.request("/v1/history/status")["history"]
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        artifacts, transitions, trace_gaps = [], [], []
        def record(relative, value):
            path = output / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            save(path, value)
            artifacts.append({"path": relative, "bytes": path.stat().st_size,
                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        # Only selected nonsecret metadata is copied; never copy credential files.
        record("session.json", {"session_id": initial["session_id"], "launch": manifest["launch"]["spec"],
                               "source": manifest["launch"]["source"], "live": initial})
        if (directory / "task.json").is_file():
            record("task.json", json.loads((directory / "task.json").read_text()))
        cursor, count = 0, 0
        first_tick, last_tick, previous_after = None, None, None
        incomplete = []
        while True:
            page = remote.request(f"/v1/history?after_seq={cursor}&limit=200")
            for summary in page["records"]:
                entry = remote.request("/v1/history/" + quote(summary["id"], safe=""))["record"]
                seq = entry["seq"]
                record(f"history/{seq:08d}.json", entry)
                count += 1
                if entry["kind"] not in ("action", "gateway_start"):
                    transitions.append({"seq": seq, "kind": entry["kind"]})
                if entry["outcome"] not in ("completed", "committed_without_observation"):
                    incomplete.append({"seq": seq, "outcome": entry["outcome"]})
                b = (entry.get("before") or {}).get("state") or {}
                a = (entry.get("after") or {}).get("state") or {}
                bt, at = b.get("tick"), a.get("tick")
                if first_tick is None:
                    first_tick = bt
                if previous_after is not None and bt != previous_after:
                    transitions.append({"seq": seq, "kind": "tick_gap", "previous_after": previous_after, "before": bt})
                if bt is not None and at is not None and at < bt:
                    transitions.append({"seq": seq, "kind": "tick_rewind", "before": bt, "after": at})
                previous_after, last_tick = at, at
                if include_traces and entry["kind"] == "action":
                    for suffix in ("trace", "trace/events"):
                        trace_cursor, part = 0, 0
                        while True:
                            trace = remote.request(f"/v1/history/{quote(entry['id'], safe='')}/{suffix}?after_seq={trace_cursor}&limit=128")
                            record(f"traces/{seq:08d}/{suffix.replace('/', '-')}-{part:06d}.json", trace)
                            if not trace.get("available"):
                                trace_gaps.append({"seq": seq, "kind": suffix, "reason": trace.get("reason")})
                            if not trace.get("has_more"):
                                break
                            next_cursor = trace["next_after_seq"]
                            if next_cursor <= trace_cursor:
                                raise RuntimeError("Trace pagination did not advance")
                            trace_cursor, part = next_cursor, part+1
            if not page["has_more"]:
                break
            if page["next_after_seq"] <= cursor:
                raise RuntimeError("History pagination did not advance")
            cursor = page["next_after_seq"]
        after = remote.request("/v1/history/status")["history"]
        final = remote.session()
        stable_keys = ("session_id", "last_seq", "retained_records", "evicted_records", "recording_fault")
        stable = all(before.get(k) == after.get(k) for k in stable_keys) and initial["session_id"] == final["session_id"] and final["paused"]
        report = {"schema": "gar.evidence-export.v1", "created_at": utc(),
                  "session_id": initial["session_id"], "stable_export": stable,
                  "retention_before": before, "retention_after": after,
                  "retained_records_exported": count, "trace_requested": include_traces,
                  "trace_gaps": trace_gaps, "incomplete_records": incomplete,
                  "recorded_discontinuities": transitions, "starts_at_tick": first_tick,
                  "ends_at_tick": last_tick, "artifacts": artifacts,
                  "claims": {
                      "uninterrupted_rollout_verified": False,
                      "reason": "Command history alone cannot prove the absence of unrecorded/direct manipulation.",
                      "video_created": False,
                      "physics_trace_completeness": "Inspect each trace header; retention/missing steps are not repaired.",
                      "missing_frames_recreated": False}}
        save(output / "manifest.json", report)
        if not stable:
            raise RuntimeError("History changed during export; retained export is marked unstable")
        return report
