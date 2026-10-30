"""One bounded policy rollout on an explicitly selected persistent session.

This is orchestration, not a solver. Plugins receive only policy observations.
Local Python plugins are trusted code, not an OS security sandbox; use the
restricted HTTP API in a separate process/host for untrusted agents.
"""
from copy import deepcopy
import base64
from datetime import datetime, timezone
import hashlib
from importlib import import_module
import inspect
import json
from pathlib import Path
import time
from uuid import uuid4

from .remote import RemoteEnv
from . import sessions


def utc():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def agent_factory(reference):
    parts = reference.split(":")
    if len(parts) != 2 or not all(parts):
        raise ValueError("Agent must be module:factory")
    module = import_module(parts[0])
    factory = getattr(module, parts[1])
    if not callable(factory):
        raise ValueError("Agent factory is not callable")
    source = inspect.getsourcefile(factory)
    return factory, {"entry_point": reference, "source_path": source,
                     "source_sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest() if source else None}


def capture(remote, observation, directory, index):
    """Save exact observation-ID images before any subsequent action."""
    result = deepcopy(observation)
    folder = directory / f"observation-{index:06d}"
    folder.mkdir()
    save(folder / "observation.json", observation)
    for number, (name, info) in enumerate(result["images"].items()):
        route = info["url"]
        if not route.startswith("/v1/cameras/") or route.startswith("//"):
            raise ValueError("Observation camera reference is not a same-server camera route")
        payload = remote.request(route, binary=True)
        # Image metadata, not an unsafe camera name, determines no filesystem path.
        path = folder / f"camera-{number:03d}.bin"
        path.write_bytes(payload)
        info["data_base64"] = base64.b64encode(payload).decode("ascii")
        info.pop("url")
        save(folder / f"camera-{number:03d}.json", {
            "name": name, "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest(),
            "mime_type": info.get("mime_type"), "observation_id": observation["observation_id"]})
    return result


def run(session, *, agent, max_actions, output, resume=False, options=None, timeout_s=None):
    if type(max_actions) is not int or max_actions < 1:
        raise ValueError("max_actions must be a positive integer; never clamped")
    if type(resume) is not bool:
        raise ValueError("resume must be boolean")
    options = {} if options is None else options
    if not isinstance(options, dict):
        raise ValueError("Agent options must be an object")
    options = json.loads(json.dumps(options, allow_nan=False))
    directory = sessions.resolve(session)
    manifest = json.loads((directory / "session.json").read_text())
    keys = json.loads(Path(manifest["launch"]["credential_file"]).read_text())
    if not keys.get("policy"):
        raise ValueError("A restricted credential is required for benchmark actions")
    owner = sessions.client(directory, timeout_s=timeout_s)      # None: the session's declared client timeout
    policy = RemoteEnv(owner.url, api_key=keys["policy"], timeout_s=owner.timeout_s)
    result = None
    admitted = False
    try:
        identity = owner.session()
        if identity["paused"] and not resume:
            raise ValueError("Session is paused; use explicit --resume to begin the benchmark")
        help_contract = policy.request("/v1/help")
        if help_contract.get("role") != "policy":
            raise ValueError("Benchmark action credential must be restricted")
        evaluation = owner.evaluate()
        if not evaluation["available"]:
            raise ValueError("This configuration has no shared task evaluator; no actions executed")
        factory, provenance = agent_factory(agent)
        task = policy.request("/v1/task")
        for info in task.get("reference_images", {}).values():
            route = info.pop("url")
            if not route.startswith("/v1/task/images/"):
                raise ValueError("Task picture reference is not a same-server task image route")
            payload = policy.request(route, binary=True)
            info.update(data_base64=base64.b64encode(payload).decode("ascii"), bytes=len(payload),
                        sha256=hashlib.sha256(payload).hexdigest())
        context = {"schema": "gar.agent-context.v1",
                   "task": task,
                   "capabilities": policy.request("/v1/capabilities"),
                   "help": help_contract}
        policy_fn = factory(deepcopy(context), deepcopy(options))
        if not callable(policy_fn):
            raise ValueError("Factory must return callable(observation) -> action or None")
        output = Path(output).resolve()
        output.mkdir(parents=True, exist_ok=False)
        before = owner.get_state()
        history_before = owner.request("/v1/history/status")["history"]
        result = {"schema": "gar.benchmark-run.v1", "run_id": uuid4().hex,
                  "session_id": identity["session_id"], "started_at": utc(),
                  "status": "running", "max_actions": max_actions, "actions_completed": 0,
                  "agent": provenance, "agent_options": options,
                  "source": sessions._source(), "launch": manifest["launch"]["spec"],
                  "history_before": history_before, "evaluation_before": evaluation,
                  "image_sampling": "Initial and after each command; no intermediate physics-step frames",
                  "intervention": "robot_actions_only", "reset_or_restore_performed": False}
        task_path = directory / "task.json"
        if task_path.exists():
            result["task_at_session_launch"] = json.loads(task_path.read_text())
            result["task_identity_note"] = "Launch task may differ after owner reset/variation; live policy instruction is in context.json."
        save(output / "started.json", result)
        save(output / "context.json", context)
        admitted = True
        start = time.monotonic()
        if identity["paused"]:
            owner.resume()
        observation = policy.obs()
        request_ids = []
        for index in range(max_actions):
            visible = capture(policy, observation, output, index)
            action = policy_fn(visible)
            if action is None:
                result["stop_reason"] = "agent_finished"
                break
            if not isinstance(action, dict) or "actions" in action:
                raise ValueError("Agent must return one JSON action, not a batch, or None")
            json.dumps(action, allow_nan=False)
            key = uuid4().hex
            save(output / f"action-{index:06d}.json", {"action": action, "idempotency_key": key, "requested_at": utc()})
            response = policy.act(action, idempotency_key=key)
            save(output / f"response-{index:06d}.json", response)
            request_ids.append(response.get("request_id"))
            result["actions_completed"] += 1
            # A committed command without its camera result must never be retried.
            if response.get("committed") is not True or response.get("observation") is None:
                raise RuntimeError("Action response lacks confirmed commitment/observation; inspect history, do not retry")
            observation = response["observation"]
        else:
            result["stop_reason"] = "max_actions"
            capture(policy, observation, output, max_actions)
        owner.pause()
        result.update(status="completed", wall_s=time.monotonic()-start,
                      evaluation=owner.evaluate(), history_after=owner.request("/v1/history/status")["history"],
                      request_ids=request_ids)
        cursor = history_before.get("last_seq") or 0
        rows = []
        while True:
            page = owner.request(f"/v1/history?after_seq={cursor}&limit=200")
            rows.extend(page["records"])
            if not page["has_more"]:
                break
            if page["next_after_seq"] <= cursor:
                raise RuntimeError("History pagination did not advance")
            cursor = page["next_after_seq"]
        actual_ids = list(dict.fromkeys(r["request_id"] for r in rows))
        start_seq = history_before.get("last_seq") or 0
        end_seq = result["history_after"].get("last_seq") or 0
        attributable = (
            result["history_after"]["session_id"] == history_before["session_id"]
            and [r["seq"] for r in rows] == list(range(start_seq + 1, end_seq + 1))
            and all(r["kind"] == "action" and r["role"] == "policy" for r in rows)
            and actual_ids == request_ids and all(request_ids)
        )
        result["exclusive_recorded_interval"] = attributable
        if not attributable:
            raise RuntimeError("Concurrent mutation, history loss or missing correlation; outcome cannot be attributed to this agent")
        after = owner.get_state()
        time_key = next((k for k in ("time_s", "sim_time_s") if k in before and k in after), None)
        simulated = after[time_key]-before[time_key] if time_key else None
        result["simulated_s"] = simulated
        result["wall_sim_ratio"] = result["wall_s"]/simulated if simulated and simulated > 0 else None
        result["batch_size"] = 1
        result["tick_before"], result["tick_after"] = before.get("tick"), after.get("tick")
    except BaseException as exc:
        if result is None:
            raise
        result.update(status="failed", error={"type": type(exc).__name__, "message": str(exc)})
    finally:
        if admitted:
            try:
                result["final_session"] = owner.pause()
            except Exception as exc:
                result["pause_error"] = {"type": type(exc).__name__, "message": str(exc)}
                result["status"] = "failed"
            result["finished_at"] = utc()
            save(output / "result.json", result)
        policy.close()
        owner.close()
    return result
