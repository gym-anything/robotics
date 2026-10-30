"""Self-describing HTTP access to one persistent robotics environment."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from copy import deepcopy
from contextlib import contextmanager
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Any, Mapping
import urllib.parse
from uuid import uuid4

from .action_codec import ACTION_HELP, actions_from_request
from .actions import arm_names
from .action_history import ActionHistory, HistoryRecorder, DEFAULT_MAX_BYTES, DEFAULT_MAX_RECORDS
from .physics_trace import PhysicsTraceStore
from .access import APIKeys, AccessDeniedError, INTERNAL, Principal
from .env import RoboticsEnv
from .errors import (
    ActionRejectedError,
    BackendProtocolError,
    ObservationUnavailableAfterCommitError,
    RoboticsEnvironmentError,
    UnsupportedActionError,
)
from .types import ImageFrame, Observation


DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8080
MAX_BODY_BYTES = 1_000_000
OBSERVATION_HISTORY = 8
IDEMPOTENCY_HISTORY = 256
API_VERSION = "v1"


class ActionExecutionError(RuntimeError):
    """An exception of no kind the gateway classifies (not a refusal, runtime, OS or value error), raised while a
    command's backend call was running. That is a server fault, not a bad request: it is answered 500 with its type,
    not the generic handler's KeyError/TypeError 400 "Invalid request". It does not affect whether the session
    latches a runtime fault."""

    def __init__(self, cause: Exception, *, action_index: int, actions: int, completed_actions: int,
                 request_id: str) -> None:
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.cause = cause
        self.action_index = action_index
        self.actions = actions
        self.completed_actions = completed_actions
        self.request_id = request_id

    def response(self, restricted: bool) -> dict[str, Any]:
        text = (f"Server error ({type(self.cause).__name__}) while command {self.action_index + 1} of {self.actions} "
                "was executing: that command stopped before it finished and the robot may have moved; "
                f"{self.completed_actions} earlier command(s) of this request completed and no later one ran. "
                "This is a fault in the server, not in the request.")
        return {"ok": False, "error_type": type(self).__name__,
                "error": text if restricted else f"{text} {self}",
                "completed_actions": self.completed_actions, "failed_action_index": self.action_index,
                "request_id": self.request_id}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("HTTP responses cannot contain non-finite numbers")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    if hasattr(value, "item"):
        return _json_safe(value.item())
    raise TypeError(f"value of type {type(value).__name__} is not JSON serializable")


class GARHTTPService:
    """Serialize HTTP operations against exactly one live ``RoboticsEnv``."""
    # Opt-in for the evaluation protocol; ordinary servers retain their contract.
    # A command rejected before execution, after earlier commands of the same request completed, ends only the
    # request: the response lists what ran and what did not, and the session continues (no runtime fault).
    recover_known_partial_batches = True

    def __init__(self, env: RoboticsEnv, *, history: ActionHistory | None = None) -> None:
        self.env = env
        self.history = HistoryRecorder(history if history is not None else ActionHistory(), env.get_state)
        self._lock = threading.RLock()
        self._observation_id = 0
        self._observations: OrderedDict[int, dict[str, ImageFrame]] = OrderedDict()
        self._observation_owners: dict[int, str] = {}
        self._idempotent_actions: OrderedDict[
            tuple[str, str], tuple[str, dict[str, Any]]
        ] = OrderedDict()
        self._idempotent_starts: OrderedDict[
            str, tuple[str, dict[str, Any]]
        ] = OrderedDict()
        self._observation_fault: str | None = None
        self._runtime_fault: tuple[str, str] | None = None
        if env.last_observation is not None:
            self._publish_observation(env.last_observation)

    def _publish_observation(
        self, observation: Observation, principal: Principal = INTERNAL,
    ) -> dict[str, Any]:
        if principal.restricted:
            observation = self.env.policy_access().observation(observation)
        self._observation_id += 1
        observation_id = self._observation_id
        self._observations[observation_id] = dict(observation.images)
        self._observation_owners[observation_id] = principal.namespace
        while len(self._observations) > OBSERVATION_HISTORY:
            expired_id, _ = self._observations.popitem(last=False)
            self._observation_owners.pop(expired_id, None)

        encoded = observation.to_dict()
        encoded["observation_id"] = observation_id
        for name, image in encoded["images"].items():
            image.pop("path", None)
            image["url"] = (
                f"/v1/cameras/{urllib.parse.quote(name, safe='')}"
                f"?observation_id={observation_id}"
            )
        published = _json_safe(encoded)
        self._observation_fault = None
        return published

    def _committed_without_observation(
        self,
        error: ObservationUnavailableAfterCommitError,
        principal: Principal = INTERNAL,
    ) -> dict[str, Any]:
        cause = error.cause
        self._observation_fault = f"{type(cause).__name__}: {cause}"
        if principal.restricted:
            return {
                "ok": True, "committed": True, "observation": None,
                "observation_error": {
                    "type": "ObservationUnavailable",
                    "message": "Command executed; robot observation unavailable. Do not repeat the command.",
                },
            }
        return _json_safe(
            {
                "ok": True,
                "committed": True,
                "operation": error.operation,
                "sequence": error.sequence,
                "observation": None,
                "observation_error": {
                    "type": type(cause).__name__,
                    "message": str(cause),
                },
                "reward": error.reward,
                "terminated": error.terminated,
                "truncated": error.truncated,
                "info": error.info,
            }
        )

    def _record_runtime_fault(self, error: Exception) -> None:
        self._runtime_fault = (type(error).__name__, str(error))

    def _clear_runtime_fault(self) -> None:
        self._runtime_fault = None

    def health(self) -> dict[str, Any]:
        ready = (
            not self.env.closed
            and self._observation_fault is None
            and self._runtime_fault is None
            and self.history.recording_fault is None
        )
        return {
            "ok": ready,
            "service": "gym-anything-robotics-http",
            "api_version": API_VERSION,
            "environment_open": not self.env.closed,
            "ready": ready,
            "observation_pipeline_ok": self._observation_fault is None,
            "runtime_ok": self._runtime_fault is None,
            "history_ok": self.history.recording_fault is None,
            "runtime_fault_type": (
                None if self._runtime_fault is None else self._runtime_fault[0]
            ),
        }

    def capabilities(self, principal: Principal = INTERNAL) -> dict[str, Any]:
        if principal.restricted:
            return {"ok": True, "capabilities": self.env.policy_access().capabilities()}
        return {"ok": True, "capabilities": self.env.capabilities.to_dict()}

    def help(self, principal: Principal = INTERNAL) -> dict[str, Any]:
        if principal.restricted:
            return self.env.policy_access().help()
        return {
            "ok": True,
            "service": "Gym Anything for Robotics HTTP API",
            "api_version": API_VERSION,
            "role": "privileged",
            "authentication": {
                "header": "Authorization: Bearer <token>",
                "required_except": ["GET /v1/health"],
            },
            "routes": {
                "GET /v1/health": (
                    "Unauthenticated process and observation-pipeline readiness."
                ),
                "GET /v1/help": "This complete machine-readable contract.",
                "GET /v1/capabilities": "Supported actions, cameras, and reset features.",
                "GET /v1/task": "Human-readable current instruction, without evaluator diagnostics.",
                "GET /v1/task/images/{name}": "PNG picture that is part of the task, when /v1/task lists one.",
                "GET /v1/observation": "Fresh policy-visible proprioception and camera URLs.",
                "GET /v1/cameras/{name}?observation_id=N": (
                    "Binary image from the exact referenced observation."
                ),
                "GET /v1/state": "Complete privileged simulator state.",
                "GET /v1/evaluation": "Read-only owner task evaluator; unavailable is not failure or success.",
                "GET /v1/history?after_seq=0&limit=100&request_id=...": (
                    "Privileged paginated command summaries, oldest first; request_id filter is optional."
                ),
                "GET /v1/history/status": "Recording session, retention limits, gaps and storage faults.",
                "GET /v1/history/{id}": "Recorded command and full before/after privileged state snapshots.",
                "GET /v1/history/{id}/trace": "Privileged per-physics-step samples and explicit completeness; after_seq/start_tick/end_tick/limit filters.",
                "GET /v1/history/{id}/trace/events": "Privileged contact, connector, controller-stop and error timeline; same pagination filters.",
                "GET /v1/geometry": (
                    "Index of all simulator-authored objects and mesh assets."
                ),
                "GET /v1/geometry/{geometry_id}": (
                    "Complete analytic/mesh/collision descriptor for one object."
                ),
                "GET /v1/geometry/assets/{asset_id}": (
                    "Exact content-addressed vertices and triangle indices."
                ),
                "GET /v1/render": "Fresh binary diagnostic scene render.",
                "POST /v1/action": (
                    "One action object, or {'actions': [...]} for an ordered sequence."
                ),
                "POST /v1/reset": "Body: {'seed': int|null, 'options': object|null}.",
                "POST /v1/recover": "Owner only: acknowledge uncertain actions, verify finite controller state and fresh cameras, and clear runtime fault without reset/replay. Body: {'acknowledge_uncertain_actions': true}.",
                "GET /v1/variations": "Supported selectors, constraints, and current episode.",
                "POST /v1/variations/select": "Preview a matching episode and task without changing state.",
                "POST /v1/variations/start": "Select and load a matching episode on this server; replaces the current episode.",
                "GET /v1/checkpoints": "List persistent physics checkpoints.",
                "POST /v1/checkpoints/save": "Body: {'label': string}.",
                "POST /v1/checkpoints/restore": "Body: {'checkpoint': string}.",
            },
            "transaction_semantics": {
                "action_response": (
                    "The returned observation is captured after the complete action. "
                    "If capture fails after commit, the response remains ok=true and "
                    "committed=true with observation_error, so clients must not retry "
                    "the action as though it never executed."
                ),
                "camera_consistency": (
                    "Every camera URL contains the matching monotonic observation_id."
                ),
                "serialization": "All state-changing requests execute one at a time.",
                "timeout_recovery": "For traced Newton actions the socket watchdog waits longer only while durable physics steps advance, without resending commands. No progress still faults. POST /v1/recover lets the owner acknowledge uncertain outcomes and verify state/cameras without reset or replay. Client HTTP deadlines remain client-controlled.",
                "idempotency": (
                    "POST /v1/action and POST /v1/variations/start optionally accept "
                    "an Idempotency-Key header; an identical retry returns the original "
                    "response without repeating the operation. Action histories are scoped "
                    "by credential. Histories last for this HTTP process only."
                ),
            },
            "information_boundary": {
                "policy": "/v1/observation",
                "privileged": "/v1/state",
                "privileged_geometry": "/v1/geometry",
                "privileged_history": "/v1/history",
                "diagnostic_camera": "/v1/render",
            },
            "action_types": deepcopy(ACTION_HELP),
            "history": {
                "scope": "Each decoded HTTP action, plus HTTP resets and checkpoint restores. Both roles are recorded.",
                "correlation": "Action responses include request_id; X-GAR-Request-ID also accompanies action errors. Each executed array member has its own record id and zero-based action_index.",
                "granularity": "Before/after every command. Instrumented backends additionally record every physics step; inspect each record's trace and completeness. No extra physics steps or images are produced.",
                "limits": "Retention and missing/incomplete evidence are explicit in /v1/history/status. No history exists for earlier unrecorded actions or operations through other services.",
            },
            "variation_selection": self.env.variation_contract(),
            "effective_capabilities": self.env.capabilities.to_dict(),
        }

    def variations(self) -> dict[str, Any]:
        with self._lock:
            return {"ok": True, "variations": _json_safe(self.env.variation_contract())}

    def select_variation(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            return {"ok": True, "committed": False, **_json_safe(self.env.select_variation(payload))}

    def start_variation(
        self, payload: Mapping[str, Any], *, idempotency_key: str | None,
    ) -> dict[str, Any]:
        if idempotency_key is not None and not 1 <= len(idempotency_key) <= 200:
            raise ValueError("Idempotency-Key must contain 1 to 200 characters")
        canonical = self._canonical_request(payload)
        with self._lock:
            if idempotency_key is not None and idempotency_key in self._idempotent_starts:
                previous_request, response = self._idempotent_starts[idempotency_key]
                if previous_request != canonical:
                    raise ValueError("Idempotency-Key was already used for a different variation start")
                return deepcopy(response)
            # Selection/validation must finish before reset or fault bookkeeping.
            selected = _json_safe(self.env.select_variation(payload))
            response = {**self.reset(selected["reset_request"]), **selected}
            if idempotency_key is not None:
                self._idempotent_starts[idempotency_key] = (canonical, deepcopy(response))
                while len(self._idempotent_starts) > IDEMPOTENCY_HISTORY:
                    self._idempotent_starts.popitem(last=False)
            return response

    def evaluation(self) -> dict[str, Any]:
        with self._lock:
            if self._runtime_fault is not None:
                raise BackendProtocolError("Service has an unresolved runtime fault")
            try:
                return {"ok": True, "evaluation": _json_safe(self.env.evaluate())}
            except (RoboticsEnvironmentError, OSError, ValueError) as error:
                self._record_runtime_fault(error)
                raise

    def state(self) -> dict[str, Any]:
        with self._lock:
            try:
                state = _json_safe(self.env.get_state())
            except (RoboticsEnvironmentError, OSError, ValueError) as error:
                self._record_runtime_fault(error)
                raise
            return {"ok": True, "state": state}

    def recover(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Owner reconciliation, not a reset, retry, or unconditional ready flag."""
        if dict(payload) != {"acknowledge_uncertain_actions": True} or payload.get("acknowledge_uncertain_actions") is not True:
            raise ValueError("Recovery requires acknowledge_uncertain_actions:true; review history before resuming")
        with self._lock:
            if self.history.recording_fault is not None:
                raise RoboticsEnvironmentError("History storage remains faulted; repair it before recovery")
            previous_fault = self._runtime_fault
            try:
                with self.history.span(kind="owner_recovery", payload=payload,
                                       request_id=str(uuid4()), role="privileged"):
                    state = _json_safe(self.env.get_state())
                    if state.get("finite") is False:
                        raise RoboticsEnvironmentError("Controller reports non-finite state; recovery refused")
                    observation = self.env.obs()
                    published = self._publish_observation(observation)
                if self.history.recording_fault is not None:
                    raise RoboticsEnvironmentError("Recovery could not be durably recorded")
            except (RoboticsEnvironmentError, OSError, ValueError) as error:
                self._record_runtime_fault(error)
                raise
            self._clear_runtime_fault()
            return {"ok": True, "recovered": True, "reset_performed": False,
                    "previous_fault": previous_fault, "state": state,
                    "observation": published, "health": self.health()}

    def task(self) -> dict[str, Any]:
        with self._lock:
            access = self.env.policy_access()
            result = {"ok": True, "instruction": access.instruction()}
            if access.task_images is not None:
                result["reference_images"] = {
                    name: {"url": f"/v1/task/images/{urllib.parse.quote(name)}", "mime_type": "image/png",
                           "description": description}
                    for name, (description, _) in access.task_images().items()
                }
            return result

    def task_image(self, name: str) -> bytes | None:
        with self._lock:
            images = self.env.policy_access().task_images
            image = images().get(name) if images is not None else None
            return None if image is None else image[1]

    def observation(self, principal: Principal = INTERNAL) -> dict[str, Any]:
        with self._lock:
            return {"ok": True, "observation": self._publish_observation(self.env.obs(), principal)}

    def geometry_index(self) -> dict[str, Any]:
        with self._lock:
            return {"ok": True, "geometry": _json_safe(self.env.geometry_index())}

    def geometry(self, geometry_id: str) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "geometry": _json_safe(self.env.geometry(geometry_id)),
            }

    def geometry_asset(self, asset_id: str) -> dict[str, Any]:
        with self._lock:
            return {
                "ok": True,
                "asset": _json_safe(self.env.geometry_asset(asset_id)),
            }

    def reset(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(payload)
        unexpected = set(values) - {"seed", "options"}
        if unexpected:
            raise ValueError(f"reset has unexpected fields: {sorted(unexpected)}")
        seed = values.get("seed")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError("reset seed must be an integer or null")
        options = values.get("options")
        if options is not None and not isinstance(options, Mapping):
            raise ValueError("reset options must be a JSON object or null")
        with self._lock:
            try:
                with self.history.span(kind="reset", payload=values, request_id=str(uuid4()),
                                       role="privileged", allow_missing_before=True):
                    observation, info = self.env.reset(seed=seed, options=options)
            except ObservationUnavailableAfterCommitError as error:
                self._clear_runtime_fault()
                return self._committed_without_observation(error)
            except (ActionRejectedError, UnsupportedActionError):
                # Invalid reset options neither reset the simulator nor repair
                # an existing fault. Preserve the previous readiness state.
                raise
            except (RoboticsEnvironmentError, OSError, ValueError) as error:
                self._record_runtime_fault(error)
                raise
            self._clear_runtime_fault()
            return {
                "ok": True,
                "committed": True,
                "observation": self._publish_observation(observation),
                "info": _json_safe(info),
            }

    @staticmethod
    def _canonical_request(payload: Mapping[str, Any]) -> str:
        return json.dumps(payload, sort_keys=True, separators=(",", ":"))

    def action(
        self,
        payload: Mapping[str, Any],
        *,
        idempotency_key: str | None,
        principal: Principal = INTERNAL,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        received_perf_s = time.perf_counter()   # request receipt: backends that budget wall time per command (a
                                                # minimum command length) can start their clock here, before the
                                                # durable history commits that precede execution
        if idempotency_key is not None:
            if not 1 <= len(idempotency_key) <= 200:
                raise ValueError("Idempotency-Key must contain 1 to 200 characters")
        canonical = self._canonical_request(payload)
        cache_key = (principal.namespace, idempotency_key)
        with self._lock:
            # A backend may declare arm names of its own (RoboticsEnv.extra_arm_names); none by default.
            declared = getattr(self.env, "extra_arm_names", None)
            declared = declared() if callable(declared) else ()
            with arm_names(declared if isinstance(declared, (tuple, list, set, frozenset)) else ()):
                action = (self.env.policy_access().decode_actions(payload) if principal.restricted
                          else actions_from_request(payload))
            if idempotency_key is not None and cache_key in self._idempotent_actions:
                previous_request, previous_response = self._idempotent_actions[
                    cache_key
                ]
                if previous_request != canonical:
                    raise ValueError(
                        "Idempotency-Key was already used for a different action"
                    )
                return deepcopy(previous_response)

            if self._runtime_fault is not None:
                raise RoboticsEnvironmentError("Owner recovery is required before further motion; do not resend uncertain actions")
            request_id = request_id or str(uuid4())
            commands = payload["actions"] if "actions" in payload else [payload]
            completed_actions = 0
            executing = None                      # index of the command whose backend call has begun

            @contextmanager
            def action_scope(index: int):
                nonlocal completed_actions, executing
                with self.history.span(
                    kind="action", payload=commands[index], request_id=request_id,
                    role=principal.role, action_index=index, idempotency_key=idempotency_key,
                ) as entry:
                    try:
                        try:
                            entry["trace"] = self.env.begin_action_trace({"action_id": entry["id"], "request_id": request_id,
                                                                          "received_perf_s": received_perf_s})
                        except Exception as error:
                            entry["trace"] = {"available": False, "error": f"{type(error).__name__}: {error}"}
                        self.history.store.write(entry, update=True)
                        executing = index
                        yield
                        completed_actions += 1
                    finally:
                        try:
                            summary = self.env.end_action_trace()
                            if summary is not None:
                                entry.setdefault("trace", {})["finalization"] = summary
                        except Exception as error:
                            entry.setdefault("trace", {})["finalization_error"] = f"{type(error).__name__}: {error}"

            try:
                execute = self.env.act_policy if principal.restricted else self.env.act
                result = execute(action, action_scope=action_scope)
            except ObservationUnavailableAfterCommitError as error:
                response = self._committed_without_observation(error, principal)
            except (ActionRejectedError, UnsupportedActionError) as error:
                if completed_actions:
                    if self.recover_known_partial_batches:
                        # Every prior command completed, and this rejection guarantees
                        # the failing command did not execute. There is no uncertain
                        # motion to replay, nor a physics fault to clear.
                        response = {"ok": False, "committed": True, "partial": True,
                                    "completed_actions": completed_actions,
                                    "rejected_action_index": completed_actions,
                                    "unexecuted_action_indices": list(range(completed_actions, len(commands))),
                                    "error_type": type(error).__name__,
                                    "error": "A command was rejected. Earlier commands completed; do not replay the batch.",
                                    "observation": None}
                        try:
                            response["observation"] = self._publish_observation(self.env.obs(), principal)
                        except Exception as observation_error:
                            response["observation_error"] = {
                                "error_type": type(observation_error).__name__,
                                "error": "Observation unavailable; request another observation, not the motion."}
                        response["request_id"] = request_id
                        response = _json_safe(response)
                        if idempotency_key is not None:
                            self._idempotent_actions[cache_key] = (canonical, response)
                            owned = [key for key in self._idempotent_actions if key[0] == principal.namespace]
                            for expired in owned[:-IDEMPOTENCY_HISTORY]:
                                del self._idempotent_actions[expired]
                        return deepcopy(response)
                    # Earlier commands have already run. Never describe the
                    # entire batch as unexecuted or permit an uncertain replay.
                    failure = RoboticsEnvironmentError(
                        f"Action sequence partially executed: {completed_actions} command(s) completed "
                        "before a command was rejected; review history and recover before further motion"
                    )
                    self._record_runtime_fault(failure)
                    raise failure from error
                # Explicit backend guarantee: no target writes or physics steps.
                # Unknown ValueErrors below still latch a runtime fault.
                raise
            except (RoboticsEnvironmentError, OSError, ValueError) as error:
                self._record_runtime_fault(error)
                raise
            except (NotImplementedError, sqlite3.Error):
                raise
            except Exception as error:
                if executing is None:             # nothing had started executing: handled as before
                    raise
                raise ActionExecutionError(error, action_index=executing, actions=len(commands),
                                           completed_actions=completed_actions, request_id=request_id) from error
            else:
                try:
                    response = {
                        "ok": True, "committed": True,
                        "observation": self._publish_observation(result.observation, principal),
                    }
                    if not principal.restricted:
                        response.update(reward=result.reward, terminated=result.terminated,
                                        truncated=result.truncated, info=_json_safe(result.info))
                except Exception as error:
                    response = self._committed_without_observation(
                        ObservationUnavailableAfterCommitError(
                            operation="action", sequence=result.observation.sequence,
                            info=result.info, cause=error, reward=result.reward,
                            terminated=result.terminated, truncated=result.truncated,
                        ), principal,
                    )
            response["request_id"] = request_id
            response = _json_safe(response)
            if idempotency_key is not None:
                self._idempotent_actions[cache_key] = (canonical, response)
                owned = [key for key in self._idempotent_actions if key[0] == principal.namespace]
                for expired in owned[:-IDEMPOTENCY_HISTORY]:
                    del self._idempotent_actions[expired]
            return deepcopy(response)

    def camera(
        self,
        name: str,
        *,
        observation_id: int | None,
        principal: Principal = INTERNAL,
    ) -> tuple[ImageFrame, int]:
        with self._lock:
            if not self._observations:
                raise ValueError("no camera observation has been captured")
            selected_id = (
                next(reversed(self._observations))
                if observation_id is None
                else observation_id
            )
            if principal.restricted:
                if observation_id is None:
                    owned = [key for key in self._observations
                             if self._observation_owners[key] == principal.namespace]
                    if not owned:
                        raise AccessDeniedError("Capture a policy observation first")
                    selected_id = owned[-1]
                if self._observation_owners.get(selected_id) != principal.namespace:
                    raise AccessDeniedError("Camera observation is not available to this credential")
            try:
                frames = self._observations[selected_id]
            except KeyError as error:
                raise ValueError(
                    f"observation_id {selected_id} is unavailable; "
                    f"retained IDs are {list(self._observations)}"
                ) from error
            try:
                frame = frames[name]
            except KeyError as error:
                raise ValueError(
                    f"camera {name!r} is unavailable; expected one of {sorted(frames)}"
                ) from error
            if not frame.path.is_file():
                raise FileNotFoundError(f"camera frame is missing: {frame.path}")
            if principal.restricted and not self.env.policy_access().camera_allowed(frame):
                raise AccessDeniedError("Camera is not available to this credential")
            return frame, selected_id

    def render(self, label: str | None = None) -> ImageFrame:
        # not under the service lock: RoboticsEnv.render takes the environment's own lock, and a backend with
        # prepare_render runs its slow offline render outside it (a policy command sent during an owner's diagnostic
        # render would otherwise wait 13.5-26.6 s behind this lock)
        return self.env.render(label="http" if label is None else label)

    def checkpoints(self) -> dict[str, Any]:
        with self._lock:
            return {"ok": True, "checkpoints": _json_safe(self.env.list_checkpoints())}

    def save_checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(payload)
        if set(values) != {"label"} or not isinstance(values["label"], str):
            raise ValueError("checkpoint save requires exactly one string label")
        with self._lock:
            return {
                "ok": True,
                "checkpoint": _json_safe(
                    self.env.save_checkpoint(label=values["label"])
                ),
            }

    def restore_checkpoint(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        values = dict(payload)
        if set(values) != {"checkpoint"} or not isinstance(
            values["checkpoint"], str
        ):
            raise ValueError("checkpoint restore requires exactly one string checkpoint")
        with self._lock:
            try:
                with self.history.span(kind="checkpoint_restore", payload=values,
                                       request_id=str(uuid4()), role="privileged", allow_missing_before=True):
                    observation, info = self.env.restore_checkpoint(values["checkpoint"])
            except ObservationUnavailableAfterCommitError as error:
                self._clear_runtime_fault()
                return self._committed_without_observation(error)
            except (ActionRejectedError, UnsupportedActionError):
                # A restore refused before touching the simulator (e.g. a checkpoint
                # from another variation) neither changes the world nor repairs an
                # existing fault: preserve the readiness state, as reset does.
                raise
            except (RoboticsEnvironmentError, OSError, ValueError) as error:
                self._record_runtime_fault(error)
                raise
            self._clear_runtime_fault()
            return {
                "ok": True,
                "committed": True,
                "observation": self._publish_observation(observation),
                "info": _json_safe(info),
            }


class GARHTTPServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        service: GARHTTPService,
        *,
        api_token: str,
        policy_api_token: str | None = None,
    ) -> None:
        self.keys = APIKeys(api_token, policy_api_token)
        if policy_api_token is not None:
            service.env.policy_access()  # Unsupported backends fail before binding.
        self.service = service
        super().__init__(address, GARHTTPHandler)
        self.api_token = api_token

    def server_close(self) -> None:
        super().server_close()
        self.service.history.store.close()


class GARHTTPHandler(BaseHTTPRequestHandler):
    server: GARHTTPServer

    def log_message(self, _format: str, *_args: object) -> None:
        return

    def _send_json(self, value: Any, status: int = HTTPStatus.OK) -> None:
        payload = json.dumps(
            _json_safe(value), separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            request_id = value.get("request_id") if isinstance(value, dict) else None
            request_id = request_id or getattr(self, "request_id", None)
            if request_id is not None:
                self.send_header("X-GAR-Request-ID", request_id)
            self.end_headers()
            self.wfile.write(payload)
        except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
            # The completed command/history/idempotency result still exists.
            # A disconnected HTTP reader is not a simulator runtime failure.
            self.close_connection = True

    def _error(self, status: int, error: Exception | str) -> None:
        principal = getattr(self, "principal", None)
        if isinstance(error, ActionExecutionError):
            self._send_json(error.response(principal is not None and principal.restricted), status)
            return
        rejected = (
            status == HTTPStatus.BAD_REQUEST and isinstance(error, ActionRejectedError)
        ) or (
            status == HTTPStatus.UNPROCESSABLE_ENTITY and isinstance(error, UnsupportedActionError)
        )
        if rejected:
            # A backend may attach `policy_reason` to a rejection: text derived only from the command and the robot's
            # own state (what a real controller reports to its operator, e.g. "joint jump 0.40 rad"). It is the only
            # backend text a restricted caller sees; without it the generic sentence stands alone.
            reason = getattr(error, "policy_reason", None)
            reason = f" Reason: {reason[:400]}" if isinstance(reason, str) and reason.strip() else ""
            self._send_json({
                "ok": False, "committed": False, "error_type": type(error).__name__,
                "error": (
                    ("Robot command is unsupported; no action executed. Consult the robot capabilities."
                     if isinstance(error, UnsupportedActionError) else
                     "Robot command rejected before execution; adjust the target or command parameters." + reason)
                    if principal is not None and principal.restricted else str(error)
                ),
            }, status)
            return
        if principal is not None and principal.restricted:
            self._send_json({
                "ok": False,
                "error_type": "AccessDenied" if status == HTTPStatus.FORBIDDEN else "RequestError",
                "error": {
                    HTTPStatus.FORBIDDEN: "Not permitted for this credential.",
                    HTTPStatus.BAD_REQUEST: "Invalid request; consult /v1/help.",
                    HTTPStatus.UNPROCESSABLE_ENTITY: "Unsupported robot command.",
                    HTTPStatus.BAD_GATEWAY: "Robot service unavailable; do not blindly retry a motion.",
                }.get(status, "Request failed."),
            }, status)
            return
        self._send_json(
            {
                "ok": False,
                "error_type": (
                    type(error).__name__ if isinstance(error, Exception) else "Error"
                ),
                "error": str(error),
            },
            status,
        )

    def _authorized(self) -> bool:
        self.principal = self.server.keys.authenticate(self.headers.get("Authorization", ""))
        if self.principal is not None:
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", "Bearer")
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _route_allowed(self, method: str, path: str) -> bool:
        if not self.principal.restricted:
            return True
        allowed = (
            method == "GET" and (path in {
                "/", "/help", "/v1/help", "/v1/capabilities", "/v1/task", "/v1/observation",
            } or path.startswith(("/v1/cameras/", "/v1/task/images/")))
        ) or (method == "POST" and path == "/v1/action")
        if not allowed:
            self._error(HTTPStatus.FORBIDDEN, "Not permitted")
        return allowed

    def _body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as error:
            raise ValueError("invalid Content-Length") from error
        if not 0 <= length <= MAX_BODY_BYTES:
            raise ValueError(f"request body must be at most {MAX_BODY_BYTES} bytes")
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body must be valid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _send_image(
        self,
        frame: ImageFrame,
        *,
        observation_id: int | None = None,
    ) -> None:
        body = frame.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header(
            "Content-Type",
            frame.mime_type
            or mimetypes.guess_type(frame.path.name)[0]
            or "application/octet-stream",
        )
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if observation_id is not None:
            self.send_header("X-GAR-Observation-ID", str(observation_id))
        self.send_header("X-GAR-Deployable", "true" if frame.deployable else "false")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/v1/health":
            self._send_json(self.server.service.health())
            return
        if not self._authorized():
            return
        if not self._route_allowed("GET", parsed.path):
            return
        try:
            if parsed.path in {"/", "/help", "/v1/help"}:
                self._send_json(self.server.service.help(self.principal))
            elif parsed.path == "/v1/capabilities":
                self._send_json(self.server.service.capabilities(self.principal))
            elif parsed.path == "/v1/task":
                self._send_json(self.server.service.task())
            elif parsed.path.startswith("/v1/task/images/"):
                body = self.server.service.task_image(urllib.parse.unquote(parsed.path[len("/v1/task/images/"):]))
                if body is None:
                    self._error(HTTPStatus.NOT_FOUND, "This task has no picture with that name")
                    return
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
            elif parsed.path == "/v1/evaluation":
                self._send_json(self.server.service.evaluation())
            elif parsed.path == "/v1/state":
                self._send_json(self.server.service.state())
            elif parsed.path == "/v1/history/status":
                self._send_json({"ok": True, "history": self.server.service.history.status()})
            elif parsed.path == "/v1/history":
                query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                if set(query) - {"after_seq", "limit", "request_id"} or any(len(v) != 1 for v in query.values()):
                    raise ValueError("Unexpected or repeated history query parameter")
                self._send_json({"ok": True, **self.server.service.history.store.list(
                    after_seq=int(query.get("after_seq", ["0"])[0]),
                    limit=int(query.get("limit", ["100"])[0]),
                    request_id=query.get("request_id", [None])[0],
                )})
            elif parsed.path.startswith("/v1/history/"):
                suffix = parsed.path[len("/v1/history/"):].split("/")
                try:
                    record = self.server.service.history.store.get(suffix[0])
                except KeyError:
                    self._error(HTTPStatus.NOT_FOUND, "History record unknown or evicted; inspect retention status")
                    return
                if len(suffix) == 1:
                    self._send_json({"ok": True, "record": record})
                elif suffix[1:] in (["trace"], ["trace", "events"]):
                    reference = record.get("trace", {})
                    if not reference.get("available"):
                        self._send_json({"ok": True, "available": False, "complete": False,
                                         "reason": "Substep trace was not recorded for this action", "details": reference})
                        return
                    query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                    if set(query) - {"after_seq", "start_tick", "end_tick", "limit"} or any(len(v) != 1 for v in query.values()):
                        raise ValueError("Unexpected or repeated trace query parameter")
                    options = {key: int(value[0]) for key, value in query.items()}
                    trace_path = Path(reference["store_path"])
                    if not trace_path.is_file():
                        self._send_json({"ok": True, "available": False, "complete": False,
                                         "reason": "Trace store is missing", "details": reference})
                        return
                    store = PhysicsTraceStore(trace_path, readonly=True)
                    try:
                        page = store.page(record["id"], kind="event" if suffix[-1] == "events" else "sample", **options)
                    except KeyError:
                        self._send_json({"ok": True, "available": False, "complete": False,
                                         "reason": "Trace not started, header evicted, or recording failed", "details": reference})
                        return
                    finally:
                        store.close()
                    self._send_json({"ok": True, "available": True, **page})
                else:
                    self._error(HTTPStatus.NOT_FOUND, "unknown history route")
            elif parsed.path == "/v1/variations":
                self._send_json(self.server.service.variations())
            elif parsed.path == "/v1/geometry":
                self._send_json(self.server.service.geometry_index())
            elif parsed.path.startswith("/v1/geometry/assets/"):
                asset_id = urllib.parse.unquote(
                    parsed.path[len("/v1/geometry/assets/") :]
                )
                if not asset_id:
                    raise ValueError("geometry asset id must not be empty")
                self._send_json(self.server.service.geometry_asset(asset_id))
            elif parsed.path.startswith("/v1/geometry/"):
                geometry_id = urllib.parse.unquote(
                    parsed.path[len("/v1/geometry/") :]
                )
                if not geometry_id:
                    raise ValueError("geometry id must not be empty")
                self._send_json(self.server.service.geometry(geometry_id))
            elif parsed.path == "/v1/observation":
                self._send_json(self.server.service.observation(self.principal))
            elif parsed.path.startswith("/v1/cameras/"):
                name = urllib.parse.unquote(parsed.path[len("/v1/cameras/") :])
                query = urllib.parse.parse_qs(parsed.query)
                raw_id = query.get("observation_id", [None])[0]
                observation_id = None if raw_id is None else int(raw_id)
                frame, selected_id = self.server.service.camera(
                    name,
                    observation_id=observation_id,
                    principal=self.principal,
                )
                self._send_image(frame, observation_id=selected_id)
            elif parsed.path == "/v1/render":
                # optional ?label=<camera>: the backend decides which labels it renders (default: its own "http" view)
                query = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
                if set(query) - {"label"} or any(len(v) != 1 for v in query.values()):
                    raise ValueError("Unexpected or repeated render query parameter (only label=<camera>)")
                self._send_image(self.server.service.render(query.get("label", [None])[0]))
            elif parsed.path == "/v1/checkpoints":
                self._send_json(self.server.service.checkpoints())
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown API route")
        except AccessDeniedError as error:
            self._error(HTTPStatus.FORBIDDEN, error)
        except UnsupportedActionError as error:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, error)
        except NotImplementedError as error:
            # A backend capability it does not offer (e.g. checkpoints).
            self._error(HTTPStatus.NOT_IMPLEMENTED, error)
        except (RoboticsEnvironmentError, OSError, sqlite3.Error) as error:
            self._error(HTTPStatus.BAD_GATEWAY, error)
        except (KeyError, TypeError, ValueError) as error:
            self._error(HTTPStatus.BAD_REQUEST, error)

    def do_POST(self) -> None:  # noqa: N802
        if not self._authorized():
            return
        parsed = urllib.parse.urlparse(self.path)
        if not self._route_allowed("POST", parsed.path):
            return
        try:
            payload = self._body()
            if parsed.path == "/v1/action":
                self.request_id = str(uuid4())
                response = self.server.service.action(
                    payload,
                    idempotency_key=self.headers.get("Idempotency-Key"),
                    principal=self.principal,
                    request_id=self.request_id,
                )
            elif parsed.path == "/v1/reset":
                response = self.server.service.reset(payload)
            elif parsed.path == "/v1/recover":
                response = self.server.service.recover(payload)
            elif parsed.path == "/v1/variations/select":
                response = self.server.service.select_variation(payload)
            elif parsed.path == "/v1/variations/start":
                response = self.server.service.start_variation(
                    payload, idempotency_key=self.headers.get("Idempotency-Key"),
                )
            elif parsed.path == "/v1/checkpoints/save":
                response = self.server.service.save_checkpoint(payload)
            elif parsed.path == "/v1/checkpoints/restore":
                response = self.server.service.restore_checkpoint(payload)
            else:
                self._error(HTTPStatus.NOT_FOUND, "unknown API route")
                return
            self._send_json(response)
        except ActionExecutionError as error:
            self._error(HTTPStatus.INTERNAL_SERVER_ERROR, error)
        except AccessDeniedError as error:
            self._error(HTTPStatus.FORBIDDEN, error)
        except UnsupportedActionError as error:
            self._error(HTTPStatus.UNPROCESSABLE_ENTITY, error)
        except NotImplementedError as error:
            # A backend capability it does not offer (e.g. checkpoints).
            self._error(HTTPStatus.NOT_IMPLEMENTED, error)
        except (RoboticsEnvironmentError, OSError, sqlite3.Error) as error:
            self._error(HTTPStatus.BAD_GATEWAY, error)
        except (KeyError, TypeError, ValueError) as error:
            self._error(HTTPStatus.BAD_REQUEST, error)


def _backend_options(value: str) -> dict[str, Any]:
    try:
        result = json.loads(value)
    except json.JSONDecodeError as error:
        raise argparse.ArgumentTypeError(
            "--backend-options-json must be valid JSON"
        ) from error
    if not isinstance(result, dict):
        raise argparse.ArgumentTypeError("--backend-options-json must be a JSON object")
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("environment")
    parser.add_argument("--backend", default="newton")
    parser.add_argument("--robot")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--api-token")
    parser.add_argument("--policy-api-token")
    parser.add_argument("--generate-policy-api-token", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--history-path", type=Path)
    parser.add_argument("--history-max-bytes", type=int, default=DEFAULT_MAX_BYTES)
    parser.add_argument("--history-max-records", type=int, default=DEFAULT_MAX_RECORDS)
    parser.add_argument(
        "--backend-options-json",
        type=_backend_options,
        default={},
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    from .registry import create

    args = parse_args(argv)
    backend_options = dict(args.backend_options_json)
    if args.device is not None:
        if "device" in backend_options:
            raise ValueError(
                "device was supplied by both --device and --backend-options-json"
            )
        backend_options["device"] = args.device
    configured_token = (args.api_token if args.api_token is not None
                        else os.environ.get("GAR_HTTP_API_TOKEN"))
    generated_token = configured_token is None
    api_token = secrets.token_urlsafe(32) if generated_token else configured_token
    policy_api_token = (args.policy_api_token if args.policy_api_token is not None
                       else os.environ.get("GAR_HTTP_POLICY_API_TOKEN"))
    if args.generate_policy_api_token:
        if policy_api_token is not None:
            raise ValueError("Policy token was both configured and requested for generation")
        policy_api_token = secrets.token_urlsafe(32)
    APIKeys(api_token, policy_api_token)  # Reject invalid credentials before creating an environment.

    history_path = args.history_path or (
        Path(__file__).resolve().parents[2] / ".runtime-cache" / "http-history" /
        f"{urllib.parse.quote(args.environment, safe='')}-{args.port}.sqlite3"
    )
    history = ActionHistory(history_path, max_bytes=args.history_max_bytes,
                            max_records=args.history_max_records)
    env = None
    server = None
    try:
        env = create(args.environment, backend=args.backend, robot=args.robot, **backend_options)
        service = GARHTTPService(env, history=history)
        with service.history.span(
            kind="gateway_start", payload={"environment": args.environment, "backend": args.backend,
                                          "robot": args.robot, "attached": "attach_ready_file" in backend_options},
            request_id=str(uuid4()), role="privileged", allow_missing_before=True,
        ):
            pass  # Read-only initial evidence, not a robot command or simulator reset.
        server = GARHTTPServer(
            (args.host, args.port),
            service,
            api_token=api_token,
            policy_api_token=policy_api_token,
        )
        print(
            json.dumps(
                {
                    "ready": True,
                    "url": f"http://{args.host}:{server.server_port}",
                    "environment": args.environment,
                    "backend": args.backend,
                    "robot": args.robot,
                    "generated_api_token": api_token if generated_token else None,
                    "policy_access_enabled": policy_api_token is not None,
                    "generated_policy_api_token": policy_api_token if args.generate_policy_api_token else None,
                    "help": "/v1/help",
                },
                separators=(",", ":"),
            ),
            flush=True,
        )
        server.serve_forever(poll_interval=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        if server is not None:
            server.server_close()
        else:
            history.close()
        if env is not None:
            env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ActionExecutionError", "GARHTTPHandler", "GARHTTPServer", "GARHTTPService", "main"]
