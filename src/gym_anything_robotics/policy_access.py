"""Allowlisted policy observations and commands; no simulator state passthrough.

Backends explicitly opt in with a reviewed PolicyAccess specification. Adding a
new backend observation field or action does not automatically expose it here.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Any

from .access import AccessDeniedError
from .action_codec import ACTION_HELP, actions_from_request
from .actions import arm_names
from .types import CameraSpec, ImageFrame, Observation


ROBOT_ACTIONS = frozenset({
    "cartesian_pose", "cartesian_delta", "bimanual_cartesian", "joint_position",
    "bimanual_joint_position", "actuator_position", "gripper", "keyboard",
    "named_joint_position",
    "named_joint_target",
    "tool_io",
    "base_twist",
})
SAFETY_FIELDS = frozenset({
    "stop_on_contact", "stop_on_arm_arm_contact", "stop_on_robot_contact",
    "stop_on_grip_loss",
})


@dataclass(frozen=True)
class PolicyAccess:
    robot: str
    cameras: tuple[CameraSpec, ...]
    # Each leaf is either a numeric scalar (None) or a fixed-length vector.
    proprioception: Mapping[tuple[str, ...], int | None]
    action_types: frozenset[str]
    instruction: Callable[[], str]
    hardware: Mapping[str, Any]
    keyboard_keys: tuple[str, ...] = ()
    # Arm names this backend accepts besides 'left' and 'right' (actions.arm_names); none by default.
    extra_arm_names: tuple[str, ...] = ()
    # Pictures that are part of the task itself, e.g. a drawing to copy: name -> (description, PNG bytes).
    # None for instruction-only tasks, whose /v1/task payload is then unchanged.
    task_images: Callable[[], Mapping[str, tuple[str, bytes]]] | None = None

    def camera_allowed(self, frame: ImageFrame) -> bool:
        return frame.deployable and frame.name in {spec.name for spec in self.cameras}

    def observation(self, observation: Observation) -> Observation:
        proprioception: dict[str, Any] = {}
        for path, length in self.proprioception.items():
            source = observation.proprioception
            for name in path:
                source = source[name]
            def number(value):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ValueError("Invalid robot feedback")
                if not math.isfinite(value):
                    raise ValueError("Invalid robot feedback")
                return value
            if length is None:
                value = number(source)
            else:
                if not isinstance(source, (tuple, list)) or len(source) != length:
                    raise ValueError("Invalid robot feedback")
                value = [number(item) for item in source]
            target = proprioception
            for name in path[:-1]:
                target = target.setdefault(name, {})
            target[path[-1]] = value
        images = {}
        for spec in self.cameras:
            frame = observation.images.get(spec.name)
            if frame is None or not self.camera_allowed(frame):
                raise ValueError("Robot camera unavailable")
            # Calibration comes from the static hardware specification, not
            # backend image metadata that may contain per-episode information.
            images[spec.name] = replace(frame, camera=spec)
        return replace(observation, robot=self.robot, proprioception=proprioception,
                       images=images, metadata={})

    def decode_actions(self, payload: Mapping[str, Any]):
        """Validate the whole batch before any command can execute.

        Omitted simulator guards default OFF for this interface. Explicit guard
        requests are rejected, never silently ignored or weakened.
        """
        values = deepcopy(dict(payload))
        if "actions" in values:
            if set(values) != {"actions"} or not isinstance(values["actions"], list):
                raise ValueError("Invalid action sequence")
            items = values["actions"]
        else:
            items = [values]
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Invalid action")
            kind = item.get("type")
            if not isinstance(kind, str) or kind not in self.action_types & ROBOT_ACTIONS:
                raise AccessDeniedError("Action is not available to policy credentials")
            if "safety" in item:
                safety = item["safety"]
                if not isinstance(safety, dict) or set(safety) - SAFETY_FIELDS:
                    raise AccessDeniedError("Simulator guards are not available")
                for field, value in safety.items():
                    if (field == "stop_on_grip_loss" and value is not None) or (
                        field != "stop_on_grip_loss" and value is not False
                    ):
                        raise AccessDeniedError("Simulator guards are not available")
            if "stop_on_arm_arm_contact" in item and item["stop_on_arm_arm_contact"] is not False:
                raise AccessDeniedError("Simulator guards are not available")
            if kind in {"cartesian_pose", "cartesian_delta", "joint_position", "actuator_position"}:
                item.setdefault("safety", {})["stop_on_arm_arm_contact"] = False
            if kind in {"bimanual_cartesian", "bimanual_joint_position"}:
                item["stop_on_arm_arm_contact"] = False
        with arm_names(self.extra_arm_names):
            return actions_from_request(values)

    def capabilities(self) -> dict[str, Any]:
        return {
            "robot": self.robot,
            "action_types": sorted(self.action_types & ROBOT_ACTIONS),
            "privileged_state": False, "diagnostic_render": False, "seeded_reset": False,
            "robot_cameras": [spec.name for spec in self.cameras],
            "camera_specs": [spec.to_dict() for spec in self.cameras],
            "proprioception_fields": [".".join(path) for path in self.proprioception],
            "keyboard_keys": list(self.keyboard_keys),
            "hardware": deepcopy(dict(self.hardware)),
        }

    def help(self) -> dict[str, Any]:
        actions = {name: deepcopy(ACTION_HELP[name]) for name in sorted(self.action_types & ROBOT_ACTIONS)}
        for action in actions.values():
            action["optional"] = [name for name in action.get("optional", [])
                                  if name not in {"safety", "stop_on_arm_arm_contact"}]
        routes = {
            "GET /v1/health": "Service readiness.",
            "GET /v1/help": "Permitted routes and commands for this credential.",
            "GET /v1/capabilities": "Robot hardware and supported commands.",
            "GET /v1/task": "Human-readable instruction for the current episode.",
            "GET /v1/observation": "Robot feedback and mounted-camera URLs.",
            "GET /v1/cameras/{name}?observation_id=N": "Image from your referenced observation.",
            "POST /v1/action": "One command or {'actions': [...]} in order.",
        }
        if self.task_images is not None:
            routes["GET /v1/task/images/{name}"] = "PNG picture that is part of the task; /v1/task lists them."
        return {
            "ok": True, "api_version": "v1", "role": "policy",
            "authentication": {"header": "Authorization: Bearer <token>",
                               "required_except": ["GET /v1/health"]},
            "routes": routes,
            "action_types": actions,
            "effective_capabilities": self.capabilities(),
            "execution": {
                "simulator_contact_guards": False,
                "collision_avoidance": "No hidden-scene collision avoidance; contacts remain physical.",
                "rewards": "Retained by evaluator; not returned to the policy.",
                "idempotency": "Use an Idempotency-Key per action request; retries replay the response, not the command. Last 256 retained per credential until HTTP restart.",
                "commit": "ok=true, committed=true means the command ran, even if observation is null. Never retry with a new key to recover a missing image.",
                "lifecycle": "The privileged evaluator starts and resets episodes.",
                "request_id": "Save request_id from action responses and X-GAR-Request-ID from errors for evaluator correlation. It grants no extra permissions.",
                "timeouts": "Wall time may exceed simulated duration. After a client timeout the command may already have executed.",
            },
        }
