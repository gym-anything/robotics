"""Strict JSON decoding for backend-independent robot actions."""

from __future__ import annotations

from typing import Any, Mapping

from .actions import (
    ActuatorPosition,
    ArmCartesianTarget,
    BaseTwist,
    BimanualCartesian,
    BimanualJointPosition,
    CartesianDelta,
    CartesianPose,
    CartesianWrench,
    Gripper,
    JointEffort,
    JointPosition,
    Keyboard,
    MotionSafety,
    NamedJointPosition,
    NamedJointTarget,
    RawAction,
    RobotAction,
    ToolIO,
)


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return dict(value)


def _fields(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str] = frozenset(),
    name: str,
) -> dict[str, Any]:
    result = dict(value)
    missing = required - set(result)
    unexpected = set(result) - required - optional
    if missing:
        raise ValueError(f"{name} is missing fields: {sorted(missing)}")
    if unexpected:
        raise ValueError(f"{name} has unexpected fields: {sorted(unexpected)}")
    return result


def _safety(value: Any) -> MotionSafety:
    if value is None:
        return MotionSafety()
    fields = _fields(
        _object(value, "safety"),
        required=set(),
        optional={
            "stop_on_contact",
            "stop_on_arm_arm_contact",
            "stop_on_robot_contact",
            "stop_on_grip_loss",
        },
        name="safety",
    )
    return MotionSafety(**fields)


def _arm_target(value: Any, name: str) -> ArmCartesianTarget:
    fields = _fields(
        _object(value, name),
        required=set(),
        optional={
            "position_m",
            "delta_m",
            "orientation_wxyz",
            "gripper_half_target_m",
        },
        name=name,
    )
    return ArmCartesianTarget(**fields)


def action_from_dict(value: Mapping[str, Any]) -> RobotAction:
    """Decode one exact, typed action without ignoring unknown fields."""

    payload = _object(value, "action")
    action_type = payload.get("type")
    if not isinstance(action_type, str) or not action_type:
        raise ValueError("action.type must be a non-empty string")

    common_safety = {"duration_s", "gripper_half_target_m", "safety"}
    if action_type == "named_joint_position":
        fields = _fields(payload, required={"type", "targets_rad", "ticks"}, name=action_type)
        return NamedJointPosition(_object(fields["targets_rad"], "targets_rad"), fields["ticks"])
    if action_type == "named_joint_target":
        fields = _fields(payload, required={"type", "targets_si", "ticks"}, name=action_type)
        return NamedJointTarget(_object(fields["targets_si"], "targets_si"), fields["ticks"])
    if action_type == "base_twist":
        fields = _fields(payload, required={"type", "duration_s"}, optional={"vx_m_s", "vy_m_s", "wz_rad_s"},
                         name=action_type)
        fields.pop("type")
        return BaseTwist(**fields)
    if action_type == "cartesian_pose":
        fields = _fields(
            payload,
            required={"type", "arm", "position_m"},
            optional={"orientation_wxyz", *common_safety},
            name=action_type,
        )
        fields.pop("type")
        fields["safety"] = _safety(fields.get("safety"))
        return CartesianPose(**fields)
    if action_type == "cartesian_delta":
        fields = _fields(
            payload,
            required={"type", "arm", "delta_m"},
            optional=common_safety,
            name=action_type,
        )
        fields.pop("type")
        fields["safety"] = _safety(fields.get("safety"))
        return CartesianDelta(**fields)
    if action_type == "bimanual_cartesian":
        fields = _fields(
            payload,
            required={"type", "left", "right"},
            optional={"duration_s", "stop_on_arm_arm_contact"},
            name=action_type,
        )
        return BimanualCartesian(
            left=_arm_target(fields["left"], "left"),
            right=_arm_target(fields["right"], "right"),
            **{
                key: fields[key]
                for key in ("duration_s", "stop_on_arm_arm_contact")
                if key in fields
            },
        )
    if action_type == "joint_position":
        fields = _fields(
            payload,
            required={"type", "arm", "positions_rad"},
            optional=common_safety,
            name=action_type,
        )
        fields.pop("type")
        fields["safety"] = _safety(fields.get("safety"))
        return JointPosition(**fields)
    if action_type == "bimanual_joint_position":
        fields = _fields(
            payload,
            required={"type", "left_positions_rad", "right_positions_rad"},
            optional={
                "duration_s",
                "left_gripper_half_target_m",
                "right_gripper_half_target_m",
                "stop_on_arm_arm_contact",
            },
            name=action_type,
        )
        fields.pop("type")
        return BimanualJointPosition(**fields)
    if action_type == "actuator_position":
        fields = _fields(
            payload,
            required={"type", "arm", "positions_rad"},
            optional=common_safety,
            name=action_type,
        )
        fields.pop("type")
        fields["safety"] = _safety(fields.get("safety"))
        return ActuatorPosition(**fields)
    if action_type == "gripper":
        fields = _fields(
            payload,
            required={"type", "arm"},
            optional={"half_width_m", "inward_effort_n", "duration_s"},
            name=action_type,
        )
        fields.pop("type")
        return Gripper(**fields)
    if action_type == "keyboard":
        fields = _fields(
            payload,
            required={"type", "keys", "ticks"},
            optional={
                "arm",
                "rate_hz",
                "linear_speed_m_s",
                "angular_speed_rad_s",
                "gripper_half_speed_m_s",
            },
            name=action_type,
        )
        fields.pop("type")
        return Keyboard(**fields)
    if action_type == "joint_effort":
        fields = _fields(
            payload,
            required={"type", "arm", "efforts_nm", "duration_s"},
            name=action_type,
        )
        fields.pop("type")
        return JointEffort(**fields)
    if action_type == "cartesian_wrench":
        fields = _fields(
            payload,
            required={"type", "arm", "force_n", "moment_nm", "duration_s"},
            name=action_type,
        )
        fields.pop("type")
        return CartesianWrench(**fields)
    if action_type == "tool_io":
        fields = _fields(
            payload,
            required={"type", "arm", "outputs"},
            optional={"ticks"},
            name=action_type,
        )
        fields.pop("type")
        return ToolIO(fields["arm"], _object(fields["outputs"], "outputs"), fields.get("ticks", 1))
    if action_type == "raw":
        fields = _fields(
            payload,
            required={"type", "request"},
            name=action_type,
        )
        return RawAction(_object(fields["request"], "request"))
    raise ValueError(f"unknown action type: {action_type!r}")


def actions_from_request(value: Mapping[str, Any]) -> RobotAction | tuple[RobotAction, ...]:
    """Decode one action object or an ordered ``actions`` array."""

    payload = _object(value, "action request")
    if "actions" not in payload:
        return action_from_dict(payload)
    fields = _fields(
        payload,
        required={"actions"},
        name="action request",
    )
    values = fields["actions"]
    if not isinstance(values, list) or not values:
        raise ValueError("actions must be a non-empty JSON array")
    return tuple(action_from_dict(_object(item, "action")) for item in values)


ACTION_HELP: dict[str, Any] = {
    "named_joint_target": {
        "required": ["type", "targets_si", "ticks"],
        "note": "Named hardware targets: radians for revolute joints, metres for prismatic joints. Read per-joint units and limits in capabilities. Exact positive integer physics ticks; empty targets holds current commands.",
    },
    "base_twist": {
        "required": ["type", "duration_s"],
        "optional": ["vx_m_s", "vy_m_s", "wz_rad_s"],
        "note": "Mobile-base velocity in the base frame; accepted only when advertised by capabilities.action_types.",
    },
    "named_joint_position": {
        "required": ["type", "targets_rad", "ticks"],
        "note": "Named hardware joint targets in radians, exact integer physics ticks; empty targets holds existing commands.",
    },
    "cartesian_pose": {
        "required": ["type", "arm", "position_m"],
        "optional": [
            "orientation_wxyz",
            "duration_s",
            "gripper_half_target_m",
            "safety",
        ],
        "example": {
            "type": "cartesian_pose",
            "arm": "right",
            "position_m": [0.60, 0.00, 0.85],
            "duration_s": 0.5,
        },
    },
    "cartesian_delta": {
        "required": ["type", "arm", "delta_m"],
        "optional": ["duration_s", "gripper_half_target_m", "safety"],
        "example": {
            "type": "cartesian_delta",
            "arm": "right",
            "delta_m": [0.0, 0.0, 0.005],
            "duration_s": 0.5,
        },
    },
    "bimanual_cartesian": {
        "required": ["type", "left", "right"],
        "optional": ["duration_s", "stop_on_arm_arm_contact"],
        "arm_target": {
            "exactly_one": ["position_m", "delta_m"],
            "optional": ["orientation_wxyz", "gripper_half_target_m"],
        },
        "example": {
            "type": "bimanual_cartesian",
            "left": {"delta_m": [0.0, 0.0, 0.002]},
            "right": {"delta_m": [0.0, 0.0, -0.002]},
            "duration_s": 0.5,
        },
    },
    "joint_position": {
        "required": ["type", "arm", "positions_rad"],
        "optional": ["duration_s", "gripper_half_target_m", "safety"],
        "positions_rad_length": 6,
    },
    "bimanual_joint_position": {
        "required": ["type", "left_positions_rad", "right_positions_rad"],
        "optional": [
            "duration_s",
            "left_gripper_half_target_m",
            "right_gripper_half_target_m",
            "stop_on_arm_arm_contact",
        ],
        "positions_rad_length_per_arm": 6,
    },
    "actuator_position": {
        "required": ["type", "arm", "positions_rad"],
        "optional": ["duration_s", "gripper_half_target_m", "safety"],
        "positions_rad_length": 6,
    },
    "gripper": {
        "required": ["type", "arm"],
        "at_least_one": ["half_width_m", "inward_effort_n"],
        "optional": ["duration_s"],
    },
    "keyboard": {
        "required": ["type", "keys", "ticks"],
        "optional": [
            "arm",
            "rate_hz",
            "linear_speed_m_s",
            "angular_speed_rad_s",
            "gripper_half_speed_m_s",
        ],
    },
    "joint_effort": {
        "required": ["type", "arm", "efforts_nm", "duration_s"],
        "efforts_nm_length": 6,
        "note": "Accepted only when advertised by capabilities.action_types.",
    },
    "cartesian_wrench": {
        "required": ["type", "arm", "force_n", "moment_nm", "duration_s"],
        "vector_lengths": {"force_n": 3, "moment_nm": 3},
        "note": "Accepted only when advertised by capabilities.action_types.",
    },
    "tool_io": {
        "required": ["type", "arm", "outputs"],
        "optional": ["ticks"],
        "note": "Robot controller outputs to the process tool the arm carries (e.g. a welding power source: start, gas, touch sensing, wire feed and voltage set points, process job), then exact integer physics ticks holding the current joint commands. Output names, units and ranges are tool-specific: read capabilities. Accepted only when advertised by capabilities.action_types.",
        "example": {
            "type": "tool_io",
            "arm": "right",
            "outputs": {"arc_on": True},
            "ticks": 1,
        },
    },
    "raw": {
        "required": ["type", "request"],
        "note": "Backend-specific operation; consult capabilities and backend documentation.",
    },
    "safety": {
        "optional": [
            "stop_on_contact",
            "stop_on_arm_arm_contact",
            "stop_on_robot_contact",
            "stop_on_grip_loss",
        ]
    },
}


__all__ = ["ACTION_HELP", "action_from_dict", "actions_from_request"]
