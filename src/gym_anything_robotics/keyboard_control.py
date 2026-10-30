"""Pure conversion from game-style keys to one Newton controller request.

This module contains no simulator state and performs no I/O.  Both the public
Python backend and the browser teleoperation bridge use it so a key chord has
one definition everywhere.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

from .actions import Keyboard
from .errors import UnsupportedActionError, UnsupportedConfigurationError


YAM_MAXIMUM_GRIPPER_HALF_WIDTH_M = 0.037524
NEWTON_PHYSICS_HZ = 480.0
NEWTON_CONTROL_SUBSTEPS = 16

KEY_TRANSLATION = {
    "A": (-1.0, 0.0, 0.0),
    "D": (1.0, 0.0, 0.0),
    "W": (0.0, 1.0, 0.0),
    "S": (0.0, -1.0, 0.0),
    "R": (0.0, 0.0, 1.0),
    "F": (0.0, 0.0, -1.0),
}
KEY_ROTATION = {
    "I": (1.0, 0.0, 0.0),
    "K": (-1.0, 0.0, 0.0),
    "J": (0.0, 1.0, 0.0),
    "L": (0.0, -1.0, 0.0),
    "U": (0.0, 0.0, 1.0),
    "O": (0.0, 0.0, -1.0),
}
SUPPORTED_KEYS = tuple(
    sorted(
        set(KEY_TRANSLATION)
        | set(KEY_ROTATION)
        | {"Q", "E", "H", "G", "B", "P", "T", "ALT", "SHIFT"}
    )
)


def _normalize(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    magnitude = math.sqrt(sum(value * value for value in vector))
    if magnitude <= 1.0e-12:
        return (0.0, 0.0, 0.0)
    return tuple(value / magnitude for value in vector)


def _sum_directions(
    keys: set[str], mapping: Mapping[str, tuple[float, float, float]]
) -> tuple[float, float, float]:
    return _normalize(
        tuple(
            sum(mapping[key][axis] for key in keys if key in mapping)
            for axis in range(3)
        )
    )


def _quaternion_multiply(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    aw, ax, ay, az = first
    bw, bx, by, bz = second
    result = (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )
    norm = math.sqrt(sum(value * value for value in result))
    if norm <= 1.0e-12:
        raise ValueError("quaternion composition produced zero norm")
    return tuple(value / norm for value in result)


def _quaternion_from_rotation_vector(
    vector: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    angle = math.sqrt(sum(value * value for value in vector))
    if angle <= 1.0e-12:
        return (1.0, 0.0, 0.0, 0.0)
    scale = math.sin(0.5 * angle) / angle
    return (
        math.cos(0.5 * angle),
        scale * vector[0],
        scale * vector[1],
        scale * vector[2],
    )


def keyboard_request(
    action: Keyboard, state: Mapping[str, Any]
) -> tuple[dict[str, Any], float]:
    """Convert one validated key chord and current robot state into a move.

    The returned effort is the physical inward effort that must be configured
    immediately before issuing the returned controller request.
    """

    keys = set(action.keys)
    unknown = keys - set(SUPPORTED_KEYS)
    if unknown:
        raise UnsupportedActionError(
            f"unsupported YAM keyboard keys: {sorted(unknown)}"
        )
    conflicts = (
        ("A", "D"),
        ("W", "S"),
        ("R", "F"),
        ("I", "K"),
        ("J", "L"),
        ("U", "O"),
        ("Q", "E"),
        ("G", "B"),
        ("P", "T"),
        ("ALT", "SHIFT"),
    )
    for first, second in conflicts:
        if {first, second} <= keys:
            raise ValueError(f"conflicting keyboard keys: {first}, {second}")
    if (keys & {"G", "B"}) and "H" not in keys:
        raise UnsupportedActionError(
            "G/B force modes require H on the current YAM controller"
        )
    if (keys & {"P", "T"}) and not (keys & {"Q", "E"}):
        raise ValueError("P/T gripper-speed modifiers require Q or E")
    if (keys & {"ALT", "SHIFT"}) and not (
        keys & (set(KEY_TRANSLATION) | set(KEY_ROTATION))
    ):
        raise ValueError("ALT/SHIFT arm-speed modifiers require a motion key")

    duration_s = action.ticks / action.rate_hz
    if duration_s > 10.0:
        raise UnsupportedConfigurationError(
            "this Newton controller accepts at most 10 seconds per keyboard "
            "action; the requested ticks/rate_hz was not split"
        )

    tcp = state["tcp"][action.arm]
    position = tuple(float(value) for value in tcp["position_m"])
    orientation = tuple(float(value) for value in tcp["orientation_wxyz"])
    arm_multiplier = 0.2 if "ALT" in keys else (6.0 if "SHIFT" in keys else 1.0)
    translation = _sum_directions(keys, KEY_TRANSLATION)
    rotation = _sum_directions(keys, KEY_ROTATION)
    position = tuple(
        position[axis]
        + translation[axis]
        * action.linear_speed_m_s
        * arm_multiplier
        * duration_s
        for axis in range(3)
    )
    rotation_vector = tuple(
        rotation[axis]
        * action.angular_speed_rad_s
        * arm_multiplier
        * duration_s
        for axis in range(3)
    )
    orientation = _quaternion_multiply(
        _quaternion_from_rotation_vector(rotation_vector), orientation
    )

    joints = state["robot_joint_coordinates"]
    half_gap = float(joints[6 if action.arm == "left" else 14])
    gripper_multiplier = 0.2 if "P" in keys else (10.0 if "T" in keys else 1.0)
    gripper_direction = (1 if "E" in keys else 0) - (1 if "Q" in keys else 0)
    half_gap = min(
        YAM_MAXIMUM_GRIPPER_HALF_WIDTH_M,
        max(
            0.0,
            half_gap
            + gripper_direction
            * action.gripper_half_speed_m_s
            * gripper_multiplier
            * duration_s,
        ),
    )

    request: dict[str, Any] = {
        "op": "move",
        "arm": action.arm,
        "position_m": position,
        "orientation_wxyz": orientation,
        "duration_s": duration_s,
        "gripper_half_target_m": half_gap,
        "stop_on_arm_arm_contact": True,
    }
    # The persistent controller normally groups 16 unchanged 480 Hz physics
    # steps into one 30 Hz control frame.  Interactive rates above 30 Hz must
    # shorten that group as well as the Cartesian delta; otherwise a nominal
    # 120 Hz input still blocks for a complete 30 Hz simulation frame.
    short_frame_substeps = round(duration_s * NEWTON_PHYSICS_HZ)
    if short_frame_substeps < NEWTON_CONTROL_SUBSTEPS:
        request["physics_substeps_per_control_frame"] = max(
            1, short_frame_substeps
        )
    effort = 0.0
    if "H" in keys:
        effort = 0.75 if "G" in keys else (10.0 if "B" in keys else 1.5)
    return request, effort


__all__ = [
    "KEY_ROTATION",
    "KEY_TRANSLATION",
    "NEWTON_CONTROL_SUBSTEPS",
    "NEWTON_PHYSICS_HZ",
    "SUPPORTED_KEYS",
    "YAM_MAXIMUM_GRIPPER_HALF_WIDTH_M",
    "keyboard_request",
]
