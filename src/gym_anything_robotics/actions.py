"""Typed, backend-independent robot actions.

Actions describe robot commands only.  They never expose object pose writes or
simulator-specific constraint controls.
"""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
import math
from typing import Any, Iterable, Iterator, Mapping, Sequence


ARM_NAMES = frozenset(("left", "right"))
# Arm names one backend declares besides the two above (a humanoid whose hand IK can include the waist offers
# "left+waist"). They are accepted only inside ``arm_names``; everywhere else an action names 'left' or 'right'.
_EXTRA_ARM_NAMES: ContextVar[frozenset[str]] = ContextVar("gar_extra_arm_names", default=frozenset())


@contextmanager
def arm_names(extra: Iterable[str] | None) -> Iterator[frozenset[str]]:
    """Accept a backend's own arm names, besides 'left' and 'right', for the actions built inside the block.

    The HTTP service decodes a session's requests inside it with what the session's backend declares
    (``RoboticsEnv.extra_arm_names``), so the names are valid for that backend alone; the backend still decides what
    each name means and refuses the action types it does not offer on it. With nothing declared the check is
    unchanged.
    """

    names = frozenset(str(name).lower() for name in (extra or ()))
    token = _EXTRA_ARM_NAMES.set(names)
    try:
        yield ARM_NAMES | names
    finally:
        _EXTRA_ARM_NAMES.reset(token)


def _arm(value: str) -> str:
    result = str(value).lower()
    if result not in ARM_NAMES:
        extra = _EXTRA_ARM_NAMES.get()
        if result not in extra:
            raise ValueError("arm must be 'left' or 'right'"
                             + (f" (this robot also accepts {', '.join(repr(n) for n in sorted(extra))})"
                                if extra else ""))
    return result


def _finite_tuple(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(item) for item in value)
    if len(result) != length or not all(math.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain {length} finite values")
    return result


def _duration(value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError("duration_s must be positive and finite")
    return result


def _optional_half_gap(value: float | None) -> float | None:
    if value is None:
        return None
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError("gripper_half_target_m must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class MotionSafety:
    """Backend-neutral stop conditions for guarded robot motion."""

    stop_on_contact: bool = False
    stop_on_arm_arm_contact: bool = True
    stop_on_robot_contact: bool = False
    stop_on_grip_loss: str | None = None

    def __post_init__(self) -> None:
        if self.stop_on_grip_loss is not None and not self.stop_on_grip_loss:
            raise ValueError("stop_on_grip_loss must name the held object")


@dataclass(frozen=True)
class CartesianPose:
    arm: str
    position_m: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float] | None = None
    duration_s: float = 0.5
    gripper_half_target_m: float | None = None
    safety: MotionSafety = field(default_factory=MotionSafety)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(
            self, "position_m", _finite_tuple(self.position_m, 3, "position_m")
        )
        if self.orientation_wxyz is not None:
            orientation = _finite_tuple(
                self.orientation_wxyz, 4, "orientation_wxyz"
            )
            if math.sqrt(sum(value * value for value in orientation)) <= 1.0e-12:
                raise ValueError("orientation_wxyz must be nonzero")
            object.__setattr__(self, "orientation_wxyz", orientation)
        object.__setattr__(self, "duration_s", _duration(self.duration_s))
        object.__setattr__(
            self,
            "gripper_half_target_m",
            _optional_half_gap(self.gripper_half_target_m),
        )


@dataclass(frozen=True)
class CartesianDelta:
    arm: str
    delta_m: tuple[float, float, float]
    duration_s: float = 0.5
    gripper_half_target_m: float | None = None
    safety: MotionSafety = field(default_factory=MotionSafety)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(self, "delta_m", _finite_tuple(self.delta_m, 3, "delta_m"))
        object.__setattr__(self, "duration_s", _duration(self.duration_s))
        object.__setattr__(
            self,
            "gripper_half_target_m",
            _optional_half_gap(self.gripper_half_target_m),
        )


@dataclass(frozen=True)
class ArmCartesianTarget:
    """One arm's portion of an atomic bimanual Cartesian command."""

    position_m: tuple[float, float, float] | None = None
    delta_m: tuple[float, float, float] | None = None
    orientation_wxyz: tuple[float, float, float, float] | None = None
    gripper_half_target_m: float | None = None

    def __post_init__(self) -> None:
        if (self.position_m is None) == (self.delta_m is None):
            raise ValueError("provide exactly one of position_m or delta_m")
        if self.position_m is not None:
            object.__setattr__(
                self,
                "position_m",
                _finite_tuple(self.position_m, 3, "position_m"),
            )
        if self.delta_m is not None:
            object.__setattr__(
                self, "delta_m", _finite_tuple(self.delta_m, 3, "delta_m")
            )
        if self.orientation_wxyz is not None:
            orientation = _finite_tuple(
                self.orientation_wxyz, 4, "orientation_wxyz"
            )
            if math.sqrt(sum(value * value for value in orientation)) <= 1.0e-12:
                raise ValueError("orientation_wxyz must be nonzero")
            object.__setattr__(self, "orientation_wxyz", orientation)
        object.__setattr__(
            self,
            "gripper_half_target_m",
            _optional_half_gap(self.gripper_half_target_m),
        )


@dataclass(frozen=True)
class BimanualCartesian:
    left: ArmCartesianTarget
    right: ArmCartesianTarget
    duration_s: float = 0.5
    stop_on_arm_arm_contact: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.left, ArmCartesianTarget) or not isinstance(
            self.right, ArmCartesianTarget
        ):
            raise TypeError("left and right must be ArmCartesianTarget values")
        object.__setattr__(self, "duration_s", _duration(self.duration_s))


@dataclass(frozen=True)
class JointPosition:
    arm: str
    positions_rad: tuple[float, ...]
    duration_s: float = 0.5
    gripper_half_target_m: float | None = None
    safety: MotionSafety = field(default_factory=MotionSafety)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(
            self,
            "positions_rad",
            _finite_tuple(self.positions_rad, 6, "positions_rad"),
        )
        object.__setattr__(self, "duration_s", _duration(self.duration_s))
        object.__setattr__(
            self,
            "gripper_half_target_m",
            _optional_half_gap(self.gripper_half_target_m),
        )


@dataclass(frozen=True)
class BimanualJointPosition:
    left_positions_rad: tuple[float, ...]
    right_positions_rad: tuple[float, ...]
    duration_s: float = 0.5
    left_gripper_half_target_m: float | None = None
    right_gripper_half_target_m: float | None = None
    stop_on_arm_arm_contact: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "left_positions_rad",
            _finite_tuple(self.left_positions_rad, 6, "left_positions_rad"),
        )
        object.__setattr__(
            self,
            "right_positions_rad",
            _finite_tuple(self.right_positions_rad, 6, "right_positions_rad"),
        )
        object.__setattr__(self, "duration_s", _duration(self.duration_s))
        object.__setattr__(
            self,
            "left_gripper_half_target_m",
            _optional_half_gap(self.left_gripper_half_target_m),
        )
        object.__setattr__(
            self,
            "right_gripper_half_target_m",
            _optional_half_gap(self.right_gripper_half_target_m),
        )


@dataclass(frozen=True)
class ActuatorPosition:
    arm: str
    positions_rad: tuple[float, ...]
    duration_s: float = 0.5
    gripper_half_target_m: float | None = None
    safety: MotionSafety = field(default_factory=MotionSafety)

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(
            self,
            "positions_rad",
            _finite_tuple(self.positions_rad, 6, "positions_rad"),
        )
        object.__setattr__(self, "duration_s", _duration(self.duration_s))
        object.__setattr__(
            self,
            "gripper_half_target_m",
            _optional_half_gap(self.gripper_half_target_m),
        )


@dataclass(frozen=True)
class Gripper:
    arm: str
    half_width_m: float | None = None
    inward_effort_n: float | None = None
    duration_s: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        if self.half_width_m is None and self.inward_effort_n is None:
            raise ValueError("Gripper requires half_width_m or inward_effort_n")
        object.__setattr__(
            self, "half_width_m", _optional_half_gap(self.half_width_m)
        )
        if self.inward_effort_n is not None:
            effort = float(self.inward_effort_n)
            if not math.isfinite(effort) or effort < 0.0:
                raise ValueError("inward_effort_n must be finite and nonnegative")
            object.__setattr__(self, "inward_effort_n", effort)
        object.__setattr__(self, "duration_s", _duration(self.duration_s))


@dataclass(frozen=True)
class Keyboard:
    """One game-style chord held for an exact number of controller ticks."""

    keys: tuple[str, ...]
    ticks: int
    arm: str = "right"
    rate_hz: float = 30.0
    linear_speed_m_s: float = 0.005
    angular_speed_rad_s: float = 0.04
    gripper_half_speed_m_s: float = 0.005

    def __post_init__(self) -> None:
        normalized = tuple(str(key).upper() for key in self.keys)
        if not normalized or any(not key for key in normalized):
            raise ValueError("Keyboard requires at least one non-empty key")
        if len(set(normalized)) != len(normalized):
            raise ValueError("a keyboard chord cannot contain duplicate keys")
        if isinstance(self.ticks, bool) or int(self.ticks) != self.ticks:
            raise ValueError("ticks must be an integer")
        if int(self.ticks) <= 0:
            raise ValueError("ticks must be positive")
        object.__setattr__(self, "keys", normalized)
        object.__setattr__(self, "ticks", int(self.ticks))
        object.__setattr__(self, "arm", _arm(self.arm))
        for name in (
            "rate_hz",
            "linear_speed_m_s",
            "angular_speed_rad_s",
            "gripper_half_speed_m_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)


@dataclass(frozen=True)
class JointEffort:
    """Direct arm-joint torque request, where supported by the backend."""

    arm: str
    efforts_nm: tuple[float, ...]
    duration_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(
            self, "efforts_nm", _finite_tuple(self.efforts_nm, 6, "efforts_nm")
        )
        object.__setattr__(self, "duration_s", _duration(self.duration_s))


@dataclass(frozen=True)
class CartesianWrench:
    """TCP force/moment request, where supported by the backend."""

    arm: str
    force_n: tuple[float, float, float]
    moment_nm: tuple[float, float, float]
    duration_s: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        object.__setattr__(self, "force_n", _finite_tuple(self.force_n, 3, "force_n"))
        object.__setattr__(
            self, "moment_nm", _finite_tuple(self.moment_nm, 3, "moment_nm")
        )
        object.__setattr__(self, "duration_s", _duration(self.duration_s))


@dataclass(frozen=True)
class NamedJointPosition:
    """Robot-independent named joint targets held for exact physics ticks."""

    targets_rad: Mapping[str, float]
    ticks: int

    def __post_init__(self) -> None:
        targets = dict(self.targets_rad)
        if any(not isinstance(k, str) or not k or isinstance(v, bool)
               or not isinstance(v, (int, float)) or not math.isfinite(v) for k,v in targets.items()):
            raise ValueError("targets_rad must map joint names to finite radians")
        if isinstance(self.ticks, bool) or not isinstance(self.ticks, int) or self.ticks < 1:
            raise ValueError("ticks must be a positive integer")
        object.__setattr__(self, "targets_rad", targets)


@dataclass(frozen=True)
class NamedJointTarget:
    """Named targets: radians for revolute, metres for prismatic joints.

    Hardware capabilities declare each joint's unit and limits. This additive
    action avoids labelling a gripper's metre opening as radians.
    """

    targets_si: Mapping[str, float]
    ticks: int

    def __post_init__(self) -> None:
        targets = dict(self.targets_si)
        if any(not isinstance(k, str) or not k or isinstance(v, bool)
               or not isinstance(v, (int, float)) or not math.isfinite(v) for k, v in targets.items()):
            raise ValueError("targets_si must map names to finite hardware-unit values")
        if type(self.ticks) is not int or self.ticks < 1:
            raise ValueError("ticks must be a positive integer")
        object.__setattr__(self, "targets_si", targets)


@dataclass(frozen=True)
class ToolIO:
    """Outputs of the robot controller to a process tool it carries (a welding power source, a solder feeder),
    then exact physics ticks with the existing joint commands held.

    Mirrors a real robot-to-tool interface (digital starts, analog set points, job selection). Output names, units
    and ranges are tool-specific and declared in the backend's capabilities; the backend rejects unknown names and
    out-of-range values before any side effect. ``arm`` names the arm carrying the tool.
    """

    arm: str
    outputs: Mapping[str, Any]
    ticks: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm", _arm(self.arm))
        outputs = dict(self.outputs)
        if not outputs:
            raise ValueError("outputs must name at least one tool output")
        for key, value in outputs.items():
            if not isinstance(key, str) or not key:
                raise ValueError("tool output names must be non-empty strings")
            if not isinstance(value, (bool, int, float, str)) or (
                isinstance(value, float) and not math.isfinite(value)
            ):
                raise ValueError(f"tool output {key} must be a finite number, boolean or string")
        if type(self.ticks) is not int or self.ticks < 1:
            raise ValueError("ticks must be a positive integer")
        object.__setattr__(self, "outputs", outputs)


@dataclass(frozen=True)
class BaseTwist:
    """Mobile-base velocity command in the base frame (v_x forward, v_y left, w_z counter-clockwise), held for
    duration_s. The backend's base controller enforces the platform's own limits and refuses requests beyond them."""

    vx_m_s: float = 0.0
    vy_m_s: float = 0.0
    wz_rad_s: float = 0.0
    duration_s: float = 0.5

    def __post_init__(self) -> None:
        for name in ("vx_m_s", "vy_m_s", "wz_rad_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
            object.__setattr__(self, name, float(value))
        object.__setattr__(self, "duration_s", _duration(self.duration_s))


@dataclass(frozen=True)
class RawAction:
    """Escape hatch for a documented backend operation."""

    request: Mapping[str, Any]

    def __post_init__(self) -> None:
        request = dict(self.request)
        if not isinstance(request.get("op"), str) or not request["op"]:
            raise ValueError("RawAction request must contain a non-empty op")
        object.__setattr__(self, "request", request)


RobotAction = (
    CartesianPose
    | CartesianDelta
    | BimanualCartesian
    | JointPosition
    | BimanualJointPosition
    | ActuatorPosition
    | Gripper
    | Keyboard
    | JointEffort
    | CartesianWrench
    | RawAction
    | NamedJointPosition
    | NamedJointTarget
    | ToolIO
    | BaseTwist
)


__all__ = [
    "arm_names",
    "ActuatorPosition",
    "ArmCartesianTarget",
    "BaseTwist",
    "BimanualCartesian",
    "BimanualJointPosition",
    "CartesianDelta",
    "CartesianPose",
    "CartesianWrench",
    "Gripper",
    "JointEffort",
    "JointPosition",
    "Keyboard",
    "MotionSafety",
    "NamedJointPosition",
    "NamedJointTarget",
    "RawAction",
    "RobotAction",
    "ToolIO",
]
