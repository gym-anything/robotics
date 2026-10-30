"""Backend-declared semantic action channels for LaundryFold.

Normalized actions are commands to finite-bandwidth robot/fixture controllers;
they are never permission to write cloth particle poses.  The bimanual schema
is robot-agnostic: a robot adapter may realize the requested tool twist with
joint torques, differential IK plus impedance, or teleoperation retargeting.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class ActionChannel:
    name: str
    unit: str
    physical_rate_per_s: float
    description: str
    minimum: float = -1.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if not self.name or not self.unit or not self.description:
            raise ValueError("Action channels need a name, unit, and description")
        if (
            not math.isfinite(self.physical_rate_per_s)
            or self.physical_rate_per_s <= 0.0
            or not math.isfinite(self.minimum)
            or not math.isfinite(self.maximum)
            or self.minimum >= self.maximum
        ):
            raise ValueError("Action channel ranges and physical rates must be finite")


@dataclass(frozen=True)
class ActionSchema:
    id: str
    channels: tuple[ActionChannel, ...]
    control_hz: float
    command_type: str = "normalized_target_velocity"
    calibration_status: str = "engineering-prior control rates"

    def __post_init__(self) -> None:
        if not self.id or not self.channels or not self.command_type:
            raise ValueError("Action schemas need an id, channels, and command type")
        names = tuple(channel.name for channel in self.channels)
        if len(names) != len(set(names)):
            raise ValueError("Action channel names must be unique")
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("Control frequency must be positive")

    @property
    def shape(self) -> tuple[int]:
        return (len(self.channels),)

    @property
    def low(self) -> np.ndarray:
        return np.asarray([channel.minimum for channel in self.channels], dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        return np.asarray([channel.maximum for channel in self.channels], dtype=np.float32)

    def validate(self, action, *, strict: bool = True) -> np.ndarray:
        value = np.asarray(action, dtype=np.float32)
        if value.shape != self.shape:
            raise ValueError(f"Expected action shape {self.shape}, got {value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError("Action must be finite")
        if strict and (np.any(value < self.low) or np.any(value > self.high)):
            raise ValueError("Action components lie outside the declared schema")
        return np.clip(value, self.low, self.high)

    def rate_commands(self, action) -> dict[str, float]:
        value = self.validate(action, strict=False)
        return {
            channel.name: float(component) * channel.physical_rate_per_s
            for component, channel in zip(value, self.channels, strict=True)
        }

    def describe(self) -> dict[str, object]:
        return {
            "id": self.id,
            "shape": self.shape,
            "control_hz": self.control_hz,
            "command_type": self.command_type,
            "calibration_status": self.calibration_status,
            "channels": [
                {
                    "index": index,
                    "name": channel.name,
                    "unit": channel.unit,
                    "minimum": channel.minimum,
                    "maximum": channel.maximum,
                    "physical_rate_per_s": channel.physical_rate_per_s,
                    "description": channel.description,
                }
                for index, channel in enumerate(self.channels)
            ],
        }


def make_bimanual_cartesian_action_schema(
    *,
    tool_names: tuple[str, str] = ("left", "right"),
    control_hz: float = 60.0,
    translation_rate_m_s: float = 0.30,
    rotation_rate_rad_s: float = 1.50,
    jaw_rate_m_s: float = 0.040,
) -> ActionSchema:
    """Return a robot-neutral two-tool twist-and-jaw command contract.

    These are target rates, not cloth forces and not kinematic pose writes.
    A concrete robot adapter remains responsible for joint limits, collision
    avoidance, actuator saturation, and torque/impedance control.
    """

    if len(tool_names) != 2 or len(set(tool_names)) != 2 or any(
        not name for name in tool_names
    ):
        raise ValueError("Bimanual control requires two distinct nonempty tool names")
    if min(
        control_hz,
        translation_rate_m_s,
        rotation_rate_rad_s,
        jaw_rate_m_s,
    ) <= 0.0:
        raise ValueError("Bimanual controller rates must be positive")
    channels: list[ActionChannel] = []
    for tool_name in tool_names:
        channels.extend(
            (
                ActionChannel(
                    f"{tool_name}_vx",
                    "m/s",
                    translation_rate_m_s,
                    f"Command {tool_name} tool target linear velocity along table/world X.",
                ),
                ActionChannel(
                    f"{tool_name}_vy",
                    "m/s",
                    translation_rate_m_s,
                    f"Command {tool_name} tool target linear velocity along table/world Y.",
                ),
                ActionChannel(
                    f"{tool_name}_vz",
                    "m/s",
                    translation_rate_m_s,
                    f"Command {tool_name} tool target linear velocity along table normal/world Z.",
                ),
                ActionChannel(
                    f"{tool_name}_wx",
                    "rad/s",
                    rotation_rate_rad_s,
                    f"Command {tool_name} tool target angular velocity about X.",
                ),
                ActionChannel(
                    f"{tool_name}_wy",
                    "rad/s",
                    rotation_rate_rad_s,
                    f"Command {tool_name} tool target angular velocity about Y.",
                ),
                ActionChannel(
                    f"{tool_name}_wz",
                    "rad/s",
                    rotation_rate_rad_s,
                    f"Command {tool_name} tool target angular velocity about Z.",
                ),
                ActionChannel(
                    f"{tool_name}_jaw",
                    "m/s",
                    jaw_rate_m_s,
                    f"Command {tool_name} symmetric jaw aperture rate; contact generates force.",
                ),
            )
        )
    return ActionSchema(
        id="laundry_bimanual_cartesian_twist_v1",
        channels=tuple(channels),
        control_hz=control_hz,
        command_type="normalized_operational_space_target_rate",
        calibration_status=(
            "robot-neutral engineering-prior limits; concrete joint-space controller "
            "and hardware limits must be calibrated per robot"
        ),
    )


FIXTURE_VELOCITY_ACTION_SCHEMA = ActionSchema(
    id="laundry_fixture_velocity_v1",
    control_hz=60.0,
    calibration_status=(
        "P6-enveloping engineering-prior rates; not hardware calibrated"
    ),
    channels=(
        ActionChannel("gripper_x", "m/s", 0.50, "Translate the folding gripper in world X."),
        ActionChannel("gripper_z", "m/s", 0.30, "Translate the folding gripper in world Z."),
        ActionChannel("gripper_wrist_y", "rad/s", 1.50, "Rotate the gripper wrist about world/local Y."),
        ActionChannel("gripper_gap", "m/s", 0.040, "Open (+) or close (-) the symmetric finger half-gap."),
        ActionChannel("roller_x", "m/s", 0.15, "Translate the crease-roller carriage in world X."),
        ActionChannel("roller_z", "m/s", 0.40, "Raise (+) or lower (-) the crease roller."),
    ),
)


BIMANUAL_CARTESIAN_ACTION_SCHEMA = make_bimanual_cartesian_action_schema()
