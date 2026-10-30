"""Shared observations, capabilities, and step results."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class CameraSpec:
    """Geometry and provenance for one physically authored robot camera."""

    name: str
    mount_frame: str
    width_px: int
    height_px: int
    native_width_px: int
    native_height_px: int
    sensor_size_m: tuple[float, float]
    focal_length_m: tuple[float, float]
    modalities: tuple[str, ...] = ("rgb",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "mount_frame": self.mount_frame,
            "resolution_px": [self.width_px, self.height_px],
            "native_resolution_px": [
                self.native_width_px,
                self.native_height_px,
            ],
            "sensor_size_m": list(self.sensor_size_m),
            "focal_length_m": list(self.focal_length_m),
            "modalities": list(self.modalities),
        }


@dataclass(frozen=True)
class Capabilities:
    environment_id: str
    backend: str
    robot: str
    action_types: frozenset[str]
    observation_channels: frozenset[str]
    privileged_state: bool
    seeded_reset: bool
    diagnostic_render: bool
    robot_cameras: tuple[str, ...] = ()
    camera_specs: tuple[CameraSpec, ...] = ()
    keyboard_keys: tuple[str, ...] = ()
    notes: Mapping[str, str] = field(default_factory=dict)
    robot_configuration: Mapping[str, Any] = field(default_factory=dict)

    def supports_action(self, name: str) -> bool:
        return str(name) in self.action_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "environment_id": self.environment_id,
            "backend": self.backend,
            "robot": self.robot,
            "action_types": sorted(self.action_types),
            "observation_channels": sorted(self.observation_channels),
            "privileged_state": self.privileged_state,
            "seeded_reset": self.seeded_reset,
            "diagnostic_render": self.diagnostic_render,
            "robot_cameras": list(self.robot_cameras),
            "camera_specs": [camera.to_dict() for camera in self.camera_specs],
            "keyboard_keys": list(self.keyboard_keys),
            "notes": dict(self.notes),
            "robot_configuration": deepcopy(dict(self.robot_configuration)),
        }


@dataclass(frozen=True)
class ImageFrame:
    name: str
    path: Path
    deployable: bool
    mime_type: str = "image/png"
    camera: CameraSpec | None = None

    def read_bytes(self) -> bytes:
        return self.path.read_bytes()

    def _repr_png_(self) -> bytes | None:
        if self.mime_type == "image/png" and self.path.exists():
            return self.read_bytes()
        return None


@dataclass(frozen=True)
class Observation:
    sequence: int
    sim_time_s: float
    robot: str
    proprioception: Mapping[str, Any]
    images: Mapping[str, ImageFrame] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence,
            "sim_time_s": self.sim_time_s,
            "robot": self.robot,
            "proprioception": deepcopy(dict(self.proprioception)),
            "images": {
                name: {
                    "path": str(frame.path),
                    "deployable": frame.deployable,
                    "mime_type": frame.mime_type,
                    "camera": (
                        None if frame.camera is None else frame.camera.to_dict()
                    ),
                }
                for name, frame in self.images.items()
            },
            "metadata": deepcopy(dict(self.metadata)),
        }


@dataclass(frozen=True)
class BackendStep:
    state: Mapping[str, Any]
    info: Mapping[str, Any] = field(default_factory=dict)
    reward: float | None = None
    terminated: bool = False
    truncated: bool = False


@dataclass(frozen=True)
class StepResult:
    observation: Observation
    reward: float | None
    terminated: bool
    truncated: bool
    info: Mapping[str, Any]

    def gym_tuple(self) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        return (
            self.observation.to_dict(),
            0.0 if self.reward is None else float(self.reward),
            self.terminated,
            self.truncated,
            dict(self.info),
        )


__all__ = [
    "BackendStep",
    "CameraSpec",
    "Capabilities",
    "ImageFrame",
    "Observation",
    "StepResult",
]
