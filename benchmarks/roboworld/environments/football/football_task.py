"""Continuous goal adjudication, trace schema, and flight invariants."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Iterable

from .aerodynamics import BallKinematics, ForceBreakdown
from .football_math import Vec3, is_finite, lerp
from .football_specs import BallSpec, FieldSpec


@dataclass(frozen=True)
class FlightSample:
    step: int
    time_s: float
    kinematics: BallKinematics
    forces: ForceBreakdown

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GoalCrossing:
    time_s: float
    center_world_m: Vec3
    velocity_world_m_s: Vec3
    whole_ball_crossed: bool
    inside_opening: bool
    scored: bool
    lateral_clearance_m: float
    lower_clearance_m: float
    upper_clearance_m: float
    reason: str


class GoalDetector:
    """Detect the whole ball crossing the goal plane between the frame.

    Field coordinates put the shooter at negative X and the goal behind X=0,
    so the ball travels toward +X.  Scoring occurs when its trailing surface,
    ``center_x - radius``, crosses the plane.  Linear interpolation gives a
    continuous event even when a fast shot traverses the plane in one step.

    The Newton 1.5 football/goal task does not score with this plane test; its
    verifier is football/newton15/task.py (frame-plane opening test, back edge
    of the drawn goal line, kick and handball rules, progress reward).
    """

    def __init__(
        self,
        ball: BallSpec = BallSpec(),
        field: FieldSpec = FieldSpec(),
        lower_surface_tolerance_m: float = 0.002,
    ) -> None:
        if lower_surface_tolerance_m < 0.0 or not math.isfinite(lower_surface_tolerance_m):
            raise ValueError("lower surface tolerance must be finite and non-negative")
        self.ball = ball
        self.field = field
        # A ground shot is legal.  Its rigid centre can sit slightly below the
        # nominal z=0 plane because the validated pitch law and shell both
        # deform.  This tolerance affects only the grass boundary, never the
        # post or crossbar clearances.
        self.lower_surface_tolerance_m = lower_surface_tolerance_m
        self.previous_time_s: float | None = None
        self.previous_position_m: Vec3 | None = None
        self.previous_velocity_m_s: Vec3 | None = None
        self.crossing: GoalCrossing | None = None

    @property
    def center_crossing_x_m(self) -> float:
        return self.field.goal_plane_x_m + self.ball.radius_m

    def reset(self) -> None:
        self.previous_time_s = None
        self.previous_position_m = None
        self.previous_velocity_m_s = None
        self.crossing = None

    def update(
        self,
        time_s: float,
        position_world_m: Vec3,
        velocity_world_m_s: Vec3,
    ) -> GoalCrossing | None:
        if self.crossing is not None:
            return self.crossing
        if not is_finite(position_world_m) or not is_finite(velocity_world_m_s) or not math.isfinite(time_s):
            raise ValueError("goal detector received non-finite state")

        previous_position = self.previous_position_m
        previous_velocity = self.previous_velocity_m_s
        previous_time = self.previous_time_s
        self.previous_position_m = position_world_m
        self.previous_velocity_m_s = velocity_world_m_s
        self.previous_time_s = time_s
        if previous_position is None or previous_velocity is None or previous_time is None:
            return None

        plane = self.center_crossing_x_m
        before = previous_position[0] <= plane
        after = position_world_m[0] > plane
        if not (before and after):
            return None
        denominator = position_world_m[0] - previous_position[0]
        alpha = 0.0 if denominator == 0.0 else (plane - previous_position[0]) / denominator
        alpha = min(1.0, max(0.0, alpha))
        center = lerp(previous_position, position_world_m, alpha)
        velocity = lerp(previous_velocity, velocity_world_m_s, alpha)
        crossing_time = previous_time + alpha * (time_s - previous_time)

        lateral_clearance = self.field.half_goal_width_m - (abs(center[1]) + self.ball.radius_m)
        lower_clearance = center[2] - self.ball.radius_m
        upper_clearance = self.field.goal_inside_height_m - (center[2] + self.ball.radius_m)
        inside = (
            lateral_clearance >= 0.0
            and lower_clearance >= -self.lower_surface_tolerance_m
            and upper_clearance >= 0.0
        )
        reason = "whole ball crossed inside the opening" if inside else "whole ball crossed outside the opening"
        self.crossing = GoalCrossing(
            time_s=crossing_time,
            center_world_m=center,
            velocity_world_m_s=velocity,
            whole_ball_crossed=True,
            inside_opening=inside,
            scored=inside,
            lateral_clearance_m=lateral_clearance,
            lower_clearance_m=lower_clearance,
            upper_clearance_m=upper_clearance,
            reason=reason,
        )
        return self.crossing


@dataclass(frozen=True)
class InvariantViolation:
    step: int
    time_s: float
    code: str
    detail: str


def validate_sample(sample: FlightSample, ball: BallSpec = BallSpec()) -> tuple[InvariantViolation, ...]:
    violations: list[InvariantViolation] = []
    state = sample.kinematics
    force = sample.forces
    for name, value in (
        ("position", state.position_world_m),
        ("velocity", state.velocity_world_m_s),
        ("angular_velocity", state.angular_velocity_world_rad_s),
        ("total_force", force.total_force_n),
        ("aerodynamic_torque", force.aerodynamic_torque_n_m),
    ):
        if not is_finite(value):
            violations.append(InvariantViolation(sample.step, sample.time_s, "non_finite", name))
    if force.speed_m_s < -1.0e-12 or force.reynolds_number < -1.0e-12:
        violations.append(InvariantViolation(sample.step, sample.time_s, "negative_scale", "speed or Reynolds number"))
    if not 0.0 <= force.wake.amplitude <= 1.0:
        violations.append(InvariantViolation(sample.step, sample.time_s, "wake_bounds", str(force.wake.amplitude)))
    if ball.mass_kg <= 0.0:
        violations.append(InvariantViolation(sample.step, sample.time_s, "invalid_ball", "non-positive mass"))
    return tuple(violations)


@dataclass
class FlightTrace:
    metadata: dict[str, Any]
    samples: list[FlightSample] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    violations: list[InvariantViolation] = field(default_factory=list)

    def append(self, sample: FlightSample) -> None:
        self.samples.append(sample)
        self.violations.extend(validate_sample(sample))

    def add_event(self, event_type: str, time_s: float, **payload: Any) -> None:
        self.events.append({"type": event_type, "time_s": time_s, **payload})

    def summary(self) -> dict[str, Any]:
        final = self.samples[-1] if self.samples else None
        max_speed = max((sample.forces.speed_m_s for sample in self.samples), default=0.0)
        max_reynolds = max((sample.forces.reynolds_number for sample in self.samples), default=0.0)
        return {
            "schema": "football-flight-trace-v1",
            "sample_count": len(self.samples),
            "duration_s": 0.0 if final is None else final.time_s,
            "max_speed_m_s": max_speed,
            "max_reynolds_number": max_reynolds,
            "event_count": len(self.events),
            "violation_count": len(self.violations),
            "final_position_world_m": None if final is None else final.kinematics.position_world_m,
            "final_velocity_world_m_s": None if final is None else final.kinematics.velocity_world_m_s,
        }

    def to_dict(self, include_samples: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": "football-flight-trace-v1",
            "metadata": self.metadata,
            "summary": self.summary(),
            "events": self.events,
            "violations": [asdict(item) for item in self.violations],
        }
        if include_samples:
            payload["samples"] = [sample.to_dict() for sample in self.samples]
        return payload

    def write_json(self, path: str | Path, include_samples: bool = True) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(include_samples), indent=2) + "\n", encoding="utf-8")
        return destination

    def write_jsonl(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        header = {"schema": "football-flight-trace-v1", "metadata": self.metadata}
        records: Iterable[dict[str, Any]] = (
            {"record": "sample", **sample.to_dict()} for sample in self.samples
        )
        with destination.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(header) + "\n")
            for record in records:
                stream.write(json.dumps(record) + "\n")
            for event in self.events:
                stream.write(json.dumps({"record": "event", **event}) + "\n")
            for violation in self.violations:
                stream.write(json.dumps({"record": "violation", **asdict(violation)}) + "\n")
        return destination
