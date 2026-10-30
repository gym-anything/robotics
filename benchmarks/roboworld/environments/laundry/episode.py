"""History-aware half-fold task, rewards, and anti-cheat termination."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import math

import numpy as np

from .mesh import RectangularMesh
from .observation import LaundryObservation
from .task import HalfFoldEvaluator, HalfFoldThresholds, TaskResult


class EpisodePhase(str, Enum):
    READY = "ready"
    GRASPED = "grasped"
    FOLDING = "folding"
    PLACED = "placed"
    RELEASED = "released"
    SETTLING = "settling"
    SUCCESS = "success"
    FAILED = "failed"
    TRUNCATED = "truncated"


@dataclass(frozen=True)
class EpisodeStep:
    phase: EpisodePhase
    task_result: TaskResult
    reward: float
    terminated: bool
    truncated: bool
    success: bool
    info: dict[str, object]


@dataclass
class HalfFoldEpisodeTask:
    mesh: RectangularMesh
    thickness_m: float
    thresholds: HalfFoldThresholds = field(default_factory=HalfFoldThresholds)
    timeout_s: float = 45.0
    require_grasp_history: bool = True
    require_fixture_clearance: bool = True
    phase: EpisodePhase = EpisodePhase.READY
    saw_grasp: bool = False
    saw_release_after_grasp: bool = False
    peak_strain: float = 0.0
    maximum_intersections: int = 0
    maximum_runtime_pose_writes: int = 0
    maximum_external_force_applications: int = 0
    previous_potential: float | None = None
    previous_released: bool = True
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.thickness_m <= 0.0 or self.timeout_s <= 0.0:
            raise ValueError("Task thickness and timeout must be positive")
        self.evaluator = HalfFoldEvaluator(
            self.mesh,
            self.thickness_m,
            thresholds=self.thresholds,
        )

    def reset(self) -> None:
        self.phase = EpisodePhase.READY
        self.saw_grasp = False
        self.saw_release_after_grasp = False
        self.peak_strain = 0.0
        self.maximum_intersections = 0
        self.maximum_runtime_pose_writes = 0
        self.maximum_external_force_applications = 0
        self.previous_potential = None
        self.previous_released = True
        self.failure_reason = None

    def update(
        self,
        observation: LaundryObservation,
        action: np.ndarray | None = None,
    ) -> EpisodeStep:
        observation.validate(particle_count=self.mesh.vertex_count)
        self.peak_strain = max(
            self.peak_strain,
            float(observation.maximum_structural_strain),
        )
        self.maximum_intersections = max(
            self.maximum_intersections,
            int(observation.nonlocal_intersection_count),
        )
        self.maximum_runtime_pose_writes = max(
            self.maximum_runtime_pose_writes,
            int(observation.runtime_pose_writes),
        )
        self.maximum_external_force_applications = max(
            self.maximum_external_force_applications,
            int(observation.external_particle_force_applications),
        )

        grasped_now = observation.pinched_particle_count > 0
        new_grasp = grasped_now and not self.saw_grasp
        self.saw_grasp = self.saw_grasp or grasped_now
        released_after_grasp = self.saw_grasp and observation.released
        new_release = released_after_grasp and not self.saw_release_after_grasp
        self.saw_release_after_grasp = (
            self.saw_release_after_grasp or released_after_grasp
        )
        settled = observation.maximum_speed_m_s <= self.thresholds.settled_max_speed_m_s

        result = self.evaluator.evaluate(
            observation.particle_positions_m,
            observation.particle_velocities_m_s,
            maximum_strain=self.peak_strain,
            invalid_intersections=self.maximum_intersections,
            invalid_pose_writes=self.maximum_runtime_pose_writes,
            released=(
                released_after_grasp
                if self.require_grasp_history
                else observation.released
            ),
            settled=settled,
        )

        if self.maximum_intersections > 0:
            self.failure_reason = "nonlocal cloth self-intersection"
        elif self.maximum_runtime_pose_writes > 0:
            self.failure_reason = "runtime cloth-pose authoring"
        elif self.maximum_external_force_applications > 0:
            self.failure_reason = "direct external particle force application"
        elif self.peak_strain > self.thresholds.maximum_strain:
            self.failure_reason = "peak structural strain exceeded the task bound"

        history_ok = self.saw_grasp or not self.require_grasp_history
        clearance_ok = observation.fixtures_clear or not self.require_fixture_clearance
        success = bool(result.success and history_ok and clearance_ok)
        hard_failure = self.failure_reason is not None
        timed_out = (
            observation.time_s >= self.timeout_s and not success and not hard_failure
        )

        if success:
            self.phase = EpisodePhase.SUCCESS
        elif hard_failure:
            self.phase = EpisodePhase.FAILED
        elif timed_out:
            self.phase = EpisodePhase.TRUNCATED
        elif self.saw_release_after_grasp:
            self.phase = EpisodePhase.SETTLING if not settled else EpisodePhase.RELEASED
        elif grasped_now and result.metrics.footprint_length_fraction <= 0.62:
            self.phase = EpisodePhase.PLACED
        elif grasped_now and result.potential > 0.60:
            self.phase = EpisodePhase.FOLDING
        elif grasped_now:
            self.phase = EpisodePhase.GRASPED
        else:
            self.phase = EpisodePhase.READY

        reward = self._reward(
            result,
            action=action,
            new_grasp=new_grasp,
            new_release=new_release,
            success=success,
            hard_failure=hard_failure,
            timed_out=timed_out,
        )
        self.previous_potential = result.potential
        self.previous_released = observation.released
        terminated = success or hard_failure
        info: dict[str, object] = {
            "phase": self.phase.value,
            "success": success,
            "failure_reason": self.failure_reason,
            "saw_grasp": self.saw_grasp,
            "saw_release_after_grasp": self.saw_release_after_grasp,
            "fixtures_clear": observation.fixtures_clear,
            "peak_structural_strain": self.peak_strain,
            "maximum_nonlocal_intersections": self.maximum_intersections,
            "maximum_runtime_pose_writes": self.maximum_runtime_pose_writes,
            "maximum_external_particle_force_applications": (
                self.maximum_external_force_applications
            ),
            "potential": result.potential,
            "failed_geometry_or_state_gates": result.failed_gates,
            "metrics": result.metrics.as_dict(),
            "thresholds": asdict(self.thresholds),
            "calibration_status": self.thresholds.calibration_status,
        }
        return EpisodeStep(
            phase=self.phase,
            task_result=result,
            reward=reward,
            terminated=terminated,
            truncated=timed_out,
            success=success,
            info=info,
        )

    def _reward(
        self,
        result: TaskResult,
        *,
        action: np.ndarray | None,
        new_grasp: bool,
        new_release: bool,
        success: bool,
        hard_failure: bool,
        timed_out: bool,
    ) -> float:
        previous = result.potential if self.previous_potential is None else self.previous_potential
        # Potential-difference shaping cannot be farmed by holding a static
        # intermediate pose.  Absolute success remains a conjunction of hard
        # physical and history gates.
        reward = 4.0 * (result.potential - previous) - 1.0e-3
        if action is not None:
            value = np.asarray(action, dtype=np.float64)
            if not np.all(np.isfinite(value)):
                raise ValueError("Reward action must be finite")
            reward -= 2.0e-4 * float(np.dot(value, value))
        if new_grasp:
            reward += 0.10
        if new_release:
            reward += 0.20 if result.potential >= 0.70 else -0.20
        if success:
            reward += 5.0
        elif hard_failure:
            reward -= 5.0
        elif timed_out:
            reward -= 1.0
        if not math.isfinite(reward):
            raise RuntimeError("Task reward became non-finite")
        return float(reward)
