"""Public environment facade."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from contextlib import AbstractContextManager, nullcontext
import threading
from typing import Any, Callable, Mapping, Sequence

from .actions import RobotAction
from .backends.base import RoboticsBackend
from .errors import EnvironmentClosedError, ObservationUnavailableAfterCommitError
from .policy_access import PolicyAccess
from .types import Capabilities, ImageFrame, Observation, StepResult


class RoboticsEnv:
    """A serialized, interactive robot-environment session.

    ``obs`` exposes policy-visible robot observations. ``get_state`` is an
    intentionally separate privileged diagnostics channel.
    """

    def __init__(self, backend: RoboticsBackend, *, auto_reset: bool = True) -> None:
        self._backend = backend
        self._closed = False
        self._lock = threading.RLock()
        self._sequence = 0
        self._last_observation: Observation | None = None
        if auto_reset:
            try:
                self.reset()
            except Exception:
                self._closed = True
                self._backend.close()
                raise

    @property
    def capabilities(self) -> Capabilities:
        provider=getattr(self._backend,"robot_configuration",None)
        actual=provider() if provider else {}
        return replace(self._backend.capabilities,robot_configuration=actual) if actual else self._backend.capabilities

    @property
    def closed(self) -> bool:
        return self._closed

    def session_paused(self, paused: bool) -> None:
        """Lifecycle hint from a session layer: while paused, a backend may release costly resources (a render
        worker) and restore them on resume. Backends without a ``session_paused`` hook are unaffected."""
        hook = getattr(self._backend, "session_paused", None)
        if callable(hook) and not self._closed:
            hook(bool(paused))

    @property
    def last_observation(self) -> Observation | None:
        return self._last_observation

    def _require_open(self) -> None:
        if self._closed:
            raise EnvironmentClosedError("environment is closed")

    def extra_arm_names(self) -> tuple[str, ...]:
        """Arm names the backend accepts besides 'left' and 'right' (``actions.arm_names``); none by default."""
        names = getattr(self._backend, "extra_arm_names", None)
        return tuple(names) if isinstance(names, (tuple, list, set, frozenset)) else ()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Observation, dict[str, Any]]:
        with self._lock:
            self._require_open()
            state, info = self._backend.reset(seed=seed, options=options)
            self._sequence = 0
            try:
                observation = self._backend.observation_from_state(
                    state, sequence=self._sequence
                )
            except Exception as error:
                raise ObservationUnavailableAfterCommitError(
                    operation="reset",
                    sequence=self._sequence,
                    info=info,
                    cause=error,
                ) from error
            self._last_observation = observation
            return observation, dict(info)

    def get_state(self) -> dict[str, Any]:
        """Return a detached copy of privileged simulator state."""

        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.get_state()))

    def evaluate(self) -> dict[str, Any]:
        """Privileged read-only task evaluation; missing criteria are not failure."""
        from .evaluation import normalize
        with self._lock:
            self._require_open()
            return normalize(self._backend.evaluate())

    def obs(self) -> Observation:
        """Capture policy-visible robot observations without advancing physics."""

        with self._lock:
            self._require_open()
            observation = self._backend.observe(sequence=self._sequence)
            self._last_observation = observation
            return observation

    def variation_contract(self) -> dict[str, Any]:
        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.variation_contract()))

    def select_variation(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Preview a matching variation and its reset request; do not reset."""

        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.select_variation(request)))

    observe = obs

    def geometry_index(self) -> dict[str, Any]:
        """List geometry objects and content-addressed mesh assets."""

        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.get_geometry_index()))

    def geometry(self, geometry_id: str) -> dict[str, Any]:
        """Return one simulator-authored object geometry descriptor."""

        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.get_geometry(str(geometry_id))))

    def geometry_asset(self, asset_id: str) -> dict[str, Any]:
        """Return exact triangle arrays for one referenced mesh asset."""

        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.get_geometry_asset(str(asset_id))))

    def act(self, action: RobotAction | Sequence[RobotAction], *,
            action_scope: Callable[[int], AbstractContextManager] | None = None) -> StepResult:
        """Execute one action or an ordered sequence of actions.

        ``action_scope`` optionally brackets each backend command for diagnostics;
        it does not add observation captures, alter durations, or change early stops.
        """

        return self._act(action, restricted=False, action_scope=action_scope)

    def policy_access(self) -> PolicyAccess:
        access=self._backend.policy_access()
        provider=getattr(self._backend,"robot_configuration",None)
        actual=provider() if provider else {}
        return replace(access,hardware={**access.hardware,"robot_configuration":actual}) if actual else access

    def begin_action_trace(self, context: Mapping[str, str]) -> dict[str, Any]:
        with self._lock:
            return self._backend.begin_action_trace(context)

    def end_action_trace(self) -> dict[str, Any] | None:
        with self._lock:
            return self._backend.end_action_trace()

    def act_policy(self, action: RobotAction | Sequence[RobotAction], *,
                   action_scope: Callable[[int], AbstractContextManager] | None = None) -> StepResult:
        """Execute audited robot commands without hidden-success early stopping."""

        return self._act(action, restricted=True, action_scope=action_scope)

    def _act(
        self, action: RobotAction | Sequence[RobotAction], *, restricted: bool,
        action_scope: Callable[[int], AbstractContextManager] | None = None,
    ) -> StepResult:
        with self._lock:
            self._require_open()
            actions = (
                tuple(action)
                if isinstance(action, Sequence) and not isinstance(action, (str, bytes))
                else (action,)
            )
            if not actions:
                raise ValueError("act requires at least one action")
            validate = getattr(self._backend, "validate_action", None)
            if validate is not None and len(actions) > 1:
                # optional backend hook: refuse a batch before its first command runs (nothing executes on refusal)
                for item in actions:
                    validate(item)
            result = None
            for index, item in enumerate(actions):
                with action_scope(index) if action_scope is not None else nullcontext():
                    result = (self._backend.act_policy(item) if restricted
                              else self._backend.act(item))
                self._sequence += 1
                if not restricted and (result.terminated or result.truncated):
                    break
            assert result is not None
            try:
                observation = self._backend.observation_from_state(
                    result.state, sequence=self._sequence
                )
            except Exception as error:
                raise ObservationUnavailableAfterCommitError(
                    operation="action",
                    sequence=self._sequence,
                    info=result.info,
                    cause=error,
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                ) from error
            self._last_observation = observation
            return StepResult(
                observation=observation,
                reward=result.reward,
                terminated=result.terminated,
                truncated=result.truncated,
                info=deepcopy(dict(result.info)),
            )

    step = act

    def render(self, *, label: str = "interactive") -> ImageFrame:
        """Capture a diagnostic render; it is not a robot-policy camera.

        A backend may declare ``prepare_render(label=...)``: called under the environment's lock, it copies what the
        render needs and returns a callable that renders it, which then runs outside the lock (a slow offline render
        does not hold up policy commands), or None to render under the lock as usual."""

        with self._lock:
            self._require_open()
            prepare = getattr(self._backend, "prepare_render", None)
            job = prepare(label=label) if callable(prepare) else None
            if job is None:
                path = self._backend.render(label=label)
                return ImageFrame(name="diagnostic_scene", path=path, deployable=False)
        return ImageFrame(name="diagnostic_scene", path=job(), deployable=False)

    def save_checkpoint(self, *, label: str) -> dict[str, Any]:
        """Save a restartable physics checkpoint without advancing time."""

        with self._lock:
            self._require_open()
            return deepcopy(dict(self._backend.save_checkpoint(label=label)))

    def restore_checkpoint(
        self, checkpoint: str
    ) -> tuple[Observation, dict[str, Any]]:
        """Restore a checkpoint and begin a new client-side observation branch."""

        with self._lock:
            self._require_open()
            state, info = self._backend.restore_checkpoint(checkpoint)
            self._sequence = 0
            try:
                observation = self._backend.observation_from_state(
                    state, sequence=self._sequence
                )
            except Exception as error:
                raise ObservationUnavailableAfterCommitError(
                    operation="checkpoint_restore",
                    sequence=self._sequence,
                    info=info,
                    cause=error,
                ) from error
            self._last_observation = observation
            return observation, deepcopy(dict(info))

    def list_checkpoints(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._require_open()
            return tuple(
                deepcopy(dict(checkpoint))
                for checkpoint in self._backend.list_checkpoints()
            )

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            try:
                self._backend.close()
            finally:
                self._closed = True

    def __enter__(self) -> "RoboticsEnv":
        self._require_open()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["RoboticsEnv"]
