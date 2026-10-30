"""Public exceptions for the robotics environment API."""

from __future__ import annotations

from typing import Any, Mapping


class RoboticsEnvironmentError(RuntimeError):
    """Base class for runtime failures reported by an environment."""


class EnvironmentClosedError(RoboticsEnvironmentError):
    """Raised when an operation is attempted after :meth:`close`."""


class BackendUnavailableError(RoboticsEnvironmentError):
    """Raised when an environment backend cannot be started or contacted."""


class BackendTimeoutError(BackendUnavailableError):
    """No reply/progress within the watchdog interval; execution is uncertain."""


class BackendProtocolError(RoboticsEnvironmentError):
    """Raised when a backend returns a malformed or failed response."""


class ActionRejectedError(ValueError):
    """One command was rejected before changing targets or advancing physics.

    Backends must only raise this from pre-execution validation/planning.
    It says nothing about earlier commands in the same request batch.
    """


class UnsupportedActionError(RoboticsEnvironmentError):
    """Unsupported command rejected before target writes or physics execution.

    Earlier commands in a batch may have executed. Failures after beginning
    execution must use a runtime error such as BackendProtocolError instead.
    """


class UnsupportedConfigurationError(RoboticsEnvironmentError):
    """Raised rather than silently ignoring an unsupported configuration."""


class ObservationUnavailableAfterCommitError(RoboticsEnvironmentError):
    """A state change committed, but its required observation could not be captured."""

    def __init__(
        self,
        *,
        operation: str,
        sequence: int,
        info: Mapping[str, Any],
        cause: Exception,
        reward: float | None = None,
        terminated: bool = False,
        truncated: bool = False,
    ) -> None:
        super().__init__(
            f"{operation} committed, but post-commit observation capture failed: "
            f"{type(cause).__name__}: {cause}"
        )
        self.operation = operation
        self.sequence = int(sequence)
        self.info = dict(info)
        self.cause = cause
        self.reward = reward
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)
