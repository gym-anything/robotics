"""Backend contract for robotics environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Mapping

from ..actions import RobotAction
from ..errors import ActionRejectedError
from ..policy_access import PolicyAccess
from ..types import BackendStep, Capabilities, Observation


class RoboticsBackend(ABC):
    capabilities: Capabilities

    def robot_configuration(self):
        """Live shared robot handles, when this adapter constructs them."""
        owner=getattr(self,"env",None)
        if owner is None:
            owner=getattr(getattr(self,"session",None),"world",None)
        builder=getattr(owner,"builder",None);model=getattr(owner,"model",None)
        handles=getattr(builder,"_gar_robots",{})
        if not handles or model is None:return {}
        for handle in handles.values():
            if handle.model is not model:handle.bind(model)
        result={"instances":{name:handle.capabilities() for name,handle in handles.items()}}
        setup=getattr(getattr(owner,"assembly",None),"robot_setup",None)
        if setup:result.update(setup_id=setup.id,setup_fingerprint=setup.fingerprint)
        return result

    def evaluate(self):
        """Read the environment-owned evaluator, never update its temporal state."""
        from ..errors import BackendProtocolError
        state = self.get_state()
        if state.get("ready") is False or state.get("finite") is False or state.get("fault"):
            raise BackendProtocolError("Cannot evaluate a faulted physics state")
        return state.get("evaluation")

    def policy_access(self) -> PolicyAccess:
        """Return an audited restricted interface, or fail closed by default."""
        from ..errors import UnsupportedConfigurationError

        raise UnsupportedConfigurationError("this backend has no restricted policy interface")

    def act_policy(self, action: RobotAction) -> BackendStep:
        from ..errors import UnsupportedActionError

        raise UnsupportedActionError("this backend has no restricted policy controller")

    def begin_action_trace(self, context: Mapping[str, str]) -> dict[str, Any]:
        """Attach diagnostics to a command without changing its control parameters."""
        return {"available": False, "reason": "backend_has_no_substep_recorder"}

    def end_action_trace(self) -> dict[str, Any] | None:
        return None

    @abstractmethod
    def reset(
        self,
        *,
        seed: int | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> Mapping[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def observation_from_state(
        self, state: Mapping[str, Any], *, sequence: int
    ) -> Observation:
        raise NotImplementedError

    def observe(self, *, sequence: int) -> Observation:
        return self.observation_from_state(self.get_state(), sequence=sequence)

    def get_geometry_index(self) -> Mapping[str, Any]:
        """Return the simulator-authored geometry catalog when supported."""

        raise NotImplementedError("this backend does not expose geometry")

    def variation_contract(self) -> Mapping[str, Any]:
        """Describe supported dataset selectors without changing the environment."""

        return {"supported": False}

    def select_variation(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        from ..errors import UnsupportedActionError

        raise UnsupportedActionError("this backend does not support variation selection")

    def get_geometry(self, geometry_id: str) -> Mapping[str, Any]:
        """Return one complete geometry descriptor when supported."""

        raise NotImplementedError("this backend does not expose geometry")

    def get_geometry_asset(self, asset_id: str) -> Mapping[str, Any]:
        """Return one content-addressed geometry asset when supported."""

        raise NotImplementedError("this backend does not expose geometry")

    @abstractmethod
    def act(self, action: RobotAction) -> BackendStep:
        raise NotImplementedError

    @abstractmethod
    def render(self, *, label: str = "interactive") -> Path:
        raise NotImplementedError

    def save_checkpoint(self, *, label: str) -> Mapping[str, Any]:
        """Persist a restartable simulator checkpoint when supported."""

        raise NotImplementedError("this backend does not support checkpoints")

    def restore_checkpoint(
        self, checkpoint: str
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        """Restore a previously persisted simulator checkpoint."""

        raise NotImplementedError("this backend does not support checkpoints")

    def list_checkpoints(self) -> tuple[Mapping[str, Any], ...]:
        """List restartable checkpoints visible to this backend."""

        raise NotImplementedError("this backend does not support checkpoints")

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError


def _same(a, b):
    import numpy as np
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b))
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    return a == b


def reject_unexecuted(execute, snapshot):
    """Run one command. A ValueError raised while snapshot() is unchanged (no target written, no tick advanced) comes
    from the backend's pre-execution validation, such as a target outside the published joint limits: it is the
    command's rejection (ActionRejectedError), which the server cancels without faulting the session. Any failure
    after a write or a step propagates unchanged and still latches a runtime fault."""
    before = snapshot()
    try:
        return execute()
    except ActionRejectedError:
        raise
    except ValueError as error:
        if _same(snapshot(), before):
            raise ActionRejectedError(str(error)) from error
        raise
