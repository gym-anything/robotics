from __future__ import annotations

import numpy as np
import pytest

from laundry.newton_style3d_ipc import (
    IPCStyle3DSelfContactConfig,
    IPCStyle3DSelfContactController,
)


def test_ipc_style3d_defaults_state_the_operator_split() -> None:
    config = IPCStyle3DSelfContactConfig()
    assert config.activation_distance_m == pytest.approx(4.0e-3)
    assert config.minimum_distance_m == 0.0
    assert config.enable_barrier_force
    assert config.enable_ccd
    assert config.verify_after_every_step


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"activation_distance_m": 0.0}, "activation"),
        ({"minimum_distance_m": 4.0e-3}, "minimum"),
        ({"barrier_stiffness_prior": 0.0}, "stiffness"),
        ({"topological_filter_graph_hops": -1}, "hops"),
        ({"ccd_safety_factor": 1.0}, "safety"),
        ({"enable_barrier_force": False, "enable_ccd": False}, "enable"),
    ),
)
def test_ipc_style3d_config_fails_closed(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        IPCStyle3DSelfContactConfig(**kwargs)


class _ExactPairAuditStub:
    def __init__(self, pairs: np.ndarray) -> None:
        self.pairs = np.asarray(pairs, dtype=np.int32).reshape((-1, 2))
        self.calls = 0

    def surface_intersection_pairs(self, positions: np.ndarray) -> np.ndarray:
        assert positions.shape == (4, 3)
        self.calls += 1
        return self.pairs


def _controller_with_exact_pair_stub(pairs: np.ndarray):
    controller = object.__new__(IPCStyle3DSelfContactController)
    controller.integrity_auditor = _ExactPairAuditStub(pairs)
    return controller


def test_integrity_validation_accepts_only_an_empty_exact_pair_set() -> None:
    controller = _controller_with_exact_pair_stub(np.empty((0, 2), dtype=np.int32))
    controller.validate_positions(np.zeros((4, 3)), context="unit state")
    assert controller.integrity_auditor.calls == 1


def test_integrity_validation_reports_the_exact_crossing_pair_count() -> None:
    controller = _controller_with_exact_pair_stub(np.asarray(((1, 7), (3, 9))))
    with pytest.raises(RuntimeError, match=r"2 confirmed triangle crossing pair"):
        controller.validate_positions(np.zeros((4, 3)), context="unit state")
    assert controller.integrity_auditor.calls == 1
