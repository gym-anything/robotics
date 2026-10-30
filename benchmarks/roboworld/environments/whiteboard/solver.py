"""Version-pinned GPU contact ordering without unsafe CCD atomic rewriting.

Warp 1.16's counting pass suppresses scratch writes. MuJoCo-Warp 3.11's
GJK/EPA CCD kernel computes its contact allocations from those scratch writes,
so applying that transformation can overwrite/drop contacts. Run only those
CCD kernels with native atomics, then canonicalize the complete contact rows
before deterministic constraint construction/solving. No contacts are invented,
removed, merged, or rounded. CPU physics is unchanged.
"""

from contextlib import contextmanager
from dataclasses import fields
from importlib.metadata import version
import threading

import numpy as np
import warp as wp
from newton.solvers import SolverMuJoCo


_GPU_EXECUTION = threading.RLock()
_CONTACT_KEYS = (
    "worldid",
    "geom",
    "pos",
    "frame",
    "dist",
    "dim",
    "includemargin",
    "friction",
    "solref",
    "solreffriction",
    "solimp",
    "type",
)


def contact_order(arrays, count):
    """Exact lexicographic ordering of physical contact data, with no tolerance."""
    if count < 0:
        raise ValueError("Negative native contact count")
    if count == 0:
        return np.empty(0, dtype=np.int64)
    columns = []
    for name in _CONTACT_KEYS:
        value = arrays[name]
        if len(value) < count:
            raise ValueError("Contact count exceeds native capacity")
        active = value[:count].reshape(count, -1)
        if not np.isfinite(active).all():
            raise FloatingPointError(f"Nonfinite native contact field: {name}")
        columns.extend(active[:, i] for i in range(active.shape[1]))
    return np.lexsort(tuple(reversed(columns)))


def canonical_contacts(model, data):
    """Host ordering only; collision detection and constraint solve stay on GPU."""
    count = int(data.nacon.numpy()[0])
    if count < 0 or count > data.naconmax:
        raise FloatingPointError("GPU contact capacity overflow")
    if count < 1:
        return
    native = {
        field.name: getattr(data.contact, field.name)
        for field in fields(data.contact)
        if isinstance(getattr(data.contact, field.name), wp.array)
        and not (
            model.nflex == 0
            and field.name in ("flex", "elem", "vert")
            and getattr(data.contact, field.name).shape == (0,)
        )
    }
    arrays = {name: value.numpy().copy() for name, value in native.items()}
    if any(value.shape[0] != data.naconmax for value in arrays.values()):
        raise RuntimeError("Unsupported native contact field layout")
    order = contact_order(arrays, count)
    if np.array_equal(order, np.arange(count)):
        return
    for name, value in arrays.items():
        value[:count] = value[order]
        native[name].assign(value)


@contextmanager
def native_ccd_compilation():
    """Scoped factory adaptation; never edit an installed Newton/Warp source."""
    from mujoco_warp._src import collision_convex, collision_driver

    factory = collision_convex.ccd_kernel_builder
    broadphase_factory = collision_driver._nxn_broadphase

    def native_factory(*args, **kwargs):
        kernel = factory(*args, **kwargs)
        # These module internals are deliberately restricted to the locked
        # Warp version. Mark modified only when the compilation option changes.
        if kernel.module.options.get("deterministic") != wp.DeterministicMode.NOT_GUARANTEED:
            kernel.module.options["deterministic"] = wp.DeterministicMode.NOT_GUARANTEED
            kernel.module.mark_modified()
        return kernel

    def bounded_broadphase(*args, **kwargs):
        kernel = broadphase_factory(*args, **kwargs)
        # Pinned _nxn_broadphase calls _add_geom_pair at most once per
        # thread. That function performs exactly one ncollision atomic_add.
        # Do not inherit Newton's constraint-row bound (thousands), which
        # otherwise multiplies every candidate geometry pair and overflows
        # Warp's int32 counter-buffer capacity in the full classroom.
        if kernel.module.options.get("deterministic_max_records") != 1:
            kernel.module.options["deterministic_max_records"] = 1
            kernel.module.mark_modified()
        return kernel

    collision_convex.ccd_kernel_builder = native_factory
    collision_driver._nxn_broadphase = bounded_broadphase
    try:
        yield
    finally:
        collision_convex.ccd_kernel_builder = factory
        collision_driver._nxn_broadphase = broadphase_factory


class WhiteboardSolver(SolverMuJoCo):
    def __init__(self, *args, **kwargs):
        with _GPU_EXECUTION:
            super().__init__(*args, **kwargs)
            self.ordered_gpu_contacts = (
                not self.use_mujoco_cpu
                and self._deterministic != wp.DeterministicMode.NOT_GUARANTEED
            )
            if not self.use_mujoco_cpu:
                if version("warp-lang") != "1.16.0" or version("mujoco-warp") != "3.11.0":
                    raise RuntimeError(
                        "GPU contact/readback adapters require the pinned Warp/MuJoCo-Warp runtime"
                    )
            if self.ordered_gpu_contacts:
                import mujoco

                if self.mj_model.nflex or np.any(
                    self.mj_model.geom_type == mujoco.mjtGeom.mjGEOM_HFIELD
                ):
                    raise ValueError(
                        "This contact adapter does not cover native flex/heightfield CCD"
                    )
                if self.mjw_model.callback.contactfilter is not None:
                    raise ValueError("Unexpected pre-existing native contact callback")
                self.mjw_model.callback.contactfilter = canonical_contacts

    def step(self, *args, **kwargs):
        # Newton's module compilation options and CCD kernel factory are global
        # within the process. Serialize owned GPU calls, not worlds within a batch.
        if self.use_mujoco_cpu:
            return super().step(*args, **kwargs)
        with _GPU_EXECUTION:
            return super().step(*args, **kwargs)

    def _mujoco_warp_step(self):
        if not self.ordered_gpu_contacts:
            return super()._mujoco_warp_step()
        with native_ccd_compilation():
            return super()._mujoco_warp_step()
