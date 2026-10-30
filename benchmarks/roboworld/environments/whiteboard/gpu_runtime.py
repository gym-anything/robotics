"""Explicit MuJoCo-Warp diagnostics and integration-state checkpoint adapter."""

import numpy as np


def constraint_readback_layout(addresses, dimensions, pyramidal, prefix, nefc):
    """Map active native rows to MuJoCo's contiguous contact layout.

    Inactive collision candidates retain address -1 and consume no rows.
    This is a host diagnostic mapping, not a constraint/physics modification.
    """
    indices = list(range(prefix))
    offsets = np.full(len(dimensions), -1, dtype=np.int32)
    for i, dim in enumerate(dimensions):
        if addresses[i, 0] < 0:
            continue
        width = max(1, 2 * (int(dim) - 1)) if pyramidal else int(dim)
        rows = addresses[i, :width]
        if len(rows) != width or np.any(rows < 0) or np.any(rows >= nefc):
            raise ValueError("Invalid active native contact constraint addresses")
        offsets[i] = len(indices)
        indices.extend(rows.tolist())
    indices = np.asarray(indices, np.int32)
    if len(indices) != nefc or not np.array_equal(np.sort(indices), np.arange(nefc)):
        raise ValueError("Native constraint readback is not a complete row permutation")
    return indices, offsets


def get_data_into(result, model, data, world_id=0):
    """Native readback plus a pinned fix for inactive-contact row indexing.

    MuJoCo-Warp 3.11 get_data_into assigns every collision candidate an active
    address, including rows with native address -1. It gathers -1 as the last
    allocated row and can create out-of-bounds CPU contactForce reads. Preserve
    native integration/constraints; repair the host permutation only.
    """
    import mujoco
    import mujoco_warp

    if model.flg_adhesion or np.any(model.geom_adhesion) or np.any(model.pair_adhesion):
        raise ValueError("Native MuJoCo-Warp contact adhesion is not supported by this adapter")
    count = int(data.nacon.numpy()[0])
    nefc = int(data.nefc.numpy()[world_id])
    if count > data.naconmax or nefc > data.njmax:
        raise FloatingPointError("GPU contact/constraint capacity overflow")
    worlds = data.contact.worldid.numpy()[:count]
    selected = np.flatnonzero(worlds == world_id)
    addresses = data.contact.efc_address.numpy()[selected]
    dimensions = data.contact.dim.numpy()[selected]
    prefix = sum(int(getattr(data, name).numpy()[world_id]) for name in ("ne", "nf", "nl"))
    indices, offsets = constraint_readback_layout(
        addresses, dimensions, model.opt.cone == mujoco.mjtCone.mjCONE_PYRAMIDAL, prefix, nefc
    )
    mujoco_warp.get_data_into(result, model, data, world_id=world_id)
    initialize_absent_adhesion(result, model)
    if np.array_equal(result.contact.efc_address, offsets):
        return
    result.contact.efc_address[:] = offsets
    for name in (
        "type",
        "id",
        "pos",
        "margin",
        "D",
        "vel",
        "aref",
        "frictionloss",
        "state",
        "force",
        "island",
    ):
        getattr(result, "efc_" + name)[:] = getattr(data.efc, name).numpy()[world_id, indices]
    if nefc:
        if mujoco.mj_isSparse(model):
            dense = np.zeros((nefc, model.nv))
            mujoco.mju_sparse2dense(
                dense,
                data.efc.J.numpy()[world_id, 0],
                data.efc.J_rownnz.numpy()[world_id, :nefc],
                data.efc.J_rowadr.numpy()[world_id, :nefc],
                data.efc.J_colind.numpy()[world_id, 0],
            )
            mujoco.mju_dense2sparse(
                result.efc_J,
                dense[indices],
                result.efc_J_rownnz,
                result.efc_J_rowadr,
                result.efc_J_colind,
            )
        else:
            result.efc_J[: nefc * model.nv] = data.efc.J.numpy()[
                world_id, indices, : model.nv
            ].flatten()


def initialize_absent_adhesion(result, model):
    """Initialize fields missing from MJWarp 3.11's nonadhesive host transfer.

    MuJoCo 3.11 contactForce subtracts contact.adhesion even for a nonadhesive
    model. get_data_into does not populate that field after reallocating the
    host contact buffer. Its uninitialized bytes must not be interpreted as
    physical force. The model must actually specify zero native adhesion.
    """
    if model.flg_adhesion or np.any(model.geom_adhesion) or np.any(model.pair_adhesion):
        raise ValueError("Cannot zero configured contact adhesion")
    result.contact.adhesion[:] = 0.0
    result.qfrc_adhesion[:] = 0.0


def _arrays(value, prefix=""):
    """Include native contact/constraint caches, not just public integration fields.

    These are numeric arrays only. No device pointers, kernels, or pickle objects
    enter checkpoints. Version/model compatibility is checked by the disk codec.
    """
    import dataclasses
    import warp as wp

    for field in dataclasses.fields(value):
        item = getattr(value, field.name)
        name = prefix + field.name
        if isinstance(item, wp.array):
            yield name, item
        elif dataclasses.is_dataclass(item):
            yield from _arrays(item, name + ".")


def readback(solver):
    """Read the actual GPU solve, never run a CPU substitute step or contact solve."""
    d = solver.mjw_data
    if d.nacon.numpy()[0] > d.naconmax or np.any(d.nefc.numpy() > d.njmax):
        raise FloatingPointError(
            "GPU contact/constraint capacity overflow; no partial contact evidence"
        )
    get_data_into(solver.mj_data, solver.mj_model, d)
    if not np.isfinite(solver.mj_data.qpos).all() or not np.isfinite(solver.mj_data.qvel).all():
        raise FloatingPointError("GPU nonfinite integration state")


def snapshot(solver):
    return {name: array.numpy().copy() for name, array in _arrays(solver.mjw_data)}


def restore(solver, data):
    arrays = dict(_arrays(solver.mjw_data))
    if set(data) != set(arrays):
        raise ValueError("GPU checkpoint field layout differs")
    if any(tuple(value.shape) != tuple(arrays[name].numpy().shape) for name, value in data.items()):
        raise ValueError("GPU checkpoint array layout differs")
    for name, value in data.items():
        arrays[name].assign(value)
    readback(solver)
