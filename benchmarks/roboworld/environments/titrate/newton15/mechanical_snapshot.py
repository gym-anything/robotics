"""Explicit, data-only exact mechanical checkpoints; never used implicitly.

Both Newton buffers, actuator state and the rigid backend's integration state
are preserved. Restoring is a declared discontinuity, not continuous rollout
evidence. The caller is responsible for serializing access to the live world.
"""

from dataclasses import asdict
import hashlib
from pathlib import Path
import numpy as np
import mujoco
from .runtime import require_newton


def identity(world):
    files = ("mechanics.py", "mechanical_snapshot.py", "laboratory.py", "lab_config.py",
             "apparatus.py", "geometry.py", "rigid_contact.py", "panda.py", "bored_geometry.py",
             "pipette_geometry.py", "room.py", "assets.py", "actuator_targets.py", "pipette_filler.py")
    source = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in files}
    from .assets import MANIFEST
    source["panda_manifest.json"] = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    return {"schema": "titration.mechanics.snapshot.v2", "config": asdict(world.lab.config),
            "source": source, "body_count": world.model.body_count, "shape_count": world.model.shape_count}


def arrays(namespace, prefix=""):
    newton, wp = require_newton()
    result = {}
    for name, value in vars(namespace).items():
        path = prefix+name
        if isinstance(value, wp.array):
            result[path] = value
        elif isinstance(value, newton.Model.AttributeNamespace):
            result.update(arrays(value, path+"/"))
    return result


def capture(world):
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(world.solver.mj_model, spec))
    mujoco.mj_getState(world.solver.mj_model, world.solver.mj_data, state, spec)
    return {"identity": identity(world), "tick": world.tick, "time_s": world.time_s,
            "fault": world.fault, "mujoco_integration": state,
            "actuator_targets": world.actuator_targets.capture(),
            **{name: {k: v.numpy().copy() for k, v in arrays(getattr(world, name)).items()}
               for name in ("state", "next_state", "control")}}


def restore(world, snapshot):
    expected = {"identity", "tick", "time_s", "fault", "mujoco_integration", "state", "next_state", "control", "actuator_targets"}
    if set(snapshot) != expected or snapshot["identity"] != identity(world):
        raise ValueError("Checkpoint schema/source/configuration does not match this mechanics instance")
    if type(snapshot["tick"]) is not int or snapshot["tick"] < 0 or not np.isfinite(snapshot["time_s"]):
        raise ValueError("Invalid checkpoint clock")
    writes = []
    actuator_targets = world.actuator_targets.validate(snapshot['actuator_targets'])
    # Validate every field before touching live arrays.
    for name in ("state", "next_state", "control"):
        targets = arrays(getattr(world, name))
        if set(targets) != set(snapshot[name]):
            raise ValueError(f"Checkpoint {name} array layout differs")
        for key, target in targets.items():
            incoming = np.asarray(snapshot[name][key])
            actual = target.numpy()
            if incoming.shape != actual.shape or incoming.dtype != actual.dtype or not np.isfinite(incoming).all():
                raise ValueError(f"Invalid checkpoint array {name}/{key}")
            writes.append((target, incoming))
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    backend = np.asarray(snapshot["mujoco_integration"])
    if backend.shape != (mujoco.mj_stateSize(world.solver.mj_model, spec),) or not np.isfinite(backend).all():
        raise ValueError("Invalid rigid integration checkpoint")
    setpoint = np.asarray(snapshot['control']['joint_target_q'])
    if not np.array_equal(setpoint, actuator_targets['setpoint'].astype(setpoint.dtype)):
        raise ValueError('Checkpoint motor setpoint and controller disagree')
    for target, incoming in writes:
        target.assign(incoming)
    mujoco.mj_setState(world.solver.mj_model, world.solver.mj_data, backend, spec)
    world.actuator_targets.restore_validated(actuator_targets)
    world.tick, world.time_s, world.fault = snapshot["tick"], snapshot["time_s"], snapshot["fault"]
