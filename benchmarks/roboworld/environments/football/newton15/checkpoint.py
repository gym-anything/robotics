"""Version-fingerprinted exact native/constitutive state, without pickle."""

from dataclasses import asdict, is_dataclass
import hashlib
import math
from pathlib import Path
import numpy as np
from gym_anything_robotics.physics_trace import pack, unpack
from football.impact_model import ContactState
from football.pitch_model import PitchContactState
from football.aerodynamics import WakeState
from football.football_task import GoalCrossing

TYPES = {
    c.__name__: c for c in (ContactState, PitchContactState, WakeState, GoalCrossing)
}
EXCLUDE = {
    "ball_view",
    "foot_view",
    "net",
    "ball",
    "patch",
    "prior",
    "model",
    "foot_proxy",
    "foot_inverse_mass_provider",
    "foot_contact_geometry_provider",
    "contact_geometry_audit",
}


def encode(value):
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, np.generic):
        return encode(value.item())
    if is_dataclass(value):
        return {"_type": type(value).__name__, "fields": encode(vars(value))}
    if isinstance(value, float) and not math.isfinite(value):
        return {"_float": repr(value)}
    if isinstance(value, set):
        return {"_set": [encode(v) for v in sorted(value)]}
    if isinstance(value, tuple):
        return {"_tuple": [encode(v) for v in value]}
    if isinstance(value, list):
        return [encode(v) for v in value]
    if isinstance(value, dict):
        return {k: encode(v) for k, v in value.items()}
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"Unsupported checkpoint field {type(value)}")


def decode(value):
    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        return value
    if "_type" in value:
        return TYPES[value["_type"]](**decode(value["fields"]))
    if "_float" in value:
        if value["_float"] not in {"inf", "-inf", "nan"}:
            raise ValueError("Invalid checkpoint float")
        return float(value["_float"])
    if "_tuple" in value:
        return tuple(decode(v) for v in value["_tuple"])
    if "_set" in value:
        return set(decode(v) for v in value["_set"])
    return {k: decode(v) for k, v in value.items()}


def fingerprint(w):
    h = hashlib.sha256(repr(asdict(w.config)).encode())
    root = Path(__file__).parents[1]
    for path in sorted([*root.glob("*.py"), *Path(__file__).parent.glob("*.py")]):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def save(w):
    import mujoco

    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    native = np.empty(mujoco.mj_stateSize(w.solver.mj_model, spec))
    mujoco.mj_getState(w.solver.mj_model, w.solver.mj_data, native, spec)
    couplers = {
        "pitch": w.pitch,
        **({"foot": w.foot} if w.foot else {}),
        **{"net_" + k: v for k, v in w.net_couplers.items()},
    }
    data = {
        "schema": "football-checkpoint-v1",
        "fingerprint": fingerprint(w),
        "tick": w.tick,
        "native": native,
        "targets": w.targets,
        "wake": w.wake,
        "goal": vars(w.goal).copy(),
        "last": w.last,
        "states": [
            {
                name: getattr(s, name).numpy()
                for name in ("body_q", "body_qd", "joint_q", "joint_qd")
            }
            for s in (w.state_0, w.state_1)
        ],
        "couplers": {
            name: {k: v for k, v in vars(c).items() if k not in EXCLUDE}
            for name, c in couplers.items()
        },
        "nets": {
            name: {"position": n.position, "velocity": n.velocity}
            for name, n in w.nets.items()
        },
    }
    data["goal"].pop("ball")
    data["goal"].pop("field")
    return pack(encode(data))


def restore(w, blob):
    import mujoco

    envelope = unpack(blob, arrays=True)
    if envelope["nonfinite_paths"]:
        raise ValueError("Non-finite physical checkpoint arrays")
    data = decode(envelope["data"])
    if data["schema"] != "football-checkpoint-v1" or data["fingerprint"] != fingerprint(
        w
    ):
        raise ValueError("Checkpoint source/configuration mismatch; no state changed")
    # Validate native and Newton array layouts before the first assignment.
    expected = mujoco.mj_stateSize(
        w.solver.mj_model, mujoco.mjtState.mjSTATE_INTEGRATION
    )
    if data["native"].shape != (expected,):
        raise ValueError("Native checkpoint layout mismatch")
    for state, values in zip((w.state_0, w.state_1), data["states"]):
        for name, value in values.items():
            if (
                value.shape != getattr(state, name).numpy().shape
                or not np.isfinite(value).all()
            ):
                raise ValueError("Checkpoint array mismatch")
    for state, values in zip((w.state_0, w.state_1), data["states"]):
        for name, value in values.items():
            getattr(state, name).assign(value)
    mujoco.mj_setState(
        w.solver.mj_model,
        w.solver.mj_data,
        data["native"],
        mujoco.mjtState.mjSTATE_INTEGRATION,
    )
    w.targets[:] = data["targets"]
    w.control.joint_target_q.assign(w.targets)
    w.wake = data["wake"]
    vars(w.goal).update(data["goal"])
    couplers = {
        "pitch": w.pitch,
        **({"foot": w.foot} if w.foot else {}),
        **{"net_" + k: v for k, v in w.net_couplers.items()},
    }
    for name, values in data["couplers"].items():
        vars(couplers[name]).update(values)
    for name, values in data["nets"].items():
        w.nets[name].position[:] = values["position"]
        w.nets[name].velocity[:] = values["velocity"]
        w.nets[name].sync_nodes()
    w.tick, w.fault, w.last = data["tick"], None, data["last"]
    w.generation += 1
    w.refresh()
    return w.state()
