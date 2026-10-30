"""Per-arm FK/IK for catalog arms mounted on the tightening bench.

Built on the shared robots.controllers.kinematics.SerialKinematics from each
catalog instance's own URDF payload; bases come from the same MOUNTS table the
scene uses. IK returns servo targets only and never moves a body.
"""

import numpy as np
from scipy.spatial.transform import Rotation

from robots.controllers.kinematics import SerialKinematics

# World-owned mount poses (position, yaw) for the bench-side arms. Both arms
# face across the bench (+y), so the shared start pose puts the two hands
# 0.86 m apart above their own tool cradles.
MOUNTS = {"left_arm_mount": ((-.43, -.26, .78), np.pi/2),
          "right_arm_mount": ((.43, -.26, .78), np.pi/2)}


def cradle_layout(arms):
    """Tool cradle x positions for the mounted arms: each preset driver lies
    beside the arm nearest its work - the green M3 driver on the right, by
    the sensor (x = 0.27 m); the yellow M4 driver on the left, by the frame
    corner (x = 0); a single arm gets both cradles on its own side."""
    arms = set(arms)
    if arms == {"left", "right"}:
        return {"M3_driver": .36, "M4_driver": -.36}
    if arms == {"right"}:
        return {"M3_driver": .36, "M4_driver": .50}
    if arms == {"left"}:
        return {"M3_driver": -.50, "M4_driver": -.36}
    raise ValueError(f"No tool layout for arms {sorted(arms)}")
# Tool centre point per catalog arm model: tip link and offset along its z.
TCP = {"fr3": ("fr3_hand_tcp", 0.), "panda": ("panda_hand", .1034)}


def base_transform(mount):
    position, yaw = MOUNTS[mount]
    t = np.eye(4)
    t[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
    t[:3, 3] = position
    return t


def validate_setup(setup):
    """Accept any catalog setup this bench can physically host: arm instances
    named "left"/"right" on their bench mounts, catalog arm models with a
    bench TCP, a Franka Hand on each flange. Returns the setup dict."""
    from robots import resolve
    data = resolve(setup).to_dict()
    arms = data["instances"]
    if any(i["model"].split("@")[0] == "g1" for i in arms.values()):
        raise ValueError(
            "The catalog G1 cannot work this bench: standing at its front edge its shoulder is "
            "~1.07 m high and its straight-arm reach to the palm ~0.43 m (URDF), while the driver "
            "grips in the cradles lie ~0.74 m away and the upward M4 is driven from below at "
            "~0.58 m; and no balance/whole-body controller is qualified for manipulation in this "
            "repository (whiteboard's G1 stands on joint servos only)")
    if not arms or set(arms) - {"left", "right"}:
        raise ValueError("tighten-codex arm instances are 'left' and/or 'right'")
    for name, inst in arms.items():
        if inst["mount"] != f"{name}_arm_mount":
            raise ValueError(f"Instance {name} must use mount {name}_arm_mount")
        model = inst["model"].split("@")[0]
        if model not in TCP:
            raise ValueError(f"No bench TCP for arm model {model}; bench arms: {', '.join(sorted(TCP))}")
    hands = {a["parent"].split("/")[0] for a in data.get("attachments", {}).values()
             if a["model"].split("@")[0] == "franka-hand"}
    if hands != set(arms):
        raise ValueError("Each arm needs a franka-hand@1 attachment on its flange")
    cradle_layout(arms)
    return data


class ArmKinematics:
    """Kinematic chains keyed by instance name ("left", "right")."""

    def __init__(self, rig, setup):
        data = setup.to_dict()
        self.chains = {}
        self.names = {}
        self.fingers = {}
        for name, handle in rig.instances.items():
            model = handle.model_id
            if model not in TCP:
                raise ValueError(f"No bench TCP definition for catalog arm {model}")
            tip, offset = TCP[model]
            mount = data["instances"][name]["mount"]
            chain = SerialKinematics(handle.metadata["urdf"], tip,
                                     base=base_transform(mount), tip_offset=(0, 0, offset))
            self.chains[name] = chain
            self.names[name] = [f"{name}/{j}" for j in chain.names]
            self.fingers[name] = [f"{name}/{model}_finger_joint{i}" for i in (1, 2)]

    def fk(self, arm, q):
        return self.chains[arm].fk(np.asarray(q, float))

    def solve(self, arm, position, orientation_xyzw, seed, *, max_evaluations=160):
        return self.chains[arm].solve(position, orientation_xyzw, seed,
                                      max_evaluations=max_evaluations)
