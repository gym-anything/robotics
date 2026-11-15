"""Intel RealSense D405 on the Panda's wrist: policy camera panda_wrist_rgb, built only for a session that selects it.

The case and bracket are real mass and collision geometry on panda_link7, with the geometry, masses and optical pose
of the Franka-wrist D405 in the other RoboWorld simulations (drilling/scene.py _wrist_camera). They are appended after
every shape of the released laboratory (laboratory.build_laboratory), so every body, joint and existing shape keeps
its index. A session without the camera builds the released model, and no module of the physics source identities
(mechanical_snapshot.identity, liquid_snapshot.source_identity) is involved in adding it.
"""

from .actuator_targets import ActuatorTargets
from .laboratory import build_laboratory
from .mechanics import LaboratoryMechanics
from .rigid_contact import configure, create_solver
from .runtime import require_newton

MOUNT_FRAME = "link7_d405_envelope"
HORIZONTAL_FOV_DEG = 87.
NATIVE_RESOLUTION_PX = (1280, 720)
# (part, centre in the wrist-link frame, full size, mass kg, render material): the D405's published 42 x 42 x 23 mm
# envelope and 60 g, and a 10 g aluminium bracket (engineering estimate); the shapes' own inertia applies.
PARTS = (("camera_case", (.075, 0., .07), (.042, .042, .023), .060, "camera_dark"),
         ("camera_bracket", (.05, 0., .046), (.050, .026, .006), .010, "aluminum"))
# Optical centre 1.5 mm in front of the case, looking along the wrist link's +z with image up along +y.
OPTICS = {"eye": (.075, -.012, .083), "target": (.075, -.012, .283), "up": (0., 1., 0.)}


def camera_name(arm):
    return f"{arm}_wrist_rgb"


def install(lab, cameras):
    """Add the D405 case and bracket to the wrist link of each robot arm whose wrist camera is in cameras; record the
    cameras' mounts as lab.camera_mounts (name: body, eye, target and up in the body frame, horizontal FOV)."""
    arms = {camera_name(arm): (arm, handle) for arm, handle in lab.builder._gar_robots.items()}
    missing = sorted(set(cameras)-set(arms))
    if missing:
        raise ValueError(f"No arm carries {', '.join(missing)}; wrist cameras: {', '.join(sorted(arms))}")
    mounts = {}
    for name in cameras:
        arm, handle = arms[name]
        body = handle.frames["wrist"]["body"]
        for part, centre, size, mass, material in PARTS:
            lab.box(f"{arm}_{part}", centre, size, material, body=body, density=mass/(size[0]*size[1]*size[2]))
        mounts[name] = dict(body=body, **OPTICS, horizontal_fov_deg=HORIZONTAL_FOV_DEG)
    lab.camera_mounts = mounts
    return mounts


def build(config, cameras):
    """build_laboratory's laboratory and model with the D405 of each selected wrist camera: the released builder, the
    parts appended, finalized and configured as build_laboratory does."""
    lab, _ = build_laboratory(config)
    install(lab, cameras)
    model = lab.builder.finalize(device=lab.config.device)
    configure(model)
    model.set_gravity((0., 0., -9.80665))
    return lab, model


class WristCameraMechanics(LaboratoryMechanics):
    """LaboratoryMechanics on the laboratory with the selected wrist cameras' D405s (the same construction steps as
    LaboratoryMechanics.__init__, on build's model)."""

    def __init__(self, config=None, cameras=()):
        newton, self.wp = require_newton()
        self.lab, self.model = build(config, tuple(cameras))
        self.state, self.next_state = self.model.state(), self.model.state()
        newton.eval_fk(self.model, self.model.joint_q, self.model.joint_qd, self.state)
        self.control = self.model.control()
        self.actuator_targets = ActuatorTargets(self.lab.robot_joints,
                                               self.control.joint_target_q.numpy())
        self.solver = create_solver(self.model)
        self.tick, self.time_s = 0, 0.
        self.fault = None
        self._validate_initial_contacts()
