"""Shared probe helpers: environment construction, code stamp, evidence output, assisted carriages.

ASSISTED: everything here that pushes a body (Carriage) is a phantom force of a probe, applied through
world.ext_wrench; it is not reachable from the backend's policy interface and every record that uses it says so.
"""

import hashlib
import json
import math
import subprocess
import time
import uuid
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

PKG = Path(__file__).resolve().parents[1]
REPO = PKG.parent
EVIDENCE = PKG/"evidence"
STAMP_EXTRA = ("gym_anything_robotics/backends/newton_concrete_finishing.py", "classroom_arrangement/world.py",
               "classroom_arrangement/scene.py", "classroom_arrangement/env.py", "classroom_arrangement/base.py",
               "classroom_arrangement/params.py", "robots/backends/newton.py", "robots/models/mobile-fr3-duo/robot.json",
               "robots/models/mobile-fr3-single/robot.json", "robots/rigs/finish-mobile-fr3-duo.json",
               "robots/rigs/finish-mobile-fr3-single.json")
PHYSICS = ("paste.py", "materials.py", "transfer.py", "coupling.py", "world.py", "scene.py", "tools.py", "layout.py", "params.py",
           "env.py", "task.py")
RUN_ID = uuid.uuid4().hex[:12]


def physics_digest():
    """Hash of the files that decide what the world does (not probes, rendering or report code)."""
    h = hashlib.sha256()
    for n in PHYSICS:
        h.update((PKG/n).read_bytes())
    for x in STAMP_EXTRA:
        p = REPO/x
        if p.is_file():
            h.update(p.read_bytes())
    return h.hexdigest()[:12]


DIGEST_AT_START = physics_digest()


def code_stamp():
    rev = subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = [ln for ln in subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", "--", "concrete_finishing",
                                          ":(exclude)concrete_finishing/evidence", *STAMP_EXTRA],
                                         capture_output=True, text=True).stdout.splitlines() if ln.strip()]
    return dict(git=rev, dirty=bool(dirty), uncommitted=[ln[3:] for ln in dirty][:12], physics=DIGEST_AT_START,
                physics_files_changed_during_run=physics_digest() != DIGEST_AT_START,
                hashed=[f"concrete_finishing/{n}" for n in PHYSICS] + list(STAMP_EXTRA),
                utc=time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    raise TypeError(type(o))


def save(name, data):
    path = EVIDENCE/name
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(data, code=code_stamp(), run_id=RUN_ID)
    path.write_text(json.dumps(data, indent=1, default=_default) + "\n")
    return path


def make_env(task="pad-darby", seed=0, *, edit=None, settle=True, **kw):
    """A task's start state. `edit(spec)` may change it for a probe (a tool placed on the paste, the robot's start)."""
    from concrete_finishing.config import Config
    from concrete_finishing.env import FinishEnv
    from concrete_finishing import layout as L
    c = Config(task=task, seed=seed, **kw)
    spec = L.setup(c.task, c.seed, variant=c.variant, **c.site())
    if edit is not None:
        edit(spec)
    return FinishEnv(c, spec=spec, settle=settle)


def surface_z(region, x, y):
    """Paste (or base) height under a world point of a horizontal region."""
    pr = region.to_region(np.array([x, y, 0.0]))
    i = int(np.clip(pr[0]/region.cell, 0, region.nu - 1)); j = int(np.clip(pr[1]/region.cell, 0, region.nv - 1))
    return float(region.o[2] + region.h[i, j])


def quat_from(R):
    return tuple(Rotation.from_matrix(R).as_quat())


class Carriage:
    """ASSISTED phantom carriage on one free body: PD toward a path across the press direction (the body's weight held
    there), a constant press along the press direction (-z by default) applied at the body's origin (a tool's sole), and
    a PD on orientation toward a target rotation; motion along the press direction free. Applied through world.ext_wrench from the state at
    the end of each tick (one tick of lag)."""

    def __init__(self, world, body, path, R_target, press, *, hz=8.0, zeta=1.0, rot_hz=10.0, vertical=None, press_dir=(0, 0, -1)):
        self.w, self.b = world, int(body)
        self.path, self.R_t, self.press = path, np.asarray(R_target, float), float(press)
        self.n = np.asarray(press_dir, float)/np.linalg.norm(press_dir)     # the press acts along n; motion along n is free
        m = float(world.model.body_mass.numpy()[self.b])
        self.m = m
        self.com_b = np.asarray(world.model.body_com.numpy()[self.b], float)
        I = np.asarray(world.model.body_inertia.numpy()[self.b], float).reshape(3, 3)
        om, omr = 2*math.pi*hz, 2*math.pi*rot_hz
        self.kp, self.kd = m*om*om, 2*m*zeta*om
        self.kr, self.kw = I*omr*omr, 2*I*zeta*omr
        self.vertical = vertical            # None: free; else (kp, kd, z(t)) for a probe that holds height
        self.t0 = world.sim_time
        self.log = []
        world.paste_recorders.append(self)

    def __call__(self, w):
        q = w.state.body_q.numpy()[self.b].astype(float)
        qd = w.state.body_qd.numpy()[self.b].astype(float)
        t = w.sim_time - self.t0
        xd, vd = self.path(t)
        n = self.n
        xd = np.asarray(xd, float); vd = np.asarray(vd, float)
        if len(xd) == 2:                                                 # a planar path: the press is vertical
            xd = np.array([xd[0], xd[1], q[2]]); vd = np.array([vd[0], vd[1], 0.0])
        e = xd - q[:3]; ev = vd - qd[:3]
        Pp = np.eye(3) - np.outer(n, n)                                  # PD across the press direction only
        # the weight is held across the press direction (as a hand holds a tool on a wall: without it the PD spring
        # sagged g/omega^2 = 6.9 mm at 6 Hz and set a jointer's rod on the block's arris below a bed joint, B7); along
        # the press direction it adds to the press, as a tool's weight does on a floor
        F = Pp @ (self.kp*e + self.kd*ev + np.array([0.0, 0.0, self.m*9.81])) + self.press*n
        if self.vertical is not None:
            kz, cz, zf = self.vertical
            F[2] += kz*(zf(t) - q[2]) - cz*qd[2]
        R = Rotation.from_quat(q[3:7]).as_matrix()
        err = Rotation.from_matrix(self.R_t @ R.T).as_rotvec()
        Iw = R @ self.kr @ R.T
        Cw = R @ self.kw @ R.T
        tau = Iw @ err - Cw @ qd[3:6]
        # the press acts at the sole (the body origin), as a hand pushes a tool onto its working face: applied at the
        # centre of mass, 24 mm out from a jointer's rod, 20 N toppled the tool about its rod faster than the
        # orientation hold resisted (B7); the wrench is taken about the centre of mass
        com = q[:3] + R @ self.com_b
        tau = tau + np.cross(q[:3] - com, self.press*n)
        w.ext_wrench[self.b, :3] = F
        w.ext_wrench[self.b, 3:] = tau
        self.F = F

    def detach(self):
        if self in self.w.paste_recorders:
            self.w.paste_recorders.remove(self)
        self.w.ext_wrench[self.b] = 0.0
