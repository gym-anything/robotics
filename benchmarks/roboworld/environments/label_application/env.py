"""Persistent label-application environment: robot-only actions, exact ticks, explicit faults.

Policy-visible controls are joint position targets (arms and Franka Hand fingers).
One environment tick = one label step = `label_substeps` rigid ticks (2 ms default).
Assisted fixtures use World directly and are labelled; they are never
reachable through this class's policy methods.
"""

from dataclasses import asdict, replace
import copy
import math
import threading
from pathlib import Path
import numpy as np

from .config import Config, REPAIR_TASKS, START_CREDIT_TASKS
from .world import World
from .task import Evaluator, INSTRUCTIONS
from . import artwork

from . import layouts

PAYLOADS = layouts.PAYLOADS          # seed-0 payloads; layouts.payload(task, seed) for every seed


def prepare_start(world):
    """The task's constructed start state before the settle: press-down's laid label; the ladder's repair rungs'
    partly applied label and ready pose (ladder.py). Other tasks start from the printer's presented label."""
    if world.config.task == "press-down":
        lay_label_on_carton(world)
        return {}
    if world.config.task in REPAIR_TASKS:
        from . import ladder
        return ladder.build_start(world)
    return {}


def make_art(config, folder):
    payload = layouts.payload(config.task, config.seed)
    if config.preset == "parcel":
        return artwork.shipping_label(folder, payload, gs1=config.task == "gs1-case-side", seed=config.seed)
    if config.preset == "specimen":
        return artwork.specimen_label(folder, payload, seed=config.seed)
    return artwork.part_label(folder, payload, seed=config.seed)


def lay_label_on_carton(world):
    """press-down START STATE (constructed, not a robot action): the label as a co-worker left it -- laid flat,
    adhesive down and untensioned, on the carton top at the seeded position inside the carrier zone of the flap
    half toward the robots (layouts.laid_label), touching, so it tacks on under its own physics during the
    settle. Nothing holds it afterwards. Same placement as the assisted demos' fixture (demos/assisted_apply.place)."""
    lab = world.label
    g = world.scene.task_geometry["carton"]
    lay = world.scene.task_geometry["laid_label"]
    p, R = world.body_pose("carton")
    (uc, vc), yaw = lay["centre"], lay["yaw"]
    c, s_ = np.cos(yaw), np.sin(yaw)
    md = R @ np.array([c, s_, 0.])                  # label length (6 in) along the carton
    cd = R @ np.array([-s_, c, 0.])
    centre = p + R @ np.array([uc, vc, g["H"]/2])
    n = np.cross(md, cd)
    X = lab.X
    lab.x = (centre[None, :] + (X[:, 0:1] - lab.length/2)*md[None, :] + (X[:, 1:2] - lab.width/2)*cd[None, :]
             + n[None, :]*(.5*lab.stock.thickness))
    lab.v[:] = 0.
    lab.bond[:] = -1; lab.phi[:] = 0.; lab.damage[:] = 0.; lab.gas[:] = 0.
    lab.theta_rest[:] = 0.
    world.label_history.clear()


class LabelEnv:
    def __init__(self, config=None, *, output_dir=None):
        self.config = config or Config()
        self.lock = threading.RLock()
        self.output = Path(output_dir or Path(__file__).resolve().parent/"output"/"session")
        self.output.mkdir(parents=True, exist_ok=True)
        self.art = make_art(self.config, self.output/"art")
        self.world = World(self.config)
        self.start_info = prepare_start(self.world)
        self.generation = 0
        self.trace_callback = None
        before = {n: self.world.body_pose(n)[0].copy() for n in self.world.scene.items}
        self._settle()
        # diagnostic: how far each item moved while the start settled (a start that disturbs its item is a bad start)
        self.start_info["settle_item_displacement_m"] = {
            n: float(np.linalg.norm(self.world.body_pose(n)[0] - p)) for n, p in before.items()}
        if self.config.task in START_CREDIT_TASKS:
            # the repair rungs' and press-down's progress is the work done on what was left to do (task.py,
            # Evaluator.progress): each label node's credit in the settled start state, fixed for the session
            self.world.scene.task_geometry["start_credit"] = Evaluator(self.world, self.art).node_credit()
        # the label's damage record (world.set_peak) starts with the episode: building and settling the start is not
        # part of it
        self.world.set_peak = np.abs(self.world.label.theta_rest)
        self.initial = self.checkpoint()
        self.evaluator = Evaluator(self.world, self.art)
        self.renderer = None

    # ------------------------------------------------------------------ lifecycle
    def _settle(self):
        """Printer presentation settles under gravity before the task starts (robot holds still); press-down:
        the laid label tacks onto the carton; smooth-corner / finish-securing: the partly applied label comes to
        rest with its lifted areas up."""
        n = int(round(self.config.settle_s/self.config.label_dt))
        for _ in range(n):
            self.world.step()
        self.world.tick = 0
        self.world.steps = 0
        self.world.label.time = 0.
        self.world.timing = dict(rigid_s=0., label_s=0., couple_s=0.)

    def checkpoint(self):
        w = self.world
        lab = w.label
        return dict(
            rigid={a: getattr(w.state, a).numpy().copy() for a in ("body_q", "body_qd", "joint_q", "joint_qd")},
            targets=w.targets.copy(), control=w.control.joint_target_q.numpy().copy(), tick=w.tick, steps=w.steps,
            wrench=w.body_wrench.copy(),
            label={k: copy.deepcopy(getattr(lab, k)) for k in (
                "x", "v", "bond", "bond_surface", "anchor", "anchor_n", "phi", "p_sustained", "bond_time", "damage", "gas",
                "theta_rest", "torn", "time", "contact_force", "events", "_pair_lam", "pressure",
                "alive", "k_hinge", "boundary", "_pinch_held", "pockets", "vertex_alive", "rebond_ok")},
            grip_force=dict(w.grip_force), grasp_latched=dict(w.grasp_latched), label_frozen=copy.deepcopy(w.label_frozen),
            set_peak=w.set_peak.copy(),
            colliders=[(c.R.copy(), c.p.copy(), c.R_prev.copy(), c.p_prev.copy(), copy.deepcopy(c.pressure_force))
                       for c in lab.colliders])

    def restore(self, snap):
        w = self.world
        w.label_history.clear()          # the settle window must not span a restore
        for a, v in snap["rigid"].items():
            getattr(w.state, a).assign(v)
            getattr(w.next_state, a).assign(v)
        w.targets = snap["targets"].copy()
        w.control.joint_target_q.assign(snap["control"])
        w.tick = snap["tick"]
        w.steps = snap["steps"]
        w.body_wrench = snap["wrench"].copy()
        w.grip_force = dict(snap["grip_force"])
        w.grasp_latched = dict(snap.get("grasp_latched", {}))
        w.label_frozen = copy.deepcopy(snap.get("label_frozen"))
        w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
        w.solver._mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
        for k, v in snap["label"].items():
            setattr(w.label, k, copy.deepcopy(v))
        for c, (R, p, Rp, pp, f) in zip(w.label.colliders, snap["colliders"]):
            c.R, c.p, c.R_prev, c.p_prev, c.pressure_force = R.copy(), p.copy(), Rp.copy(), pp.copy(), f
        # checkpoints written before the damage record existed carry only the present set: start the record there
        w.set_peak = (np.asarray(snap["set_peak"], float).copy() if "set_peak" in snap
                      else np.abs(w.label.theta_rest).astype(float))
        w.label.fault = None
        w.fault = None
        w.label._pockets()

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            self.evaluator = Evaluator(self.world, self.art)
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.joints

    def act(self, targets, *, ticks, guard=None):
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            if self.world.fault:
                raise RuntimeError(self.world.fault)
            self.world.set_targets(targets)
            for _ in range(ticks):
                self.world.step()
                if self.trace_callback:
                    self.trace_callback(self)
                if guard and guard(self):
                    break
            return self.get_state()

    # ------------------------------------------------------------------ state
    def get_state(self):
        with self.lock:
            w = self.world
            jq, jv = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
            q = w.state.body_q.numpy()
            ev = self.evaluator.evaluate() if not w.fault else dict(success=False, score=0., checks={}, metrics={})
            return {
                "tick": w.steps, "generation": self.generation, "time_s": w.steps*self.config.label_dt,
                "ready": w.fault is None, "fault": w.fault, "config": asdict(self.config),
                "instruction": INSTRUCTIONS[self.config.task],
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                               "target": float(w.targets[s["q"]])} for n, s in w.joints.items()},
                "objects": {n: {"position_m": q[i["body"], :3].tolist(), "quaternion_xyzw": q[i["body"], 3:].tolist()}
                            for n, i in w.scene.items.items()},
                "label": {k: v for k, v in w.label.summary().items() if k != "stats"},
                "pad_loads_n": {w.model.body_label[b]: f for b, f in getattr(w, "last_pad_loads", {}).items()},
                "velocity_reflexes": len(w.reflexes),
                "label_solver": {k: w.label.stats.get(k) for k in ("steps", "unconverged", "extended", "unresolved",
                                                                  "max_last_update_m")},
                "terminated": w.label_frozen is not None,
                "termination": w.label_frozen,
                "evaluation": ev,
            }

    def close(self):
        if self.renderer:
            self.renderer.close()
            self.renderer = None
