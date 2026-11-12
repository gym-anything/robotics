"""filleting-tasks-v2 evaluator: task_set_evaluator.py with the render-check fixes (take-knife success is the original
right-hand verdict and its progress measures the knife in either hand; only head-off accepts a knife put down).
fillet-whole-fish's evaluation (tasks/fillet-whole-fish/verifier.py) is never overridden.

Each task's own grading is in its verifier.py (tasks/<name>/), read as in task_set_evaluator.py; a cutting task whose
knife may be put down safely sets KNIFE_PUT_DOWN (head-off).
"""
import copy
from collections import deque

import numpy as np

from . import layout as LY
from .task import Evaluator, check
from .task_set_v2_catalogue import CONTRACT, LIMITS
from .task_set_progress import Connections, clip, gained, ramp


class TaskSetV2Evaluator(Evaluator):
    contract = CONTRACT

    def __init__(self, env, task):
        super().__init__(env)
        self.task = task
        self.knife_history = deque(maxlen=int(np.ceil(3.2/env.config.rigid_dt)) + 2)
        self.hand_history = {}
        self.weigh_windows = {}
        self.weighed = set()
        self.initial_faces = set(env.world.cut.cut_faces)
        self.graph = None
        self.source = self.target = None
        self.r0 = 0
        self.raw_start = 0.
        w = env.world
        if getattr(self.folder, "PRODUCT", None) is not None:
            self.graph = Connections(w)
            self._prime_product()
            checks, metrics, raw = self._cut_task()
            self.raw_start = raw
        elif hasattr(self.folder, "start"):
            self.folder.start(self)
        self.monitor()

    @property
    def folder(self):
        """The task's verifier.py (tasks/<name>/): every task of the set has its folder."""
        return LY.task_verifier(self.task)

    def _prime_product(self):
        """The task's product at the start (product in its verifier.py): a separation's terminals (source and target
        tets) and their connections at the start, or the number of slices the start allows."""
        w, g = self.env.world, self.graph
        nt = g.nt
        ref = np.asarray(w.assembly["ref_tet"][:nt])
        reg = np.asarray(w.assembly["region_tet"][:nt])
        if self.folder.PRODUCT == "slices":
            self.target_count = self.folder.product(self, ref, reg)
        else:
            source, target = self.folder.product(self, ref, reg)
            self.source, self.target = source, target
            self.r0 = g.remaining(source, target, start=True)

    def _targets(self, source, candidates):
        """A separation's target tets: the candidates more than two mesh lengths from the source tets."""
        from scipy.spatial import cKDTree
        w, g = self.env.world, self.graph
        if len(source) and len(candidates):
            d = cKDTree(g.centres[source]).query(g.centres[candidates])[0]
            target = candidates[d > 2*w.config.mesh_h]
        else:
            target = np.zeros(0, int)
        return target

    def snapshot(self):
        return copy.deepcopy(dict(knife=list(self.knife_history), hands=self.hand_history,
                                  windows=self.weigh_windows, weighed=self.weighed))

    def restore(self, snap):
        snap = copy.deepcopy(snap)
        self.knife_history = deque(snap["knife"], maxlen=int(np.ceil(3.2/self.env.config.rigid_dt)) + 2)
        self.hand_history, self.weigh_windows, self.weighed = snap["hands"], snap["windows"], snap["weighed"]

    def knife_pose(self):
        from .workpiece import blade_segment_poses
        poses = blade_segment_poses(self.env.world)
        # First edge-centre minus half a segment gives the actual flexible blade heel.
        R, p = poses[0]
        heel = p - R[:, 0]*self.env.world.scene.blade.seg_len/2
        centres = self.env.world.tissue.body_states()["p"][1:1 + len(poses)]
        return heel, R, np.asarray(centres)

    def record_knife(self):
        """Append the knife handle's pose now to knife_history (a task's monitor calls it); returns it (p, R)."""
        env, w = self.env, self.env.world
        p, R = w.body_pose("knife_handle")
        self.knife_history.append((env.sim_time, np.array(p), np.array(R)))
        return p, R

    def monitor(self):
        """Called after every engine tick, not after a client action/evaluation: the task's own records (monitor in its
        verifier.py)."""
        hook = getattr(self.folder, "monitor", None)
        if hook is not None:
            hook(self)

    def evaluate(self):
        """The episode's outcome under this contract: the task's verifier.py (tasks/<name>/) decides, usually through
        evaluate_graded."""
        return self.folder.verify(self)

    def evaluate_graded(self, grade=None):
        """This contract's evaluation: grade(evaluator) -> (checks, metrics, progress) from the task's verifier.py, or
        None for a cutting task (_cut_task: progress gained over the start's). Success needs every check, the time limit
        among them; the score is 1 on success, else 0.95 x progress."""
        if self.env.world.fault:
            return dict(schema="gar.evaluation.v1", task=self.task, contract=CONTRACT,
                        success=False, score=0., checks={}, metrics={}, fault=self.env.world.fault)
        if grade is not None:
            checks, metrics, progress = grade(self)
        else:
            checks, metrics, raw = self._cut_task()
            progress = gained(raw, self.raw_start)
        checks["time"] = check("time", self.env.sim_time, LIMITS[self.task], "s",
                                self.env.sim_time <= LIMITS[self.task])[1]
        success = bool(checks) and all(c["passed"] for c in checks.values())
        progress = clip(progress)
        metrics.update(progress=dict(value=progress, failure_factor=.95), crack_faces=self.crack_faces())
        return dict(schema="gar.evaluation.v1", task=self.task, contract=CONTRACT, success=success,
                    score=1. if success else .95*progress, checks=checks, metrics=metrics,
                    time_s=self.env.sim_time, note=None)

    def _workmanship(self, metrics, frags):
        mass_g = sum(f["mass"] for f in frags.values())*1000
        allowance = self.DEBRIS_BUDGET*mass_g
        W = ramp(metrics["debris_g"], allowance, max(3*allowance, 1e-12))
        W *= ramp(metrics["piece_speed_max_mps"], 1., 3.)
        if metrics["tissue_nodes_below_bench"]:
            W *= .5
        if self.env.blade_robot_contacts:
            W *= .5
        return W

    def _sever(self, free_share, edges):
        remaining = self.graph.remaining(self.source, self.target, edges)
        part = clip(1 - remaining/self.r0) if self.r0 else 0.
        return clip(free_share + (1 - free_share)*part), remaining

    def _cut_task(self):
        """A cutting task's checks, metrics and raw progress: the task's released checks (released_checks in its
        verifier.py), the knife check if the task sets KNIFE_PUT_DOWN, the blade-contact check of this contract, and its
        progress (progress in its verifier.py; a separation's from the connections it severs, slices' from the products
        and the best unfinished cut) times the workmanship multiplier."""
        w, folder = self.env.world, self.folder
        checks, metrics = folder.released_checks(self)
        lab, frags, main = self._parts()
        edges = self.graph.edges()
        # head-off only (spec: "knife held or put down"): either hand, or a safe set-down, as fillet-whole-fish allows.
        # portion, medallions, excise-defect and remove-chain keep their original checks unchanged.
        if getattr(folder, "KNIFE_PUT_DOWN", False) and "knife_not_dropped" in checks:
            x = w.tissue.x.numpy()
            okay, where = self._knife_put_down([c for c, f in frags.items() if f["mass"] >= .001], frags, x)
            checks["knife_not_dropped"] = check("knife_not_dropped", float(okay), 1., "bool", okay, where)[1]
        checks["no_blade_robot_contact"] = check("no_blade_robot_contact", self.env.blade_robot_contacts, 0.,
                                                  "ticks", self.env.blade_robot_contacts == 0)[1]
        if folder.PRODUCT == "slices":
            partial = self.graph.partial_slice(lab, frags, edges, w.config.mesh_h)
            tilt = ramp(checks["cut_tilt_deg"]["value"], 10., 25.)
            raw = folder.progress(self, checks, metrics, frags, partial, tilt)
            raw = max(0., raw - .05*checks["no_tag_ends"]["value"])
            metrics.update(target_count=self.target_count, partial_cut=partial)
        else:
            raw, remaining = folder.progress(self, checks, metrics, edges)
            metrics.update(remaining_connections=remaining, starting_connections=self.r0)
        W = self._workmanship(metrics, frags)
        metrics["workmanship_multiplier"] = W
        return checks, metrics, clip(W*raw)
