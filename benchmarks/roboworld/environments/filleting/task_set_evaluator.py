"""Opt-in task-set evaluator. Evaluator._eval_A3 (fillet-whole-fish) is never overridden."""
import copy
from collections import deque

import numpy as np

from . import layout as LY
from .task import Evaluator, REF, check, near_boxes
from .task_set_catalogue import CONTRACT, LIMITS, NEW_TASK
from .task_set_progress import Connections, angle, clip, gained, ramp, knife_return_geometry


class TaskSetEvaluator(Evaluator):
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
        if task in ("A1", "B3", "C1", "B5", "C3"):
            self.graph = Connections(w)
            self._prime_product()
            checks, metrics, raw = self._cut_task()
            self.raw_start = raw
        elif task == NEW_TASK:
            from .workpiece import blade_segment_poses
            heel, R, segs = self.knife_pose()
            target, target_R = w.scene.block_knife_pose
            last_R, last_p = blade_segment_poses(w)[-1]
            self.start_distance = float(np.linalg.norm(last_p + last_R[:, 0]*w.scene.blade.seg_len/2 -
                                                       (target + target_R[:, 0]*w.scene.blade.L)))
            self.return_start = self._return_raw(heel, R, segs)[0]
        self.monitor()

    def _prime_product(self):
        w, g = self.env.world, self.graph
        ts, nt = w.tissue, g.nt
        ref = np.asarray(w.assembly["ref_tet"][:nt])
        reg = np.asarray(w.assembly["region_tet"][:nt])
        if self.task == "B5":
            # Leave two 5 g end pieces, as the product definition does. A fixed
            # denominator from this seed's initial loin mass, never remaining mass.
            mass = float(ts.htmass[:nt][ref == self._ref_id("loin")].sum())
            self.target_count = max(1, int((mass - .010)/.150))
        elif self.task == "C3":
            # Nominal thick 100 mm of the 350 mm loin, scaled by this seed's length.
            self.target_count = max(1, int(round(4*w.workpiece.extra["model"].fL)))
        else:
            from scipy.spatial import cKDTree
            if self.task == "A1":
                source = np.flatnonzero(ref == REF["head"])
                candidates = np.flatnonzero((ref == REF["fillet_upper"]) | (ref == REF["fillet_lower"]))
            elif self.task == "B3":
                source = np.flatnonzero(reg == self._region_id("melanin_spot"))
                candidates = np.flatnonzero(reg != self._region_id("melanin_spot"))
            else:
                source = np.flatnonzero(reg == self._region_id("chain"))
                candidates = np.flatnonzero(reg == self._region_id("main"))
            if len(source) and len(candidates):
                d = cKDTree(g.centres[source]).query(g.centres[candidates])[0]
                target = candidates[d > 2*w.config.mesh_h]
            else:
                target = np.zeros(0, int)
            if self.task == "A1":
                names = [b["name"] for b in w.workpiece.rigid_bodies]
                source = np.append(source, nt + w.assembly["n_knife_bodies"] + names.index("head"))
                # Include the posterior vertebra so an intact backbone cannot vanish from the graph.
                segs = sorted(n for n in names if n.startswith("segment_"))
                if segs:
                    target = np.append(target, nt + w.assembly["n_knife_bodies"] + names.index(segs[-1]))
            self.source, self.target = source, target
            self.r0 = g.remaining(source, target, start=True)

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

    def monitor(self):
        """Called after every engine tick, not after a client action/evaluation."""
        env, w = self.env, self.env.world
        if self.task in ("A0", NEW_TASK):
            p, R = w.body_pose("knife_handle")
            self.knife_history.append((env.sim_time, np.array(p), np.array(R)))
        if self.task == "A0":
            pinch = w.pinch_forces()
            for arm in w.rig.instances:
                hist = self.hand_history.setdefault(arm, dict(since=None, poses=deque()))
                if min(pinch.get(arm, (0., 0.))) <= .5:
                    hist["since"] = None
                    hist["poses"].clear()
                    continue
                if hist["since"] is None:
                    hist["since"] = env.sim_time
                hand = w.kin.forward(arm)
                relp, relR = (p - hand[:3, 3]) @ hand[:3, :3], hand[:3, :3].T @ R
                hist["poses"].append((env.sim_time, relp, relR))
                while len(hist["poses"]) > 1 and hist["poses"][1][0] < env.sim_time - 3.:
                    hist["poses"].popleft()
        if self.task == "B5" and env.steps % 50 == 0:
            self._monitor_scale()

    def _monitor_scale(self):
        w = self.env.world
        reading = w.scale_reading_g()
        if reading is None or reading < 5.:
            self.weigh_windows.clear()
            return
        _, frags, _ = self._parts()
        x = w.tissue.x.numpy()
        supported = []
        for f in frags.values():
            if f["mass"] < .005:
                continue
            p = x[self.fragment_nodes(f)]
            if (self.in_footprint(p, LY.SCALE).all() and
                    abs(float(p[:, 2].min()) - LY.TABLE["top_z"] - LY.SCALE["size"][2]) <= .003 and
                    not self._robot_touch(p).any() and not self._blade_touch(p).any()):
                supported.append(f)
        # One independently supported piece must account for the scale's entire load.
        if len(supported) != 1:
            self.weigh_windows.clear()
            return
        f = supported[0]
        key = tuple(sorted(int(t) for t in f["tets"]))
        win = self.weigh_windows.get(key, []) + [float(reading)]
        win = win[-3:]
        self.weigh_windows = {key: win}
        if len(win) == 3 and max(win) - min(win) < 1. and abs(np.mean(win) - f["mass"]*1000) <= 2.:
            self.weighed.add(key)

    def evaluate(self):
        if self.env.world.fault:
            return dict(schema="gar.evaluation.v1", task=self.task, contract=CONTRACT,
                        success=False, score=0., checks={}, metrics={}, fault=self.env.world.fault)
        if self.task == "A0":
            checks, metrics, progress = self._take_knife()
        elif self.task == NEW_TASK:
            checks, metrics, progress = self._put_away()
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

    def _take_knife(self):
        from .scene import INSERT
        from .workpiece import blade_segment_poses
        w, env = self.env.world, self.env
        heel, R, centres = self.knife_pose()
        edge = np.array([p for _, p in blade_segment_poses(w)])
        _, handle_R = w.body_pose("knife_handle")
        tilt = angle(handle_R[:, 2], np.array([0., 0., 1.]))
        over = self.in_footprint(edge, LY.BOARD) & (edge[:, 2] > LY.BOARD_TOP_Z + .01)
        # Distance-to-board credit is continuous, rather than counting segments/checks passed.
        dx = np.maximum(np.abs(edge[:, 0] - LY.BOARD["center"][0]) - LY.BOARD["size"][0]/2, 0)
        dy = np.maximum(np.abs(edge[:, 1] - LY.BOARD["center"][1]) - LY.BOARD["size"][1]/2, 0)
        dz = np.maximum(LY.BOARD_TOP_Z + .01 - edge[:, 2], 0)
        board = float(np.mean(np.clip(1 - np.sqrt(dx*dx + dy*dy + dz*dz)/.15, 0., 1.)))
        target, target_R = w.scene.block_knife_pose
        slot = knife_return_geometry(heel, R[:, 0], centres, target, target_R, w.scene.blade.L, w.scene.blade.H)
        out = not slot["inside"]
        rows = []
        for arm, h in self.hand_history.items():
            if h["since"] is None or not h["poses"]:
                continue
            poses = h["poses"]
            held = env.sim_time - h["since"]
            slip = max(np.linalg.norm(q[1] - poses[-1][1]) for q in poses)
            angular_slip = max(angle(q[2][:, 0], poses[-1][2][:, 0]) for q in poses)
            if arm == "right":
                offset, axis = w.knife_in_grip()
                lateral = float(np.hypot(offset[1], offset[2]))
                seated = axis <= 10. and lateral <= INSERT["lip_height"]
                seat = ramp(axis, 10., 30.)*ramp(lateral, INSERT["lip_height"], 2*INSERT["lip_height"])
            else:
                # No knife insert on the other hand: a stable physical pinch is sufficient.
                seated = angular_slip <= 10.
                seat = ramp(angular_slip, 10., 30.)
            stable = seat*ramp(slip, .002, .010)
            progress = (clip(held/3.) + stable + ramp(tilt, 30., 90.)*board)/3. if out else 0.
            rows.append((held >= 3. and seated and slip <= .002 and out, progress, arm, held, slip))
        best = max(rows, key=lambda r: (r[0], r[1]), default=(False, 0., None, 0., None))
        checks = dict([check("stable_hand_hold", best[3], 3., "s; <=2 mm slip, <=10 deg axis", best[0]),
                       check("edge_down_deg", tilt, 30., "deg", tilt <= 30.),
                       check("blade_over_board", float(over.all()), 1., "bool; >10 mm above board", over.all()),
                       check("blade_never_touched_robot", env.blade_robot_contacts, 0., "ticks", env.blade_robot_contacts == 0)])
        return checks, dict(holding_hand=best[2], slip_m=best[4], out_of_block=out), best[1]*(.5 if env.blade_robot_contacts else 1.)

    def _return_raw(self, heel, R, centres):
        from .workpiece import blade_segment_poses
        w = self.env.world
        target, target_R = w.scene.block_knife_pose
        last_R, last_p = blade_segment_poses(w)[-1]
        tip = last_p + last_R[:, 0]*w.scene.blade.seg_len/2
        geom = knife_return_geometry(heel, tip - heel, centres, target, target_R, w.scene.blade.L, w.scene.blade.H)
        distance = float(np.linalg.norm(tip - (target + target_R[:, 0]*w.scene.blade.L)))
        approach = clip(1 - distance/max(self.start_distance, 1e-12))
        raw = .3*approach + .7*geom["insertion"]*ramp(geom["axis_error"], 5., 30.)
        return raw, geom

    def _put_away(self):
        w, env = self.env.world, self.env
        heel, R, centres = self.knife_pose()
        raw, geom = self._return_raw(heel, R, centres)
        p, _ = w.body_pose("knife_handle")
        history = [h for h in self.knife_history if h[0] >= env.sim_time - 1. - 1e-9]
        motion = max((float(np.linalg.norm(h[1] - p)) for h in history), default=None)
        window = bool(history and history[0][0] <= env.sim_time - 1. + 1e-9)
        pinch = w.pinch_forces()
        robot_force, _ = w._knife_contacts()
        released = max((max(f) for f in pinch.values()), default=0.) <= .5 and robot_force <= .2
        # Also inspect free-knife and blade geometry against the robot's tissue colliders.
        released &= not near_boxes(w, env._robot_boxes, np.vstack((centres, p)), margin=.002).any()
        intact = not (set(w.cut.cut_faces) - self.initial_faces)
        above_bench = bool(np.min(np.vstack((centres, p))[:, 2]) >= LY.TABLE["top_z"] - .005)
        checks = dict([check("heel_position_mm", geom["heel_error"]*1000, 10., "mm", geom["heel_error"] <= .010),
                       check("slot_axis_deg", geom["axis_error"], 5., "deg", geom["axis_error"] <= 5.),
                       check("blade_in_slot", float(geom["inside"]), 1., "bool; finite slot", geom["inside"]),
                       check("released", float(released), 1., "bool", released),
                       check("rest_mm", None if motion is None else 1000*motion, 1., "mm over trailing 1 s",
                             window and motion <= .001),
                       check("workpiece_uncut", len(set(w.cut.cut_faces) - self.initial_faces), 0., "new faces", intact),
                       check("blade_never_touched_robot", env.blade_robot_contacts, 0., "ticks", env.blade_robot_contacts == 0)])
        progress = gained(raw, self.return_start) if intact and above_bench else 0.
        if env.blade_robot_contacts:
            progress *= .5
        return checks, dict(**geom, robot_contact_n=robot_force), progress

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
        task, w = self.task, self.env.world
        checks, metrics = getattr(super(), f"_eval_{task}")()
        lab, frags, main = self._parts()
        edges = self.graph.edges()
        # All cutting tasks use either hand and allow safe tool set-down, as fillet-whole-fish does.
        if "knife_not_dropped" in checks:
            x = w.tissue.x.numpy()
            okay, where = self._knife_put_down([c for c, f in frags.items() if f["mass"] >= .001], frags, x)
            checks["knife_not_dropped"] = check("knife_not_dropped", float(okay), 1., "bool", okay, where)[1]
        checks["no_blade_robot_contact"] = check("no_blade_robot_contact", self.env.blade_robot_contacts, 0.,
                                                  "ticks", self.env.blade_robot_contacts == 0)[1]
        if task == "A1":
            separated = checks["head_separated"]["value"]
            sever, remaining = self._sever(separated, edges)
            raw = .8*sever + .2*separated*ramp(metrics["meat_with_head_kg"]*1000, 8., 30.)*checks["fish_on_board"]["value"]
            metrics.update(remaining_connections=remaining, starting_connections=self.r0)
        elif task == "B3":
            total = metrics["spot_kg"]
            sever, remaining = self._sever(clip(metrics["spot_removed_kg"]/max(total, 1e-12)), edges)
            raw = sever*ramp(metrics["good_removed_kg"]/max(total, 1e-12), 2., 6.)*ramp(checks["skin_hole_mm2"]["value"], 1., 400.)
            if total <= 0:
                raw = 0.
            checks["spot_exists"] = check("spot_exists", total, 0., "kg; strictly positive", total > 0)[1]
            metrics.update(remaining_connections=remaining, starting_connections=self.r0)
        elif task == "C1":
            sever, remaining = self._sever(clip(metrics["chain_separated_kg"]/max(metrics["chain_kg"], 1e-12)), edges)
            raw = sever*ramp(checks["main_loss"]["value"], .03, .10)*ramp(checks["tail_split_mm"]["value"], 25., 60.)
            if metrics["chain_kg"] <= 0:
                raw = 0.
            checks["chain_exists"] = check("chain_exists", metrics["chain_kg"], 0., "kg; strictly positive", metrics["chain_kg"] > 0)[1]
            metrics.update(remaining_connections=remaining, starting_connections=self.r0)
        else:
            partial = self.graph.partial_slice(lab, frags, edges, w.config.mesh_h)
            tilt = ramp(checks["cut_tilt_deg"]["value"], 10., 25.)
            if task == "B5":
                pieces = sorted(((c, f) for c, f in frags.items() if f["mass"] >= .005),
                                key=lambda cf: float(self.graph.centres[cf[1]["tets"], 0].mean()))
                full = pieces[1:-1]
                weighed = [tuple(sorted(int(t) for t in f["tets"])) in self.weighed for _, f in full]
                share = float(np.mean(weighed)) if weighed else 0.
                checks["weighed"] = check("weighed", share, 1., "fraction; each piece physically on scale",
                                           bool(weighed) and all(weighed))[1]
                quality = sum(ramp(abs(f["mass"]*1000 - 150.), 7., 40.) for _, f in full)
                raw = clip((quality + .5*partial)/self.target_count)*tilt*(.8 + .2*share)
                metrics["weighed_piece_count"] = sum(weighed)
            else:
                meds = [r for r in metrics["pieces"] if r["both_ends_cut"] and r["thickness_mm"] <= 50.]
                quality = sum(ramp(abs(r["thickness_mm"] - 25.), 5., 12.) for r in meds)
                spread = checks["thickness_spread_mm"]["value"]
                raw = clip((quality + .5*partial)/self.target_count)*tilt*(ramp(spread, 5., 12.) if meds else 1.)
            raw = max(0., raw - .05*checks["no_tag_ends"]["value"])
            metrics.update(target_count=self.target_count, partial_cut=partial)
        W = self._workmanship(metrics, frags)
        metrics["workmanship_multiplier"] = W
        return checks, metrics, clip(W*raw)
