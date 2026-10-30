"""Task verifier adapter. The original Evaluator is called unchanged for bag-purchase."""
import math

import numpy as np

from . import task as T
from . import task_set_scoring as S
from .task_set_catalog import CONTRACT, evaluation_contract, selected_contract


def signed_tilt(R):
    return math.degrees(math.acos(max(-1., min(1., float(R[2, 2])))))


def rectangle_coverage(points, hx, hy):
    """Area of a convex polygon clipped to the base rectangle / base area (continuous in pose)."""
    poly = [np.asarray(p, float) for p in points]
    for axis, bound, sign in ((0, hx, 1), (0, -hx, -1), (1, hy, 1), (1, -hy, -1)):
        old, poly = poly, []
        if not old:
            return 0.
        for a, b in zip(old, old[1:] + old[:1]):
            da, db = sign*(a[axis] - bound), sign*(b[axis] - bound)
            if da <= 0:
                poly.append(a)
            if (da <= 0) != (db <= 0):
                poly.append(a + da/(da - db)*(b - a))
    area = abs(sum(a[0]*b[1] - b[0]*a[1] for a, b in zip(poly, poly[1:] + poly[:1])))/2
    return S.clip(area/(4*hx*hy))


class Evaluator(T.Evaluator):
    def __init__(self, env, selection):
        self.selection = selection
        self.named_task = selection.task
        self.modern = selected_contract(selection.task, selection.contract) == CONTRACT
        super().__init__(env)
        if self.modern:
            # Capture geometry after the ordinary reset settle, before any policy action.
            _, self.start_metrics = self.measure()

    def measure(self):
        c, m = {}, {}
        rows = list(self.history.rows)
        c["hands_off_window"] = len(rows) >= self.history.n and not any(r["contact"] for r in rows)
        m["hands_off_window_s"] = len(rows)*self.env.config.dt
        name = self.named_task
        if name in ("open-bag", "bag-candle"):
            # Internal task is bag-purchase: reuse every geometry correction of its evaluator, including signed normals.
            self._eval_bag(c, m, rows)
            self._eval_tear(c, m, 0, "bag", T.TEAR_TOL_MM)
            self._bag_progress(c, m)
            if name == "open-bag":
                for k in ("candle_in_bag", "candle_upright", "candle_on_board"):
                    c.pop(k)
        elif name in ("lift-lid", "put-lid-on"):
            self._eval_lid(c, m)
            self._lid_geometry(m)
            if name == "put-lid-on":
                for k in ("lid_off", "lid_on_table", "base_not_lifted", "base_not_tipped"):
                    c.pop(k)
                c["lid_on"] = m["lid_fits"] and m["lid_overlap_mm"] >= T.LID_CLOSED_OVERLAP_MM
                c["base_upright_on_table"] = (m["base_tilt_signed_deg"] <= T.LID_BASE_TILT_DEG
                                               and -1. <= m["base_lowest_above_table_mm"] <= T.LID_BASE_LIFT_MM)
        elif name == "unwrap-gift":
            self._eval_unwrap(c, m)
        elif name in ("open-carton", "open-flaps"):
            from . import carton as CT
            self._eval_carton(c, m, rows)
            if name == "open-flaps":
                c.pop("seam_open")
                m["seam_tape_present"] = False
            angles = CT.flap_angles(self.world.ipc)
            m.update({f"{f}_angle_deg": a for f, a in angles.items()})
            pc, Rc = self.world.ipc.body_pose(self.body["carton_body"])
            m["carton_tilt_signed_deg"] = signed_tilt(Rc)
            c["carton_upright"] = m["carton_tilt_signed_deg"] <= T.CARTON_TILT_DEG
            origin = pc - Rc @ self.world.ipc.bodies[self.body["carton_body"]].render["com_offset"]
            ds = []
            for n in CT.CONTENTS:
                p, _ = self.world.ipc.body_pose(self.body[n])
                rel = Rc.T @ (p - origin)
                nearest = np.clip(rel, (-CT.L_IN/2, -CT.W_IN/2, 2*CT.T_B), (CT.L_IN/2, CT.W_IN/2, CT.Z_END))
                ds.append(1e3*float(np.linalg.norm(rel - nearest)))
            m["contents_outside_distance_mm"] = ds
        return {k: bool(v) for k, v in c.items()}, m

    def _lid_geometry(self, m):
        ipc, g = self.world.ipc, self.geo["box"]
        ib, il = self.body["gift_base"], self.body["gift_lid"]
        pb, Rb = ipc.body_pose(ib)
        pl, Rl = ipc.body_pose(il)
        ob = pb - Rb @ ipc.bodies[ib].render["com_offset"]
        ol = pl - Rl @ ipc.bodies[il].render["com_offset"]
        hx, hy = g["hx"], g["hy"]
        lx, ly = hx + g["clearance"] + g["wall"], hy + g["clearance"] + g["wall"]
        corners = np.array([[-lx, -ly, 0.], [lx, -ly, 0.], [lx, ly, 0.], [-lx, ly, 0.]])
        local = (corners @ Rl.T + ol - ob) @ Rb
        m["lid_plan_fraction"] = rectangle_coverage(local[:, :2], hx, hy)
        # All four upper base corners must fit INSIDE the lid, below its roof and above its skirt bottom.
        rim = np.array([[x, y, g["base_h"]] for x in (-hx, hx) for y in (-hy, hy)])
        in_lid = (rim @ Rb.T + ob - ol) @ Rl
        m["lid_fits"] = bool(np.all(np.abs(in_lid[:, 0]) <= hx + g["clearance"] + 1e-6)
                             and np.all(np.abs(in_lid[:, 1]) <= hy + g["clearance"] + 1e-6)
                             and np.all(in_lid[:, 2] >= T.LID_CLOSED_OVERLAP_MM*1e-3)
                             and np.all(in_lid[:, 2] <= g["lid_h"] - g["wall"] + .0005)
                             and float(Rb[:, 2] @ Rl[:, 2]) >= math.cos(math.radians(10.)))
        m["base_tilt_signed_deg"] = signed_tilt(Rb)
        m["base_lowest_above_table_mm"] = 1e3*(self._lowest_z(ib) - self.geo_table_z())
        pf, _ = ipc.body_pose(self.body["photo_frame"])
        rf = Rb.T @ (pf - ob)
        nearest = np.clip(rf, (-hx, -hy, 0.), (hx, hy, g["base_h"]))
        m["frame_outside_distance_mm"] = 1e3*float(np.linalg.norm(rf - nearest))

    def evaluate(self):
        label = evaluation_contract(self.selection.task, self.selection.contract)
        if not self.modern:
            return dict(super().evaluate(), contract=label)
        c, m = self.measure()
        name = self.named_task
        if name in ("open-bag", "bag-candle"):
            progress = S.bag(m, self.start_metrics, purchase=name == "bag-candle")
        elif name in ("lift-lid", "put-lid-on"):
            frame = 1. - .5*S.ramp(m["frame_outside_distance_mm"], 0., 50.)
            progress = (S.lid_off if name == "lift-lid" else S.lid_on)(m, self.start_metrics, frame)
        elif name == "unwrap-gift":
            progress = S.unwrap(m, self.start_metrics)
        else:
            from .carton import FLAPS
            progress = S.carton(m, self.start_metrics, FLAPS, taped=name == "open-carton")
        success = all(c.values())
        m["progress"] = S.clip(progress)
        return dict(success=success, score=S.score(success, progress), checks=c, metrics=m, contract=label)
