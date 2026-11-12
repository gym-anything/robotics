"""gift-bags-tasks-v2 verifier: task_set_evaluator.py with the render-check fixes.

Changes: the lid_skirt_engagement_mm metric (put-lid-on's engagement credit needs the base rim inside the lid's
skirt) and a `task` field in every evaluation, whose `contract` is the bare gift-bags-tasks-v2."""
import math

import numpy as np

from . import task as T
from .config import task_verifier
from . import task_set_v2_scoring as S
from .task_set_v2_catalog import CONTRACT, evaluation_contract, selected_contract


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
        """(checks, metrics) of the selected task: the hands-off window, then the task's measure(self, checks, metrics,
        rows) from its verifier.py (tasks/<task>/)."""
        c, m = {}, {}
        rows = list(self.history.rows)
        c["hands_off_window"] = len(rows) >= self.history.n and not any(r["contact"] for r in rows)
        m["hands_off_window_s"] = len(rows)*self.env.config.dt
        task_verifier(self.named_task).measure(self, c, m, rows)
        return {k: bool(v) for k, v in c.items()}, m

    def _carton_geometry(self, c, m):
        """open-carton and open-flaps: the flap angles, the carton's signed tilt (carton_upright) and how far each item
        of the contents is outside the carton's inner volume."""
        from . import carton as CT
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
        inside = bool(np.all(np.abs(in_lid[:, 0]) <= hx + g["clearance"] + 1e-6)
                      and np.all(np.abs(in_lid[:, 1]) <= hy + g["clearance"] + 1e-6))
        # Skirt engagement: how far the base's rim reaches up inside the lid's skirt, counted only while all four rim
        # corners are inside the skirt in plan. lid_overlap_mm (task.py) is the vertical overlap alone, which also
        # reads ~55 mm for a lid lying on the table beside the box.
        m["lid_skirt_engagement_mm"] = 1e3*max(0., float(in_lid[:, 2].min())) if inside else 0.
        m["lid_fits"] = bool(inside
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
        """Legacy and bag-purchase-v1 selections: the released evaluation (task.Evaluator.evaluate) with their contract;
        a gift-bags-tasks-v2 selection: its task's verifier.py (tasks/<task>/), which grades through evaluate_task."""
        label = evaluation_contract(self.selection.task, self.selection.contract)
        if not self.modern:
            return dict(super().evaluate(), contract=label)
        return task_verifier(self.named_task).verify(self)

    def evaluate_task(self, released=None, *, score_of=None, progress_of=None):
        """A gift-bags-tasks-v2 selection: measure() and the task's progress_of(self, S, metrics) on this contract's
        score arithmetic (S); success needs every check. Other selections: the released evaluation (task.Evaluator)."""
        if not self.modern:
            return super().evaluate_task(released, score_of=score_of)
        label = evaluation_contract(self.selection.task, self.selection.contract)
        c, m = self.measure()
        name = self.named_task
        progress = progress_of(self, S, m)
        success = all(c.values())
        m["progress"] = S.clip(progress)
        return dict(success=success, score=S.score(success, progress), checks=c, metrics=m, task=name, contract=label)
