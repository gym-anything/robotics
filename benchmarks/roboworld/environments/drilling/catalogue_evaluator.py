"""State-only task scoring. No simulator construction or action access.

The old evaluator supplies tick-rate measurements and common constraints. The new contract is opt-in.
"""

import math
from types import MethodType

import numpy as np

from . import params as P
from .catalogue import CONTRACT, LEGACY, instruction, validate_contract
from .task import Evaluator, FAIL_SCORE_FACTOR, _angle_deg, _check, _in_plane, _ramp


def clip(x):
    x = float(x)
    if not math.isfinite(x):
        raise ValueError("non-finite task measurement")
    return max(0., min(1., x))


def mean(xs):
    xs = list(xs)
    return sum(xs)/len(xs) if xs else 0.


def depth_share(h, diameter, thickness):
    return 1. if h.through_at(diameter) else clip(h.full_depth(diameter)/thickness)


def area_depth_share(h, diameter, thickness):
    """Union of the stages' cylindrical volumes, normalized to the requested bore.

    A through 3 mm pilot is 1/4 of a 6 mm bore; an uncut wider stage adds nothing.
    Integrating annuli avoids double counting a pilot, and the credit rises continuously from 1/4 to 1 as the
    wider stage cuts deeper, with no jump.
    """
    widths = sorted({0., *(min(diameter, s["d_mm"]) for s in h.stages)})
    return clip(sum((b*b - a*a)/(diameter*diameter)*depth_share(h, b, thickness)
                    for a, b in zip(widths, widths[1:])))


class CatalogueEvaluator(Evaluator):
    task_contract = LEGACY

    @classmethod
    def wrap(cls, old, contract):
        if not isinstance(old, cls):
            new = cls.__new__(cls)
            new.__dict__.update(old.__dict__)  # keep start poses, tick claims and checkpoint schema
            old = new
        old.task_contract = validate_contract(old.task, contract)
        old.instruction = instruction(old.task, old.task_contract)
        return old

    def _intact(self, h):
        broken = {n for n, it in self.w.hm.items.items() if it.kind == "bit" and it.state["broken"]}
        return h.blocked_at is None and not (broken & set(h.tools))

    def _bore(self, h, error, diameter, thickness, tol, max_d=None, axis_tol=1.):
        if not self._intact(h):
            return 0.
        value = depth_share(h, diameter, thickness)*_ramp(error, tol, 3*tol)
        value *= _ramp(_angle_deg(h.a, -h.face_normal), axis_tol, 3.)
        if max_d is not None:
            value *= _ramp(h.d_mm + 2e3*h.max_offset_m, max_d, max_d + 0.5)
        return value

    def _blind(self, h, error):
        fd = h.full_depth(6.8)
        d = clip(fd/0.015)*_ramp(fd, 0.0155, 0.018)
        return (0. if h.through or not self._intact(h) else d*_ramp(error, 0.0005, 0.0015)
                *_ramp(_angle_deg(h.a, -h.face_normal), 1., 3.))

    def _panel(self, h, error):
        rnd = (h.d_mm + 2e3*h.wander_m)/max(h.d_mm, 1e-9)
        return (area_depth_share(h, 6., 0.0015)*_ramp(error, 0.001, 0.003)
                *_ramp(_angle_deg(h.a, -h.face_normal), 1., 3.)*_ramp(rnd, 1.10, 1.30)
                if self._intact(h) else 0.)

    def nearest(self, name, p_local, kinds=("drilled",)):
        if self.task_contract == LEGACY or kinds != ("drilled",):
            return super().nearest(name, p_local, kinds)
        # A shallow abandoned start must not hide an acceptable finished opening.
        candidates = []
        for h in self.holes(name):
            e = _in_plane(h, p_local)
            if self.task == "press-blind":
                quality = self._blind(h, e)
                valid = not h.through and 0.015 <= h.full_depth(6.8) <= 0.0155 and e <= 0.0005
            elif self.task == "panel-holes":
                quality = self._panel(h, e)
                valid = h.through_at(6.) and e <= 0.001 and (h.d_mm + 2e3*h.wander_m)/max(h.d_mm, 1e-9) <= 1.10
            else:
                quality = self._bore(h, e, 9., 0.005 if name == "angle_bracket" else 0.006, 0.0005, 9.25)
                valid = h.through_at(9.) and e <= 0.0005 and h.d_mm + 2e3*h.max_offset_m <= 9.25
            valid = valid and _angle_deg(h.a, -h.face_normal) <= 1.
            candidates.append(((bool(valid), quality, h.full_depth(), -e, -h.id), h, e))
        return max(candidates, key=lambda c: c[0])[1:] if candidates else None

    def _finish(self):
        p = self.w.scene.tools["drill_press"]
        rpm = abs(p.spindle.speed())*60/(2*math.pi)
        stopped = (0. if p.s_on else _ramp(rpm, 1., 1000.))
        up = _ramp(-p.j["quill"].pos(self.w), 0.002, 0.080)
        # Continuous bit withdrawal from every current slot, read from its actual tip pose.
        tip, _ = self.w.hm.item_world(p.item)
        clear = 1.
        for name in p.item.slots:
            d = self.w.hm.drillables[name]
            pos, rot = self.w.body_pose(d.body)
            local = rot.T @ (tip - pos)
            above = local[2] - (d.center[2] + d.half[2])
            clear = min(clear, clip(1. + above/0.080))
        return mean((stopped, up, clear))

    def _take_progress(self):
        s, w = self.s, self.w
        held = 0. if s["held_since"] is None else clip((w.sim_time - s["held_since"])/3.)
        placed = 0. if s["placed_since"] is None else clip((w.sim_time - s["placed_since"])/3.)
        it = w.scene.tools["drill"].item
        if it.state["broken"]:
            return 0.
        tip, axis = w.hm.item_world(it)
        angle = math.degrees(math.acos(float(np.clip(-axis[2]/np.linalg.norm(axis), -1., 1.))))
        height = tip[2] - P.BENCH_TOP_Z
        height_credit = min(clip(height/0.030), _ramp(height, 0.070, 0.150))
        # Fade geometry credit with the hold, so lifting/contact onset does not jump the score.
        return mean((held, placed, held*_ramp(angle, 5., 25.), held*height_credit))

    def _punch_progress(self):
        dimples = [h for h in self.w.hm.drillables["flat_bar"].holes if h.kind == "dimple"]
        credits, used = [], set()
        for mk in self.L["marks"]:
            candidates = [(float(np.linalg.norm((h.p0 - np.asarray(mk["local"]))[:2])), h) for h in dimples]
            e, h = min(candidates, key=lambda c: (c[0], c[1].id)) if candidates else (math.inf, None)
            credits.append(_ramp(e, 0.0003, 0.0015))
            if h is not None:
                used.add(h.id)
        stray = sum(h.id not in used and any(np.linalg.norm((h.p0 - np.asarray(m["local"]))[:2]) < 0.010
                                            for m in self.L["marks"]) for h in dimples)
        return clip(mean(credits) - 0.25*stray/len(self.L["marks"]))

    def _panel_damage(self):
        chosen = [self.nearest("cover_panel", m["local"]) for m in self.L["marks"]]
        # Re-entry inside a bit radius is the same opening, as in the ledger-anchor rule.
        return [h for h in self.holes("cover_panel") if not any(
            hit is not None and hit[1] <= 0.003 and _in_plane(h, hit[0].p0) <= 0.003 for hit in chosen)]

    def _panel_holes(self):
        out = super()._panel_holes()
        if self.task_contract == CONTRACT:
            bad = [h.id for h in self._panel_damage() if h.through]
            out.append(_check("no stray through-hole", not bad, bad, kind="constraint"))
        return out

    def _slot(self, item, name):
        slot = item.slots.get(name)
        if not slot or slot[1] != "hole":
            return None
        return next((h for h in self.holes(name) if h.id == slot[2]), None)

    def _overlap(self, item, name, h):
        """Current shank length inside this hole, not the stale insertion history."""
        if h is None or self._slot(item, name) is not h:
            return 0.
        d = self.w.hm.drillables[name]
        tip, axis = self.w.hm.item_world(item)
        pos, rot = self.w.body_pose(d.body)
        tip_local, axis_local = rot.T @ (tip - pos), rot.T @ axis
        z = float((tip_local - h.p0) @ h.a)
        base = z - item.length*float(axis_local @ h.a)
        lo, hi = sorted((base, z))
        lo, hi = max(lo, 0.), min(hi, h.thickness)
        if hi <= lo:
            return 0.
        dot = float(axis_local @ h.a)
        if abs(dot) < 1e-9:
            return 0.
        # Slot ids are necessary but not enough: reject a stale slot beside the part or an
        # impossible shank across an undersize bore. 0.6 mm is holes.CAPTURE_TOL (chamfer capture).
        radius = 0.5e-3*h.d_mm + h.max_offset_m - item.radius + 0.0006
        radial = []
        for at in (lo, hi):
            offset = tip_local - h.p0 + axis_local*((at-z)/dot) - h.a*at
            radial.append(float(np.linalg.norm(offset)))
        if radius < 0. or max(radial) > radius:
            return 0.
        return clip((hi-lo)/h.thickness)

    def _timber_lower_mark(self, top):
        d1, d0 = (self.w.hm.drillables[n] for n in ("board1", "board0"))
        p1, r1 = self.w.body_pose(d1.body)
        p0, r0 = self.w.body_pose(d0.body)
        mark = np.asarray(self.L["marks"][0]["local"]) if top is None else top.p0
        axis = np.array([0., 0., -1.]) if top is None else top.a
        ray, direction = r0.T @ (p1 + r1 @ mark - p0), r0.T @ r1 @ axis
        # Follow the bore line to the lower face, rather than imposing an additional vertical
        # mark: a valid top entry 2 mm off with a 1-degree bore shifts its exit by another 1.3 mm.
        if abs(direction[2]) < 1e-9:
            return ray
        return ray + direction*((d0.center[2] + d0.half[2] - ray[2])/direction[2])

    def _timber_state(self):
        hm = self.w.hm
        mark = np.asarray(self.L["marks"][0]["local"])
        picked, bore = {}, []
        for name in ("board1", "board0"):
            target = mark if name == "board1" else self._timber_lower_mark(picked["board1"])
            candidates = []
            for h in self.holes(name):
                e = _in_plane(h, target)
                q = self._bore(h, e, 14.3, 0.038, 0.002, axis_tol=P.TIMBER_AXIS_TOL_DEG)
                q *= _ramp(h.scorch_max_c, 200., 300.)
                valid = (h.through_at(14.3) and e <= 0.002 and
                         _angle_deg(h.a, -h.face_normal) <= P.TIMBER_AXIS_TOL_DEG and h.scorch_max_c < 200.)
                candidates.append(((valid, q, -e, -h.id), h, q))
            _, h, q = max(candidates, key=lambda c: c[0]) if candidates else (None, None, 0.)
            picked[name], bore = h, [*bore, q]
        it = hm.items["carriage_bolt"]
        shares = [self._overlap(it, n, h) for n, h in picked.items()]
        return picked, mean(bore), mean(shares), shares

    def _timber_bolt(self):
        if self.task_contract == LEGACY:
            return super()._timber_bolt()
        picked, _, _, shares = self._timber_state()
        out = []
        mark = np.asarray(self.L["marks"][0]["local"])
        marks = {"board1": mark, "board0": self._timber_lower_mark(picked["board1"])}
        for name, h in picked.items():
            out.append(_check(f"{name}: through at 14.3 mm", h is not None and h.through_at(14.3)))
            if h is not None:
                out += [_check(f"{name}: entry within 2 mm of the marked bore", _in_plane(h, marks[name]) <= 0.002),
                        _check(f"{name}: square to the faces", _angle_deg(h.a, -h.face_normal) <= P.TIMBER_AXIS_TOL_DEG),
                        _check(f"{name}: no charred wall (< 200 C)", h.scorch_max_c < 200.)]
        out.append(_check("carriage bolt spans both marked through holes", all(x >= 1. - 1e-6 for x in shares)))
        force = self.w.hm.items["carriage_bolt"].state["max_axial_n"]
        out.append(_check("bolt pushed in with less than 50 N", force < 50., force, "< 50 N", kind="constraint"))
        return out

    def _match_state(self):
        upper = [self.nearest("angle_bracket", m["local"]) for m in self.L["marks"]]
        lower = sorted(self.holes("flat_bar"), key=lambda h: (
            self._bore(h, 0., 9., 0.006, 0.0005, 9.25), h.through_at(9.), -h.id), reverse=True)[:2]
        bores = [0. if h is None else self._bore(h[0], h[1], 9., 0.005, 0.0005, 9.25) for h in upper]
        bores += [self._bore(h, 0., 9., 0.006, 0.0005, 9.25) for h in lower] + [0.]*(2-len(lower))
        rows = []
        for k in range(2):
            it = self.w.hm.items[f"m8_bolt{k}"]
            ha, hb = self._slot(it, "angle_bracket"), self._slot(it, "flat_bar")
            mark = next((i for i, hit in enumerate(upper) if hit and hit[0] is ha), None)
            eligible = mark is not None and hb is not None and any(hb is h for h in lower)
            share = mean((self._overlap(it, "angle_bracket", ha), self._overlap(it, "flat_bar", hb))) if eligible else 0.
            rows.append((mark, None if hb is None else hb.id, share))
        # Two bolts in one pair of holes cannot stand for the two fastenings.
        distinct = rows[0][0] != rows[1][0] and rows[0][1] != rows[1][1]
        bolt = mean(r[2] for r in rows) if distinct else max(r[2] for r in rows)/2.
        return upper, lower, mean(bores), bolt, distinct and all(r[2] >= 1. - 1e-6 for r in rows)

    def _match_bolt(self):
        out = super()._match_bolt()
        if self.task_contract == LEGACY:
            return out
        _, lower, _, _, spans = self._match_state()
        out.append(_check("two 9 mm bores in the flat bar, square and within 9.25 mm diameter", len(lower) == 2 and all(
            h.through_at(9.) and _angle_deg(h.a, -h.face_normal) <= 1. and
            h.d_mm + 2e3*h.max_offset_m <= 9.25 for h in lower)))
        out.append(_check("two bolts span distinct pairs of matching through holes", spans))
        return out

    def _progress(self):
        if self.task == "take-drill":
            return self._take_progress()
        if self.task == "punch-layout":
            return self._punch_progress()
        if self.task in ("press-drill", "press-blind"):
            blind = self.task == "press-blind"
            hit = self.nearest("alu_block" if blind else "flat_bar", self.L["marks"][0]["local"])
            bore = 0. if hit is None else (self._blind(*hit) if blind else self._bore(*hit, 9., 0.006, 0.0005, 9.25))
            return bore*(0.85 + 0.15*self._finish())
        if self.task == "panel-holes":
            credits = [self._panel(*hit) if (hit := self.nearest("cover_panel", m["local"])) else 0.
                       for m in self.L["marks"]]
            return clip(mean(credits) - sum(area_depth_share(h, 6., 0.0015) for h in self._panel_damage())/4.)
        if self.task == "timber-bolt":
            _, bore, bolt, _ = self._timber_state()
            return (2*bore + bolt)/3.
        if self.task == "match-bolt":
            upper, _, bore, bolt, _ = self._match_state()
            spacing = 0. if any(h is None for h in upper) else _ramp(
                abs(float(np.linalg.norm(upper[0][0].p0 - upper[1][0].p0)) - 0.060), 0.0005, 0.0015)
            return (2*bore + bolt*spacing)/3.
        raise ValueError(f"No continuous score for {self.task}")

    def evaluate(self):
        if self.task_contract == LEGACY:
            return super().evaluate()  # includes the ledger-anchor output, byte-for-byte semantics
        if self.w.fault or not np.isfinite(self.w.bq).all() or not np.isfinite(self.w.jq).all():
            return dict(task=self.task, contract=CONTRACT, success=False, score=None, evaluated=False,
                        reason=self.w.fault or "non-finite state", instruction=self.instruction)
        out = super().evaluate()
        progress = self._progress()
        # Explicit defect costs and damaged-bore terms already account for these constraints.
        penalize = [c for c in out["checks"] if c["kind"] == "constraint" and not c["passed"]
                    and c["name"] != "no bit broken" and "stray" not in c["name"]]
        progress = clip(progress*(0.5 if penalize else 1.))
        out.update(contract=CONTRACT, progress=progress, evaluated=True,
                   scoring="1 on success, else 0.9 x continuous work-product progress",
                   score=1. if out["success"] else FAIL_SCORE_FACTOR*progress)
        if self.task == "press-drill":
            out["evaluator_note"] = "The outcome is judged: a bar held without turning may pass without vise clamping."
        return out


def _get_state(env):
    state = env._catalogue_get_state()
    if env.evaluator.task_contract == CONTRACT:
        ev = state["evaluation"]
        if state.get("fault") or not state.get("finite", True) or "error" in ev:
            ev = dict(ev, success=False, score=None, evaluated=False,
                      reason=state.get("fault") or ev.get("error") or "non-finite state")
        state["evaluation"] = dict(ev, contract=CONTRACT)
    return state


def configure_environment(env, contract):
    """Install only evaluator/state-report hooks on this instance; no global monkeypatch or scene changes.

    Binding get_state here covers act/reset/checkpoint/trace callers inside DrillingEnv as well as the backend.
    The original initial checkpoint's measurement schema is retained.
    """
    env.evaluator = CatalogueEvaluator.wrap(env.evaluator, contract)
    if not hasattr(env, "_catalogue_get_state"):
        env._catalogue_get_state = env.get_state
        env.get_state = MethodType(_get_state, env)
    return env
