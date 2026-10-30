"""Mechanical room B-012 as one world: the rigid world of classroom_arrangement (Newton 1.5 + MuJoCo on the CPU, 0.5 ms
ticks, the Mobile FR3 Duo with its servos, reflexes and base controller), the wet materials as 2.5-D regions
(paste.py) and the coupling between them (coupling.py).

Order within a tick:
  1. every contactor's support plate is placed under its sole from the paste and the force the plate carried in the
     previous tick (coupling.Coupler.place_plates); MuJoCo's geom_pos / geom_quat of the plate are written (the plate
     geoms have geom_sameframe cleared);
  2. the rigid world steps with the drag and tack of the paste of the previous tick and any assisted wrench applied;
  3. each plate's normal force is read from the solver's contacts, and the paste under every sole is cut, filled,
     worked and its forces computed (Coupler.after_step);
  4. every SLOW_DT the material clock advances: strength, bleeding, evaporation, blisters, slow flow (Region.slow_step).

Plates touch nothing but their own sole: each has a contype/conaffinity bit of its own that only its sole geom also
carries in its contype (a bit added to what Newton compiled, never one taken away). The plate's
priority is above every other geom's, so its contact is frictionless (condim 1) with its own soft solref (M7).
"""

import math
import time
import numpy as np
from scipy.spatial.transform import Rotation
import classroom_arrangement.world as CW
from classroom_arrangement.world import World
from classroom_arrangement import params as CP
from . import params as P
from .scene import FinishScene
from .coupling import Contactor, Coupler

SLOW_EVERY = int(round(P.SLOW_DT/CP.DT))
PLATE_SOLREF = (0.01, 2.0)       # M7 case b/d: soft plate contact cut lost-contact ticks by two thirds
PLATE_PRIORITY = 3
ROOM_MARGIN = 0.30
BODY_SPEED_FAULT = 12.0


class FinishWorld(World):
    # The consolidated classroom World builds `self.scene_cls` (machine-2 hook); the module-global swap below is the
    # machine-3 hook this environment was built against. Both select the same FinishScene.
    scene_cls = FinishScene

    def __init__(self, config, spec):
        original = CW.Scene
        CW.Scene = FinishScene
        try:
            super().__init__(config, spec)
        finally:
            CW.Scene = original
        self.ext_wrench = np.zeros_like(self.body_wrench)        # ASSISTED probes add phantom forces here
        self.regions = spec["regions"]
        self.timing["paste_s"] = 0.0
        self.paste_recorders = []
        self._body_com = self.model.body_com.numpy().astype(float)
        self._build_contactors()
        sc = self.scene
        self.tool_bodies = np.array([t["body"] for t in sc.tools.values()], int)
        self.tool_names = list(sc.tools)
        # forms and kickers of the pad: the largest horizontal contact force on any of them is recorded (a braced form
        # moves at about FORM_DISTURB_N; the evaluator reads it)
        labels = list(self.model.shape_label) if hasattr(self.model, "shape_label") else []
        self._form_geoms = np.array(sorted(self._shape_geom[i] for i, lab in enumerate(labels)
                                           if lab.startswith(("form_", "kicker_")) and i in self._shape_geom), int)
        self.form_peak_n = 0.0
        self.form_force_n = 0.0
        # the rag (tool-sealant): a loaded blade rubbed on it leaves its load there
        self._rag_geoms = np.array(sorted(self._shape_geom[i] for i, lab in enumerate(labels)
                                          if lab == "rag" and i in self._shape_geom), int)

    # ------------------------------------------------------------------ construction
    def _build_contactors(self):
        m, sc = self.mjm, self.scene
        # Newton compiles each collision group as contype = its bit, conaffinity = every other bit: a bit that is no
        # geom's contype is free; a plate with only that bit in its conaffinity meets only geoms carrying it in contype
        used = 0
        for g in range(m.ngeom):
            used |= int(m.geom_contype[g])
        free = [k for k in range(30, 0, -1) if not used & (1 << k)]
        specs = sc.contactor_specs
        if len(free) < len(specs):
            raise RuntimeError(f"{len(specs)} contactors need their own contact bits; only {len(free)} are free")
        self.contactors = []
        self.plate_geom = {}
        self._plate_of_geom = {}
        for sp, bit in zip(specs, free):
            pg = self._shape_geom[sp["plate_shape"]]
            sole_geoms = [self._shape_geom[s] for s in sp["sole_shapes"]]
            m.geom_contype[pg] = 0
            m.geom_conaffinity[pg] = 1 << bit
            for g in sole_geoms:
                # the added bit also opens the sole to the other geoms of its own group: only allowed where those are
                # of its own body or its parent (MuJoCo filters both)
                b0 = int(m.geom_bodyid[g])
                for h in np.flatnonzero(np.asarray(m.geom_contype) == m.geom_contype[g]):
                    bh = int(m.geom_bodyid[h])
                    if bh != b0 and bh != int(m.body_parentid[b0]) and int(m.body_parentid[bh]) != b0:
                        raise RuntimeError(f"contactor {sp['name']}: its sole's collision group also holds body {bh}")
                m.geom_contype[g] = int(m.geom_contype[g]) | (1 << bit)
            m.geom_priority[pg] = PLATE_PRIORITY
            m.geom_condim[pg] = 1
            m.geom_solref[pg] = PLATE_SOLREF
            m.geom_solimp[pg][:3] = CP.CONTACT_SOLIMP
            m.geom_margin[pg] = 0.0
            m.geom_gap[pg] = 0.0
            if hasattr(m, "geom_sameframe"):
                m.geom_sameframe[pg] = 0
            c = Contactor(sp["name"], sp["body"], sp["p_ls"], sp["R_ls"], sp["sole"], sp["mass"], tool=sp["tool"],
                          geom_sole=sole_geoms, plate_geom=pg, bear=sp.get("bear"))
            c.park = np.array(sp["park"], float)
            c.bit = bit
            self.contactors.append(c)
            self.plate_geom[c.name] = pg
            self._plate_of_geom[pg] = c.name
        # each tool's other sides (its back, ends and long sides): a second contactor on the tool's plate, serving while
        # the working sole is turned away from the paste. Its plate bit is given to the tool's other geoms only while it
        # serves, so the plate under a working sole still meets that sole alone (a float dropped back-first
        # passed into the fresh pad without displacing it)
        main = {c.name: c for c in self.contactors}
        self.side_contactors = []
        for sp in getattr(sc, "side_specs", ()):
            mc = main[sp["main"]]
            geoms = [self._shape_geom[s] for s in sp["shapes"] if s in self._shape_geom]
            for g in geoms:
                b0 = int(m.geom_bodyid[g])
                for h in np.flatnonzero(np.asarray(m.geom_contype) == m.geom_contype[g]):
                    bh = int(m.geom_bodyid[h])
                    if bh != b0 and bh != int(m.body_parentid[b0]) and int(m.body_parentid[bh]) != b0:
                        raise RuntimeError(f"contactor {sp['name']}: its tool's collision group also holds body {bh}")
            c = Contactor(sp["name"], sp["body"], np.zeros(3), np.eye(3), sp["faces"][0]["sole"], sp["mass"],
                          geom_sole=geoms, plate_geom=mc.plate_geom)
            c.side_of, c.faces = mc, sp["faces"]
            c.use_face(0)
            c.park, c.bit = mc.park, mc.bit
            c.extra_geoms = [g for g in geoms if g not in mc.geom_sole]
            c.bits_on = False
            self.contactors.append(c)
            self.side_contactors.append(c)
            self.plate_geom[c.name] = mc.plate_geom
        self._plate_geoms = np.array(sorted(self._plate_of_geom), int)
        self._plate_owner = {}                                   # plate geom -> the contactor that placed it this tick
        # every sole geom -> its contactor: the sole's whole press on the room (paste plates and static geoms: a
        # jointer's rod rides the block arrises, a float strikes off on the slab) is what a hand or a wrist sensor feels
        self._sole_of_geom = {g: c.name for c in self.contactors if c.side_of is None for g in c.geom_sole}
        self._sole_geoms = np.array(sorted(set(self._sole_of_geom) | {g for c in self.side_contactors for g in c.geom_sole}), int)
        self.sole_force = {c.name: 0.0 for c in self.contactors}
        self.coupler = Coupler(self.regions, self.contactors, CP.DT)
        self.coupler.body_com = self.model.body_com.numpy().astype(float)
        self.plate_force = {c.name: 0.0 for c in self.contactors}
        self._park_all()

    def _park_all(self):
        for c in self.contactors:
            self.mjm.geom_pos[c.plate_geom] = c.park
            self.mjm.geom_quat[c.plate_geom] = (1.0, 0.0, 0.0, 0.0)

    # ------------------------------------------------------------------ stepping
    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        t0 = time.perf_counter()
        try:
            q = self.state.body_q.numpy().astype(float)
            placed = self.coupler.place_plates(self, q)
            self._plate_owner = {}
            for c in self.contactors:
                pl = placed.get(c.name)
                if pl is None:
                    if c.plate_geom not in self._plate_owner:        # a shared plate stays where its other user put it
                        self.mjm.geom_pos[c.plate_geom] = c.park
                        self.mjm.geom_quat[c.plate_geom] = (1.0, 0.0, 0.0, 0.0)
                else:
                    centre, R = pl
                    x, y, z, w = Rotation.from_matrix(R).as_quat()
                    self.mjm.geom_pos[c.plate_geom] = centre
                    self.mjm.geom_quat[c.plate_geom] = (w, x, y, z)
                    self._plate_owner[c.plate_geom] = c.name
            self._side_bits()
            self.body_wrench[:] = self.ext_wrench
            for c in self.contactors:
                if c.point is not None and np.any(c.wrench[:3]):
                    b = c.body
                    Rb = Rotation.from_quat(q[b, 3:7]).as_matrix()
                    com = q[b, :3] + Rb @ self._body_com[b]
                    self.body_wrench[b, :3] += c.wrench[:3]
                    self.body_wrench[b, 3:] += np.cross(c.point - com, c.wrench[:3])
            self.timing["paste_s"] += time.perf_counter() - t0
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise
        super().step()
        try:
            t1 = time.perf_counter()
            forces = self._plate_forces()
            self.plate_force = forces
            self._form_forces()
            q = self.state.body_q.numpy().astype(float)
            qd = self.state.body_qd.numpy().astype(float)
            self.coupler.after_step(self, q, qd, forces, CP.DT)
            if self.tick % SLOW_EVERY == 0:
                for reg in self.regions.values():
                    reg.slow_step(P.SLOW_DT)
                    if not np.isfinite(reg.tb).all() or not np.isfinite(reg.ts).all():
                        raise FloatingPointError(f"Non-finite paste in region {reg.name} at tick {self.tick}")
            if len(self._rag_geoms):
                self._wipe(qd)
            self._room_guard(q, qd)
            for rec in self.paste_recorders:
                rec(self)
            self.timing["paste_s"] += time.perf_counter() - t1
        except Exception as exc:
            self.fault = f"{type(exc).__name__}: {exc}"
            raise

    def _side_bits(self):
        """A tool's other geoms carry its plate's bit while its side contactor holds the plate, and only then."""
        m = self.mjm
        for c in self.side_contactors:
            on = self._plate_owner.get(c.plate_geom) == c.name
            if on != c.bits_on:
                for g in c.extra_geoms:
                    m.geom_contype[g] = (int(m.geom_contype[g]) | (1 << c.bit)) if on else (int(m.geom_contype[g]) & ~(1 << c.bit))
                c.bits_on = on

    def _plate_forces(self):
        """Plate force of each contactor (the paste's support), and into self.sole_force the normal force of all its
        sole's contacts with static geoms (plates included). A plate's force is its present user's (a tool's working
        sole or, while that is turned away, its side contactor, whose 'sole' is the whole tool)."""
        d, mj = self.mjd, self._mujoco
        out = {c.name: 0.0 for c in self.contactors}
        sole = {c.name: 0.0 for c in self.contactors}
        self.sole_force = sole
        if not d.ncon:
            return out
        g = np.asarray(d.contact.geom[:d.ncon])
        f6 = np.zeros(6)
        hit = np.flatnonzero(np.isin(g, self._sole_geoms).any(axis=1))
        body = self.mjm.geom_bodyid
        sides = [c for c in self.side_contactors if c.region is not None]
        for i in hit:
            a, b = int(g[i, 0]), int(g[i, 1])
            if body[b] == 0:
                tg, other = a, b
            elif body[a] == 0:
                tg, other = b, a
            else:
                continue
            plate = other in self._plate_of_geom
            names = [self._plate_owner.get(other, self._plate_of_geom[other])] if plate else \
                ([self._sole_of_geom[tg]] if tg in self._sole_of_geom else []) + [c.name for c in sides if tg in c.geom_sole]
            if not names:
                continue
            mj.mj_contactForce(self.mjm, d, int(i), f6)
            f = abs(float(f6[0]))
            for name in names:
                sole[name] += f
                if plate:
                    out[name] += f
        return out

    def _wipe(self, qd):
        """A carrying tool whose geoms touch the rag while it moves over it leaves its load there: the load falls by
        exp(-d / RAG_WIPE) over a sliding distance d (a loaded blade drawn across a shop towel)."""
        d = self.mjd
        if not d.ncon:
            return
        g = np.asarray(d.contact.geom[:d.ncon])
        hit = np.isin(g, self._rag_geoms).any(axis=1)
        if not hit.any():
            return
        other = np.where(np.isin(g[hit, 0], self._rag_geoms), g[hit, 1], g[hit, 0])
        bodies = set(int(self.mjm.geom_bodyid[x]) for x in other)
        for c in self.contactors:
            if c.roll <= 0 or c.tool is None or not c.tool.carry:
                continue
            if int(self.mjm.geom_bodyid[c.geom_sole[0]]) not in bodies:
                continue
            v = float(np.linalg.norm(qd[c.body, :3]))
            off = c.roll*(1.0 - math.exp(-v*CP.DT/P.RAG_WIPE))
            self.coupler._ensure_packet(c, c.region)
            if c.packet is not None and off > 0:
                packet = c.packet.clone(off)
                if self.coupler.rag_packet is None:
                    self.coupler.rag_packet = packet
                elif self.coupler.rag_packet.material.name == packet.material.name:
                    self.coupler.rag_packet.mix(packet)
            c.roll -= off
            self.coupler.log["wiped_m3"] = self.coupler.log.get("wiped_m3", 0.0) + off

    def _form_forces(self):
        d, mj = self.mjd, self._mujoco
        self.form_force_n = 0.0
        if not d.ncon or not len(self._form_geoms):
            return
        g = np.asarray(d.contact.geom[:d.ncon])
        hit = np.flatnonzero(np.isin(g, self._form_geoms).any(axis=1))
        f6 = np.zeros(6)
        worst = 0.0
        for i in hit:
            mj.mj_contactForce(self.mjm, d, int(i), f6)
            frame = np.asarray(d.contact[int(i)].frame).reshape(3, 3)
            f = frame.T @ f6[:3]
            worst = max(worst, float(np.hypot(f[0], f[1])))
        self.form_force_n = worst
        self.form_peak_n = max(self.form_peak_n, worst)

    def _room_guard(self, q, qd):
        """No tool is faster than BODY_SPEED_FAULT or out of the room: a fault of the world, said at once."""
        tb = self.tool_bodies
        if not len(tb):
            return
        sp = np.linalg.norm(qd[tb, :3], axis=1)
        x = q[tb, :3]
        rx, ry, rz = P.ROOM
        out = ((x[:, 0] < -ROOM_MARGIN) | (x[:, 0] > rx + ROOM_MARGIN) | (x[:, 1] < -ROOM_MARGIN) | (x[:, 1] > ry + ROOM_MARGIN)
               | (x[:, 2] < -ROOM_MARGIN) | (x[:, 2] > rz + ROOM_MARGIN))
        bad = np.flatnonzero((sp > BODY_SPEED_FAULT) | out | ~np.isfinite(sp))
        if len(bad):
            k = int(bad[0])
            raise FloatingPointError(f"Tool {self.tool_names[k]} " + ("left the room" if out[k] else f"moves at {sp[k]:.1f} m/s")
                                     + f" at tick {self.tick}")

    # ------------------------------------------------------------------ the material clock alone (wait)
    def advance_material(self, seconds, dt=None):
        """Advance every region's clock by `seconds` in steps of dt (the wait command; the rigid world does not step)."""
        dt = P.WAIT_DT if dt is None else dt
        n = int(math.ceil(seconds/dt - 1e-9))
        done = 0.0
        for k in range(n):
            h = min(dt, seconds - done)
            for reg in self.regions.values():
                reg.slow_step(h)
            done += h
        for c in self.contactors:
            if c.packet is not None and c.roll > 0:
                c.packet.values["age"] = c.packet.values.get("age", 0.)+done
        for reg in self.regions.values():
            if not np.isfinite(reg.tb).all() or not np.isfinite(reg.ts).all():
                raise FloatingPointError(f"Non-finite paste in region {reg.name} after a wait")
        return done

    # ------------------------------------------------------------------ queries
    def tool_pose(self, name):
        return self.body_pose(self.scene.tools[name]["body"])

    def snapshot(self):
        import copy
        return dict(regions={n: r.snapshot() for n, r in self.regions.items()},
                    falling=copy.deepcopy(self.coupler.falling),
                    rag_packet=copy.deepcopy(self.coupler.rag_packet),
                    contactors={c.name: c.snapshot() for c in self.contactors},
                    log=dict(self.coupler.log), plate_force=dict(self.plate_force), form_peak_n=self.form_peak_n,
                    plates={c.name: (self.mjm.geom_pos[c.plate_geom].copy(), self.mjm.geom_quat[c.plate_geom].copy())
                            for c in self.contactors})

    def restore_snapshot(self, s):
        import copy
        from .transfer import all_regions
        for n, r in self.regions.items():
            r.restore(s["regions"][n])
        idle = Contactor("idle", 0, np.zeros(3), np.eye(3), None, 1.0).snapshot()
        for c in self.contactors:
            c.restore(s["contactors"].get(c.name, idle), {r.name: r for r in all_regions(self.regions)})
            if c.name in s["plates"]:
                pos, quat = s["plates"][c.name]
                self.mjm.geom_pos[c.plate_geom] = pos
                self.mjm.geom_quat[c.plate_geom] = quat
        self._plate_owner = {c.plate_geom: c.name for c in self.contactors if c.region is not None}
        self._side_bits()
        self.coupler.log = dict(s["log"])
        self.coupler.falling = copy.deepcopy(s.get("falling", []))
        self.coupler.rag_packet = copy.deepcopy(s.get("rag_packet"))
        self.plate_force = dict(s["plate_force"])
        self.form_peak_n = s.get("form_peak_n", 0.0)
