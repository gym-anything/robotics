"""Rigid world + water: classroom_arrangement's qualified Mobile FR3 Duo world (Newton 1.5 + MuJoCo CPU, 0.5 ms ticks,
servo/reflex/base stack unchanged) with painting's contact refinements (two 0.25 ms rigid substeps, coupled fingers,
force-mode targets past closed, condim 4/6 with torsion and rolling friction on items and pads, 1 ms tool/pad contacts,
converged no-slip) and the dish room scene, plus the water system's per-tick coupling:

    before each rigid step: valve and balancer springs, jet reactions, impact impulses, buoyancy, drag and film shear
                            into world.body_wrench; finger-pad friction scheduled from the soap state of what they touch
    after each rigid step:  water network (valve openings -> flows -> jets), packets, films, species and soil, pools

Assisted fixtures (probes only) set world.fixture; they must add to world.body_wrench (classroom scripted.Phantom does).
"""

import mujoco
import numpy as np

from classroom_arrangement import params as CP
from painting.world import PaintingWorld
from . import params as P
from .scene import RinsingScene


class RinsingWorld(PaintingWorld):
    scene_cls = RinsingScene
    NO_ROLL = ("glass_rack", "peg_rack", "cutting_board", "can", "colander", "steel_bowl", "plate", "bowl")

    def __init__(self, config, spec):
        super().__init__(config, spec)
        self.water = None                      # surface_rinsing.water.system.WaterSystem, attached by the environment
        m = self.mjm
        # fixture joint stops are hard stops like the arm's (the spray lever's preloaded spring rests on its stop; a
        # handle's quarter turn ends at its stop): MuJoCo's default soft limit let a 0.7 N m preload sit beyond it
        for j in range(m.njnt):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "fx_" in name and m.jnt_limited[j]:
                m.jnt_solref[j] = (2*CP.DT, 1.)
                d0, dmax, mid, power = CP.STOP_SOLIMP
                m.jnt_solimp[j] = (d0, dmax, 1e-3, mid, power)
        # fixture contacts (handles, spout, valves, hose) at the tool/pad time constant: a 20 g lever pressed between
        # two fingers is a light body between stiff pads (as in painting)
        fx = {self._mj_body[b] for b in self.scene.fixture_bodies if self._mj_body.get(b, -1) >= 0}
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in fx:
                m.geom_solref[g] = (self.CONTACT_TC, 1.0)
        # the ware too (glass, china, steel, the can, the racks, the tomatoes and the sponge): MuJoCo's soft contact
        # penetrates by about impact speed x time constant whatever the mass, and at the 4 ms default a sponge pressed
        # onto the pan at 1 m/s went past the middle of its 4 mm floor hull and was pushed out under it (A5 scrub, caught
        # tick by tick). At 1 ms (2 ticks, MuJoCo's stability floor) the same impact stays inside the outer millimetre.
        ware = {self._mj_body[b] for it in self.scene.items.values() for b in (it["body"], it.get("top"))
                if b is not None and self._mj_body.get(b, -1) >= 0}
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in ware:
                m.geom_solref[g] = (self.CONTACT_TC, 1.0)
        self.item_bodies = {n: it["body"] for n, it in self.scene.items.items()}
        # produce rolls with a soft fruit's resistance: c_rr ~0.035 (a ripe tomato flattens 1-2 mm under its weight;
        # estimate) times its 14 mm radius = 0.5 mm, torsion 1 mm. Painting's 15 um (lacquered handles) let the
        # tomatoes roll down the colander's 1 % drainboard pitch without end (rest test: 22-86 mm in the settle)
        tom = {self._mj_body[it["body"]] for it in self.scene.items.values() if it["kind"] == "tomato"}
        for g in range(m.ngeom):
            if int(m.geom_bodyid[g]) in tom:
                m.geom_condim[g] = 6
                m.geom_friction[g][1] = 1.0e-3
                m.geom_friction[g][2] = 5.0e-4
        # MuJoCo's no-slip pass removes the creep of its soft friction under a sustained load: a pinched glass slides
        # down in the hand at 0.05 mm/s without it (P18), a glass on the drainboard's 0.3 deg pitch at 0.4 um/s and a
        # tomato on the colander's 0.6 deg floor rolls at 0.1 mm/s (P11 rest; both below their friction and rolling
        # thresholds, so a real one stays put). It costs iterations x rows^2: painting's 1000 iterations to 1e-15 took
        # 194 ms per tick among this room's ~350 constraint rows (390x wall/sim). It runs always at
        # classroom_arrangement's qualified setting, params.HOLD (100 iterations, 1e-10): the glass creep is then zero,
        # the tomatoes' 0.2 um/s.
        self._pad_geoms = set(int(g) for g in self.pad_geoms)
        self._robot_mj = {self._mj_body[b] for b in self.scene.robot_bodies if self._mj_body.get(b, -1) >= 0}
        self.noslip_hold = (int(P.HOLD["noslip_iterations"]), float(P.HOLD["noslip_tolerance"]))
        self.mjm.opt.noslip_tolerance = self.noslip_hold[1]
        self.mjm.opt.noslip_iterations = self.noslip_hold[0]
        self.holding = False
        self.slip_max, self.noslip_off_ticks = 0.0, 0

    def step(self):
        if self.fault:
            raise RuntimeError(self.fault)
        self.body_wrench[:] = 0.0
        if self.water is not None:
            try:
                self.water.pre_step()
            except Exception as exc:
                self.fault = f"{type(exc).__name__}: {exc}"
                raise
        self._couple_fingers()
        self._force_mode_targets()
        self._noslip_schedule()
        super(PaintingWorld, self).step()
        if self.water is not None:
            try:
                self.water.post_step()
            except Exception as exc:
                self.fault = f"{type(exc).__name__}: {exc}"
                raise

    def set_lumped_mass(self, body, extra_mass, extra_com_local, extra_inertia=None):
        """painting's lumped liquid (mass at its centroid, its own inertia neglected: right for a pool that does not turn
        with its container), plus, when given, the liquid's own inertia (diagonal, body frame): the sponge's pore water
        is held by capillarity and turns with the sponge. Without it the wet sponge (8 g dry, ~70 g of water) turned
        like the dry one, a tenth of its real inertia, and rocked at a few hundred hertz under the friction of a scrub
        until an edge went through the pan's floor (A5 scrub, caught tick by tick)."""
        super().set_lumped_mass(body, extra_mass, extra_com_local)
        if extra_inertia is not None:
            k = self._mj_body[body]
            self.mjm.body_inertia[k] = np.asarray(self.mjm.body_inertia[k], float) + np.asarray(extra_inertia, float)

    def _noslip_schedule(self):
        """The no-slip pass removes soft friction's creep at (nearly) zero slip: items resting or held. During real
        sliding it rewrites large friction forces after the main solve while leaving the normal forces as solved, and
        a light body pressed and slid (the 76 g wet sponge scrubbed at 25-60 N) rocked at a few hundred hertz, sank
        through the pan's floor hull and left through the sink (A5 scrub; in a press-and-drag test 3 of 12 trials
        tripped the speed guard with the pass, none without it). So every 10 ticks the fastest tangential slip at any
        contact is measured, and the pass runs only while none slides faster than params.HOLD slip_off (5 mm/s; it
        comes back below slip_on, 2 mm/s). Creep is micrometres per second, scrubbing centimetres to decimetres."""
        d, m = self.mjd, self.mjm
        if self.tick % 10 == 0:
            import mujoco
            vmax = 0.0
            v1, v2 = np.zeros(6), np.zeros(6)
            for i in range(d.ncon):
                con = d.contact[i]
                b1, b2 = int(m.geom_bodyid[con.geom1]), int(m.geom_bodyid[con.geom2])
                p = np.array(con.pos)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b1, v1, 0)
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, b2, v2, 0)
                u1 = v1[3:] + np.cross(v1[:3], p - d.xpos[b1])
                u2 = v2[3:] + np.cross(v2[:3], p - d.xpos[b2])
                vr = u1 - u2
                n = np.array(con.frame[:3])
                vt = vr - (vr @ n)*n
                vmax = max(vmax, float(np.linalg.norm(vt)))
            self.slip_max = vmax
            on = m.opt.noslip_iterations > 0
            if on and vmax > P.HOLD["slip_off"]:
                m.opt.noslip_iterations = 0
            elif not on and vmax < P.HOLD["slip_on"]:
                m.opt.noslip_iterations = self.noslip_hold[0]
        if m.opt.noslip_iterations == 0:
            self.noslip_off_ticks += 1
        hold = False
        for i in range(d.ncon):
            g1, g2 = int(d.contact[i].geom[0]), int(d.contact[i].geom[1])
            if g1 in self._pad_geoms and int(m.geom_bodyid[g2]) not in self._robot_mj or \
                    g2 in self._pad_geoms and int(m.geom_bodyid[g1]) not in self._robot_mj:
                hold = True
                break
        self.holding = hold                       # (reported; the no-slip pass is gated on slip, above)

    def joint_value(self, label):
        """Position of a fixture joint (single-dof) by its Newton label."""
        if not hasattr(self, "_jlabel"):
            labels = list(self.model.joint_label)
            qs = self.model.joint_q_start.numpy()
            ds = self.model.joint_qd_start.numpy()
            self._jlabel = {lab: (int(qs[k]), int(ds[k])) for k, lab in enumerate(labels)}
        q0, d0 = self._jlabel[label]
        return float(self.state.joint_q.numpy()[q0]), float(self.state.joint_qd.numpy()[d0])
