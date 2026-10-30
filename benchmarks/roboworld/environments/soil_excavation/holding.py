"""Tool holding on the classroom Mobile FR3 Duo stack: a frozen copy of painting.world.PaintingWorld's additions to
classroom_arrangement.world.World, without its paint hooks.

The additions are copied, not imported, so the soil environment's physics is decided by files of this package and the
classroom stack only:

  tool geoms condim 6 with pad-patch torsion and rolling resistance, finger pads condim 4; tool and finger-pad contacts
  at a 1 ms time constant; MuJoCo's no-slip pass converged; the Franka Hand's spindle as a stiff common-mode coupling
  of the two fingers; force-mode grip targets set past closed so the fingers apply the set force on any handle width.

Each comment below is painting's.
"""

import numpy as np

from classroom_arrangement import params as CP
from classroom_arrangement.world import World


class HoldingWorld(World):

    # torsional friction of a finger pad on a tool handle: mu (2/3) r_patch with mu 0.8 and a ~4 mm patch. With the
    # classroom's condim 3 a pinch on a round handle is two point contacts on one line and the tool pivots freely
    # about it (a round brush creeps 6.5 mm in 30 s at rest); condim 4 adds the patch's spin resistance.
    TORSION_M = 0.002
    # rolling resistance of tools on the table, cloth and floor (MuJoCo rolling friction, m = torque per normal force):
    # c_rr ~ 0.003 for a lacquered wooden handle on laminate times a ~4 mm radius (estimate). Without it a dropped round
    # brush rolls across the table at a constant 2.1 mm/s for 40 s. Pad contacts keep the
    # pads' own condim 4 (pads have contact priority over tools).
    ROLL_M = 1.5e-5
    CONTACT_TC = 0.001                         # s: tool and finger-pad contact time constant (2 x the 0.5 ms tick)
    NO_ROLL = ("table", "board", "palette", "spray_box", "crate", "stencil", "tray")

    def __init__(self, config, spec):
        super().__init__(config, spec)
        tools = {it["body"] for it in self.scene.items.values()}
        # furniture and props that stand on their feet or lie flat (the craft table on its four capsule legs, the board,
        # palette, booth, crate side, stencil, tray) get no rolling dimensions: the 15 um rolling coefficient lets the
        # table's legs roll and the table creeps at a constant 0.0025 mm/min with its board and palette (2.9e-8 m/s,
        # 2.4e-11 m/s without them)
        flat = {it["body"] for it in self.scene.items.values() if it.get("kind") in self.NO_ROLL}
        fingers = {i for i in self.scene.robot_bodies if "finger" in self.model.body_label[i]}
        tool_mj = {self._mj_body[b] for b in tools if self._mj_body[b] >= 0}
        flat_mj = {self._mj_body[b] for b in flat if self._mj_body[b] >= 0}
        finger_mj = {self._mj_body[b] for b in fingers if self._mj_body[b] >= 0}
        for g in range(self.mjm.ngeom):
            body = int(self.mjm.geom_bodyid[g])
            if body in flat_mj:
                self.mjm.geom_condim[g] = 3
            elif body in tool_mj:
                self.mjm.geom_condim[g] = 6
                self.mjm.geom_friction[g][1] = self.TORSION_M
                self.mjm.geom_friction[g][2] = self.ROLL_M
            elif body in finger_mj:
                self.mjm.geom_condim[g] = 4
                self.mjm.geom_friction[g][1] = self.TORSION_M
        # Tool and finger contacts at the stiffest stable time constant (2 ticks). MuJoCo's soft contacts scale their
        # compliance with the bodies' effective mass: with the classroom's 4-5 ms a 10 g round brush pinched at 14-35 N sits
        # 1-2 mm deep in both pads on two soft points each, and a legal fast wrist swing walks it out of the pinch (the
        # pads shearing it in opposite directions at their friction limit while its inertial load is 0.3 N); the can,
        # the sash brush likewise. At 1 ms the same swing moves the brush 2 mm in the hand.
        for g in range(self.mjm.ngeom):
            body = int(self.mjm.geom_bodyid[g])
            if body in tool_mj or body in finger_mj:
                self.mjm.geom_solref[g] = (self.CONTACT_TC, 1.0)
        # MuJoCo's no-slip pass converged: with the classroom's 100 iterations to 1e-10 a pinched tool still slides in
        # the hand at a constant rate (roller 0.11 deg/min for minutes; 0.26 deg/s with no pass at all, 5x slower with
        # stiffer pads). 1000 iterations to 1e-15 hold it (4e-6 deg/s) for ~6 % more time.
        self.mjm.opt.noslip_iterations = 1000
        self.mjm.opt.noslip_tolerance = 1e-15
        # the aerosol button's slide is a hard stop at both ends, like the arm, finger and Spine stops of the classroom
        # stack: with MuJoCo's default soft limit the 10 N spring preload holds it 0.65 mm above its top stop and 45 N
        # pushes it 1.06 mm through the bottom one
        import mujoco
        from classroom_arrangement import params as CP
        for j in range(self.mjm.njnt):
            name = mujoco.mj_id2name(self.mjm, mujoco.mjtObj.mjOBJ_JOINT, j) or ""
            if "button_joint" in name and self.mjm.jnt_limited[j]:
                self.mjm.jnt_solref[j] = (2*CP.DT, 1.)
                d0, dmax, mid, power = CP.STOP_SOLIMP
                self.mjm.jnt_solimp[j] = (d0, dmax, 2e-5, mid, power)

    # The Franka Hand's two fingers are driven by one motor through a spindle: they always move symmetrically about the
    # hand. The classroom stack drives them as two independent joints; in force mode each pushes with a constant force,
    # so the pair (and the tool between them) has no stiffness against sliding together along the finger axis, and a
    # held tool's weight component makes it drift at a constant 1-1.5 um/s forever (painting's P15/P11 rest creep: the finger
    # joints of the holding hand moving +-1.3 um/s in step, the contacts at 4-17 % of their friction cone). The spindle
    # is modelled as a stiff common-mode coupling between the finger pair and the hand (internal: forces sum to zero).
    FINGER_COUPLING = dict(k=5.0e4, c=40.0)       # N/m, N s/m (estimate: a geared drive; 5 N gives 0.1 mm)

    def _finger_pairs(self):
        inv = {v: k for k, v in self._mj_body.items()}
        out = []
        for arm in self.arms:
            j1, j2 = self.groups[f"{arm}_gripper"]
            bl, br = self.scene.robot_body[f"{arm}_fr3_leftfinger"], self.scene.robot_body[f"{arm}_fr3_rightfinger"]
            hand = inv.get(int(self.mjm.body_parentid[self._mj_body[bl]]))
            out.append((self.joints[j1], self.joints[j2], bl, br, hand))
        return out

    def _couple_fingers(self):
        pairs = getattr(self, "_fpairs", None)
        if pairs is None:
            pairs = self._fpairs = self._finger_pairs()
        q, qd = self.state.joint_q.numpy(), self.state.joint_qd.numpy()
        bq = self.state.body_q.numpy()
        k, c = self.FINGER_COUPLING["k"], self.FINGER_COUPLING["c"]
        for J1, J2, bl, br, hand in pairs:
            u = bq[bl, :3] - bq[br, :3]
            n = float(np.linalg.norm(u))
            if n < 1e-6 or hand is None:
                continue
            u = u/n
            delta = 0.5*(float(q[J1["q"]]) - float(q[J2["q"]]))
            ddot = 0.5*(float(qd[J1["dof"]]) - float(qd[J2["dof"]]))
            F = -(k*delta + c*ddot)
            self.body_wrench[bl, :3] += F*u
            self.body_wrench[br, :3] += F*u
            self.body_wrench[hand, :3] -= 2*F*u

    def _force_mode_targets(self):
        """In force mode the classroom controller pushes kp (ref - q) with the reference walking toward the target and
        clipped to q -+ force/kp; with the target at closed (0) the push on a thin handle stops at kp x its half width
        (12.3 N on the 9 mm round brush when 35 N is set). A closing force-mode target is set
        past closed so the fingers apply the set force on any width."""
        q = None
        for arm, f in self.grip_force.items():
            if not f:
                continue
            q = self.state.joint_q.numpy() if q is None else q
            for n in self.gripper_joints(arm):
                j = self.joints[n]["q"]
                if self.targets[j] <= 0.0005:
                    # only while something holds the finger open: closed on nothing, the target stays at closed so
                    # the hand does not report a latched grasp of air
                    self.targets[j] = -0.03 if float(q[j]) > 0.001 else 0.0
