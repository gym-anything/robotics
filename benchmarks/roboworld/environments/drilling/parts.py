"""Parts: fasteners, cleaning tools, the ball-peen hammer and the workpieces.

Hole-capable parts (bolt and anchor shanks, the vacuum nozzle, the brush) carry their shank on a welded link that is
excluded from drillable solids and meets holes through the proxies (holes.Item). Dimensions are read from the
product documents named per part; masses follow from the shapes' densities.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from . import params as P
from . import tools as T
from .holes import Item


def _pose(spec):
    return np.asarray(spec["xyz"], float), Rotation.from_euler("xyz", spec.get("rpy", (0., 0., 0.)))


def _assembly(sc, name, root, welded, free=True, pose=None):
    """Joints for a root link (free, or fixed to the world at `pose`) and links welded to it at their build poses."""
    b, wp = sc.b, sc.wp
    I = wp.transform_identity()
    rp = pose
    joints = [b.add_joint_free(root, label=f"{name}_free") if free else
              b.add_joint_fixed(-1, root, parent_xform=wp.transform(tuple(map(float, rp[0])), T.wq(rp[1])), child_xform=I,
                                label=f"{name}_fixed")]
    for child, child_pose in welded:
        joints.append(b.add_joint_fixed(root, child, parent_xform=T.rel_xform(sc, pose, child_pose), child_xform=I,
                                        label=f"{name}_weld_{child}"))
    b.add_articulation(joints, label=f"{name}_assembly")


def bolt(sc, spec, carriage=False):
    """M8 x 30 hex bolt (ISO 4017: head 13 AF x 5.3 mm) or a 1/2 x 4 in carriage bolt (ASME B18.5: dome 1.1 in x 0.27 in,
    square neck 1/2 in). Frame: x along the shank towards its end; origin under the head."""
    name = spec["name"]
    d, L = spec["d"], spec["length"]
    pose = _pose(spec)
    pr = T.Parts(sc)
    H = T.link(sc, f"{name}_head", pose)
    if carriage:
        dome = pr.cyl(H, f"{name}_dome", (-0.0035, 0., 0.), 0.0140, 0.0035, "x", "zinc", density=7850., roll=True)
        neck = sc.box(f"{name}_neck", (0.0035, 0., 0.), (0.0035, 0.00635, 0.00635), "zinc", body=H, density=7850.)
        sc.loose_shapes.update([dome, neck])
        shank_start = 0.007
    else:
        # hex head 13 mm across flats, 5.3 mm high (ISO 4017 M8), a hex prism
        head = pr.hex_prism(H, f"{name}_hex_head", (-0.00265, 0., 0.), "x", 0.013, 0.00265, "zinc", density=0.5*7850.)
        sc.stiff_shapes.update(head)                     # rolling lever arm and contact priority of the metal faces ...
        sc.loose_shapes.update(head)                     # ... with the repo's standard law, which rules where it lies
        shank_start = 0.
    Sp = T.compose(pose, (shank_start, 0., 0.))
    S = T.link(sc, f"{name}_shank", Sp)
    ls = L - shank_start
    pr.cyl(S, f"{name}_shank_body", (0.5*ls, 0., 0.), 0.5*d, 0.5*ls, "x", "zinc", density=7850., mu=P.MU["steel_steel"])
    proud = 0.
    if carriage and P.BOLT_END_FEET["proud_m"] > 0.:
        # the flat end it stands on in the backer's hole: three points, not a flat face on a flat floor
        # (params.BOLT_END_FEET)
        ft = P.BOLT_END_FEET
        proud = ft["proud_m"]
        for k_ in range(3):
            a_ = 2.*math.pi*k_/3.
            pr.sphere(S, f"{name}_end_foot{k_}", (ls - ft["foot_r_m"] + proud, ft["circle_r_m"]*math.cos(a_),
                                                  ft["circle_r_m"]*math.sin(a_)), ft["foot_r_m"], "zinc", density=0.,
                      mu=P.MU["steel_steel"])
    _assembly(sc, name, H, [(S, Sp)], pose=pose)
    sc.mutual_filter([H, S])
    sc.add_item(Item(name, "bolt", S, axis=(1., 0., 0.), tip=(ls + proud, 0., 0.), radius=0.5*d, length=ls + proud))
    sc.bodies[name] = H
    sc.parts[name] = dict(kind="carriage_bolt" if carriage else "bolt", body=H, shank=S, d=d, length=L)


NUT_AF_M, NUT_H_M = P.NUT_AF_M, P.NUT_H_M              # the anchor's hex nut and washer (params.py)
WASHER_T_M, WASHER_R_M = P.WASHER_T_M, P.WASHER_R_M


def anchor(sc, spec):
    """Hilti KWIK Bolt TZ2 1/2 in carbon steel wedge anchor (ESR-4266 3.1): a stud with a 1/2-13 thread at the top and a
    tapered mandrel (the cone) at the tip, an expansion clip on the mandrel, a washer and a hex nut. Frame: x towards the
    tip; origin at the top of the thread. The washer's bearing face sits `seat` from the tip as delivered, so the
    washer meets the fixture when the anchor is hammered to h_nom.

    Links: the stud (`top`, the free root, with the threaded stub; its shank on a welded link that is the hole item),
    the nut (on a carrier: a slide along the stud and a turn about it) and the washer (turning on the nut's bearing
    face). The clip is drawn on the shank; it has no collider and no joint of its own: its state is the cone's travel
    into it, and how it holds the stud in the hole is WedgeAnchor's law (friction on the stud's free joint, and once
    the cone is drawn in, an elastic tie of the stud's tip to the concrete). Setting is physical end to end: a torque on
    the nut (a socket on its hex) turns it against the solved thread and bearing friction; the thread runs it along the
    stud by the pitch; the washer stops it on the fixture, so the stud is drawn out of the hole and the cone into the
    clip, which expands against the wall until the cone's resistance balances the stud tension (the preload)."""
    from newton._src.solvers.mujoco.equality import _add_equality_constraint
    name = spec["name"]
    L = spec.get("length", P.KBTZ2_WALL_LENGTH_M)
    seat = spec.get("seat", P.KBTZ2["h_nom"])
    letter = "L" if L > 0.15 else "G"
    pose = _pose(spec)
    pr = T.Parts(sc)
    b, wp = sc.b, sc.wp
    r = 0.5*P.KBTZ2["d"]
    I = wp.transform_identity()
    x_w = L - seat                                          # washer bearing face from the top
    top = T.link(sc, f"{name}_top", pose)
    pr.cyl(top, f"{name}_stub", (0.5*x_w, 0., 0.), r, 0.5*x_w, "x", "zinc", density=7850.)
    Sp = T.compose(pose, (x_w, 0., 0.))
    S = T.link(sc, f"{name}_shank", Sp)
    pr.cyl(S, f"{name}_shank_body", (0.5*seat, 0., 0.), r, 0.5*seat, "x", "zinc", density=7850.,
           mu=P.MU["steel_concrete"])
    # the clip on the mandrel, 2-22 mm from the tip (drawn; its mass on the shank)
    pr.cyl(S, f"{name}_clip", (seat - 0.002 - 0.5*P.CLIP_LEN_M, 0., 0.), r + 0.0004, 0.5*P.CLIP_LEN_M, "x",
           "steel_bright", collide=False)
    x_n = x_w - WASHER_T_M - 0.5*NUT_H_M
    Np = T.compose(pose, (x_n, 0., 0.))
    NC = T.link(sc, f"{name}_nut_carrier", Np, mass=1e-3, inertia=(1e-8, 1e-8, 1e-8))
    N = T.link(sc, f"{name}_nut", Np)
    # the hex below the nut's chamfered top; the top NUT_CHAMFER_H_M (its outer face, away from the washer), where the
    # 30 degree chamfer has taken the corners off, as a hex whose corners lie on the across-flats circle, so that it
    # enters a socket's hex at any angle (a socket met a full-height hex's corners with its rim; E20 pushon_turned)
    hc = P.NUT_CHAMFER_H_M
    # (steel on steel friction: a socket or a wrench on the plated nut; with the repo's default 0.5 a socket's rim
    # dragged the loose nut round with it, E20 pushon_turned)
    nut_shapes = pr.hex_prism(N, f"{name}_hex_nut", (0.5*hc, 0., 0.), "x", NUT_AF_M, 0.5*(NUT_H_M - hc), "zinc",
                              density=0.5*7850., mu=P.MU["steel_steel"])
    nut_shapes += pr.hex_prism(N, f"{name}_hex_nut_top", (-0.5*NUT_H_M + 0.5*hc, 0., 0.), "x",
                               NUT_AF_M*math.sqrt(3.)/2., 0.5*hc, "zinc", density=0.5*7850., mu=P.MU["steel_steel"])
    sc.stiff_shapes.update(nut_shapes)
    Wp = T.compose(pose, (x_w - 0.5*WASHER_T_M, 0., 0.))
    W = T.link(sc, f"{name}_washer_link", Wp)
    # the washer is drawn round; it meets a fixture through an octagon (inradius 15.9 mm, its corners on the 17.2 mm
    # rim): the union of 4 boxes 45 deg apart, each the octagon's full width across two flats by one side, so that it
    # bears on a face with face contacts: a round rim meets a box face with one contact point (MuJoCo, the cylinder-box
    # pair), and the setting preload through one rim point tipped the stud over
    pr.cyl(W, f"{name}_washer", (0., 0., 0.), WASHER_R_M, 0.5*WASHER_T_M, "x", "zinc", collide=False)
    r_in = WASHER_R_M*math.cos(math.pi/8)
    half_side = r_in*math.tan(math.pi/8)
    washer_shapes = [sc.box(f"{name}_washer_pad{k}", (0., 0., 0.), (0.5*WASHER_T_M, r_in, half_side), "collision_hidden",
                            body=W, rot=T.wq(Rotation.from_euler("x", 45.*k, degrees=True)), density=0.5*7850.)
                     for k in range(4)]                     # the 4 boxes cover the octagon twice: half density
    sc.stiff_shapes.update(washer_shapes)
    washer_shape = washer_shapes[0]
    thread_len = P.KBTZ2_THREAD_LEN_M[letter]
    j_free = b.add_joint_free(top, label=f"{name}_free")
    joints = [j_free,
              b.add_joint_fixed(top, S, parent_xform=T.rel_xform(sc, pose, Sp), child_xform=I, label=f"{name}_weld_{S}")]
    j_ns = b.add_joint_prismatic(top, NC, parent_xform=T.rel_xform(sc, pose, Np), child_xform=I, axis=(1., 0., 0.),
                                 limit_lower=-(x_n - 0.5*NUT_H_M), limit_upper=thread_len - (x_n + 0.5*NUT_H_M),
                                 label=f"{name}_nut_slide")
    j_nt = b.add_joint_revolute(NC, N, parent_xform=I, child_xform=I, axis=(1., 0., 0.), limit_lower=-1e4,
                                limit_upper=1e4, label=f"{name}_nut_turn")
    j_wt = b.add_joint_revolute(N, W, parent_xform=T.rel_xform(sc, Np, Wp), child_xform=I, axis=(1., 0., 0.),
                                label=f"{name}_washer_turn")
    joints += [j_ns, j_nt, j_wt]
    b.add_articulation(joints, label=f"{name}_assembly")
    stiff = {"mujoco:eq_solref": (2*P.DT, 1.), "mujoco:eq_solimp": (.9999, .9999, .001, .5, 2.)}
    # the thread: the nut's place along the stud follows its turn (right-handed: clockwise seen from the nut's open
    # end, + about x, runs it towards the tip), a single-joint equality whose target WedgeAnchor sets each tick
    # (built with the nut's turn as its second joint and zero coefficients, and made single-joint in bind: Newton turns
    # a missing second joint into the model's last joint, and when that was a free joint - a part built after the
    # anchor - MuJoCo refused the model: 'only HINGE and SLIDE joint allowed in constraint')
    eq_thread = _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.JOINT, joint1=j_ns,
                                         joint2=j_nt, polycoef=[0., 0., 0., 0., 0.], label=f"{name}_thread",
                                         custom_attributes=stiff)
    # the cone drawn into the gripping clip: an elastic tie of the stud's tip to the concrete, off until the cone meets
    # the clip (WedgeAnchor sets its stiffness to ESR-4266's axial stiffness)
    eq_cone = _add_equality_constraint(b, constraint_type=sc.newton.solvers.SolverMuJoCo.EqType.WELD, body1=top,
                                       body2=-1, enabled=False, label=f"{name}_cone", custom_attributes=stiff)
    bodies = [top, S, NC, N, W]
    sc.mutual_filter(bodies)
    # a deep socket's bore clears the stud that passes into it (the socket meets only the nut)
    for t_ in sc.tools.values():
        if type(t_).__name__ == "ImpactWrench":
            sc.filter_bodies(t_.anvil, top)
            sc.filter_bodies(t_.anvil, S)
    it = sc.add_item(Item(name, "anchor", S, axis=(1., 0., 0.), tip=(seat, 0., 0.), radius=r, length=seat))
    it.clip_length = P.CLIP_LEN_M
    it.assembly_bodies = bodies
    sc.bodies[name] = top
    mech = WedgeAnchor(name, it, dict(top=top, shank=S, nut=N, washer=W, nut_carrier=NC), dict(
        free=T.JointRef(j_free, "free"), nut_slide=T.JointRef(j_ns, "nut_slide"), nut_turn=T.JointRef(j_nt, "nut_turn"),
        washer_turn=T.JointRef(j_wt, "washer_turn")), eq_thread, eq_cone, washer_shapes, nut_shapes, seat, L)
    sc.add_tool(mech)
    sc.parts[name] = dict(kind="anchor", body=top, shank=S, nut=N, washer=W, length=L, seat=seat, mechanism=mech.name)


class WedgeAnchor(T.Tool):
    """The setting mechanics of one KB-TZ2 (parts.anchor), stepped with the tools (a Tool without a housing, registered
    as '<anchor>_setting'). Every law reads last tick's solved state.

    A REDUCED MODEL, and what in it is an input. The nut, its thread and the washer are bodies and constraints the
    solver solves. The expansion (clip, cone, concrete) is not geometry: it is the three laws below. Two of the
    setting's numbers follow from inputs and are not predictions: at a nut torque T the stud tension is T/(K d),
    because the thread's and the bearing's friction were solved from the nut factor K (NUT_FACTOR_K: 26.8 kN at
    68 N m); and the stud rises CLIP_PLAY_M + F/beta, because the expansion is a tie with ESR-4266's axial stiffness
    (WEDGE_BETA_N_M: 1.46 mm at 26.8 kN). What the model decides is which torque the wrench's blows reach, what a
    wrong hole, dust or a badly seated socket do to the setting, and when the anchor fails.

    Nut and thread (ISO 16047 / VDI 2230 torque split, T = F (P/2 pi + 0.577 mu_G d2) on the thread + mu_K r_b F under
    the nut): the nut's place along the stud follows its turn by the pitch (a single-joint equality on the nut's slide,
    its target P/2 pi x the nut's turn, set every tick); that equality's solved force is the stud tension F (the vise
    screw's pattern, tools.Vise); the thread's pitch and friction torque (P/2 pi + 0.577 mu_G d2) F act as friction on
    the nut's turn and the bearing friction mu_K r_b F on the washer's turn. The pitch term is applied as a resistance:
    the thread is self-locking (0.577 mu_G d2 > P/2 pi), so a load on the stud never turns the nut back (the equality
    acts on the slide only). Turned back by a wrench the nut meets the same friction as turned on (the real nut would
    come loose at the friction less the pitch term, 0.75 of it; no task loosens an anchor).
    The installation torque is recorded from the solve: the two friction forces that resisted the nut (the thread's on
    its turn, the bearing's on the washer's turn), summed, in the ticks in which the nut was turning on
    (NUT_TURNING_RAD_S) with the cone gripping; the largest is the anchor's installation torque
    (state 'max_turn_torque_nm'). A tension raised by anything but a turning nut (a pull on the fixture, a neighbouring
    anchor) is no torque.

    Clip and cone (state: the cone's travel s into the clip; 0 with the clip against its collar). The clip's ring
    presses on the hole wall with k_clip (r_free - R) when the hole is smaller than the clip's free diameter (its drag,
    ANCHOR_CLIP_N in the nominal hole; there is no other resistance to driving the anchor, so a steady push above the
    drag seats it without a hammer: see ANCHOR_CLIP_N). The clip is taken to sit far firmer on the wall than on the
    cone (the condition for follow-up expansion; CLIP_WALL_MU and CONE_MU are assumed values), so while it bears on
    the wall it stays put and the stud moves through it: s follows the stud's rise in the hole. After the clip's play
    the cone meets the clip. From there the expansion is ONE elastic tie of the stud's tip to the concrete with
    ESR-4266's axial stiffness beta (scaled by the clip's engaged length): the stud tension it carries rises by beta
    per metre of stud rise. The wedge itself (the clip opening on the taper, the concrete giving way) is not computed:
    the cone's angle and friction enter only the friction that keeps the stud from turning with the nut
    (N_cone = tie load / tan(alpha + phi_c), spin friction mu_c N_cone r_c).
    Applied as: the ring's drag mu_w N_ring as friction on the stud's free joint at the two ends of the clip's play,
    each one way (against its collar the clip resists the stud being driven in; with the cone against it, the stud
    being drawn out; in between the stud slides through the clip), and the ring's spin friction mu_c N_ring r_c on
    the stud's turn; from the cone's meeting the clip, the tie (a MuJoCo weld whose static stiffness k/R is set to
    beta: d = 1 - 1/(tc^2 beta invweight); a connect let the stud pivot about its tip and tip over) and the cone's
    spin friction. A clip that does not touch the wall (a hole beyond its free diameter) holds nothing: the stud
    turns and pulls up with the nut and no torque builds - an anchor that spins. The concrete is fixed (the clip's tie
    and frictions act against the world).

    Capacity. The tie breaks when it carries the stud's steel strength (WEDGE_STEEL_N: ESR-4266's N_sa, 50.0 kN, the
    neck's fracture; 127 N m at the nut): the anchor has then failed for good, holds nothing and cannot be set again
    (state 'failed'). It also lets go if the cone travels CLIP_TRAVEL_M (12.7 mm, which the steel strength forestalls).
    LIMITS (not modelled): the neck's yield before fracture (42.2 kN: the tie stays elastic up to 50 kN); concrete
    breakout under a pull on the fixture (ACI 318 with ESR-4266's k_uncr = 24: about 38 kN nominal in C25 concrete at
    h_ef 83 mm; the preload alone cannot cause it, the washer bears on the same concrete); pull-through of the cone
    (ESR-4266 gives no pullout strength for the 1/2 in anchor: it does not govern) and concrete splitting, the other
    results of over-torquing R. Morgan names (STRUCTURE, Apr 2017: 'steel failure of the anchor, the anchor pulling
    through the wedges, or the entire anchor pulling out of the concrete'); the relaxation of the preload after
    setting (the same article: the loss 'can range between 50-60% of the initial anchor pre-load', beginning within
    minutes); and any effect of dust on the grip (dust only stops the anchor short of its embedment, holes.py)."""

    def __init__(self, anchor, it, bodies, joints, eq_thread, eq_cone, washer_shape, nut_shapes, seat, length):
        self.name = f"{anchor}_setting"
        self.anchor = anchor
        self.it = it
        self.b = bodies
        self.j = joints
        self.eq_thread_newton, self.eq_cone_newton = int(eq_thread), int(eq_cone)
        self.washer_shape = int(washer_shape[0]) if isinstance(washer_shape, (list, tuple)) else int(washer_shape)
        self.washer_shapes = [int(x) for x in washer_shape] if isinstance(washer_shape, (list, tuple)) else [int(washer_shape)]
        self.nut_shapes = [int(x) for x in nut_shapes]
        self.seat, self.length = seat, length
        self.tan_aphi = math.tan(math.radians(P.CONE_ANGLE_DEG) + math.atan(P.CONE_MU))
        delta0 = 0.5*P.CLIP_FREE_D_M - 0.5e-3*P.SDS_BIT_D_MM                # ring interference in the nominal 13.3 mm hole
        self.k_clip = P.ANCHOR_CLIP_N/P.CLIP_WALL_MU/delta0         # the clip's ring stiffness (N/m)
        self.c = P.KBTZ2_PITCH_M/(2*math.pi)

    def joints(self):
        return [j for k, j in self.j.items() if k != "free"]

    def bind(self, world):
        import mujoco
        super().bind(world)
        w = world
        mjm = w.solver.mj_model
        jf = mujoco.mj_name2id(mjm, mujoco.mjtObj.mjOBJ_JOINT, f"{self.anchor}_free")
        if jf < 0 or mjm.jnt_type[jf] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError(f"{self.anchor}: free joint not found in the MuJoCo model")
        self.j["free"].mj = jf
        e2n = np.asarray(w.solver.mjc_eq_to_newton_eq.numpy()).reshape(-1)
        self.eq_thread = next(i for i in range(mjm.neq) if int(e2n[i]) == self.eq_thread_newton)
        self.eq_cone = next(i for i in range(mjm.neq) if int(e2n[i]) == self.eq_cone_newton)
        # Newton converts a joint equality's missing second joint (joint2 = -1) into the model's last joint (here the
        # washer's turn, found by printing eq_obj2id); with zero coefficients the constraint is the same, but its
        # softness then includes that light joint's inverse weight (the nut slid 7 mm on its thread under a 30 N push).
        # The thread is a single-joint equality: no second joint
        mjm.eq_obj2id[self.eq_thread] = -1
        self.fdof = {k: int(mjm.jnt_dofadr[j.mj]) for k, j in self.j.items()}
        self.free_dofs = list(range(self.fdof["free"], self.fdof["free"] + 6))   # world x y z, local rx ry rz
        mj_of = {n: j for j, n in w._mj_to_newton.items()}
        self.mj_top = mj_of[self.b["top"]]
        self.mj_washer = mj_of[self.b["washer"]]
        # ENGINE (constraint regularization only). MuJoCo makes every constraint row soft, R = (1 - d)/d x diagApprox,
        # and a joint friction row that the main solve keeps soft is stopped by the world's no-slip pass afterwards
        # while the equality and contact forces in series with it keep the main solve's values (a stud crept at
        # 0.3 mm/s). The nut's and washer's turn friction, which carry
        # the setting torque in series with the thread, therefore get the stiffest impedance (d = 0.9999, the vise
        # thread's) and a 2-tick time constant, so the main solve already holds them. (Hardening the impedance of
        # the stud's free-joint rows the same way made MuJoCo diverge within 10-40 ticks; they keep the default
        # impedance and rely on no-slip; the time constant of the three translations is set below.) The
        # thread's single-joint equality yields R F/k = 0.3 mm at 27 kN with the slide's own inverse weight.
        for d_ in (self.fdof["nut_turn"], self.fdof["washer_turn"]):
            mjm.dof_solref[d_] = (2*P.DT, 1.)
            mjm.dof_solimp[d_] = (0.9999, 0.9999, 0.001, 0.5, 2.)
        # ENGINE: an inertia on the nut's turn relative to the stud (params.NUT_TURN_ARMATURE). The nut alone is
        # 1e-6 kg m2: once it bears, its own thread and bearing friction stop it within a tick whenever the socket's
        # flats lose contact, and a socket that runs it down at 2300 rpm (13.7 degrees a tick) then stands up to 11.3
        # degrees into the nut's hex for a tick (E20 dusty; 4.6 degrees with this inertia, which is about
        # that of the anvil and socket that drive the nut). Its cost: an inertia on a relative coordinate reacts on
        # the stud, which turns 2-6 degrees in its hole during a setting in a nominal hole and 15 degrees in a 13.6 mm
        # hole (0.25 without; taking the last tick's reaction off the stud again was tried: 15 -> 5.6 degrees in the
        # 13.6 mm hole, no change in the nominal one, not adopted). The wrench's blows do not
        # need it (they come through a hammer and a dilated anvil that cannot outrun the nut: E20 sound without it,
        # 0 corners passed), and the washer's turn on the nut carries no inertia at all (E20 seating_blows and
        # real_hammer without it: no fault, the same loads).
        mjm.dof_armature[self.fdof["nut_turn"]] = P.NUT_TURN_ARMATURE
        # ENGINE: an inertia on the nut's slide along the stud (a relative coordinate: it adds nothing to the anchor's
        # motion as a whole). The thread's equality answers within two ticks; a hammer blow moves the stud within one,
        # and without this inertia the 44 g nut and washer are displaced 0.03-0.35 mm along the stud for a few ticks
        # at every blow (a nut cannot move along its thread without turning; with the inertia 0.002-0.007 mm; E20
        # seating_blows and real_hammer: no fault and no stud tension either way).
        mjm.dof_armature[self.fdof["nut_slide"]] = P.NUT_SLIDE_ARMATURE_KG
        # ENGINE: the clip's drag is joint friction on the stud's three translations. A MuJoCo friction row does not
        # oppose the velocity: it steers the acceleration to -v/(time constant), up to its force, and with the default
        # 20 ms a stud leaving its seat at 0.15 m/s coasted 1.07 mm with the 150 N drag "on" (the drag even pushed it
        # along against a 20 N push). At 2 ticks the drag stops the stud within a tick, as
        # 150 N on 0.15 kg does (11 um). Only the time constant: the rows' impedance stays the default
        for d_ in self.free_dofs[:3]:
            mjm.dof_solref[d_] = (2*P.DT, 1.)
        # ENGINE: the nut's hex at MuJoCo's stiffest contact impedance (params.NUT_HEX_SOLIMP; the socket's hex walls
        # carry the same, tools.ImpactWrench.bind)
        hexes = set(self.nut_shapes)
        for g in range(mjm.ngeom):
            if g < len(w._geom_shape) and int(w._geom_shape[g]) in hexes:
                mjm.geom_solimp[g][:3] = P.NUT_HEX_SOLIMP
        self.it.state.setdefault("setting", self.fresh())

    @staticmethod
    def fresh():
        return dict(thread_n=0., max_thread_n=0., grip=False, cone_n=0., n_cone=0., n_wall=0., s_m=0.,
                    hold_n=0., spin_rad=0., spin_max_deg=0., nut_turn_rad=0., events=[], torque_nm=0., max_torque_nm=0.,
                    grip_data=None, grip_solimp=None, cone_capacity_n=0., washer_pen_m=0., pen_grip_m=0.,
                    tip_prev=None, give_prev=None, s_rate=0., max_set_n=0., max_set_torque_nm=0.,
                    turn_torque_nm=0., max_turn_torque_nm=0., failed=None)

    # ------------------------------------------------------------------ laws
    def torque_for(self, F):
        """Torque the nut needs to turn on further at stud tension F (N): pitch + thread + bearing (N m). This is the
        friction law the nut's and washer's friction are set from (K d F), a derived number; the torque the nut was
        actually turned against is recorded from the solve (post_step, 'turn_torque_nm')."""
        return F*(self.c + 0.577*P.THREAD_MU*P.KBTZ2_D2_M + P.BEARING_MU*P.NUT_BEARING_R_M)

    def ring(self, clip):
        """(engaged share, ring force N) of the clip in the hole it is in (it.state['clip'], holes.py)."""
        if not clip or clip.get("engaged", 0.) <= 0. or not clip.get("concrete"):
            return 0., 0.
        f = clip["engaged"]
        return f, self.k_clip*max(0., 0.5*P.CLIP_FREE_D_M - clip["R_m"])*f

    def axis_world(self, w):
        return Rotation.from_quat(w.bq[self.b["top"]][3:]).apply([1., 0., 0.])

    def pre_step(self, w):
        mjm, d = w.solver.mj_model, w.solver.mj_data
        st = self.it.state.setdefault("setting", self.fresh())
        F = st["thread_n"]
        fl = mjm.dof_frictionloss
        q_turn = self.j["nut_turn"].pos(w)
        # the thread: the nut's place on the stud follows its turn; pitch and thread friction resist the turn, the
        # bearing friction the washer's turn, each above the free nut's prevailing torque (NUT_PREVAILING_NM)
        mjm.eq_data[self.eq_thread][0] = self.c*q_turn
        fl[self.fdof["nut_turn"]] = P.NUT_PREVAILING_NM + (self.c + 0.577*P.THREAD_MU*P.KBTZ2_D2_M)*F
        # (the bearing face carries whatever presses the nut on its washer: the stud's tension once the nut is
        # tight, or a socket's push on a nut that stands on its washer on the fixture, read as the washer's load on
        # the fixture. With the tension alone a pushed-on socket's rim turned the loose nut with it, E20 pushon_turned)
        fl[self.fdof["washer_turn"]] = P.NUT_PREVAILING_NM + P.BEARING_MU*P.NUT_BEARING_R_M*max(F, st.get("washer_n", 0.))
        # the clip
        clip = self.it.state.get("clip")
        f, n_ring = self.ring(clip)
        n_cone = st["cone_n"]/self.tan_aphi if st["grip"] else 0.
        st["n_cone"], st["n_wall"] = n_cone, n_ring + n_cone
        a = np.abs(self.axis_world(w))
        # the stud's spin in the clip: the cone's friction mu_c N r_c, under the ring's pressure before the cone is in
        # (a clip loose in an oversize hole holds nothing, so the nut's prevailing torque turns the stud with it)
        # (the cone's load read from the tie, or from the thread's tension when that is larger: both lag a tick, and a
        # nut seated by a running wrench raises the tension within two ticks)
        n_load = max(n_cone, (F/self.tan_aphi) if st["grip"] else 0.)
        spin = P.CONE_MU*(n_ring + n_load)*P.CONE_R_M
        hold = 0.                                           # within the clip's play the stud slides through the clip
        if not st["grip"] and n_ring > 0.:
            # The clip is carried along only at the two ends of its play, and each end holds one way: against its
            # collar the clip resists the stud being driven in (its drag on the stud points out of the hole); with
            # the cone against it, it resists the stud being drawn out (its drag points into the hole). A friction
            # row holds both ways, so the drag is taken off for a tick when the last tick's drag pointed the wrong
            # way (it was holding the stud against a push the clip cannot resist): the stud then moves off that end
            # into its play. (Held both ways, a hammered stud kept its washer pressed on the face at 104 N, and a
            # 20 N push could not put a rebounded stud's washer back on the face.)
            # An end counts as reached when the cone's present speed in the clip carries it there within this tick: the drag set
            # only after the end was passed let a stud thrown back at 0.19 m/s take the clip 0.16 mm along with no
            # drag at all; set a tick early the stud stops short and closes up afterwards.
            drag_ax = st.get("clip_drag_n", 0.)             # last tick's drag on the stud along its axis, + into the hole
            s_next = st["s_m"] + st.get("s_rate", 0.)*w.dt
            if min(st["s_m"], s_next) <= P.CLIP_END_BAND_M:
                hold = P.CLIP_WALL_MU*n_ring if drag_ax < P.CLIP_WRONG_WAY_N else 0.
            elif max(st["s_m"], s_next) >= P.CLIP_PLAY_M - P.CLIP_END_BAND_M:
                hold = P.CLIP_WALL_MU*n_ring if drag_ax > -P.CLIP_WRONG_WAY_N else 0.
        st["hold_n"] = hold if not st["grip"] else st["cone_n"]
        for i in range(3):
            fl[self.free_dofs[i]] = hold*float(a[i])
        fl[self.free_dofs[3]] = spin                        # the free joint's angular dofs are in the stud's frame
        d.eq_active[self.eq_cone] = 1 if st["grip"] else 0
        if st["grip"]:
            mjm.eq_data[self.eq_cone][:11] = st["grip_data"]
            mjm.eq_solimp[self.eq_cone][:2] = st["grip_solimp"]

    def _tie(self, w, f):
        """The cone's tie of the stud's tip to the concrete, where the tip is now, with static stiffness beta x f."""
        mjm = w.solver.mj_model
        q = w.bq[self.b["top"]]
        tip = q[:3] + Rotation.from_quat(q[3:]).apply([self.length, 0., 0.])
        tc = float(mjm.eq_solref[self.eq_cone][0])
        invw = float(mjm.body_invweight0[self.mj_top][0])
        K = max(P.WEDGE_BETA_N_M*f, 1e3)
        dd = 1. - 1./(tc*tc*K*invw)
        # a MuJoCo weld (body1 the stud, body2 the world): data[0:3] the world point, data[3:6] the same point in the
        # stud's frame (its tip), data[6:10] the stud's orientation inverted (w, x, y, z), data[10] torque scale
        inv = [float(q[6]), -float(q[3]), -float(q[4]), -float(q[5])]
        return [*map(float, tip), self.length, 0., 0., *inv, 1.], (dd, dd)

    def post_step(self, w):
        import mujoco
        d = w.solver.mj_data
        st = self.it.state.setdefault("setting", self.fresh())
        F, Fc = 0., 0.
        n = d.nefc
        ax = self.axis_world(w)
        if n:
            types = np.asarray(d.efc_type[:n])
            ids = np.asarray(d.efc_id[:n])
            eqrows = types == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
            r = np.flatnonzero(eqrows & (ids == self.eq_thread))
            if len(r):
                # the thread carries the stud's tension when the washer pushes the nut back towards the stud's top:
                # the equality then pushes the nut's slide forwards (+); a push on the nut (a socket pressed onto it)
                # reads negative and is no tension
                st["thread_raw_n"] = float(d.efc_force[r[0]])
                F = max(0., st["thread_raw_n"])
            rc = np.flatnonzero(eqrows & (ids == self.eq_cone))
            if len(rc) == 6:
                Fc = abs(float(np.asarray(d.efc_force[rc[:3]], float) @ ax))   # the tie's pull along the stud
                st["cone_torque_nm"] = float(np.asarray(d.efc_force[rc[3:]], float) @ ax)
        # the nut bears only when its washer bears on a fixture: a thread force with the washer free is the nut's own
        # inertia under a hammer blow, not a stud tension that could draw the cone in
        f6 = np.zeros(6)
        wash_n, gives = 0., []
        qw = w.bq[self.b["washer"]]
        c_face = np.asarray(qw[:3], float) + ax*0.5*P.WASHER_T_M       # centre of the washer's bearing face
        for i in range(d.ncon):
            g0, g1 = int(d.contact.geom[i][0]), int(d.contact.geom[i][1])
            b0, b1 = int(w.solver.mj_model.geom_bodyid[g0]), int(w.solver.mj_model.geom_bodyid[g1])
            if self.mj_washer in (b0, b1):
                mujoco.mj_contactForce(w.solver.mj_model, d, i, f6)
                wash_n += abs(float(f6[0]))
                nrm = np.asarray(d.contact.frame[i][:3], float)          # from geom 1 to geom 2
                if abs(float(nrm @ ax)) > 0.8:                           # the bearing face on a fixture's face
                    dist = float(d.contact.dist[i])
                    pos = np.asarray(d.contact.pos[i], float)
                    # the fixture's surface under this contact, and how far the bearing face's centre is inside it
                    gives.append(float((pos + nrm*0.5*dist - c_face) @ (-nrm)) if b0 == self.mj_washer else
                                 float((pos - nrm*0.5*dist - c_face) @ nrm))
        st["washer_n"] = wash_n
        # how far the washer's bearing face stands inside the fixture's face at the stud's axis (the soft contact's
        # give; the mean over the face's contact points of the centre's depth below the fixture's surface, so that a
        # stud leaning in its hole does not count)
        pen = max(0., float(np.mean(gives))) if gives else None      # None: the bearing face touches nothing
        st["washer_pen_m"] = pen or 0.
        # the clip's drag on the stud along its axis (the friction rows on the stud's three translations)
        drag = np.zeros(3)
        f_thread = f_bearing = 0.          # the solved friction on the nut's turn and on the washer's turn (N m)
        if n:
            fr = np.flatnonzero(types == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
            for r_ in fr:
                k_ = int(ids[r_]) - self.free_dofs[0]
                if 0 <= k_ < 3:
                    drag[k_] = float(d.efc_force[r_])
                elif int(ids[r_]) == self.fdof["nut_turn"]:
                    f_thread = float(d.efc_force[r_])
                elif int(ids[r_]) == self.fdof["washer_turn"]:
                    f_bearing = float(d.efc_force[r_])
        st["clip_drag_n"] = float(drag @ ax)
        if wash_n < P.ANCHOR_SET_ON_N:
            F = 0.
        st["thread_n"] = F
        st["max_thread_n"] = max(st["max_thread_n"], F)
        st["torque_nm"] = self.torque_for(F)
        st["max_torque_nm"] = max(st["max_torque_nm"], st["torque_nm"])
        st["cone_n"] = Fc if st["grip"] else 0.
        # the torque the nut is being turned on against: the two friction forces that resist it, as solved, while it
        # turns on (0 while it stands); the installation torque is the largest of these with the cone gripping
        turning = float(w.jqd[self.j["nut_turn"].dof]) > P.NUT_TURNING_RAD_S
        st["turn_torque_nm"] = (abs(f_thread) + abs(f_bearing)) if turning else 0.
        if turning and st["grip"] and F > P.ANCHOR_SET_ON_N:
            st["max_turn_torque_nm"] = max(st.get("max_turn_torque_nm", 0.), st["turn_torque_nm"])
        # capacity: the stud's steel strength. The neck breaks, the tie is gone for good; what the nut was taken to
        # stays on record
        if st["grip"] and Fc >= P.WEDGE_STEEL_N:
            st["grip"], st["failed"] = False, "steel"
            st["cone_n"] = 0.
            st["max_set_n"] = st["max_set_torque_nm"] = 0.
            st["tip_prev"] = None
            st["events"].append(dict(tick=w.tick, event="stud_steel_failure", tie_n=round(Fc, 0), thread_n=round(F, 0)))
        # the installation: the largest stud tension the gripping cone has carried against the bearing nut (thread and
        # cone both), and the torque that tension needed at the nut. It stays when the tension later falls (a ledger
        # drawn further into its seat by a neighbouring anchor, concrete relaxing) and is lost with the grip
        if st["grip"]:
            st["max_set_n"] = max(st["max_set_n"], min(F, Fc))
            st["max_set_torque_nm"] = self.torque_for(st["max_set_n"])
        # the cone's travel: while the clip bears on the wall it stays put and the stud moves through it
        clip = self.it.state.get("clip")
        f, n_ring = self.ring(clip)
        q = w.bq[self.b["top"]]
        tip = np.asarray(q[:3], float) + ax*self.length
        s_before = st["s_m"]
        if not clip or clip.get("z_tip") is None or f <= 0.:
            st["tip_prev"] = None
            if st["grip"]:
                st["grip"] = False
                st["max_set_n"] = st["max_set_torque_nm"] = st["max_turn_torque_nm"] = 0.
                st["events"].append(dict(tick=w.tick, event="clip_left_hole"))
        else:
            # The stud's travel in the clip is its tip's travel along its axis less the change of the washer's give
            # into the fixture's face: the clip is driven in only as far as the stud goes with its washer on that face.
            # A seating blow puts the washer into the face for a few ticks (a MuJoCo contact stops a part over its
            # speed x its 2-tick time constant; steel on concrete gives micrometres): 0.05-0.08 mm under E9's blows
            # (0.45 kg at 3 m/s), 0.3-0.5 mm while the nut's slide carried a 10 kg inertia
            # (bind). Counted as travel, the contact's return draws the cone that far back into the clip with the
            # washer still on the face, and once that reaches the clip's play the tie catches and the rest of the
            # return loads it (with the 10 kg inertia a hammered anchor held 10.7 kN with no torque applied). The give
            # is known a tick after the washer first touches (the contact exists from then): on that tick its
            # appearance is not counted. While the cone grips, the give at its catching is kept.
            give = st["pen_grip_m"] if st["grip"] else pen
            if st.get("tip_prev") is not None and (n_ring > 0. or st["grip"]):
                # the cone is drawn into the clip only by the stud's tension (the nut bearing on its washer); a stud
                # that only moves back out (a rebound while it is hammered) drags the clip along at the end of its play
                s_max = P.CLIP_TRAVEL_M if (st["grip"] or F > P.ANCHOR_SET_ON_N) else P.CLIP_PLAY_M
                d_give = (give - st["give_prev"]) if (give is not None and st.get("give_prev") is not None) else 0.
                moved_in = float((tip - np.asarray(st["tip_prev"], float)) @ ax) - d_give
                st["s_m"] = float(min(max(st["s_m"] - moved_in, 0.), s_max))
            st["tip_prev"], st["give_prev"] = [float(x) for x in tip], give
            if not st["grip"] and not st.get("failed") and n_ring > 0. and \
                    st["s_m"] >= P.CLIP_PLAY_M - P.CLIP_END_BAND_M and F > P.ANCHOR_SET_ON_N:
                st["grip"] = True
                st["pen_grip_m"] = pen or 0.
                st["grip_data"], st["grip_solimp"] = self._tie(w, f)
                st["events"].append(dict(tick=w.tick, event="cone_meets_clip", thread_n=round(F, 1),
                                         s_mm=round(st["s_m"]*1e3, 4), tie_d=round(st["grip_solimp"][0], 6),
                                         invw=round(float(w.solver.mj_model.body_invweight0[self.mj_top][0]), 4)))
            elif st["grip"] and st["s_m"] >= P.CLIP_TRAVEL_M:
                st["grip"] = False
                st["max_set_n"] = st["max_set_torque_nm"] = st["max_turn_torque_nm"] = 0.
                st["tip_prev"] = None
                st["events"].append(dict(tick=w.tick, event="cone_pulled_through", thread_n=round(F, 1)))
            elif st["grip"]:
                # the cone holds the stud only against being drawn out: a stud driven back in (a hammer blow after a
                # rebound drew the cone into the clip) pushes the cone out of the clip, and the tie lets go
                deeper = float((tip - np.asarray(st["grip_data"][:3])) @ ax)
                if deeper > 2e-6:
                    st["grip"] = False
                    st["max_set_n"] = st["max_set_torque_nm"] = st["max_turn_torque_nm"] = 0.
                    st["tip_prev"] = None
                    st["s_m"] = max(0., P.CLIP_PLAY_M - deeper)
                    st["events"].append(dict(tick=w.tick, event="cone_driven_out", deeper_mm=round(deeper*1e3, 3)))
        st["s_rate"] = (st["s_m"] - s_before)/w.dt          # the cone's speed into the clip (pre_step's look-ahead)
        # the stud turning in its hole while its nut is loaded (the installer's 'anchor spins')
        omega = float(np.asarray(w.bqd[self.b["top"]][3:], float) @ ax)
        if F > P.ANCHOR_SET_ON_N and f > 0.:
            st["spin_rad"] += omega*w.dt
            st["spin_max_deg"] = max(st["spin_max_deg"], abs(math.degrees(st["spin_rad"])))
        st["nut_turn_rad"] = self.j["nut_turn"].pos(w)

    def state(self):
        w = self.w
        return dict(nut_turns=round(self.j["nut_turn"].pos(w)/(2*math.pi), 4),
                    nut_travel_mm=round(self.j["nut_slide"].pos(w)*1e3, 3))


def nozzle(sc, spec):
    """HEPA vacuum wand with a narrow nozzle: 10 mm OD x 200 mm tube on a 30 mm handle (OSHA Table 1 practice)."""
    name = spec["name"]
    pose = _pose(spec)
    pr = T.Parts(sc)
    H = T.link(sc, f"{name}_handle", pose)
    pr.cyl(H, f"{name}_grip", (-0.06, 0., 0.), 0.016, 0.06, "x", "hose_black", density=600., mu=P.MU["rubber"])
    pr.cyl(H, f"{name}_hose_stub", (-0.15, 0., 0.), 0.019, 0.03, "x", "hose_black", density=400.)
    Np = T.compose(pose, (0., 0., 0.))
    N = T.link(sc, f"{name}_tube", Np)
    pr.cyl(N, f"{name}_tube_body", (0.10, 0., 0.), 0.005, 0.10, "x", "steel_bright", density=2700.)
    _assembly(sc, name, H, [(N, Np)], pose=pose)
    sc.mutual_filter([H, N])
    sc.add_item(Item(name, "nozzle", N, axis=(1., 0., 0.), tip=(0.20, 0., 0.), radius=0.005, length=0.20))
    sc.bodies[name] = H
    sc.parts[name] = dict(kind="nozzle", body=H, tube=N)


def brush(sc, spec):
    """Hole-cleaning wire brush for 1/2 in anchors (Hilti HIT-RB class): 13 mm bristle head 100 mm on a 250 mm rod."""
    name = spec["name"]
    pose = _pose(spec)
    pr = T.Parts(sc)
    H = T.link(sc, f"{name}_handle", pose)
    pr.cyl(H, f"{name}_grip", (-0.05, 0., 0.), 0.012, 0.05, "x", "brush_handle", density=900., mu=P.MU["rubber"], roll=True)
    Rp = T.compose(pose, (0., 0., 0.))
    Rk = T.link(sc, f"{name}_rod", Rp)
    pr.cyl(Rk, f"{name}_rod_body", (0.075, 0., 0.), 0.0025, 0.075, "x", "steel_bright", density=7850.)
    # (both rims with rolling resistance: lying on its grip and its bristle head the brush rolls as a cone does, and
    # without it that rocking never died once the hammer that propped it was taken)
    pr.cyl(Rk, f"{name}_bristles", (0.20, 0., 0.), 0.0062, 0.05, "x", "brush_wire", density=2500., mu=0.3, roll=True)
    _assembly(sc, name, H, [(Rk, Rp)], pose=pose)
    sc.mutual_filter([H, Rk])
    sc.add_item(Item(name, "brush", Rk, axis=(1., 0., 0.), tip=(0.25, 0., 0.), radius=0.0062, length=0.25))
    sc.bodies[name] = H
    sc.parts[name] = dict(kind="brush", body=H, rod=Rk)


def ballpeen_head_r():
    """Radius (m) of the kit hammer's head: a steel cylinder of the head's mass and length."""
    k = P.KIT_HAMMER
    return math.sqrt(k["head_kg"]/(math.pi*k["head_len_m"]*7850.))


PEEN_FACE = np.array([P.KIT_HAMMER["head_x_m"], 0.5*P.KIT_HAMMER["head_len_m"], 0.])     # the head's flat face centre, hammer frame


def ballpeen(sc, spec):
    """The kit's 32 oz ball-peen hammer (params.KIT_HAMMER): a 0.907 kg steel head 100 mm long across a 340 mm hickory
    handle. Frame: x along the handle towards the head, y along the head: the striking face at +y (PEEN_FACE)."""
    name = spec["name"]
    pose = _pose(spec)
    pr = T.Parts(sc)
    H = T.link(sc, f"{name}", pose)
    k = P.KIT_HAMMER
    # One plain cylinder for the head and one for the handle, without rolling-resistance rows. Richer heads were
    # tried, each with a defect: rolling rows on the head jammed an anchor under the blow (clean-and-set's first
    # hole, 3.5 degrees, the clip gripping at 7 kN); a head with a ball at the peen end and a flat-sided eye section
    # started to tremble on the kit tray after half a minute (0.9 mm/s, E22). What the plain cylinder does wrong is
    # stated in E14
    pr.cyl(H, f"{name}_handle", (0.0, 0., 0.), k["handle_r_m"], 0.5*k["handle_len_m"], "x", "hickory", density=750.,
           mu=P.MU["default"])
    pr.cyl(H, f"{name}_head", (k["head_x_m"], 0., 0.), ballpeen_head_r(), 0.5*k["head_len_m"], "y", "steel_dark",
           density=7850.)
    _assembly(sc, name, H, [], pose=pose)
    sc.bodies[name] = H
    sc.parts[name] = dict(kind="ballpeen", body=H)


def workpiece(sc, spec):
    """A drillable workpiece: flat bar, block, angle (one drillable leg plus a solid leg), panel, backer or board."""
    name = spec["name"]
    pose = _pose(spec)
    half = spec["half"]
    extra = ()
    if spec["kind"] == "angle":
        # vertical leg of the equal angle (as long as the drillable horizontal leg is wide), standing on that leg's
        # edge (x+)
        extra = ((f"{name}_leg", (half[0] - half[2], 0., half[0]), (half[2], half[1], half[0] - half[2]), "steel"),)
    fixed = spec.get("fixed", False)
    d = sc.add_drillable(name, spec["material"], pose, half, static=fixed, n_slots=spec.get("slots", 4),
                         extra_boxes=extra, render_material=spec.get("render", spec["material"]))
    for (x, y) in spec.get("dimples", []):
        sc.initial_dimples.append((name, (x, y, half[2]), (0., 0., 1.), spec.get("dimple_depth", None)))
    return d


def build_part(sc, spec):
    k = spec["kind"]
    if k in ("bolt",):
        return bolt(sc, spec)
    if k == "carriage_bolt":
        return bolt(sc, spec, carriage=True)
    if k == "anchor":
        return anchor(sc, spec)
    if k == "nozzle":
        return nozzle(sc, spec)
    if k == "brush":
        return brush(sc, spec)
    if k == "ballpeen":
        return ballpeen(sc, spec)
    if k in ("bar", "block", "angle", "panel", "board"):
        return workpiece(sc, spec)
    raise ValueError(f"unknown part kind {k}")
