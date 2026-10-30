"""Task ladder: start states of the two repair rungs (smooth-corner, finish-securing).

A co-worker has stuck the 4 x 6 in shipping label down on the carton top and pressed most of it, but some areas
did not take and have curled up off the carton (layouts.partly_applied: one corner for smooth-corner, two to four
corners, ends or long sides for finish-securing). This module builds that state on the real world; nothing here
acts after the episode starts.

CONSTRUCTED START STATE (not a robot action; same standing as press-down's laid label, env.lay_label_on_carton):
  1. the label is laid flat, adhesive down and touching, at the seeded place on the flap half; each lifted area is
     the part of the label beyond a straight bond front, curled up from the front at constant curvature (at most
     100/m, below the paper's 119/m bending yield, so pressing it flat is elastic) until its farthest point stands
     60-90 deg off the carton;
  2. the label's rest shape (labelsim theta_rest, its permanent-set state) is set to that shape: the curl is a set
     the label keeps, as a label corner that did not take keeps its curl, not a force holding it up;
  3. the label's own physics then runs a few steps, so the flat part tacks on through the solver's bond rule;
  4. the flat part that bonded is given the contact fraction a standard firm application leaves (task.py:
     phi = 1 - exp(-(100 kPa + p_tack)/p_r) on the surface it is bonded to, wet-out 100 % of the standard): the
     co-worker's pressing. The lifted areas keep whatever their own physics gave them (nothing, or the tack of
     the first row at the front);
  5. env._settle runs the normal 0.4 s settle. Nothing holds the label afterwards.
No material, contact, adhesion, solver or robot parameter is touched: positions, the permanent-set state, bonds
and contact fractions are the label's state variables, and the values written are ones its own laws produce.

The robot starts in a ready pose (ready_pose): the right hand beside the label, its wrist camera looking at the
middle of the label from the first of 0.18-0.26 m at which the arm clears the carton and the label by 30 mm
(0.20-0.22 m in practice), tilted 27 deg from the vertical toward the arm, its fingers closed so the two
fingertip pads form one pressing face (the squeegee card stays in the tool tray: in this world a card does not pass
a pressing load to a label, so the fingertips are the working applicator). It is the arm's initial joint
configuration and position-servo targets; no grasp, latch or controller acts beyond the arm's ordinary servo.
"""
import math
import numpy as np

from . import layouts
from . import materials as M

READY_HEIGHT = .08            # m: hand height above the label's middle that scripted demos return to
# The ready pose's roll about the camera's view axis (ready_candidates index, 45 deg steps). Measured over 40 starts
# (smooth-corner 0-29, finish-securing 0-9; ray casts against the robot's rendered geometry): roll 0 is the only one
# that leaves every lifted area's raised part in view of the right wrist camera (>= 97 %) with the label fully
# visible overhead at every start; roll 2, the roll nearest the arm's default joints, leaves the arm's last link over
# the lifted corner at one corner position in four (12-53 % of the corner visible), and rolls 4-6 put the hand over
# the label overhead.
READY_ROLL = 0
# The parked hand keeps clear of the work: the arm's collision geometry at least READY_CLEARANCE from the carton
# (MuJoCo geom distance) and its pad and finger colliders at least that from the label, curled tips included. Roll 0
# brings the hand lower than roll 2 does, at some starts down to the carton top, so the wrist camera backs off along
# its view line, VIEW_DISTANCE_STEPS, until the hand clears. At finish-securing seed 29 with roll 0 at 0.18 m, the
# fingertip boxes 14.5 mm from the carton drew a spurious -208 mm box-box contact from MuJoCo on the first settle step,
# and the carton was pushed off the bench (a MuJoCo narrow-phase defect, reported, not repaired); 30 mm avoids it.
READY_CLEARANCE = .03         # m
VIEW_DISTANCE_STEPS = (.18, .20, .22, .24, .26)
VIEW_DISTANCE = .18           # m: ready pose, wrist camera to the middle of the label
VIEW_TILT = math.radians(27.)  # ready pose, wrist camera axis from the vertical, leaning toward the arm's side
TACK_STEPS = 5                # label steps of free physics in which the flat part tacks on (step 3)


def frame(world):
    """World placement of the partly applied label: (centre, md, cd, n) with md/cd the label's material u/v axes
    on the carton top (flip: the label lies the other way round) and n the top's outward normal."""
    g = world.scene.task_geometry["carton"]
    lay = world.scene.task_geometry["laid_label"]
    p, R = world.body_pose("carton")
    (uc, vc), yaw = lay["centre"], lay["yaw"]
    c, s_ = math.cos(yaw), math.sin(yaw)
    md, cd = R @ np.array([c, s_, 0.]), R @ np.array([-s_, c, 0.])
    if lay.get("flip"):
        md, cd = -md, -cd
    return p + R @ np.array([uc, vc, g["H"]/2]), md, cd, R[:, 2]


def lifted_masks(lab, lay):
    """Per lifted area, the label nodes beyond its bond front (material coordinates; independent of the state)."""
    return [layouts.flap_mask(lab.X, f)[0] for f in lay["regions"]]


def lay_partly_applied(world):
    """Steps 1-2 of the construction (module docstring): positions and rest shape; adhesive state cleared."""
    lab = world.label
    lay = world.scene.task_geometry["laid_label"]
    centre, md, cd, n = frame(world)
    half = .5*lab.stock.thickness
    X = lab.X
    x = (centre[None, :] + (X[:, 0:1] - lab.length/2)*md[None, :] + (X[:, 1:2] - lab.width/2)*cd[None, :]
         + n[None, :]*half)
    for f in lay["regions"]:
        m, d = layouts.flap_mask(X, f)
        e = f["normal"][0]*md + f["normal"][1]*cd             # in-plane direction from the front into the flap
        k, dd = f["curvature"], d[m]
        x[m] += (np.sin(k*dd)/k - dd)[:, None]*e[None, :] + ((1. - np.cos(k*dd))/k)[:, None]*n[None, :]
    lab.x = x
    lab.v[:] = 0.
    lab.bond[:] = -1; lab.phi[:] = 0.; lab.damage[:] = 0.; lab.gas[:] = 0.
    lab.p_sustained[:] = 0.; lab.bond_time[:] = 0.; lab.rebond_ok[:] = True
    lab.theta_rest[:] = lab._hinge_angles(x)[0]               # the curl is the label's set (rest) shape
    world.label_history.clear()


def standard_wet_out(world):
    """Step 4: the flat part that bonded to the carton gets the contact fraction of a standard firm application on
    the surface it is bonded to (the wet-out reference the evaluator uses, task.P_STANDARD_APPLICATION)."""
    from .task import P_STANDARD_APPLICATION
    lab = world.label
    lifted = np.any(lifted_masks(lab, world.scene.task_geometry["laid_label"]), 0)
    carton = [i for i, c in enumerate(lab.colliders) if c.name == "carton"]
    sel = np.flatnonzero(np.isin(lab.bond, carton) & ~lifted)
    for i in sel:
        surf = lab.colliders[lab.bond[i]].surface_list()[lab.bond_surface[i]]
        lab.phi[i] = max(lab.phi[i], 1. - math.exp(-(P_STANDARD_APPLICATION + M.P_TACK)/surf.p_rough))
    return dict(standard_nodes=int(len(sel)), lifted_nodes=int(lifted.sum()),
                lifted_bonded_nodes=int((lifted & (lab.bond >= 0)).sum()))


def set_configuration(world, values):
    """Write arm joint positions as the initial configuration (position, servo target and the rate-limited
    command all at the value, zero velocity) and bring bodies, MuJoCo data and label colliders to it."""
    q = world.state.joint_q.numpy().copy()
    t = world.control.joint_target_q.numpy().copy()
    for name, v in values.items():
        j = world.joints[name]["q"]
        q[j] = t[j] = world.targets[j] = float(v)
    qd = np.zeros_like(world.state.joint_qd.numpy())
    for st in (world.state, world.next_state):
        st.joint_q.assign(q)
        st.joint_qd.assign(qd)
    world.control.joint_target_q.assign(t)
    world.newton.eval_fk(world.model, world.state.joint_q, world.state.joint_qd, world.state)
    world.next_state.assign(world.state)
    world.solver._update_mjc_data(world.solver.mj_data, world.model, world.state)
    world._mujoco.mj_forward(world.solver.mj_model, world.solver.mj_data)
    world._pose_colliders(first=True)


def _camera_in_tcp_frame(world, arm, k):
    """The arm's wrist camera eye and viewing direction in its TCP frame (rigid on link7: scene._wrist_camera)."""
    from .labelsim import quat_to_matrix
    cam = world.scene.cameras[f"{arm}_wrist_rgb"]
    q = world.state.body_q.numpy()[cam["body"]]
    p7, R7 = q[:3].astype(float), quat_to_matrix(q[3:].astype(float))
    T = k.forward(arm)
    eye = p7 + R7 @ np.asarray(cam["eye"], float)
    look = R7 @ (np.asarray(cam["target"], float) - np.asarray(cam["eye"], float))
    return T[:3, :3].T @ (eye - T[:3, 3]), T[:3, :3].T @ (look/np.linalg.norm(look))


def ready_candidates(world, arm="right", distance=VIEW_DISTANCE, rolls=range(8)):
    """The reachable ready-pose IK solutions: `arm`'s TCP placed so its wrist camera looks at the middle of the label
    from `distance`, tilted VIEW_TILT from the vertical toward the arm's own side, at the given rolls (of eight, 45 deg
    apart) about the view axis. Returns (arm, eye, view, [dict(roll, change, joints, tcp)]); change = largest joint
    move from the arm's default joints."""
    from scipy.spatial.transform import Rotation
    from .kinematics import ArmKinematics
    if arm not in world.rig.instances:
        arm = next(iter(world.rig.instances))                  # single-arm deployments: their one arm
    k = ArmKinematics(world)
    centre, md, cd, n = frame(world)
    base = np.array([*world.scene.arm_hw[arm]["mount"][0], centre[2]])
    toward = base - centre
    toward -= (toward @ n)*n
    toward /= np.linalg.norm(toward)
    eye = centre + n*distance*math.cos(VIEW_TILT) + toward*distance*math.sin(VIEW_TILT)
    view = (centre - eye)/np.linalg.norm(centre - eye)
    c_loc, d_loc = _camera_in_tcp_frame(world, arm, k)
    now = k.joints_now(arm)
    out = []
    for j in rolls:
        # TCP orientation R with R d_loc = view, rolled j x 45 deg about the view axis
        R0 = Rotation.align_vectors([view], [d_loc])[0]
        R = (Rotation.from_rotvec(view*j*math.pi/4)*R0).as_matrix()
        tcp = eye - R @ c_loc
        try:
            sol = k.solve(arm, tcp, Rotation.from_matrix(R).as_quat(), restarts=8)
        except ValueError:
            continue
        out.append(dict(roll=j, change=float(np.abs(np.array(list(sol.values())) - now).max()), joints=dict(sol),
                        tcp=tcp))
    return arm, eye, view, out


def ready_pose(world, arm="right"):
    """The ready pose: `arm`'s hand beside the label with its fingers closed (the two fingertip pads one pressing
    face), placed so its wrist camera looks at the middle of the label (from the first of VIEW_DISTANCE_STEPS that
    clears, below), tilted VIEW_TILT from the
    vertical toward the arm's own side (a lifted area is seen in profile, not foreshortened as from straight above,
    and the hand does not sit over the label in the overhead view). Of eight rolls about the view axis, READY_ROLL is
    used, the one measured to keep every lifted area and the whole label in the robot's default views, at the first
    of VIEW_DISTANCE_STEPS where the hand clears the carton and the label by READY_CLEARANCE. If no step clears (or
    the roll is unreachable on another robot), the clearing solution nearest the arm's default joints at any roll;
    the returned `fallback` says so. Call it with the label already in place (build_start)."""
    def place(c):
        values = dict(c["joints"])
        values.update({jn: 0. for jn in world.gripper_joints(c["arm"])})  # fingers closed: pads together
        set_configuration(world, values)
        return values

    for dist in VIEW_DISTANCE_STEPS:
        arm, eye, view, cands = ready_candidates(world, arm, dist, rolls=(READY_ROLL,))
        for c in cands:
            c["arm"] = arm
            values = place(c)
            clear = arm_clearance(world, arm)
            if min(clear.values()) >= READY_CLEARANCE:
                return dict(arm=arm, tcp_target_m=c["tcp"].tolist(), camera_eye_m=eye.tolist(),
                            camera_view=view.tolist(), joints=values, roll=c["roll"], view_distance_m=dist,
                            clearance_m=clear, fallback=False)
    best = None
    for dist in VIEW_DISTANCE_STEPS:
        arm, eye, view, cands = ready_candidates(world, arm, dist)
        for c in sorted(cands, key=lambda c: c["change"]):
            c["arm"] = arm
            place(c)
            clear = arm_clearance(world, arm)
            if min(clear.values()) >= READY_CLEARANCE:
                best = (c, eye, view, dist, clear)
                break
        if best:
            break
    if best is None:
        raise RuntimeError(f"no ready pose for {arm} clears the carton and the label by {READY_CLEARANCE} m")
    c, eye, view, dist, clear = best
    values = place(c)
    return dict(arm=arm, tcp_target_m=c["tcp"].tolist(), camera_eye_m=eye.tolist(), camera_view=view.tolist(),
                joints=values, roll=c["roll"], view_distance_m=dist, clearance_m=clear, fallback=True)


def _geom_points(world, g):
    """Points on MuJoCo geom g's surface in world coordinates (box: corners and a 5 x 5 grid on each face; mesh: its
    vertices; sphere, capsule, cylinder: rings), from the posed MuJoCo data."""
    m, d, mj = world.solver.mj_model, world.solver.mj_data, world._mujoco
    R, p, t, s = d.geom_xmat[g].reshape(3, 3), d.geom_xpos[g], int(m.geom_type[g]), m.geom_size[g]
    G = mj.mjtGeom
    if t == G.mjGEOM_BOX:
        a = np.linspace(-1., 1., 5)
        U, V = np.meshgrid(a, a)
        loc = []
        for k in range(3):
            i, j = [x for x in range(3) if x != k]
            for sg in (-1., 1.):
                q = np.zeros((25, 3)); q[:, k] = sg; q[:, i] = U.ravel(); q[:, j] = V.ravel()
                loc.append(q*s)
        loc = np.concatenate(loc)
    elif t == G.mjGEOM_MESH:
        i = int(m.geom_dataid[g])
        loc = np.asarray(m.mesh_vert[m.mesh_vertadr[i]:m.mesh_vertadr[i] + m.mesh_vertnum[i]], float)
    elif t in (G.mjGEOM_SPHERE, G.mjGEOM_CAPSULE, G.mjGEOM_CYLINDER):
        r, h = s[0], (0. if t == G.mjGEOM_SPHERE else s[1])
        ang = np.linspace(0., 2*np.pi, 16, endpoint=False)
        loc = np.concatenate([np.c_[r*np.cos(ang), r*np.sin(ang), np.full(16, z)] for z in (-h - r*(t != G.mjGEOM_CYLINDER), -h, 0., h, h + r*(t != G.mjGEOM_CYLINDER))])
    else:
        return np.zeros((0, 3))
    return loc @ R.T + p


def arm_clearance(world, arm):
    """How far the parked arm is from the work, in m: `carton`, the smallest distance from the arm's MuJoCo collision
    geoms (surface points, _geom_points) to the carton, by the label solver's own carton collider; `label`, the
    smallest distance from the hand's pad and finger colliders (the label solver's) to the label's nodes, less half the
    label's thickness. (MuJoCo's own box-box distance query returns spurious zeros here, so it is not used.)"""
    m = world.solver.mj_model
    gs = world._geom_shape
    arm_shapes = {int(i) for i in world.scene.arms[arm].shape_ids}
    P = np.concatenate([_geom_points(world, g) for g in range(m.ngeom) if gs[g] >= 0 and int(gs[g]) in arm_shapes])
    lab = world.label
    dc = min(float(c.sdf(P)[0].min()) for c in lab.colliders if c.name == "carton")
    hand = [c for c in lab.colliders if c.name.startswith(arm) and c.role in ("pad", "robot")]
    dl = min(float(c.sdf(lab.x)[0].min()) for c in hand) - .5*lab.stock.thickness
    return dict(carton=dc, label=dl)


def build_start(world):
    """Steps 1-4 plus the ready pose (env.LabelEnv calls env._settle afterwards: step 5)."""
    lay_partly_applied(world)                   # first, so the ready pose can keep the hand clear of the curls
    info = dict(ready=ready_pose(world))
    for _ in range(TACK_STEPS):
        world.step()
    info.update(standard_wet_out(world))
    return info
