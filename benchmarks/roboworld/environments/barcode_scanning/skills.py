"""Operator skills for the ASSISTED scripted demonstrations (never policy actions): each composes the normal robot
controls (IK waypoints -> joint position targets, Franka Hand grasps) exactly as scripted.Operator does, or -- where
named "phantom" -- a labelled fixture from probes/fixtures.py. Every phase label says which.

take_scanner / return_scanner   right hand takes the scanner from its cradle by its head and later reseats it
aim                             held scanner looking at a label or point from `standoff`, pitched down
phantom_trigger                 ASSISTED phantom fingertip pulls the trigger (3 N on the blade, reaction on the scanner)
press_key                       one fingertip of a tilted, closed hand presses a keycap until its switch closes
phantom_present / phantom_put_back   ASSISTED phantom (a customer, patron or stocker) lifts a unit out and holds its
                                barcode towards a point, and later puts it back where it was
"""
import math
import numpy as np
from .scripted import CHUNK_S
from scipy.spatial.transform import Rotation

from .scripted import hand_quat, scanner_in_hand, tcp_for_scanner, place_held, reseat, seated_scanner_pose


def tuck(op, arm, seconds=2.0, carried=(), swing=.7):
    """The whole arm swung outward about its base (the ready pose with joint 1 turned `swing` rad away from the other
    arm), holding `carried` if given: a Cartesian park left the elbow in the middle, where the other arm's paths to
    the keypad ran into it. A checked joint-space move. In the ready pose the elbow is behind the
    base, so the swing that takes the hand outward takes the elbow inward: with both arms tucked at 0.7 rad the elbows
    overlap by 63 mm (measured, collide.py), so smaller swings (half, then none: the ready pose) are tried in turn."""
    from . import params as P_
    chain, names = op.k.arms[arm]
    x = op.w.scene.arm_hw[arm]["mount"][0][0]
    now = np.array(op.k.joints_now(arm), float)
    n = max(1, int(round(seconds/CHUNK_S)))
    first = None
    for sw in (swing, swing/2, 0.):
        goal = np.clip(np.array(P_.READY_POSE[:7], float), chain.lower, chain.upper)
        goal[0] = float(np.clip(math.copysign(sw, -x), chain.lower[0], chain.upper[0]))
        path = [now + (goal - now)*(3*a*a - 2*a*a*a) for a in np.linspace(0, 1, n + 1)[1:]]
        hit = checker(op).path_clear(arm, list(names), path, carried=carried, every=3)
        if hit is None:
            run_path(op, arm, path)
            return sw
        first = first or hit
        # a straight joint-space swing may clip something: rise 5 cm first, then swing
        pt, Rt = op.tcp(arm)
        try:
            up = _solve_path(op, arm, _cart_steps(pt, Rt, pt + np.array([0., 0., .05]), Rt), now)
        except ValueError:
            up = []
        if up and checker(op).path_clear(arm, list(names), up, carried=carried, every=1) is None:
            u = np.asarray(up[-1])
            path2 = list(up) + [u + (goal - u)*(3*a*a - 2*a*a*a) for a in np.linspace(0, 1, n + 1)[1:]]
            if checker(op).path_clear(arm, list(names), path2, carried=carried, every=3) is None:
                run_path(op, arm, path2)
                return sw
    raise ValueError(f"no collision-free tuck: {first[1][:2]}")


def park(op, arm, seconds=1.6, carried=()):
    """The arm's hand out to its own side, over the cart (holding `carried`, e.g. the scanner, if given)."""
    w = op.w
    x = w.scene.arm_hw[arm]["mount"][0][0]
    pt, Rt = op.tcp(arm)
    try:
        checked_move_joints(op, arm, np.array([math.copysign(.46, x), -.36, 1.12]), Rt.as_quat(), seconds=seconds,
                            carried=carried, what="parking")
    except ValueError:
        # the other arm's elbow in the way of the park (bulky checkout): swing this arm out about its base instead
        tuck(op, arm, carried=carried)


def take_scanner(op, arm, force=17.5, depth=.030):
    """Right hand over the scanner head, grasp it, lift it along the cup axis. Returns the grasp record."""
    from .probes.e7_hold import grasp_frame
    w = op.w
    p_g, q_g, approach, axis = grasp_frame(op.env, "head", depth)
    op.phase = "ASSISTED scripted: open hand over the scanner head"
    op.open_hand(arm)
    checked_move_joints(op, arm, p_g - .10*approach, q_g, seconds=2.0, what="reaching over the scanner")
    op.phase = "ASSISTED scripted: descend onto the head"
    checked_move(op, arm, p_g, q_g, seconds=1.2, ignore=scanner_bodies(op), what="descending onto the scanner")
    op.phase = f"ASSISTED scripted: grasp {force:g} N per finger"
    op.grasp(arm, force, seconds=1.0)
    rel = scanner_in_hand(op, arm)
    op.phase = "ASSISTED scripted: lift out of the cradle"
    lift = p_g + .08*axis
    checked_move(op, arm, lift, q_g, seconds=1.2, carried=scanner_bodies(op), ignore_labels=("cradle",),
                 what="lifting the scanner out")
    return dict(p_g=p_g, q_g=q_g, approach=approach, axis=axis, lift=lift, rel=rel)


def return_scanner(op, arm, g):
    op.phase = "ASSISTED scripted: carry back over the cradle"
    # the scanner 80 mm above its seat on the cup axis, from its measured pose in the hand (the grasp's own lift pose,
    # replayed after 40 s of held counting, put the crept scanner's handle 2 mm from the cradle's head rest)
    from .scripted import scanner_in_hand, tcp_for_scanner
    o_c, q_c = op.w.scene.insertion_pose(.08)
    pt_c, Rt_c = tcp_for_scanner(o_c, Rotation.from_quat(q_c).as_matrix(), scanner_in_hand(op, arm))

    def carry():
        checked_move_joints(op, arm, pt_c, Rt_c.as_quat(), seconds=2.2, carried=scanner_bodies(op), what="carrying the scanner back")
    try:
        carry()
    except ValueError:
        # the other arm, parked where it was, can sit in the way back to the cradle (count-by-scanning: 2.4 mm from
        # the carrying arm's path): it swings aside, then the carry is planned again
        others = [a for a in op.w.scene.arm_hw if a != arm]
        if not others:
            raise
        # first the carrying arm parks on its own side with the scanner (aimed at a unit, its elbow was turned towards
        # the other arm and every path from there to the cradle swept it through the other arm's elbow, link 4;
        # count-by-scanning); then the other arm swings aside, smallest swing first (the tuck's widest swing
        # turns that arm's elbow inwards)
        last = None
        for sw in (None, 0., .35, .7):
            try:
                if sw is None:
                    op.phase = "ASSISTED scripted: park the scanner on this arm's side"
                    park(op, arm, carried=scanner_bodies(op))
                else:
                    op.phase = "ASSISTED scripted: the other arm swings aside"
                    tuck(op, others[0], swing=sw)
            except ValueError as exc:
                last = exc
                continue
            op.phase = "ASSISTED scripted: carry back over the cradle"
            try:
                carry()
                break
            except ValueError as exc:
                last = exc
        else:
            raise last
    op.phase = "ASSISTED scripted: scanner upright over the socket, lower, release 10 mm above its floor"
    # scripted.reseat's two moves with the loop closed on the scanner's measured pose at the socket's mouth
    # (over_socket), then its release, unless a protective stop fired on the way down: then the scanner is not over its
    # seat (checkout-bulky: a crept scanner caught the cup's lip, the arm stopped, the hand opened and the scanner fell
    # to the floor). Wait out the stop, measure the grip again, retry from the mouth.
    mouth, drop = .050, .010
    for attempt in range(2):
        n0 = len(op.w.reflexes)
        off = over_socket(op, arm, mouth)
        o, q = op.w.scene.insertion_pose(drop)
        pt, Rt = tcp_for_scanner(o, Rotation.from_quat(q).as_matrix(), scanner_in_hand(op, arm))
        op.move(arm, pt + off, Rt.as_quat(), seconds=1.6)
        if len(op.w.reflexes) == n0 or attempt:
            break
        op.phase = "ASSISTED scripted: protective stop while lowering; wait, measure the grip again, retry from the mouth"
        op.wait(1.2)
        op.phase = "ASSISTED scripted: scanner upright over the socket, lower, release 10 mm above its floor"
    op.open_hand(arm, seconds=.8)
    op.phase = "ASSISTED scripted: retreat"
    checked_move(op, arm, g["p_g"] - .10*g["approach"], g["q_g"], seconds=1.0, ignore=scanner_bodies(op), what="retreating")


def over_socket(op, arm, height, seconds=1.6, corrections=3, tol_m=3e-4):
    """ASSISTED: the held scanner upright over its cradle's socket (Scene.insertion_pose), the handle's bottom `height`
    m above the socket's floor, the loop closed on the scanner's measured pose: after each move the commanded hand pose
    is shifted by the remaining error of the handle's bottom centre, the point that has to enter the socket. The socket
    leaves the handle's front face 0.75 mm from the lip (Scene.CUP_CLEAR); open loop, the arm's position servos left the
    handle 1.9-2.0 mm towards the lip at the counter and its front edge came down on the lip's top (checkout-bulky:
    two protective stops, then the scanner fell to the floor). A policy would close this loop with its
    cameras. Returns the shift (world, m) for the moves below."""
    sc = op.w.scene
    hc, hq = sc.handle_frame()
    bottom = np.asarray(hc) + Rotation.from_quat(hq).apply((0., 0., -sc.HANDLE_HALF[2]))
    o, q = sc.insertion_pose(height)
    R = Rotation.from_quat(q).as_matrix()
    goal = o + R @ bottom
    pt, Rt = tcp_for_scanner(o, R, scanner_in_hand(op, arm))
    off = np.zeros(3)
    op.move(arm, pt, Rt.as_quat(), seconds=seconds)
    for _ in range(corrections):
        op.wait(.2)
        ps, Rs = op.w.body_pose("scanner")
        err = goal - (ps + Rs @ bottom)
        if np.linalg.norm(err) < tol_m:
            break
        off = off + err
        op.move(arm, pt + off, Rt.as_quat(), seconds=.5)
    return off


def scanner_bodies(op):
    b = op.w.scene.bodies
    return (b["scanner"], b["scanner_trigger"])


def aim_pose(centre, normal, standoff, pitch_deg):
    """Scanner origin (window centre) and orientation looking at `centre` (a label facing `normal`) from `standoff`
    m, pitched down by `pitch_deg`."""
    look = -np.asarray(normal, float)
    horiz = look - (look @ np.array([0, 0, 1.]))*np.array([0, 0, 1.]); horiz /= np.linalg.norm(horiz)
    th = math.radians(pitch_deg)
    d = math.cos(th)*horiz - math.sin(th)*np.array([0, 0, 1.])
    x = d
    z = np.array([0, 0, 1.]) - (np.array([0, 0, 1.]) @ x)*x; z /= np.linalg.norm(z)
    y = np.cross(z, x)
    return np.asarray(centre, float) - standoff*d, np.column_stack([x, y, z])


def aim(op, arm, g, centre, normal, standoff=.15, pitch=20., via=None):
    """Carry the held scanner to look at `centre` (facing `normal`) from `standoff`; returns the aim errors."""
    w = op.w
    ps_aim, Rs_aim = aim_pose(centre, normal, standoff, pitch)
    pt_aim, Rt_aim = tcp_for_scanner(ps_aim, Rs_aim, g["rel"])
    pt, _ = op.tcp(arm)
    mid = via if via is not None else .5*(pt + pt_aim) + np.array([0, 0, .06])
    op.phase = "ASSISTED scripted: carry the scanner towards the label"
    checked_move_joints(op, arm, mid, Rt_aim.as_quat(), seconds=2.0, carried=scanner_bodies(op), what="carrying the scanner")
    op.phase = "ASSISTED scripted: aim at the label"
    place_held(op, arm, "scanner", ps_aim, Rs_aim, seconds=1.2)
    op.wait(.2)
    ps, Rs = w.body_pose("scanner")
    return dict(aim_error_mm=float(np.linalg.norm(ps - ps_aim)*1e3),
                aim_angle_deg=math.degrees((Rotation.from_matrix(Rs)*Rotation.from_matrix(Rs_aim).inv()).magnitude()),
                window=ps_aim, look=Rs_aim[:, 0])


class PhantomCover:
    """ASSISTED phantom fingertip on a book's front cover: a torque about the spine hinge driving the cover towards a
    target angle (PD, capped at `cap` N m, about twice the cover's gravity moment), its reaction on the book block.
    `target=None` lets go."""
    def __init__(self, world, book, cap=.15):
        m = world.model
        it = world.scene.items[book]
        self.w, self.block, self.cover = world, it["body"], it["cover_body"]
        j = it["hinge"]
        self.qi, self.di = int(m.joint_q_start.numpy()[j]), int(m.joint_qd_start.numpy()[j])
        I = float(np.max(np.linalg.eigvalsh(np.asarray(m.body_inertia.numpy()[self.cover], float))))
        mass = float(m.body_mass.numpy()[self.cover])
        self.k, self.c, self.cap = .4, 2*math.sqrt(.4*max(I, 1e-6) + .4*mass*.01), cap
        self.target = None
        self.log = []

    def angle(self):
        return float(self.w.state.joint_q.numpy()[self.qi])

    def wrench(self, wr):
        if self.target is None:
            return wr
        th = self.angle()
        om = float(self.w.state.joint_qd.numpy()[self.di])
        tau = float(np.clip(self.k*(self.target - th) - self.c*om, -self.cap, self.cap))
        _, Rb = self.w.body_pose(self.block)
        axis = Rb @ np.array([0., 0., -1.])                  # the spine joint's axis (opening positive)
        wr[self.cover, 3:] += tau*axis
        wr[self.block, 3:] -= tau*axis
        self.log.append(dict(t=self.w.sim_time, angle_deg=math.degrees(th), torque_nm=tau))
        return wr


class Fixtures:
    """The labelled phantoms of one rollout, combined into the world's fixture hook."""
    def __init__(self, world):
        from .probes.fixtures import PhantomTrigger
        self.w = world
        self.trigger = PhantomTrigger(world)
        self.holds = {}
        self.covers = {}
        self._apply()

    def _apply(self):
        from .probes.fixtures import combine
        combine(self.w, self.trigger, *self.holds.values(), *self.covers.values())

    def cover(self, book):
        if book not in self.covers:
            self.covers[book] = PhantomCover(self.w, book)
            self._apply()
        return self.covers[book]

    def hold(self, body, **kw):
        from .probes.fixtures import PhantomHold
        h = PhantomHold(self.w, body=body, **kw)
        self.holds[body] = h
        self._apply()
        return h

    def report(self):
        """How hard each phantom pushed: per hand-over hold the steps, peak and mean force (N, gravity compensation
        included), peak torque (N m) and peak tracking error (mm); per cover the steps and peak torque; the trigger
        finger's steps, peak force and largest travel."""
        def vec_peak(log, key):
            v = [float(np.linalg.norm(r[key])) for r in log if key in r]
            return (round(max(v), 3), round(float(np.mean(v)), 3)) if v else (None, None)
        out = dict(holds={}, covers={}, trigger={})
        for name, h in self.holds.items():
            fmax, fmean = vec_peak(h.log, "force_n")
            tmax, _ = vec_peak(h.log, "torque_nm")
            err = [r["pos_err_mm"] for r in h.log if "pos_err_mm" in r]
            out["holds"][str(name)] = dict(steps=len(h.log), peak_force_n=fmax, mean_force_n=fmean, peak_torque_nm=tmax,
                                           peak_tracking_error_mm=round(max(err), 3) if err else None)
        for name, c in self.covers.items():
            tq = [abs(float(r["torque_nm"])) for r in c.log]
            out["covers"][str(name)] = dict(steps=len(c.log), peak_torque_nm=round(max(tq), 4) if tq else None)
        tl = [r for r in self.trigger.log if r.get("force_n")]
        out["trigger"] = dict(steps=len(tl), peak_force_n=max((float(r["force_n"]) for r in tl), default=None),
                              max_travel_mm=round(max((r["travel_mm"] for r in tl), default=0.), 3) if tl else None)
        return out

    def save(self, path):
        """Every phantom's per-step log (time, force, torque, tracking error; trigger travel), one npz."""
        arrays = {}
        for kind, group in (("hold", self.holds), ("cover", self.covers), ("trigger", {"finger": self.trigger})):
            for name, ph in group.items():
                for key in ("t", "force_n", "torque_nm", "pos_err_mm", "angle_deg", "travel_mm"):
                    vals = [r[key] for r in ph.log if key in r]
                    if vals:
                        arrays[f"{kind}/{name}/{key}"] = np.asarray(vals, float)
        np.savez_compressed(path, **arrays)


def phantom_trigger(op, fx, seconds=.6, force=3.):
    op.phase = f"ASSISTED phantom fingertip pulls the trigger ({force:g} N)"
    fx.trigger.force = force
    op.wait(seconds)
    op.phase = "ASSISTED phantom fingertip lets go"
    fx.trigger.force = 0.
    op.wait(.4)


def fingertip(op, arm):
    """Lowest point of the arm's fingers (world), from their collision shapes: what touches a key first."""
    w = op.w
    m = w.model
    if not hasattr(op, "_finger_pts"):
        op._finger_pts = {}
    if arm not in op._finger_pts:
        labels = list(m.body_label)
        sb, st, ss, sx = (a.numpy() for a in (m.shape_body, m.shape_type, m.shape_scale, m.shape_transform))
        import newton
        pts = []
        for i in range(m.shape_count):
            b = int(sb[i])
            if b < 0 or not labels[b].startswith(f"{arm}/") or "finger" not in labels[b]:
                continue
            if int(st[i]) == int(newton.GeoType.BOX):
                h = np.asarray(ss[i], float)
                loc = np.array([[a*h[0], c*h[1], e*h[2]] for a in (-1, 1) for c in (-1, 1) for e in (-1, 1)])
            else:
                src = m.shape_source[i]
                if src is None or not hasattr(src, "vertices"):
                    continue
                loc = np.asarray(src.vertices, float)*np.asarray(ss[i], float)
            tf = sx[i]
            pts.append((b, Rotation.from_quat(tf[3:7]).apply(loc) + tf[:3]))
        op._finger_pts[arm] = pts
    best = None
    for b, loc in op._finger_pts[arm]:
        pb, Rb = w.body_pose(b)
        wp = loc @ Rb.T + pb
        k = int(np.argmin(wp[:, 2]))
        if best is None or wp[k, 2] < best[2]:
            best = wp[k]
    return best


def _carried_boxes(op, arm):
    """Oriented boxes (centre, R, half) in the TCP frame of every collision shape carried past joint 7 (link 7, the
    wrist camera and its bracket, the hand, both fingers at their current opening), and those of the fingers alone."""
    from . import grasping as G
    w = op.w
    labels = list(w.model.body_label)
    mine = {i for i, l in enumerate(labels) if l.startswith(f"{arm}/") and (l.endswith("link7") or "hand" in l or "finger" in l)}
    fingers = {i for i in mine if "finger" in labels[i]}
    pt, Rt = op.tcp(arm)
    Ri = Rt.inv().as_matrix()
    frame = lambda bx: [(Ri @ (c - pt), Ri @ R, h) for c, R, h in bx]
    return frame(G.obstacles(w, only_bodies=mine)), frame(G.obstacles(w, only_bodies=fingers))


def _finger_points_tcp(op, arm):
    """The finger collision shapes' points (box corners, hull vertices) in the TCP frame now: the set fingertip()
    takes its lowest point from."""
    fingertip(op, arm)                          # builds op._finger_pts
    pt, Rt = op.tcp(arm)
    Ri = Rt.inv().as_matrix()
    pts = []
    for b, loc in op._finger_pts[arm]:
        pb, Rb = op.w.body_pose(b)
        pts.append((loc @ Rb.T + pb - pt) @ Ri.T)
    return np.concatenate(pts)


def _corners(boxes):
    return np.array([c + R @ (np.array([sx, sy, sz])*h) for c, R, h in boxes
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])


def plan_press(op, arm, key, depth_m=.003, margins=(.004, .002), tilts=(20., 30., 40.), lead_m=.004):
    """Hand orientations for pressing `key` with one fingertip corner of the closed Franka Hand, checked against the
    scene with the hand's own collision geometry (link 7, the wrist camera, the hand, the fingers; separating-axis
    test): at the pressed depth and along the vertical descent from 60 mm above, clear of every other keycap, the
    keypad case, the host display, the shelf and the stock by `margin`, with one fingertip corner leading every other
    finger point by `lead_m`. Candidates: the approach tilted 20-40 deg from straight down towards 12 azimuths, the
    fingers turned about it in 15 deg steps. Returns (quat, tip offset in the
    TCP frame, margin, tilt) for the collision-free candidates, the largest margin and least tilt first.
    (A fixed tilt pressed the middle keys but put the wrist camera on the shelf's front beam over the top row and link
    7 on the host display over the left column.)"""
    from . import grasping as G
    from .scripted import hand_quat
    w = op.w
    kb = w.scene.keys[key]["body"]
    pk, _ = w.body_pose(kb)
    top = pk + np.array([0., 0., .004])
    boxes, _ = _carried_boxes(op, arm)
    corners = _finger_points_tcp(op, arm)     # the same points fingertip() tracks (box corners, mesh vertices)
    base_xy = np.asarray(w.scene.arm_hw[arm]["mount"][0], float)
    obs = G.obstacles(w, exclude_bodies={kb}, centre=top, radius=.35)
    out = []
    down = np.array([0., 0., -1.])
    for tilt in tilts:
        th = math.radians(tilt)
        for az in np.radians(np.arange(0, 360, 30)):
            lead = np.array([math.cos(az), math.sin(az), 0.])
            approach = down*math.cos(th) + lead*math.sin(th)
            u = np.cross(approach, (0., 0., 1.)); u /= np.linalg.norm(u); v = np.cross(approach, u)
            for roll in np.radians(np.arange(0, 360, 15)):
                fingers = math.cos(roll)*u + math.sin(roll)*v
                q = hand_quat(approach, fingers)
                R = Rotation.from_quat(q).as_matrix()
                z = (corners @ R.T)[:, 2]
                k0 = int(np.argmin(z))
                tip = corners[k0]
                # one corner must lead: every finger point more than 5 mm from it at least `lead_m` higher (a level
                # finger edge pressed two keys, and the measured lowest point flipped between its ends)
                away = np.linalg.norm(corners - tip, axis=1) > .005
                if away.any() and float(np.min(z[away]) - z[k0]) < lead_m:
                    continue
                for mg in margins:
                    ok = True
                    for dz in (-depth_m, .01, .03, .06):
                        p_tcp = top + np.array([0., 0., dz]) - R @ tip
                        if _boxes_collide(p_tcp, R, boxes, obs, mg):
                            ok = False
                            break
                    if ok:
                        # how well the wrist points back towards the cart (-y), between the host display on the
                        # keypad's left and the cradle on its right: towards either arm's base it hung over one of them
                        # (left wrist on the display, right arm's links on the cradled scanner)
                        wrist = -approach[:2]/max(np.linalg.norm(approach[:2]), 1e-9)
                        home = np.array([0., -1.])
                        out.append(dict(quat=q, tip=tip, margin=mg, tilt=tilt, az_deg=math.degrees(az), roll_deg=math.degrees(roll),
                                        homeward=float(wrist @ home)))
                        break
    out.sort(key=lambda c: (-c["margin"], -round(c["homeward"], 1), c["tilt"]))
    return out


def _boxes_collide(p_tcp, R_tcp, boxes, obs, margin):
    from .grasping import _obb_overlap
    for c, Rb, h in boxes:
        cw = p_tcp + R_tcp @ c
        Rw = R_tcp @ Rb
        for oc, oR, oh in obs:
            if np.linalg.norm(cw - oc) > np.linalg.norm(h) + np.linalg.norm(oh) + margin:
                continue
            if _obb_overlap(cw, Rw, h, oc, oR, oh, margin):
                return True
    return False


def held(op, item):
    """Bodies of a held item (a book with its cover)."""
    it = op.w.scene.items[item]
    return tuple(b for b in (it["body"], it.get("cover_body")) if b is not None)


def plan_joint_path(op, arm, position, quat, seconds=1.5, ignore=(), carried=(), restarts=16, retreat=True, fingers=None):
    """Joint-space path (smooth-step, 10 ms waypoints) to an IK goal for `position`/`quat`, the goal chosen among IK
    solutions (nearest the joints now first) whose whole path is clear of everything by MuJoCo's collision test
    (collide.Checker), the `carried` item moving with the hand; `ignore` bodies may be touched. If no direct path is
    clear, the same via a retreat: straight up 10 cm and 10 cm back towards the robot first. None if nothing is clear
    (the last collision found is left in op.last_block)."""
    chain, names = op.k.arms[arm]
    now = np.array(op.k.joints_now(arm), float)
    from . import params as P_
    rng = np.random.default_rng(11)
    seeds = [now, np.array(P_.READY_POSE[:7])] + [rng.uniform(chain.lower, chain.upper) for _ in range(restarts)]
    sols = []
    for s0 in seeds:
        try:
            sol = op.k.solve(arm, position, quat, seed=s0, tol_m=2e-3, tol_rad=.03)
        except ValueError:
            continue
        v = np.array([sol[n] for n in names])
        if all(np.abs(v - x).max() > 1e-3 for x in sols):
            sols.append(v)
    sols.sort(key=lambda v: float(np.abs(v - now).max()))
    n = max(1, int(round(seconds/CHUNK_S)))
    ramp = lambda a, b, k: [a + (b - a)*(3*t*t - 2*t*t*t) for t in np.linspace(0, 1, k + 1)[1:]]
    op.last_block = None
    starts = [(now, [])]
    if retreat:
        # via points: straight up 10 cm; or back through the arm's ready pose (a joint-space detour around the mast
        # and the other arm: a direct joint path from beside the cradle to the counter swept link 3 through the camera
        # mast)
        pt, Rt = op.tcp(arm)
        try:
            via = _solve_path(op, arm, _cart_steps(pt, Rt, pt + np.array([0., 0., .10]), Rt, step=.01), now)
            if checker(op).path_clear(arm, list(names), via, ignore_bodies=ignore, carried=carried, fingers=fingers, every=1) is None:
                starts.append((np.asarray(via[-1]), via))
        except ValueError:
            pass
        ready = np.clip(np.array(P_.READY_POSE[:7]), chain.lower, chain.upper)
        via = ramp(now, ready, max(20, n))
        if checker(op).path_clear(arm, list(names), via, ignore_bodies=ignore, carried=carried, fingers=fingers, every=3) is None:
            starts.append((ready, via))
    op.plan_log = [dict(solutions=len(sols), starts=len(starts))]
    for k, (start, pre) in enumerate(starts):
        for i, goal in enumerate(sols):
            path = list(pre) + ramp(start, goal, n)
            hit = checker(op).path_clear(arm, list(names), path, ignore_bodies=ignore, carried=carried, fingers=fingers, every=3)
            if hit is None:
                return path
            op.last_block = hit[1][:2]
            op.plan_log.append(dict(start=k, goal=i, at=f"{hit[0]}/{len(path)}", hit=hit[1][:2]))
    return None


SPEED_FRACTION = .5     # planned joint paths are timed so that no joint exceeds this fraction of its datasheet speed


def _min_seconds(op, arm, P_):
    """The shortest time for the joint path P_ (rows) with a smooth-step profile (peak 1.5x the mean rate) keeping
    every joint under SPEED_FRACTION of its datasheet limit: the world slews targets at the limits, so a faster plan
    is followed on a different path from the one that was checked (the left wrist met the display)."""
    names = op.k.arms[arm][1]
    vmax = np.array([op.w.joints[n]["velocity_limit"] for n in names], float)
    travel = np.abs(np.diff(P_, axis=0)).sum(axis=0)
    return float(np.max(1.5*travel/(SPEED_FRACTION*vmax)))


def run_joint_path(op, arm, path, seconds, start=None):
    """Execute a planned (and checked) joint path, re-timed to `seconds` (or longer, see _min_seconds) with a
    smooth-step profile along its arc length in joint space (10 ms waypoints): the joints sent are the ones that were
    checked, not a re-solve."""
    names = op.k.arms[arm][1]
    P_ = np.asarray([np.asarray(op.k.joints_now(arm), float) if start is None else np.asarray(start, float)] + [np.asarray(q, float) for q in path])
    seg = np.linalg.norm(np.diff(P_, axis=0), axis=1)
    L = np.concatenate([[0.], np.cumsum(seg)])
    if L[-1] < 1e-9:
        return
    seconds = max(seconds, _min_seconds(op, arm, P_))
    n = max(1, int(round(seconds/CHUNK_S)))
    for i in range(1, n + 1):
        a = i/n; a = 3*a*a - 2*a*a*a
        q = np.array([np.interp(a*L[-1], L, P_[:, j]) for j in range(P_.shape[1])])
        op.set_targets(dict(zip(names, map(float, q))))
        op._chunk()


def run_path(op, arm, path):
    """Execute a planned joint path (10 ms waypoints), slowed uniformly if any joint would exceed SPEED_FRACTION of
    its datasheet speed (see _min_seconds)."""
    names = op.k.arms[arm][1]
    P_ = np.asarray([np.asarray(op.k.joints_now(arm), float)] + [np.asarray(q, float) for q in path])
    planned = len(path)*CHUNK_S
    need = _min_seconds(op, arm, P_) if len(path) else 0.
    if need > planned*1.001:
        run_joint_path(op, arm, path, need, start=P_[0])
        return
    for qj in path:
        op.set_targets(dict(zip(names, map(float, qj))))
        op._chunk()


def checked_move_joints(op, arm, position, quat, seconds=1.5, ignore=(), carried=(), restarts=16, what="the move"):
    """op.move_joints, collision-checked (plan_joint_path); raises ValueError before moving if no path is clear."""
    path = plan_joint_path(op, arm, position, quat, seconds, ignore, carried, restarts)
    if path is None:
        raise ValueError(f"no collision-free joint path for {what} (e.g. {getattr(op, 'last_block', None)})")
    run_path(op, arm, path)
    return path[-1]


def checked_move(op, arm, position, quat=None, seconds=1.0, ignore=(), carried=(), what="the move", ignore_labels=()):
    """A straight-line move, planned (IK waypoints every 1 cm / 5 deg, continuous), checked with MuJoCo's collision
    test and executed as that joint path (run_joint_path). If the straight line cannot be solved near the edge of the
    workspace (a waypoint a few mm out of tolerance), a checked joint-space move to the same pose instead. Raises
    before moving if neither is clear."""
    chain, names = op.k.arms[arm]
    p0, r0 = op.tcp(arm)
    r1 = Rotation.from_quat(quat) if quat is not None else r0
    poses = _cart_steps(p0, r0, np.asarray(position, float), r1, step=.01, ang=math.radians(5.))
    try:
        path = _solve_path(op, arm, poses, np.array(op.k.joints_now(arm), float))
    except ValueError:
        path = None
    if path is not None:
        hit = checker(op).path_clear(arm, list(names), path, ignore_bodies=ignore, carried=carried, every=1,
                                     ignore_labels=ignore_labels)
        if hit is None:
            run_joint_path(op, arm, path, seconds)
            return
        op.last_block = hit[1][:2]
    jpath = plan_joint_path(op, arm, np.asarray(position, float), r1.as_quat(), seconds=seconds, ignore=ignore,
                            carried=carried, retreat=False)
    if jpath is None:
        raise ValueError(f"no collision-free move for {what} (e.g. {getattr(op, 'last_block', None)})")
    run_path(op, arm, jpath)


def checker(op):
    """The operator's whole-robot collision checker (collide.Checker), built once."""
    if getattr(op, "_checker", None) is None:
        from .collide import Checker
        op._checker = Checker(op.w)
    return op._checker


def _solve_path(op, arm, poses, seed, jump=.25):
    """Joint configurations along Cartesian `poses` [(position, quat)], each continuous with the previous (no branch
    jump > `jump` rad); raises ValueError if the IK loses the path."""
    out = []
    for i, (p_, q_) in enumerate(poses):
        _, seed = _ik_near(op, arm, p_, q_, seed, jump, f"at waypoint {i + 1}/{len(poses)}")
        out.append(seed)
    return out


def _path_clear(op, arm, joint_path, boxes, obs, margin=.002):
    """Every collision shape carried past joint 7, along the joint path (TCP by forward kinematics), clears `obs`."""
    chain = op.k.arms[arm][0]
    for qj in joint_path:
        T = chain.fk(np.asarray(qj))
        if _boxes_collide(T[:3, 3], T[:3, :3], boxes, obs, margin):
            return False
    return True


def _cart_steps(p0, r0, p1, r1, step=.005, ang=math.radians(3.)):
    from scipy.spatial.transform import Slerp
    n = max(1, int(np.ceil(max(np.linalg.norm(p1 - p0)/step, (r1*r0.inv()).magnitude()/ang))))
    sl = Slerp([0, 1], Rotation.concatenate([r0, r1]))
    return [(p0 + (p1 - p0)*i/n, sl(i/n).as_quat()) for i in range(1, n + 1)]


MIN_JOINT_MARGIN_RAD = .15     # a planned press configuration keeps every joint this far from its limits


def press_key(op, arm, key, step_m=.0005, max_mm=4., hold_s=.12, plan=None):
    """One fingertip corner of the closed Franka Hand presses keycap `key` (the closed tips are ~30 x 18 mm, the keys
    19 mm apart). The orientation comes from plan_press (collision-checked at the key); the path there is planned
    before moving and checked too (IK continuity, and the hand, camera and link 7 clear of the keypad, display and
    shelf): from over another key the hand rises to 120 mm, turns, crosses and descends in 5 mm / 3 deg steps; from
    further away it moves in joint space. Then closed loop on the measured tip, as a person watches their finger: the
    lowest fingertip point is brought 8 mm over the keycap centre, it descends in `step_m` steps until the key's
    switch closes, holds `hold_s`, and lifts."""
    from . import grasping as G
    w = op.w
    sc = w.scene
    op.set_targets({j: 0. for j in w.gripper_joints(arm)})             # fingers closed, position mode
    # the plan's fingertip corner is taken from the fingers as they are now: wait until they are shut (from fully open
    # they need ~0.8 s at the Franka Hand's 50 mm/s; a 0.3 s wait planned the first key of a sequence with half-open
    # fingers, a corner 12 mm below and 23 mm aside of the real one)
    for _ in range(15):
        op.wait(.1)
        if op.finger_width(arm) < .001:
            break
    op.wait(.2)
    kb = sc.keys[key]["body"]
    pk, _ = w.body_pose(kb)
    top = pk + np.array([0., 0., .004])
    cands = plan if plan is not None else plan_press(op, arm, key)
    if not cands:
        raise ValueError(f"no collision-free hand orientation for the {key} key")
    boxes, _ = _carried_boxes(op, arm)
    obs = G.obstacles(w, exclude_bodies={kb}, centre=top, radius=.45)
    from . import params as P_
    ready = np.array(P_.READY_POSE[:7])
    rng = np.random.default_rng(7)
    pt, Rt = op.tcp(arm)
    now = np.array(op.k.joints_now(arm), float)
    near = np.linalg.norm((pt - top)[:2]) < .15 and 0. < pt[2] - top[2] < .14
    chosen, tried = None, 0
    why = []
    names = list(op.k.arms[arm][1])
    chain = op.k.arms[arm][0]
    # from over the keypad the hand first rises straight up 50 mm (a straight line, checked), so that a joint-space
    # move starts clear of the keycaps
    lift_path = []
    if near:
        try:
            lift_path = _solve_path(op, arm, _cart_steps(pt, Rt, pt + np.array([0., 0., .05]), Rt), now)
        except ValueError:
            lift_path = []
    # prefer configurations well inside the joint limits; accept any only if none is: the keypad stands close to
    # the arm bases, and the right-column keys (3, ENTER) had no plan even at 0.05 rad for either arm
    for min_margin in (MIN_JOINT_MARGIN_RAD, 0.):
        if chosen is not None:
            break
        for c in cands[:30]:
            R = Rotation.from_quat(c["quat"]).as_matrix()
            hover = top + np.array([0., 0., .06]) - R @ c["tip"]
            tried += 1
            options = []
            if near:                                   # a short Cartesian hop from the previous key: up, turn, across, down
                try:
                    safe = top[2] + .12
                    a_ = np.array([pt[0], pt[1], max(pt[2], safe)])
                    b_ = np.array([hover[0], hover[1], max(hover[2], safe)])
                    qc = Rotation.from_quat(c["quat"])
                    poses = (_cart_steps(pt, Rt, a_, Rt) + _cart_steps(a_, Rt, a_, qc) + _cart_steps(a_, qc, b_, qc)
                             + _cart_steps(b_, qc, hover, qc))
                    options.append(_solve_path(op, arm, poses, now))
                except ValueError:
                    pass
            # joint-space moves to several IK solutions (the elbow placed differently), after the lift when near
            start = np.asarray(lift_path[-1]) if lift_path else now
            sols = []
            for s0 in [start, ready] + [ready + rng.normal(0, .6, 7) for _ in range(10)]:
                try:
                    sol = op.k.solve(arm, hover, c["quat"], seed=s0, tol_m=2e-3, tol_rad=.03)
                except ValueError:
                    continue
                v = np.array([sol[n] for n in names])
                if all(np.abs(v - x).max() > .05 for x in sols):
                    sols.append(v)
            sols.sort(key=lambda v: -_joint_margin(op, arm, v))
            for goal in sols[:4]:
                options.append(list(lift_path) + [start + (goal - start)*(3*a*a - 2*a*a*a) for a in np.linspace(0, 1, 41)[1:]])
            if not options:
                why.append("no IK")
            for path in options:
                qc = Rotation.from_quat(c["quat"])
                # a hop from key to key keeps the elbow where it is, and over a sequence of keys the arm drifted to within
                # 0.04-0.1 rad of the joint 1 and 4 limits, where the closed-loop correction jumped IK branches and the press
                # was refused (keys 6, 1, 2, then 3); the ready pose pressed the same key cleanly
                if _joint_margin(op, arm, path[-1]) < min_margin:
                    why.append("near a joint limit"); continue
                try:
                    # the descent to the press depth must stay on the same IK branch (solved in 1 cm steps: a single 7 cm
                    # step moved joints > 0.3 rad legitimately and was refused)
                    _solve_path(op, arm, _cart_steps(hover, qc, hover - np.array([0., 0., .07]), qc, step=.01), path[-1], jump=.15)
                except ValueError:
                    why.append("descent IK"); continue
                if not _path_clear(op, arm, path, boxes, obs):
                    why.append("hand boxes on the way"); continue
                hit = checker(op).path_clear(arm, names, path, ignore_bodies=(kb,))
                if hit is not None:
                    why.append(f"path {hit[1][0][:2]}"); continue      # the whole arm, with MuJoCo's geometry (collide.py)
                # and the press itself: the hand at the pressed depth, whole arm checked, also 2 mm to each side (the
                # closed-loop fingertip correction moves the hand by a few mm and ends within 0.3 mm of the keycap centre;
                # a zero-clearance pose passed the planner and then touched during the correction)
                ok = True
                centred = None
                for off in ((0., 0.), (.002, 0.), (-.002, 0.), (0., .002), (0., -.002)):
                    target = hover + np.array([off[0], off[1], -.064])   # the deepest the press goes: the 4 mm travel stop
                    try:
                        press = _solve_path(op, arm, _cart_steps(hover, qc, target, qc, step=.01), path[-1], jump=.15)
                    except ValueError:
                        ok = False; why.append("press IK"); break
                    # the descent and the pressed pose, the whole arm (with the margin)
                    hits = []
                    for qj in press[::2] + [press[-1]]:
                        hits = _press_hits(op, arm, dict(zip(names, map(float, qj))), kb)
                        if hits:
                            break
                    if hits:
                        ok = False; why.append(f"pressed pose {hits[0][:2]}"); break
                    if centred is None:
                        centred = press
                if not ok:
                    continue
                chosen = (c, hover, path, centred)
                break
            if chosen is not None:
                break
    if chosen is None:
        from collections import Counter
        raise ValueError(f"no reachable, collision-free path to the {key} key ({tried} orientations tried: "
                         f"{Counter(why).most_common(4)})")
    c, hover, path, centred = chosen
    q = c["quat"]
    names = list(op.k.arms[arm][1])
    op.phase = f"ASSISTED scripted: fingertip to the {key} key"
    run_joint_path(op, arm, path, .03*len(path))
    op.wait(.3)
    # the planned descent's joints (hover to the travel stop, 1 cm apart) seed the closed-loop corrections so that the
    # 7-joint arm keeps the checked elbow (re-solved from the current joints, the elbow drifted and link 5 met the
    # cradled scanner)
    guide = [(float(hover[2]) - .01*i, np.asarray(j, float)) for i, j in enumerate([path[-1]] + list(centred))]
    rec = _press_from_hover(op, arm, key, top, q, hover, np.array(path[-1]), step_m, max_mm, hold_s, guide=guide,
                            tip_local=np.asarray(c["tip"], float))
    rec["hand"] = dict(tilt_deg=c["tilt"], azimuth_deg=c["az_deg"], roll_deg=c["roll_deg"], clearance_mm=c["margin"]*1e3,
                       orientations_tried=tried)
    return rec


def press_key_either(op, key, arms, **kw):
    """Press `key` with the first of `arms` that has a collision-free plan; before another arm tries, the arm that
    could not is tucked aside (both hands empty). Returns the record with the arm used and the arms that declined."""
    declined = []
    for i, arm in enumerate(arms):
        try:
            try:
                rec = press_key(op, arm, key, **kw)
                if not rec["closed"]:
                    raise ValueError(f"the {key} key's switch did not close (fingertip {rec['tip_offset_mm']} mm)")
            except ValueError as exc0:
                # once more from the ready pose (a clean start for the IK), if the arm can get there
                _rise_clear(op, arm)
                if not _to_ready(op, arm):
                    raise
                declined.append(dict(arm=arm, why=str(exc0)[:300], retried_from="ready pose"))
                rec = press_key(op, arm, key, **kw)
                if not rec["closed"]:
                    raise ValueError(f"the {key} key's switch did not close (fingertip {rec['tip_offset_mm']} mm)")
            rec["arm"], rec["declined"] = arm, declined
            return rec
        except ValueError as exc:
            declined.append(dict(arm=arm, why=str(exc)[:300]))
            if i + 1 < len(arms):
                _rise_clear(op, arm)
                try:
                    tuck(op, arm)
                except ValueError as exc2:
                    raise ValueError(f"{exc2} (after the {arm} arm declined the {key} key: {declined[-1]['why']})") from exc
    raise ValueError(f"no arm has a collision-free plan for the {key} key: {declined}")


def _to_ready(op, arm, seconds=2.0):
    """A checked joint-space move to the environment's ready pose (P.READY_POSE); False if its path is not clear."""
    from . import params as P_
    chain, names = op.k.arms[arm]
    now = np.array(op.k.joints_now(arm), float)
    goal = np.clip(np.array(P_.READY_POSE[:7], float), chain.lower, chain.upper)
    n = max(1, int(round(seconds/CHUNK_S)))
    path = [now + (goal - now)*(3*a*a - 2*a*a*a) for a in np.linspace(0, 1, n + 1)[1:]]
    if checker(op).path_clear(arm, list(names), path, every=3) is not None:
        return False
    run_path(op, arm, path)
    return True


def _swing_other_arm_aside(op, arm):
    """The other arm swings out of the shared workspace (tuck); True if it moved. For a plan that found no clear path:
    parked wherever its last task left it, the other arm's elbow blocked the grasp of a basket item, a set-down spot
    and the way back to the cradle. Only the plan is retried, never a motion."""
    others = [a for a in op.w.scene.arm_hw if a != arm]
    if not others:
        return False
    try:
        op.phase = "ASSISTED scripted: the other arm swings aside"
        tuck(op, others[0])
        return True
    except ValueError:
        return False


def _rise_clear(op, arm, dz=.05):
    """A hand stopped over the keypad (a press refused part-way) rises straight up `dz` first, checked for touching
    only: a tuck straight from there swung link 7 into the display's edge)."""
    pt, Rt = op.tcp(arm)
    now = np.array(op.k.joints_now(arm), float)
    try:
        up = _solve_path(op, arm, _cart_steps(pt, Rt, pt + np.array([0., 0., dz]), Rt), now)
    except ValueError:
        return False
    if checker(op).path_clear(arm, list(op.k.arms[arm][1]), up, clearance=0., every=1) is not None:
        return False
    run_path(op, arm, up)
    return True


def _press_hits(op, arm, joints, kb, margin=True):
    """Contacts of a pressing pose other than with the pressed key: touching anything (the other keycaps sit 2.5 mm
    apart and are only ever checked for touching), and when planning (`margin`) anything else within the checker's
    clearance; during execution only touching counts (the clearance is a planning allowance for tracking error: a
    correction step 2.1 mm from the cradled scanner was refused although nothing touched)."""
    return checker(op).contacts(arm, joints, ignore_bodies=(kb,), clearance=None if margin else 0.)


def _touching(w, body):
    """Shape labels in contact with `body` now (MuJoCo's contact list)."""
    d = w.solver.mj_data
    labels = w.scene.b.shape_label
    sb = w.scene.b.shape_body
    out = set()
    for i in range(d.ncon):
        s_ = [int(w._geom_shape[x]) if 0 <= x < len(w._geom_shape) else -1 for x in d.contact[i].geom]
        b_ = [int(sb[x]) if x >= 0 else -1 for x in s_]
        if body in b_:
            other = s_[1] if b_[0] == body else s_[0]
            out.add(labels[other] if other >= 0 else "world")
    return sorted(out)


def _joint_margin(op, arm, joints):
    chain = op.k.arms[arm][0]
    return float(np.min(np.minimum(np.asarray(joints) - chain.lower, chain.upper - np.asarray(joints))))


def _ik_near(op, arm, target, q, seed, jump, what):
    """IK for `target` continuous with `seed`: the solver from the seed, else from 8 seeds jittered by 0.05 rad around
    it, the solution nearest the seed; refused if every one is more than `jump` rad from it."""
    best = None
    rng = np.random.default_rng(3)
    for s0 in [seed] + [seed + rng.normal(0, .05, len(seed)) for _ in range(8)]:
        try:
            t = op.k.solve(arm, target, q, seed=s0, tol_m=2e-3, tol_rad=.03)
        except ValueError:
            continue
        jt = np.array(list(t.values()))
        d = float(np.abs(jt - seed).max())
        if best is None or d < best[0]:
            best = (d, t, jt)
        if d <= jump:
            break
    if best is None:
        raise ValueError(f"IK unreachable {what}")
    if best[0] > jump:
        raise ValueError(f"IK branch jump {what}: {best[0]:.2f} rad")
    return best[1], best[2]


def _press_from_hover(op, arm, key, top, q, hover, seed, step_m, max_mm, hold_s, guide=None, tip_local=None):
    w = op.w

    def tip_now():
        # the planned fingertip corner itself (fixed in the closed hand), located from the measured hand pose, as a
        # person watches the finger they press with; the lowest point of the fingers (the earlier rule) depends on
        # every other corner staying higher, which a planning error or a few degrees of tracking error can break
        if tip_local is None:
            return fingertip(op, arm)
        pt, Rt = op.tcp(arm)
        return pt + Rt.apply(tip_local)
    tcp_t = hover.copy()

    kb = w.scene.keys[key]["body"]

    def go(target, settle, jump=.15):
        nonlocal seed
        start = seed
        if guide:
            # seed from the planned joints at this height, continuity still measured against the joints now
            zs = np.array([g_[0] for g_ in guide])
            start = guide[int(np.argmin(np.abs(zs - float(target[2]))))][1]
        t, jt = _ik_near(op, arm, target, q, start, jump + .1 if guide else jump, f"pressing {key}")
        if np.abs(jt - seed).max() > jump + .1:
            raise ValueError(f"IK branch jump pressing {key}: {np.abs(jt - seed).max():.2f} rad")
        hits = _press_hits(op, arm, t, kb, margin=False)
        if hits:
            raise ValueError(f"collision predicted pressing {key} at {np.round((target - hover)*1e3, 1).tolist()} mm "
                             f"from the hover pose: {hits[:2]}")
        seed = jt
        op.set_targets(t)
        op.wait(settle)
    # bring the measured fingertip 8 mm above the keycap centre, in <= 3 mm Cartesian steps
    trace = []
    for _ in range(4):
        tip = tip_now()
        err = (top + np.array([0., 0., .008])) - tip
        pt_now, _ = op.tcp(arm)
        trace.append(dict(err_mm=np.round(err*1e3, 1).tolist(), tcp_lag_mm=round(float(np.linalg.norm(pt_now - tcp_t)*1e3), 1),
                          tip_from_tcp_mm=np.round((tip - pt_now)*1e3, 1).tolist(),
                          lowest_finger_point_from_tip_mm=np.round((fingertip(op, arm) - tip)*1e3, 1).tolist()))
        n = max(1, int(np.ceil(np.linalg.norm(err)/.003)))
        for i in range(1, n + 1):
            go(tcp_t + err*i/n, .35 if i == n else .03)
        tcp_t = tcp_t + err
        if np.linalg.norm(err) < .0015:
            break
    tip = tip_now()
    rec = dict(key=key, tip_offset_mm=np.round((tip - top)*1e3, 1).tolist(), approach_trace=trace)
    op.phase = f"ASSISTED scripted: press {key}"
    sw = w.key_switch[key]
    closed_at = None
    for k in range(1, int((8 + max_mm)/(step_m*1e3)) + 1):
        go(tcp_t - np.array([0., 0., k*step_m]), .04)
        if sw.state:
            closed_at = float((top[2] - tip_now()[2])*1e3)
            break
    op.wait(hold_s)
    others = [k_ for k_, s_ in w.key_switch.items() if k_ != key and s_.state]
    rec["touching"] = {k_: _touching(w, w.scene.keys[k_]["body"]) for k_ in [key] + others}
    op.phase = f"ASSISTED scripted: release {key}"
    go(tcp_t + np.array([0., 0., .02]), .3)
    rec.update(closed=closed_at is not None, tip_below_key_top_mm=closed_at, other_keys_down=others)
    return rec


def phantom_present(op, fx, body, label, centre, normal, hold_up=.25, seconds=1.6, name="the unit"):
    """ASSISTED phantom: lift a unit straight up by `hold_up` m, then hold it with its barcode (label) centre at
    `centre` facing `normal` (keeping the unit as upright as that allows). Returns the hold (to put the unit back)."""
    w = op.w
    b = w.scene.bodies[body] if isinstance(body, str) else body
    p0, R0 = w.body_pose(b)
    h = fx.hold(body, kp=400., kr=2.)
    h.origin = (p0.copy(), R0.copy())
    op.phase = f"ASSISTED phantom takes {name} up"
    h.move(p0 + np.array([0., 0., hold_up]), Rotation.from_matrix(R0).as_quat(), 1.2)
    op.wait(1.3)
    n_item, c_item = np.asarray(label["normal"]), np.asarray(label["center"])
    d = np.asarray(normal, float); d /= np.linalg.norm(d)
    Rn = Rotation.align_vectors([d], [Rotation.from_matrix(R0).apply(n_item)])[0]*Rotation.from_matrix(R0)
    p_goal = np.asarray(centre, float) - Rn.apply(c_item)
    op.phase = f"ASSISTED phantom turns {name}'s barcode towards the scanner"
    h.move(p_goal, Rn.as_quat(), seconds)
    op.wait(seconds + .4)
    return h


def phantom_put_back(op, fx, h, seconds=1.8, name="the unit"):
    p0, R0 = h.origin
    op.phase = f"ASSISTED phantom puts {name} back"
    h.move(p0 + np.array([0., 0., .20]), Rotation.from_matrix(R0).as_quat(), seconds)
    op.wait(seconds + .2)
    h.move(p0 + np.array([0., 0., .003]), Rotation.from_matrix(R0).as_quat(), 1.2)
    op.wait(1.5)
    h.enabled = False
    op.wait(.5)


# ------------------------------------------------------------------------------------------ items: take, present, drop
HANDOVER = np.array([-.22, -.04, 1.13])       # where a customer / patron holds an item up (counter and library desk)


def upright(sku):
    """Orientation in which an item is held up: cylinders axis vertical (cap up), boxes and books standing on their
    bottom face (catalogue height vertical), narrow side towards the robot."""
    if sku.shape == "box":
        w_, d_, h_ = sku.dims
        return Rotation.from_euler("z", 0. if d_ <= w_ else math.pi/2)
    return Rotation.identity()


def take_item(op, fx, arm, item, rec, force=None, open_cover_deg=None):
    """The robot grasps `item` where it lies if grasping.plan finds a collision-free grasp; otherwise a labelled
    phantom (the customer or patron) lifts it out and holds it up at HANDOVER first, and lets go once the fingers
    have closed. Returns the grasp used."""
    from . import grasping as G
    from .catalog import SKUS
    w = op.w
    it = w.scene.items[item]; s = SKUS[it["sku"]]
    shut = (it["cover_body"],) if it.get("cover_body") is not None and open_cover_deg is None else ()
    plans = G.plan(w, op.k, arm, item, s, extra_exclude=shut)     # a shut cover is closed on with the block
    rec["grasps_where_it_lies"] = len(plans)
    hold = None
    if not plans:
        # the phantom (a customer, a patron) only hands over what is within its reach: an item on the floor behind the
        # counter stays there (without this rule the "customer" picked up a jar the robot had dropped, and the
        # evaluator then found nothing on the floor)
        lowest = float(_item_points(w, item)[:, 2].min())
        if lowest < FLOOR_REACH_M:
            raise RuntimeError(f"no collision-free grasp of the {s.name} where it lies, and it is on the floor "
                               f"(lowest point {lowest*1e3:.0f} mm): no phantom hand-over")
        # straight up until the whole item clears the container's rim, turned upright only up there, across high,
        # then down to the handover point (a lying bottle turned upright while crossing the basket's wall 2 cm above
        # it: its swinging end caught the rim and dragged the basket off the counter)
        p0, R0 = w.body_pose(it["body"])
        reach = float(np.linalg.norm(np.asarray(s.dims[:3] if s.shape == "box" else (2*s.dims[0], s.dims[1]))))/2
        cont = w.scene.items.get(it.get("container")) or {}
        rim = (w.body_pose(cont["body"])[0][2] + cont["container_size"][2]/2) if cont.get("container_size") else p0[2]
        high = max(rim, p0[2]) + reach + .06
        book = it.get("cover_body") is not None
        # a book held up by its block alone: its front cover swung open to 90-100 deg on the spine and turned the book
        # 10 deg in a 2 N m/rad hold, and the planned grasp then landed on the book's top edge. The
        # patron holds the cover shut (phantom torque on the spine) and the hold bears the cover's weight too
        hold = fx.hold(item, kp=400., kr=6. if book else 2., extra=(f"{item}_cover",) if book else ())
        cov = None
        if book:
            cov = fx.cover(item)
            cov.target = 0.
        op.phase = f"ASSISTED phantom lifts the {s.name} out and holds it up"
        q0 = Rotation.from_matrix(R0).as_quat()
        hold.move(np.array([p0[0], p0[1], high]), q0, 1.2)
        op.wait(1.3)
        hold.move(np.array([p0[0], p0[1], high]), upright(s).as_quat(), .8)
        op.wait(.9)
        hp = HANDOVER*np.array([1. if arm.startswith("left") else -1., 1., 1.])    # held up to the hand taking it
        z = _phantom_crossing_height(op, it["body"], p0[:2], hp[:2], upright(s).as_matrix(), max(high, hp[2]))
        rec["handover_crossing_z_m"] = round(z, 3)
        if z > high + 1e-3:
            hold.move(np.array([p0[0], p0[1], z]), upright(s).as_quat(), .6)
            op.wait(.7)
        hold.move(np.array([hp[0], hp[1], z]), upright(s).as_quat(), 1.2)
        op.wait(1.3)
        hold.move(hp, upright(s).as_quat(), max(.8, (z - hp[2])/.25))
        op.wait(max(.8, (z - hp[2])/.25) + .4)
        if cov is not None and open_cover_deg is not None:
            # the patron opens the front cover (its hinge is vertical now: the cover stays where it is put) so that the
            # hand closes on the text block alone and the label inside the cover faces out
            op.phase = f"ASSISTED phantom fingertip opens the cover of {s.name} to {open_cover_deg:g} deg"
            cov.target = math.radians(open_cover_deg)
            op.wait(1.5)
            rec["cover_open_deg"] = math.degrees(cov.angle())
        plans = G.plan(w, op.k, arm, item, s, extra_exclude=shut)
        rec["handed_over"] = True
        rec["handed_over_from"] = dict(z_m=round(float(p0[2]), 3), container=it.get("container") if cont else None)
        # nothing but the item near the fingers in the customer's hand: open fully (a 6 mm clearance per finger landed
        # a finger on the edge of a book that moved 5 mm in the hold)
        for g_ in plans:
            g_["open_half"] = .04
    if not plans:
        raise RuntimeError("no collision-free grasp")
    mine = held(op, item)
    g = path = None
    for attempt in range(2):
        for cand in plans[:12]:
            # the whole arm, not only the hand, must reach the pre-grasp pose and descend without touching anything
            # but the item (collide.Checker), checked with the fingers open as they will be (the hand opens first)
            path = plan_joint_path(op, arm, cand["pre"], cand["quat"], seconds=2.0, ignore=mine, fingers=cand["open_half"])
            if path is None:
                continue
            try:
                poses = _cart_steps(cand["pre"], Rotation.from_quat(cand["quat"]), cand["tcp"], Rotation.from_quat(cand["quat"]), step=.01)
                down = _solve_path(op, arm, poses, np.asarray(path[-1]))
            except ValueError:
                continue
            hit = checker(op).path_clear(arm, list(op.k.arms[arm][1]), down, ignore_bodies=mine, every=1,
                                         fingers=cand["open_half"])
            if hit is None:
                g, g_down = cand, down
                break
            op.last_block = hit[1][:2]
        if g is not None or attempt or not _swing_other_arm_aside(op, arm):
            break
    if g is None:
        raise RuntimeError(f"no grasp of {len(plans)} with a collision-free arm path (e.g. {getattr(op, 'last_block', None)})")
    rec["grasp_tilt_deg"] = g["tilt_deg"]
    force = force or (35. if s.mass > .5 else 15. if s.mass > .05 else 6.)
    op.phase = f"ASSISTED scripted: open to {2*g['open_half']*1e3:.0f} mm over the {s.name}"
    op.open_hand(arm, half_width=g["open_half"])
    run_path(op, arm, path)
    op.phase = f"ASSISTED scripted: approach the {s.name}"
    run_joint_path(op, arm, g_down, 1.2)
    op.phase = f"ASSISTED scripted: grasp {force:g} N per finger"
    op.grasp(arm, force, seconds=.8)
    if hold is not None:
        hold.enabled = False
        if cov is not None and open_cover_deg is None:
            cov.target = None
        op.phase = "ASSISTED scripted: the phantom has let go; the robot holds it"
        op.wait(.3)
    rec["finger_width_mm"] = op.finger_width(arm)*1e3
    pt, Rt = op.tcp(arm)
    op.phase = f"ASSISTED scripted: lift the {s.name}"
    # out of a container: 22 cm, clear of its rim; standing on the counter (checkout-counter) or handed over: 10 cm, and
    # lower if the arm cannot reach that high (a tall item grasped at its top on the counter's far side: the 22 cm lift
    # was out of reach)
    cont_body = it.get("container") and w.scene.items.get(it["container"], {}).get("body")
    for k, dz in enumerate((.10 if hold is not None or cont_body is None else .22, .10, .06)):
        try:
            checked_move(op, arm, pt + (0, 0, dz), Rt.as_quat(), seconds=1.2, carried=mine, ignore=(cont_body,),
                         what=f"lifting the {s.name}")
            rec["lift_m"] = dz
            break
        except ValueError:
            if k == 2:
                raise
    pb, _ = w.body_pose(it["body"])
    pt, _ = op.tcp(arm)
    if np.linalg.norm(pb - pt) > .25:
        raise RuntimeError("the item is not in the hand")
    return g


def present_to_cradle(op, arm, item, label, rec, face="window", standoff=.12, sweep=.10, speed=.20, body=None, tilt_deg=12.,
                      pause_s=.3, clear_view=True):
    """Sweep a held item's barcode (label, in the frame of `body`, default the item) across the cradled scanner's
    window at `standoff` m, choosing the free rotation about the barcode normal for the most downward hand the IK
    reaches at every waypoint."""
    w = op.w
    it = w.scene.items[item]
    b = body if body is not None else it["body"]
    ps, Rs = w.body_pose("scanner")
    look = Rs[:, 0]
    horiz = np.cross(look, (0., 0., 1.)); horiz /= np.linalg.norm(horiz)
    pt, Rt = op.tcp(arm)
    pb, Rb = w.body_pose(b)
    rel_R = Rt.inv()*Rotation.from_matrix(Rb)
    rel_p = Rt.inv().apply(pb - pt)
    n_item, c_item = np.asarray(label["normal"]), np.asarray(label["center"])
    # the barcode faces the window turned `tilt_deg` about the vertical, as a cashier presents it: square to the
    # window, a glossy jar label put the LEDs' glint on the middle of its bars and never decoded
    want = Rotation.from_rotvec(np.array([0., 0., 1.])*math.radians(tilt_deg)).apply(-look if face == "window" else look)
    n_hand = rel_R.apply(n_item)
    cands = []
    for phi in np.linspace(0., 2*math.pi, 72, endpoint=False):
        Rcand = Rotation.from_rotvec(want*phi)*Rotation.align_vectors([want], [n_hand])[0]
        cands.append((-Rcand.apply((0., 0., 1.))[2], Rcand))
    cands.sort(key=lambda c: -c[0])                     # the most downward hand first
    mine = held(op, item)
    names = list(op.k.arms[arm][1])
    centre = ps + standoff*look
    # the item's other barcodes on the same body (a book's ISBN on the cover that carries its library label) stay out of
    # the imager's field of view along the approach and the sweep: the free rotation was chosen for the hand alone, a
    # book's ISBN crossed the window and was read before its library label. Items with one barcode: no effect.
    from . import params as P_
    others = [] if body is not None else [L for L in w.scene.labels if L is not label and L.get("item") == item]
    pupil = ps - P_.PUPIL_SETBACK_M*look
    ax_y, ax_z = Rs[:, 1], Rs[:, 2]
    half_h, half_v = math.radians(P_.HFOV_DEG/2), math.radians(P_.VFOV_DEG/2)

    def others_seen(Rh, tcps):
        for tcp in tcps:
            for L in others:
                o = tcp + Rh.apply(rel_p + rel_R.apply(np.asarray(L["center"], float)))
                n = Rh.apply(rel_R.apply(np.asarray(L["normal"], float)))
                v = o - pupil
                depth = float(v @ look)
                if depth <= 0 or float(n @ -look) < .2:
                    continue
                m = math.atan2(max(L.get("size") or (.03, .03))/2, depth)
                if math.atan2(abs(float(v @ ax_y)), depth) < half_h + m and math.atan2(abs(float(v @ ax_z)), depth) < half_v + m:
                    return True
        return False
    chosen = None
    rejected_for_others = 0
    # the 36 most downward hands at the asked standoff first; then the other 36, then 3 cm farther and 2 cm nearer (in a
    # library run the third book found none of the first 36, and the hand let it fall)
    tries = [(standoff, cands[:36]), (standoff, cands[36:]), (standoff + .03, cands), (standoff - .02, cands)]
    for k_try, (stand, group) in enumerate(tries):
        if chosen is not None:
            break
        centre = ps + stand*look
        if k_try:
            rec["presentation_fallback"] = dict(standoff_m=round(stand, 3), rotations="other 36" if k_try == 1 else "all 72")
        chosen = _presentation_search(op, arm, group, centre, sweep, horiz, rel_p, rel_R, c_item, others, others_seen, mine,
                                      names, rec)
    if chosen is None:
        raise RuntimeError("no reachable, collision-free presentation")
    down, Rh, start, end, path, rest, k_in = chosen
    rejected_for_others = rec.pop("_rejected_for_others", 0)
    rec["presentation_hand_from_down_deg"] = math.degrees(math.acos(max(-1., min(1., down))))
    if others:
        rec["other_barcodes_kept_out_of_view"] = dict(labels=[L.get("kind") for L in others], rotations_rejected=rejected_for_others)
    return _present_run(op, arm, item, rec, face, speed, pause_s, clear_view, w, b, ps, look, horiz, chosen, sweep, mine)


def _presentation_search(op, arm, group, centre, sweep, horiz, rel_p, rel_R, c_item, others, others_seen, mine, names, rec):
    chosen = None
    for down, Rh in group:
        tcp_for = lambda c: c - Rh.apply(rel_p + rel_R.apply(c_item))
        start, end = tcp_for(centre - sweep*horiz), tcp_for(centre + sweep*horiz)
        if others and others_seen(Rh, [start + np.array([0., 0., .08])*f for f in np.linspace(1., 0., 5)]
                                  + [start + (end - start)*f for f in np.linspace(0., 1., 17)]):
            rec["_rejected_for_others"] = rec.get("_rejected_for_others", 0) + 1
            continue
        try:
            for tp_ in (start + (0, 0, .08), start, end):
                op.k.solve(arm, tp_, Rh.as_quat(), restarts=4)
        except Exception:
            continue
        # the whole arm and the item it carries clear of the scanner, the cradle and everything else along the
        # approach and the sweep (collide.Checker; an unchecked presentation put the wrist camera's bracket on the
        # arm's own link 5)
        path = plan_joint_path(op, arm, start + np.array([0., 0., .08]), Rh.as_quat(), seconds=2.2, carried=mine)
        if path is None:
            continue
        # ... and along the turn towards the scanner, the joint path to that start: with the approach and the sweep
        # clear, a book's ISBN still came into a corner of the frame on the turn and was read in one of two renders of
        # the same motion
        if others and any(others_seen(Rotation.from_matrix(T[:3, :3]), [T[:3, 3]])
                          for T in (op.k.forward(arm, np.asarray(q)) for q in list(path[::5]) + [path[-1]])):
            rec["_rejected_for_others"] = rec.get("_rejected_for_others", 0) + 1
            continue
        try:
            poses = (_cart_steps(start + np.array([0., 0., .08]), Rh, start, Rh, step=.01)
                     + _cart_steps(start, Rh, end, Rh, step=.01))
            rest = _solve_path(op, arm, poses, np.asarray(path[-1]))
        except ValueError:
            continue
        if checker(op).path_clear(arm, names, rest, carried=mine, every=1) is not None:
            continue
        chosen = (down, Rh, start, end, path, rest, len(_cart_steps(start + np.array([0., 0., .08]), Rh, start, Rh, step=.01)))
        break
    return chosen


def _present_run(op, arm, item, rec, face, speed, pause_s, clear_view, w, b, ps, look, horiz, chosen, sweep, mine):
    down, Rh, start, end, path, rest, k_in = chosen
    op.phase = "ASSISTED scripted: turn the barcode towards the scanner"
    run_path(op, arm, path)
    run_joint_path(op, arm, rest[:k_in], 1.0)
    op.phase = f"ASSISTED scripted: sweep across the window at {speed:g} m/s"
    # the sweep at constant speed along the checked path (a smooth-step profile would peak at 1.5x the pass speed)
    names = op.k.arms[arm][1]
    sweep_path = rest[k_in:]
    n = max(1, int(round(max(.1, 2*sweep/speed)/CHUNK_S)))
    P_ = np.asarray([rest[k_in - 1]] + list(sweep_path), float)
    L = np.concatenate([[0.], np.cumsum(np.linalg.norm(np.diff(P_, axis=0), axis=1))])
    paused = pause_s <= 0.
    for i in range(1, n + 1):
        q = np.array([np.interp(i/n*L[-1], L, P_[:, j]) for j in range(P_.shape[1])])
        op.set_targets(dict(zip(names, map(float, q))))
        op._chunk()
        if not paused and i >= n//2:
            # held still in front of the window for a moment, as a cashier does with a small item (a lip balm swept
            # through at 0.2 m/s was in view for four frames, fewer than the auto-exposure needed)
            paused = True
            op.phase = f"ASSISTED scripted: hold it at the window {pause_s:g} s"
            op.wait(pause_s)
            op.phase = f"ASSISTED scripted: sweep across the window at {speed:g} m/s"
    op.wait(.4)
    if clear_view:
        rec["cleared_view"] = _clear_view(op, arm, mine, ps, look)


def _clear_view(op, arm, mine, ps, look):
    """After a presentation, carry the item out of the docked scanner's view before turning it for the set-down: up
    above the window's field of view, then back behind the window's plane (the window looks out horizontally; nothing
    behind it is seen). A battery pack turned in front of the window after its sweep showed its barcode again 1.8 s
    later and was sold twice: the same-symbol timeout runs from the last decode, as on the real scanner."""
    look_h = np.array([look[0], look[1], 0.])
    look_h /= max(np.linalg.norm(look_h), 1e-9)
    op.phase = "ASSISTED scripted: lift the item out of the scanner's view"
    for rise in (.18, .14, .10):
        pt, Rt = op.tcp(arm)
        try:
            checked_move(op, arm, pt + np.array([0., 0., rise]), Rt.as_quat(), seconds=.9, carried=mine, what="lifting clear")
            break
        except ValueError:
            continue
    else:
        return False
    pt, Rt = op.tcp(arm)
    ahead = float((pt - ps) @ look_h)                 # the hand's distance in front of the window plane
    for back in (.06, .03):
        try:
            checked_move(op, arm, pt - (ahead + back)*look_h, Rt.as_quat(), seconds=1.0, carried=mine,
                         what="carrying it behind the window")
            return True
        except ValueError:
            continue
    return "lifted only"


def drop_into(op, arm, item, container, rec, _retried=False):
    """Lower the held item into the container and let go: its lowest point 2 cm above the container's floor, or as
    low as a collision-free path allows (6, 10 cm, then just over the rim), hand forward, down or as held. (Let go
    over the rim, a 235 mm book toppled over the 160 mm bin's wall.)"""
    from . import grasping as G
    w = op.w
    it = w.scene.items[item]; c = w.scene.items[container]
    if not in_hand(op, arm, item):
        raise RuntimeError(f"{item} is no longer in the hand (finger gap {op.finger_width(arm)*1e3:.1f} mm)")
    tp = w.body_pose(c["body"])[0]
    L, Wd, H = c["container_size"]
    floor = tp[2] - H/2 + .004                      # inner floor (the wall/floor thickness of stock._open_box)
    rim = tp[2] + H/2
    pt, Rt = op.tcp(arm)
    mine = held(op, item)
    names = list(op.k.arms[arm][1])
    # the held item's collision corners in the TCP frame (it keeps its pose in the hand)
    Ri = Rt.inv().as_matrix()
    corners = np.array([Ri @ (cc + R_ @ (np.array([a, b, d])*h) - pt) for cc, R_, h in G.obstacles(w, only_bodies=set(mine))
                        for a in (-1, 1) for b in (-1, 1) for d in (-1, 1)])
    op.phase = f"ASSISTED scripted: carry it to the {container.replace('_', ' ')}"
    if not _retried:
        checked_move(op, arm, pt + (0, 0, .08), None, seconds=.8, carried=mine, what="the lift")
    chosen = None
    # release spots: the four first tried (towards the robot's side of the container), at every release height and hand
    # orientation; only if none is reachable, a grid over the rest of its floor (the bulky checkout's tote, moved 40 mm
    # for the water case, had no reachable spot among the four). With the grid in every height's inner loop,
    # a grid spot 20 mm above the floor beat the four at 176 mm: the library check-in's carry down to it pressed the
    # book against the arm's own link (a protective stop) and a book's ISBN was read before its library label
    first = [(-.05, -.02), (-.08, -.06), (0., -.06), (-.10, 0.)]
    grid = [(dx, dy) for dy in (-.08, 0., .08) for dx in (-.12, -.04, .04, .12)
            if abs(dx) <= L/2 - .06 and abs(dy) <= Wd/2 - .05 and (dx, dy) not in first]
    for spots in (first, grid):
        for gap in (.02, .06, .10, rim - floor + .02):
            for qd in (hand_quat((0., 1., 0.), (1., 0., 0.)), hand_quat((0., 0., -1.), (1., 0., 0.)), Rt.as_quat()):
                low = float(np.min((corners @ Rotation.from_quat(qd).as_matrix().T)[:, 2]))
                for dx, dy in spots:
                    cand = np.array([tp[0] + dx, tp[1] + dy, floor + gap - low])
                    above = np.array([cand[0], cand[1], max(cand[2] + .06, rim - low + .05)])
                    # joint-space to above the container (a straight Cartesian descent over the tote jumped IK branches),
                    # then straight down; the whole arm and the item it carries are checked
                    path = plan_joint_path(op, arm, above, qd, seconds=2.4, carried=mine)
                    if path is None:
                        continue
                    try:
                        down = _solve_path(op, arm, _cart_steps(above, Rotation.from_quat(qd), cand, Rotation.from_quat(qd), step=.01),
                                           np.asarray(path[-1]))
                    except ValueError:
                        continue
                    if checker(op).path_clear(arm, names, down, carried=mine, every=1) is not None:
                        continue
                    chosen = (cand, above, qd, path, gap, down)
                    rec["release_spot_mm"] = dict(dx=round(dx*1e3), dy=round(dy*1e3), from_grid=spots is grid)
                    break
                if chosen is not None:
                    break
            if chosen is not None:
                break
        if chosen is not None:
            break
    if chosen is None and not _retried and _swing_other_arm_aside(op, arm):
        return drop_into(op, arm, item, container, rec, _retried=True)
    if chosen is None:
        raise RuntimeError("no reachable, collision-free release pose in the container")
    drop, above, q_drop, path, gap, down = chosen
    rec["release_height_above_floor_mm"] = round(gap*1e3, 1)
    run_path(op, arm, path)
    run_joint_path(op, arm, down, 1.2)
    op.phase = "ASSISTED scripted: let go in the container"
    op.open_hand(arm, seconds=.6)
    for target in (above + (0, 0, .04), drop + (0, 0, .10), drop + (0, 0, .05)):
        try:
            checked_move(op, arm, target, q_drop, seconds=1.0, ignore=mine, what="leaving the container")
            break
        except ValueError:
            continue
    else:
        park(op, arm)
    op.wait(.6)
    q = w.state.body_q.numpy()
    rec["in_container"] = bool(op.env.evaluator._in_container(q, container, it["body"]))


HANDOFF = np.array([.26, .045])   # on the counter (its front edge is y = 0) in front of the tote, clear of the scanner


# the counter's front strip (its edge is y = 0), clear of the scanner's field of view (x 0.03-0.09 at this distance)
# and of the cradle: right of the scanner in front of the tote, then left of it towards the basket
HANDOFF_SPOTS = [np.array([x, .045]) for x in (.16, .20, .24, -.03, -.07, .28)]
# tried only when no spot of the first row works: 4 cm further in (the 180 mm battery pack, set down
# on its edge, found none of the first row clear of the basket's wall and the arm's own links, 72 tries)
HANDOFF_SPOTS_MORE = [np.array([x, .085]) for x in (.16, .20, .24, .28, -.03)]
EDGE_MARGIN_M = .015            # a round item set down keeps its footprint this far inside the counter's front edge
FLOOR_REACH_M = .30             # a phantom customer or patron does not reach below this height (the floor behind the counter)
CROSSING_MARGIN_M = .03         # a phantom's carry clears what it crosses by this much (its hold lags up to 15 mm)


def _phantom_crossing_height(op, body, a, b, R, z0, step=.01, top=1.8):
    """Height (>= z0) at which a phantom carries the free `body`, held at R, straight across from a to b (xy): the
    lowest at which the whole crossing (1 cm samples) touches nothing (collide.Checker), plus CROSSING_MARGIN_M. The
    customer or patron lifts an item over what stands in the way. A fixed height (the container's rim + the item's
    half-diagonal + 6 cm) let a patron's upright book sweep through the cradled scanner's head and knock it out of its
    seat (contacts scanner_head / book_2 for 0.4 s, the scanner tipped 5 deg)."""
    chk = checker(op)
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(math.ceil(np.linalg.norm(b - a)/step)) + 1)
    z = float(z0)
    while z <= top:
        if not any(chk.free_body_contacts(body, (*(a + f*(b - a)), z), R) for f in np.linspace(0., 1., n)):
            return z + CROSSING_MARGIN_M
        z += step
    raise RuntimeError(f"no clear crossing below {top} m for the phantom's carry")


def _item_points(w, item):
    """World points of the item's collision shapes (hull vertices / box corners)."""
    from .probes.e19_drop import _shape_points
    b = w.scene.items[item]["body"]
    return _shape_points(w, [b])


def in_hand(op, arm, item):
    """Is the item still between this hand's fingers (fingers not closed on nothing, item within 25 cm of the TCP)?"""
    pb, _ = op.w.body_pose(op.w.scene.items[item]["body"])
    pt, _ = op.tcp(arm)
    return op.finger_width(arm) > .002 and float(np.linalg.norm(pb - pt)) < .25


def put_down(op, arm, item, rec, spot=None, _retried=False):
    """Set the held item down on the counter at `spot` (default: the first of HANDOFF_SPOTS the whole arm reaches
    along a collision-free path; a fixed spot at x = 0.26 m had the left arm's elbow in the camera mast),
    stable and graspable again from above: a cylinder upright, a box resting on the face spanned by
    its grasped (narrow) axis and its longest axis. The hand turns the item about the grasp; the item is lowered to
    5 mm above the counter and let go."""
    from .catalog import SKUS
    w = op.w
    it = w.scene.items[item]; s = SKUS[it["sku"]]
    if not in_hand(op, arm, item):
        raise RuntimeError(f"the {s.name} is no longer in the hand (finger gap {op.finger_width(arm)*1e3:.1f} mm)")
    top = w.scene.fixtures["counter"]["top"]
    pt, Rt = op.tcp(arm)
    pb, Rb = w.body_pose(it["body"])
    rel_R = Rt.inv()*Rotation.from_matrix(Rb)
    rel_p = Rt.inv().apply(pb - pt)
    if s.shape == "box":
        dims = np.asarray(s.dims)
        g = int(np.argmax(np.abs(Rb.T @ Rt.apply((0., 1., 0.)))))      # the item axis along the fingers
        others = [i for i in range(3) if i != g]
        longest = max(others, key=lambda i: dims[i])
        vert = [i for i in others if i != longest][0]
        R_now = Rotation.from_matrix(Rb)
        up_now = R_now.apply(np.eye(3)[vert])
        R_d = Rotation.align_vectors([(0., 0., 1. if up_now[2] >= 0 else -1.)], [up_now])[0]*R_now
        half_up = dims[vert]/2
    else:
        r, h = s.dims
        a = Rotation.from_matrix(Rb).apply((0., 0., 1.))
        R_d = Rotation.align_vectors([(0., 0., 1.)], [a])[0]*Rotation.from_matrix(Rb)
        half_up = h/2
    op.phase = "ASSISTED scripted: set it down beside the scanner"
    mine = held(op, item)
    names = list(op.k.arms[arm][1])
    release = min(.04, op.finger_width(arm)/2 + .008)          # each finger 8 mm clear of the item when letting go
    chosen = fallback = None
    tried = []
    # the item may stand at any yaw (a can, or a box on its face): turning it about the vertical keeps the hand's
    # 208 mm housing off the tote's front wall and the scanner (every spot was blocked at one yaw)
    # a round item's footprint stays EDGE_MARGIN_M inside the counter's front edge (y = 0): an 85 mm jar at the 45 mm
    # spot row stood 2.5 mm from the edge (it fell over after a 3 m/s sweep had turned it in the grip, and rolled off)
    foot = s.dims[0] if s.shape != "box" else None
    for k_sp, sp in enumerate([np.asarray(spot)] if spot is not None else HANDOFF_SPOTS + HANDOFF_SPOTS_MORE):
        if spot is None and k_sp == len(HANDOFF_SPOTS) and (chosen is not None or fallback is not None):
            break
        for yaw in (0., 30., -30., 60., -60., 90., -90., 120., -120., 150., -150., 180.):
            Rh = Rotation.from_euler("z", yaw, degrees=True)*R_d*rel_R.inv()
            y_in = float(sp[1])
            if spot is None and s.shape != "box":
                # a round item stands fully on the counter (tipped, it rolls); a box standing on its edge may put a
                # corner over the edge with its centre 45 mm inside (pushing boxes back too put the 180 mm battery
                # pack's rotated footprints into the basket's wall, and no spot was left)
                y_in = max(y_in, foot + EDGE_MARGIN_M)
            centre = np.array([sp[0], y_in, top + half_up + .005])
            tcp = centre - Rh.apply(rel_p)
            path = plan_joint_path(op, arm, tcp + (0, 0, .08), Rh.as_quat(), seconds=2.2, carried=mine, restarts=8)
            if path is None:
                tried.append((round(float(sp[0]), 2), yaw, getattr(op, "last_block", None)))
                continue
            try:
                down = _solve_path(op, arm, _cart_steps(tcp + (0, 0, .08), Rh, tcp, Rh, step=.01), np.asarray(path[-1]))
            except ValueError:
                continue
            hit = checker(op).path_clear(arm, names, down, carried=mine, every=1)
            if hit is not None:
                tried.append((round(float(sp[0]), 2), yaw, hit[1][:1]))
                continue
            # and the way out: the hand, opened to release, rises 10 cm clear of the item it leaves behind (checked
            # only after the set-down, the exit was blocked twice: the hand had no clear move away)
            try:
                up = _solve_path(op, arm, _cart_steps(tcp, Rh, tcp + (0, 0, .10), Rh, step=.01), np.asarray(down[-1]))
            except ValueError:
                tried.append((round(float(sp[0]), 2), yaw, "no exit IK"))
                fallback = fallback or (tcp, path, Rh, sp, yaw, down)
                continue
            hit = checker(op).path_clear(arm, names, up, ignore_bodies=mine, every=1, fingers=release)
            if hit is not None:
                tried.append((round(float(sp[0]), 2), yaw, ("exit",) + tuple(hit[1][:1])))
                fallback = fallback or (tcp, path, Rh, sp, yaw, down)       # a clear descent, if no spot has a clear exit
                continue
            chosen = (tcp, path, Rh, sp, yaw, down)
            break
        if chosen is not None:
            break
    if chosen is None and fallback is None and not _retried and _swing_other_arm_aside(op, arm):
        return put_down(op, arm, item, rec, spot=spot, _retried=True)
    if chosen is None and fallback is not None:
        chosen = fallback
        rec["set_down_exit_unchecked"] = True
    from collections import Counter

    def reason(b):                                      # the blocking pair(s) without their depths, so like counts with like
        if b is None:
            return "no path"
        if isinstance(b, str):
            return b
        exit_ = isinstance(b, tuple) and b[:1] == ("exit",)
        pairs = b[1:] if exit_ else b
        names = "+".join(sorted({"/".join(str(x) for x in c[:2]) if isinstance(c, (tuple, list)) else str(c) for c in pairs}))
        return ("exit: " if exit_ else "") + names
    why = Counter(reason(t[2]) for t in tried)
    rec["set_down_blocked"] = dict(why)                 # every candidate tried before the one used, by what blocked it
    if chosen is None:
        rec["set_down_tried"] = [dict(x=t[0], yaw_deg=t[1], blocked_by=reason(t[2])) for t in tried]
        raise ValueError(f"no collision-free set-down spot on the counter ({len(tried)} tried: {why.most_common(5)})")
    tcp, path, Rh, sp, yaw, down = chosen
    rec["set_down_spot"] = dict(x=float(sp[0]), y=float(sp[1]), yaw_deg=yaw, tried=len(tried))
    run_path(op, arm, path)
    run_joint_path(op, arm, down, 1.0)
    op.open_hand(arm, half_width=release, seconds=.4)
    for dz in (.10, .06, .03):
        try:
            checked_move(op, arm, tcp + (0, 0, dz), Rh.as_quat(), seconds=.8, ignore=mine, what="leaving it")
            break
        except ValueError:
            continue
    else:
        _rise_clear(op, arm, dz=.03)
        if not _to_ready(op, arm):
            raise ValueError(f"no collision-free move for leaving it (e.g. {getattr(op, 'last_block', None)})")
    op.wait(.5)
    pb, Rb = w.body_pose(it["body"])
    tilt = abs(Rb[2, 2]) if s.shape != "box" else max(abs(Rb[2, k]) for k in range(3))
    rec["set_down_tilt_deg"] = float(math.degrees(math.acos(max(-1., min(1., tilt)))))
    rec["set_down_at"] = np.round(pb, 3).tolist()
