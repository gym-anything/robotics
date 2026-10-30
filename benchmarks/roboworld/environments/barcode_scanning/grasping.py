"""Grasp planning for the ASSISTED scripted operator (never a policy action): where can a Franka Hand close on a
stock item without its housing or fingers running into the basket walls, the shelf decks or neighbouring stock?

The hand is its own collision geometry measured in the TCP frame (fingers open 80 mm; world.py robot shapes):
housing x +-32 mm, y (finger axis) +-104 mm, z 37..129 mm behind the TCP; each finger x +-11 mm, y 40..66 mm,
z -45..+10 mm; the link-7 flange x, y +-44 mm, z 104..167 mm. Obstacles are the model's own collision shapes (boxes
and convex hulls as oriented boxes), except the robots and the item itself. Candidates: every finger axis along
which the item is narrower than the stroke (a cylinder across its diameter, lid to lid if short enough, a jar by
its cap), and approach directions perpendicular to it tilted up to 60 deg from straight down; each is checked at
the grasp and along the straight approach (clearance 3 mm, separating-axis test) and must be reachable by the IK.
"""
import math
import numpy as np
from scipy.spatial.transform import Rotation

STROKE = .075
FINGER_T = .026        # finger thickness along the closing axis (inner pad face to outer face)


def hand_boxes(open_half=.04):
    """(centre, half extents) in the TCP frame with each finger's pad face `open_half` from the TCP."""
    y = open_half + FINGER_T/2
    return [((0., 0., -.083), (.032, .104, .046)),         # housing
            ((0., y, -.0175), (.011, FINGER_T/2, .0275)),  # finger +y
            ((0., -y, -.0175), (.011, FINGER_T/2, .0275)), # finger -y
            ((0., 0., -.1355), (.044, .044, .0315))]       # link-7 flange


HAND = hand_boxes()


def _obb_overlap(c1, R1, h1, c2, R2, h2, margin=0.):
    """Separating-axis test for two oriented boxes (centres, rotation matrices with box axes as columns, halves)."""
    h1 = np.asarray(h1) + margin
    t = c2 - c1
    axes = [R1[:, i] for i in range(3)] + [R2[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            v = np.cross(R1[:, i], R2[:, j])
            n = np.linalg.norm(v)
            if n > 1e-9:
                axes.append(v/n)
    for L in axes:
        r1 = sum(abs(L @ R1[:, i])*h1[i] for i in range(3))
        r2 = sum(abs(L @ R2[:, i])*h2[i] for i in range(3))
        if abs(t @ L) > r1 + r2:
            return False
    return True


def obstacles(world, exclude_bodies=(), centre=None, radius=.45, only_bodies=None):
    """Oriented boxes (centre, R, half) of every collision shape near `centre`, except robots and excluded bodies
    (or, with `only_bodies`, of the shapes on those bodies only, robots included)."""
    m = world.model
    q = world.state.body_q.numpy()
    sb = m.shape_body.numpy(); st = m.shape_type.numpy(); ss = m.shape_scale.numpy(); sx = m.shape_transform.numpy()
    flags = m.shape_flags.numpy() if hasattr(m, "shape_flags") else None
    robots = set(getattr(world.scene, "robot_bodies", ()))
    out = []
    import newton
    for i in range(m.shape_count):
        b = int(sb[i])
        if only_bodies is not None:
            if b not in only_bodies:
                continue
        elif b in robots or b in exclude_bodies:
            continue
        if flags is not None and not int(flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        tf = sx[i]
        if b >= 0:
            pb, Rb = q[b, :3], Rotation.from_quat(q[b, 3:])
        else:
            pb, Rb = np.zeros(3), Rotation.identity()
        Rs = Rb*Rotation.from_quat(tf[3:7])
        if int(st[i]) == int(newton.GeoType.BOX):
            c, h = pb + Rb.apply(tf[:3]), np.asarray(ss[i], float)
        else:
            src = m.shape_source[i]
            if src is None or not hasattr(src, "vertices"):
                continue
            v = np.asarray(src.vertices)*np.asarray(ss[i], float)
            lo, hi = v.min(0), v.max(0)
            c, h = pb + Rb.apply(tf[:3]) + Rs.apply((lo + hi)/2), (hi - lo)/2
        if centre is not None and np.linalg.norm(c - centre) - np.linalg.norm(h) > radius:
            continue
        out.append((c, Rs.as_matrix(), h))
    return out


def hand_collides(p_tcp, R_tcp, obs, margin=.003, boxes=None):
    for hc, hh in (boxes or HAND):
        c = p_tcp + R_tcp @ np.asarray(hc)
        for oc, oR, oh in obs:
            if _obb_overlap(c, R_tcp, hh, oc, oR, oh, margin):
                return True
    return False


def candidates(world, item, sku):
    """(grasp point, finger axis unit vector, list of admissible approach vectors) for the item as it lies."""
    from .catalog import neck
    p, R = world.body_pose(world.scene.items[item]["body"])
    out = []
    down = np.array([0., 0., -1.])
    if sku.shape == "box":
        half = np.asarray(sku.dims)/2
        up = int(np.argmax(np.abs(R[2, :])))
        top = p + R[:, up]*np.sign(R[2, up])*max(0., half[up] - .02)    # pads 20 mm below the top face
        for k in range(3):
            if k != up and 2*half[k] <= STROKE:
                out.append((top, R[:, k], 2*half[k]))
    else:
        r, h = sku.dims
        g = neck(sku)
        a = R[:, 2]
        standing = abs(a[2]) > .9
        if g and (2*r > STROKE or standing):                        # jar, or any standing bottle: across its cap
            out.append((p + a*((g["cap"][0] + g["cap"][1])/2), None, 2*g["cap_r"]))
        elif standing:                                              # standing can: across the diameter near its top
            # ... but with the finger pads (they reach 10 mm past the TCP) clear of the printed barcode: a lip balm
            # taken 20 mm below its top had the finger across its label and never read
            depth = .02
            lab = next((L for L in world.scene.labels if L.get("item") == item and L.get("kind") == "product"), None)
            if lab is not None:
                end = np.sign(a[2])                                 # +1: the item's +z end is up
                ext = max(lab["size"])/2
                label_top = end*lab["center"][2] + ext              # the label's reach towards the upper end
                depth = float(np.clip(h/2 - label_top - .013, .006, .02))
            out.append((p + np.sign(a[2])*a*(h/2 - depth), None, 2*r))
        else:
            body_c = p + a*((g["body"][0] + g["body"][1])/2) if g else p.copy()
            out.append((body_c, None, 2*r))                         # across the diameter (any axis normal to a)
            if h <= STROKE and not g:
                out.append((p.copy(), a, h))                        # lid to lid
    res = []
    for c, f, width in out:
        if f is None:                                               # cylinder: finger axis normal to its axis
            a = R[:, 2]
            u = np.cross(a, (0., 0., 1.)) if abs(a[2]) < .95 else np.cross(a, (1., 0., 0.))
            u /= np.linalg.norm(u); v = np.cross(a, u)
            for ang in np.radians(np.arange(0, 180, 15)):
                fa = math.cos(ang)*u + math.sin(ang)*v
                res.append((c, fa, width))
        else:
            res.append((c, f/np.linalg.norm(f), width))
    return res


def plan(world, kin, arm, item, sku, max_tilt_deg=60., approach_len=.12, extra_exclude=()):
    """Collision-free top-ish grasps, best (least tilted) first: dicts with tcp, quat, approach, fingers, tilt."""
    from .scripted import hand_quat
    body = world.scene.items[item]["body"]
    obs = obstacles(world, exclude_bodies={body, *extra_exclude}, centre=world.body_pose(body)[0])
    found = []
    for c, f, width in candidates(world, item, sku):
        # open only as far as the item needs, with at least 12 mm to spare on each side for small items (the lip
        # balm, 15.6 mm, was pushed out of a 6 mm margin by the descending fingers)
        open_half = min(.04, width/2 + max(.006, .012 - width/4))
        boxes = hand_boxes(open_half)
        # approach directions normal to the finger axis, as close to straight down as allowed
        base = np.array([0., 0., -1.]) - (np.array([0., 0., -1.]) @ f)*f
        if np.linalg.norm(base) < 1e-6:
            continue
        base /= np.linalg.norm(base)
        side = np.cross(f, base)
        for tilt in np.radians(np.arange(0., max_tilt_deg + 1e-6, 10.)):
            for sgn in ((1.,) if tilt == 0 else (1., -1.)):
                app = math.cos(tilt)*base + sgn*math.sin(tilt)*side
                if app[2] > -.4:                                    # never from below the horizon
                    continue
                try:
                    q = hand_quat(app, f)
                except Exception:
                    continue
                Rm = Rotation.from_quat(q).as_matrix()
                pre = c - approach_len*app
                if any(hand_collides(c - s*app, Rm, obs, boxes=boxes) for s in (0., .03, .06, approach_len)):
                    continue
                try:
                    kin.solve(arm, c, q, restarts=4)
                    kin.solve(arm, pre, q, restarts=4)
                except Exception:
                    continue
                found.append(dict(tcp=c, quat=q, approach=app, fingers=f, pre=pre, open_half=open_half, width=width,
                                  tilt_deg=float(math.degrees(math.acos(max(-1., min(1., -app[2])))))))
        if len(found) >= 6:
            break
    found.sort(key=lambda d: d["tilt_deg"])
    return found
