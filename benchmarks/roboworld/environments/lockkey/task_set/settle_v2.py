"""lockkey-tasks-v2: hands-off settle of the constructed free-key starts (spec, "Start states").

The mid-cycle starts (turn-to-open, swing-open, remove-key) and return-key's key
on the mat are written as poses, not grown by physics. Before the task starts the
world runs SETTLE_S of ordinary native ticks with the robot commanded to hold
HOME through its usual position servos: no force, fixture, mass, contact, solver
or robot setting is touched, and the recording hook is not called (this is
construction, not a robot action). The task then starts at tick zero from the
settled state, as after the soil and laundry settles.

Setup raises, and the start is not served, unless the settle leaves the
intended state intact: the latch released exactly where intended, the key seated
for the mid-cycle starts, the shackle over its bore for swing-open (< 3.1 mm) and
swung open for remove-key (> 6.2 mm), the return-key key on its mat spot (5 mm),
the key moved at most 2 mm by the settle and slower than 5 cm/s at its end, no
contact on the key or the lock deeper than 1 mm, and the seated bow at most
0.43 mm into the unrendered `shell` collider. That last pair is the carried
round-3 overlap the unlock task needs for release (0.06-0.43 mm);
the recorded seated poses start 0.66-0.71 mm in and settle to 0.21-0.24 mm,
held by keyway friction.

The design's stricter settle bar -- every contact within 0.1 mm and key, pins and
drivers at rest -- is measured and recorded (`settle.spec`), not enforced: in the
unchanged physics two starts cannot meet it at any settle length. The pin stacks
of the locked lock (turn-to-open, return-key) chatter intermittently (pin speeds
to 0.19 m/s, keypin/driver overlaps to 0.39 mm, in bursts), and the free key on
the mat rocks on its bow (to 4 rad/s, about +-0.5 deg, 0.4-0.7 mm reported bow/mat
contact overlap), creeping about 0.2 mm/s. Both were measured over 3 s in the
render-check round; neither moves a task reading toward credit.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation

from lockkey.newton15.evaluation import OPEN_LATERAL
from . import presets_v2 as presets

SETTLE_S = .5
# Enforced.
GROSS_PENETRATION_M = 1e-3
SHELL_OVERLAP_PAIR = frozenset({"shell", "key_bow"})
MAX_SHELL_OVERLAP_M = 4.3e-4
MAX_SETTLE_MOVE_M = .002
MAX_KEY_SPEED_M_S = .05
MAT_SPOT_M = .005
# The design's bar, recorded per start (see the module docstring).
SPEC_PENETRATION_M = 1e-4
REST_SPEED_M_S = 1e-3          # key linear speed and every pin/driver joint speed at the end
REST_SPIN_RAD_S = .05          # key angular speed at the end
SETTLED = ("turn-to-open", "swing-open", "remove-key", "return-key")
RELEASED = ("swing-open", "remove-key")
MECHANISM_PREFIXES = ("key_", "plug", "keypin_", "driver_", "shackle", "shell", "lock_casing")


def _contact_depths(env):
    """Signed distances of the latest native contacts that involve the key or the lock (robot-internal pairs and
    the fixed scenery's own pairs are not start-construction contacts): signed distance of the
    two contact points along the contact normal."""
    c, labels, shape_body = env.contacts, env.builder.shape_label, env.builder.shape_body
    n = int(c.rigid_contact_count.numpy()[0])
    if not n:
        return []
    s0, s1 = c.rigid_contact_shape0.numpy()[:n], c.rigid_contact_shape1.numpy()[:n]
    p0, p1 = c.rigid_contact_point0.numpy()[:n], c.rigid_contact_point1.numpy()[:n]
    normal = c.rigid_contact_normal.numpy()[:n]
    q = env.state.body_q.numpy()

    def world(shape, point):
        body = shape_body[int(shape)]
        return point if body < 0 else q[body, :3]+Rotation.from_quat(q[body, 3:]).apply(point)
    rows = []
    for a, b, x, y, m in zip(s0, s1, p0, p1, normal):
        pair = (labels[int(a)], labels[int(b)])
        if not any(s.startswith(MECHANISM_PREFIXES) for s in pair) or all(s.startswith("panda") for s in pair):
            continue
        rows.append((float(np.dot(world(b, y)-world(a, x), m)), pair))
    return rows


def settle(env, task):
    """Run the settle on a freshly placed start (tick 0). Returns the settle record; raises if the start is invalid."""
    if task not in SETTLED:
        return None
    if env.tick != 0:
        raise RuntimeError("The start settle runs only on a newly placed start")
    key_body = env.bodies["key"]
    before = env.state.body_q.numpy()[key_body].copy()
    plug0, lateral0 = env.evaluation["plug_angle_deg"], env.evaluation["shackle_short_leg_lateral_offset_m"]
    ticks = int(round(SETTLE_S/env.config.dt))
    callback, env.trace_callback = env.trace_callback, None
    try:
        with env.lock:
            env.targets = env.initial_targets.copy()
            for _ in range(ticks):
                env._step()
    finally:
        env.trace_callback = callback
    q, qd = env.state.body_q.numpy(), env.state.body_qd.numpy()
    jq, jqd = env.state.joint_q.numpy(), env.state.joint_qd.numpy()
    key = q[key_body]
    pins = {n: float(abs(jqd[env.builder.joint_qd_start[j]])) for n, j in env.mechanism_joints.items()
            if n.startswith(("keypin", "driver"))}
    depths = _contact_depths(env)
    shell = [d for d, pair in depths if frozenset(pair) == SHELL_OVERLAP_PAIR]
    others = sorted((d, pair) for d, pair in depths if frozenset(pair) != SHELL_OVERLAP_PAIR)
    worst = others[0] if others else (0., None)
    deepest = {}
    for d, pair in others:
        deepest.setdefault("|".join(pair), round(-d, 7))
    e = env.evaluation
    record = dict(
        seconds=SETTLE_S, ticks=ticks, robot="commanded to hold HOME; native servos, no fixture or force",
        key_moved_m=float(np.linalg.norm(key[:3]-before[:3])),
        key_turned_deg=float(np.degrees((Rotation.from_quat(before[3:]).inv()*Rotation.from_quat(key[3:])).magnitude())),
        key_speed_m_s=float(np.linalg.norm(qd[key_body, :3])), key_spin_rad_s=float(np.linalg.norm(qd[key_body, 3:])),
        max_pin_speed_m_s=max(pins.values(), default=0.),
        max_penetration_m=max(0., -worst[0]), worst_contact=list(worst[1]) if worst[1] else None,
        deepest_contacts_m=dict(list(deepest.items())[:5]),
        shell_bow_overlap_m=max([0.]+[-d for d in shell]),
        plug_deg=[float(plug0), float(e["plug_angle_deg"])],
        shackle_lateral_m=[float(lateral0), float(e["shackle_short_leg_lateral_offset_m"])],
        released=bool(env.released), key_seated=bool(e["key_seated_now"]))
    record["spec"] = dict(penetration_le_0_1_mm=record["max_penetration_m"] <= SPEC_PENETRATION_M,
                          key_at_rest=(record["key_speed_m_s"] <= REST_SPEED_M_S
                                       and record["key_spin_rad_s"] <= REST_SPIN_RAD_S),
                          pins_at_rest=record["max_pin_speed_m_s"] <= REST_SPEED_M_S)
    problems = []
    if record["max_penetration_m"] > GROSS_PENETRATION_M:
        problems.append(f"contact {record['worst_contact']} penetrates {record['max_penetration_m']*1e3:.3f} mm")
    if record["shell_bow_overlap_m"] > MAX_SHELL_OVERLAP_M:
        problems.append(f"key bow {record['shell_bow_overlap_m']*1e3:.3f} mm into the shell collider")
    if record["key_moved_m"] > MAX_SETTLE_MOVE_M:
        problems.append(f"key moved {record['key_moved_m']*1e3:.1f} mm")
    if record["key_speed_m_s"] > MAX_KEY_SPEED_M_S:
        problems.append(f"key still moving at {record['key_speed_m_s']:.3f} m/s")
    if bool(env.released) != (task in RELEASED):
        problems.append(f"latch released={env.released}, intended {task in RELEASED}")
    lateral = e["shackle_short_leg_lateral_offset_m"]
    if task == "return-key":
        on_mat = (abs(key[0]-presets.MAT_KEY_XY[0]) < MAT_SPOT_M and abs(key[1]-presets.MAT_KEY_XY[1]) < MAT_SPOT_M
                  and key[2] < env.config.table_height+.01)
        if not on_mat:
            problems.append(f"key left its place on the mat: {np.round(key[:3], 4).tolist()}")
    else:
        if not e["key_seated_now"]:
            problems.append("key no longer seated")
        if task == "remove-key" and not lateral > OPEN_LATERAL:
            problems.append(f"shackle no longer open ({lateral*1e3:.2f} mm)")
        if task == "swing-open" and not lateral < OPEN_LATERAL/2:
            problems.append(f"shackle not over its bore ({lateral*1e3:.2f} mm)")
    if problems:
        raise RuntimeError(f"lockkey {task}: constructed start is not valid after the {SETTLE_S} s settle: "
                           + "; ".join(problems))
    env.tick = 0
    return record
