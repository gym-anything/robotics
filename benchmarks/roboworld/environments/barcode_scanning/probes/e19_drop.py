"""E19 -- analytic reference: free fall from 100 mm onto the counter, impact, rest, and energy accounting.

FIXTURE (the only one): after the environment's own 1 s settle (counter preset, seed 1, whose basket holds a cereal box
and a soup can), the dropped body's free-joint pose is set ONCE, before the first tick of the drop, with zero velocity:
its lowest collision point 100 mm above the counter top, over a clear part of the counter (x 0.73 m, y 0.30 m, right of
the bagging tote). Nothing else touches it: no forces, no mount, no further state writes. Every case starts from the
same settled checkpoint.

Cases
  cereal_box_flat   K03 cereal box (0.43 kg, 195 x 70 x 290 mm box) lying flat on its back (the 195 x 290 face down)
  soup_can_upright  K01 soup can (0.35 kg, 211x400 can as a 32-facet convex hull, 2 mm detection envelope), upright
  scanner_on_side   the scanner (214 g head + handle boxes, 3 g trigger on its spring-loaded pivot) lying on its right side

Measured per tick (the dropped body; for the scanner also its trigger lever)
  lowest collision point above the counter top (from the collision boxes / hull vertices at the current pose),
  COM position and velocity (Newton body_qd = COM linear velocity and angular velocity, world frame), MuJoCo contacts
  touching the body (active/inactive, signed distance, normal force from mj_contactForce, partner shapes),
  mechanical energy E = sum over its bodies of 1/2 m v^2 + 1/2 w.I.w + m g (z_com - z_counter), plus for the scanner
  the trigger spring's 1/2 k (q - q_target)^2 and its armature's 1/2 I_a qd^2.
Reference: first contact at t = sqrt(2h/g) = 0.14278 s, impact speed sqrt(2gh) = 1.4007 m/s (h = 0.100 m, g = 9.81).
Discrete reference (MuJoCo's semi-implicit Euler: v_n = g n dt, drop_n = g dt^2 n(n+1)/2): the first tick whose pose
is at or below the surface, and the free-fall energy deficit 1/2 m g^2 dt t that the position lag produces (no gain).
Energy must never exceed its initial value: any gain is reported with its size and time.

  python -m barcode_scanning.probes.e19_drop --dt 0.001 --out barcode_scanning/evidence/e19_drop
  python -m barcode_scanning.probes.e19_drop --dt 0.0005 --out barcode_scanning/evidence/e19_drop
  python -m barcode_scanning.probes.e19_drop --report --out barcode_scanning/evidence/e19_drop
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np

H = 0.100
G = 9.81
DROP_XY = (0.73, 0.30)
T_RUN = 3.0
REST_V, REST_W = 1e-3, 1e-2          # m/s, rad/s: "at rest" thresholds (time to rest = last time either is exceeded)


def _shape_points(w, bodies):
    """World points of every colliding shape on `bodies` (box corners, convex-hull vertices)."""
    import newton
    from scipy.spatial.transform import Rotation
    m = w.model
    st = m.shape_type.numpy(); sb = m.shape_body.numpy(); sc = m.shape_scale.numpy(); tf = m.shape_transform.numpy()
    flags = m.shape_flags.numpy()
    bq = w.state.body_q.numpy()
    pts = []
    for s in np.flatnonzero(np.isin(sb, list(bodies))):
        if not flags[s] & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        if st[s] == int(newton.GeoType.BOX):
            h = sc[s]
            local = np.array([[x, y, z] for x in (-h[0], h[0]) for y in (-h[1], h[1]) for z in (-h[2], h[2])])
        elif st[s] in (int(newton.GeoType.CONVEX_MESH), int(newton.GeoType.MESH)):
            local = np.asarray(m.shape_source[s].vertices, float)*sc[s]
        else:
            raise RuntimeError(f"shape {m.shape_label[s]}: unsupported type {st[s]} for the lowest-point computation")
        Rs, ps = Rotation.from_quat(tf[s, 3:]), tf[s, :3]
        b = int(sb[s])
        Rb, pb = Rotation.from_quat(bq[b, 3:].astype(float)), bq[b, :3].astype(float)
        pts.append(pb + Rb.apply(ps + Rs.apply(local)))
    return np.vstack(pts)


def _counter_top(w):
    """The counter top's upper face (an 8-vertex hull: its mesh's highest vertex)."""
    m = w.model
    s = list(m.shape_label).index("counter_top")
    src = m.shape_source[s]
    half = float(np.asarray(src.vertices)[:, 2].max()) if src is not None and hasattr(src, "vertices") else float(m.shape_scale.numpy()[s, 2])
    return float(m.shape_transform.numpy()[s, 2] + half)


def run(a):
    from barcode_scanning.manifest import launch_stamp
    code_at_launch = launch_stamp()
    import mujoco
    from scipy.spatial.transform import Rotation
    from barcode_scanning import params as P
    from barcode_scanning.config import Config
    from barcode_scanning.env import ScanEnv
    from barcode_scanning.probes.e17_stress import DarkRenderer
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    cfg = Config(preset="counter", task="checkout-present", seed=1, rigid_dt=a.dt)
    t_build = time.time()
    env = ScanEnv(cfg, output_dir=out/f"session_dt{a.dt*1e3:g}ms")
    env.render_meta = {"scanner": {"leds": [], "aim": "aim", "led_on": "led_on"}}
    env.renderer_factory = lambda: DarkRenderer()
    w = env.world
    sc = w.scene
    newton = w.newton
    model = w.model
    z_top = _counter_top(w)
    base = env.checkpoint()
    item_of = lambda sku: next(n for n, it in sc.items.items() if it.get("sku") == sku)
    cases = [("cereal_box_flat", item_of("K03"), Rotation.from_euler("x", -90, degrees=True)),
             ("soup_can_upright", item_of("K01"), Rotation.identity()),
             ("scanner_on_side", "scanner", Rotation.from_euler("x", 90, degrees=True))]
    if a.cases:
        cases = [c for c in cases if c[0] in a.cases.split(",")]
    win = tuple(float(x) for x in a.contacts_window.split(",")) if a.contacts_window else None
    jchild = model.joint_child.numpy(); jtype = model.joint_type.numpy()
    q_start = model.joint_q_start.numpy(); qd_start = model.joint_qd_start.numpy()
    mass = model.body_mass.numpy(); com = model.body_com.numpy(); inertia = model.body_inertia.numpy()
    geom_shape = w._geom_shape
    sbody = np.asarray(sc.b.shape_body)
    labels = list(model.shape_label)
    results = {}
    for name, body_name, rot in cases:
        env.restore(base)
        w.fault = None
        b = sc.items[body_name]["body"] if body_name in sc.items else sc.bodies[body_name]
        bodies = [b] + ([sc.bodies["scanner_trigger"]] if body_name == "scanner" else [])
        j = int(np.flatnonzero((jchild == b) & (jtype == int(newton.JointType.FREE)))[0])
        qs, ds = int(q_start[j]), int(qd_start[j])
        jq = w.state.joint_q.numpy().copy(); jqd = w.state.joint_qd.numpy().copy()
        bq = w.state.body_q.numpy()
        convention_err = float(np.abs(jq[qs:qs + 7] - bq[b]).max())      # free joint coords == body origin pose (world)
        # FIXTURE: the initial pose, set once before the first tick (orientation, then lift so the lowest point is at h)
        jq[qs:qs + 3] = (DROP_XY[0], DROP_XY[1], 1.3)
        jq[qs + 3:qs + 7] = rot.as_quat()
        jqd[ds:ds + 6] = 0.
        if body_name == "scanner":
            jqd[int(qd_start[w.scene.trigger_joint])] = 0.
        for _ in range(2):
            w.state.joint_q.assign(jq); w.state.joint_qd.assign(jqd); w.sync_solver()
            newton.eval_fk(model, w.state.joint_q, w.state.joint_qd, w.state)
            pts = _shape_points(w, bodies)
            bq = w.state.body_q.numpy()
            c = bq[b, :3] + Rotation.from_quat(bq[b, 3:].astype(float)).apply(com[b])
            jq[qs] += DROP_XY[0] - c[0]; jq[qs + 1] += DROP_XY[1] - c[1]
            jq[qs + 2] += z_top + H - pts[:, 2].min()
        w.state.joint_q.assign(jq); w.state.joint_qd.assign(jqd); w.sync_solver()
        newton.eval_fk(model, w.state.joint_q, w.state.joint_qd, w.state)
        w.next_state.assign(w.state)
        w.solver._update_mjc_data(w.solver.mj_data, w.model, w.state)
        mujoco.mj_forward(w.solver.mj_model, w.solver.mj_data)
        h0 = float(_shape_points(w, bodies)[:, 2].min() - z_top)
        k_t = P.TRIGGER_RATE_N_PER_M*P.TRIGGER_ARM_M**2
        arm_trig = float(model.joint_armature.numpy()[int(qd_start[w.scene.trigger_joint])])
        tdof = int(qd_start[w.scene.trigger_joint])

        def energy():
            bq = w.state.body_q.numpy().astype(float); bqd = w.state.body_qd.numpy().astype(float)
            E = dict(ke_lin=0., ke_rot=0., pe=0., spring=0.)
            for bb in bodies:
                R = Rotation.from_quat(bq[bb, 3:]).as_matrix()
                zc = float(bq[bb, 2] + (R @ com[bb])[2])
                v, om = bqd[bb, :3], bqd[bb, 3:6]
                ob = R.T @ om
                E["ke_lin"] += .5*mass[bb]*float(v @ v)
                E["ke_rot"] += .5*float(ob @ inertia[bb] @ ob)
                E["pe"] += mass[bb]*G*(zc - z_top)
            if body_name == "scanner":
                q = float(w.state.joint_q.numpy()[w.trigger_q]); qd = float(w.state.joint_qd.numpy()[tdof])
                qt = float(w.control.joint_target_q.numpy()[w.trigger_q])
                E["spring"] = .5*k_t*(q - qt)**2
                E["ke_rot"] += .5*arm_trig*qd*qd
            E["total"] = E["ke_lin"] + E["ke_rot"] + E["pe"] + E["spring"]
            return E

        def contact_detail():
            m_, d = w.solver.mj_model, w.solver.mj_data
            f6 = np.zeros(6); rows = []
            for i in range(d.ncon):
                cc = d.contact[i]
                s = [int(geom_shape[g]) if 0 <= g < len(geom_shape) else -1 for g in cc.geom]
                bb = [int(sbody[x]) if x >= 0 else -1 for x in s]
                if not (bb[0] in bodies or bb[1] in bodies):
                    continue
                mujoco.mj_contactForce(m_, d, i, f6)
                rows.append(dict(shapes=[labels[x] if x >= 0 else str(x) for x in s], pos=np.round(cc.pos, 5).tolist(),
                                 normal=np.round(cc.frame[:3], 4).tolist(), dist_mm=float(cc.dist)*1e3, exclude=int(cc.exclude),
                                 force_n=np.round(f6, 3).tolist()))
            return rows

        def contacts():
            m_, d = w.solver.mj_model, w.solver.mj_data
            f6 = np.zeros(6)
            act_dist, inact_dist, fn, partners, n_act = [], [], 0., set(), 0
            for i in range(d.ncon):
                cc = d.contact[i]
                s = [int(geom_shape[g]) if 0 <= g < len(geom_shape) else -1 for g in cc.geom]
                bb = [int(sbody[x]) if x >= 0 else -1 for x in s]
                if not (bb[0] in bodies or bb[1] in bodies):
                    continue
                other = s[1] if bb[0] in bodies else s[0]
                if bb[0] in bodies and bb[1] in bodies:
                    continue
                if cc.exclude == 0:
                    mujoco.mj_contactForce(m_, d, i, f6)
                    fn += float(f6[0]); act_dist.append(float(cc.dist)); n_act += 1
                    partners.add(labels[other] if other >= 0 else str(other))
                else:
                    inact_dist.append(float(cc.dist))
            return (min(act_dist) if act_dist else None, min(inact_dist) if inact_dist else None, fn, sorted(partners), n_act)

        E0 = energy()
        m_sys = float(sum(mass[bb] for bb in bodies))
        n = int(round(a.seconds/cfg.rigid_dt))
        C = []
        S = dict(t=[], h_low_mm=[], z_com=[], vz=[], v=[], w=[], E=[], ke=[], pe=[], spring=[], c_dist_mm=[], c_inactive_dist_mm=[],
                 c_force_n=[], c_active=[], trigger_mm=[])
        partners_all = set()
        t0 = time.time()
        R_start = Rotation.from_quat(w.state.body_q.numpy()[b, 3:].astype(float))
        for k in range(n):
            w.step()
            bq = w.state.body_q.numpy(); bqd = w.state.body_qd.numpy()
            R = Rotation.from_quat(bq[b, 3:].astype(float))
            zc = float(bq[b, 2] + R.apply(com[b])[2])
            Ek = energy()
            cd, ci, fn, prt, na = contacts()
            partners_all.update(prt)
            S["t"].append(w.sim_time); S["h_low_mm"].append(float(_shape_points(w, bodies)[:, 2].min() - z_top)*1e3)
            S["z_com"].append(zc); S["vz"].append(float(bqd[b, 2])); S["v"].append(float(np.linalg.norm(bqd[b, :3])))
            S["w"].append(float(np.linalg.norm(bqd[b, 3:6]))); S["E"].append(Ek["total"]); S["ke"].append(Ek["ke_lin"] + Ek["ke_rot"])
            S["pe"].append(Ek["pe"]); S["spring"].append(Ek["spring"])
            # MuJoCo's contacts after mj_step belong to the positions at the START of that step (t - dt)
            S["c_dist_mm"].append(None if cd is None else cd*1e3); S["c_inactive_dist_mm"].append(None if ci is None else ci*1e3)
            S["c_force_n"].append(fn); S["c_active"].append(na)
            S["trigger_mm"].append(w.trigger_travel_m()*1e3 if body_name == "scanner" else None)
            if win and win[0] <= w.sim_time - cfg.rigid_dt <= win[1]:
                C.append(dict(t_pose=round(w.sim_time - cfg.rigid_dt, 6), contacts=contact_detail(), vz=float(bqd[b, 2]),
                              w_body=bqd[b, 3:6].tolist(), E_mJ=Ek["total"]*1e3))
        wall = time.time() - t0
        R_end = Rotation.from_quat(w.state.body_q.numpy()[b, 3:].astype(float))
        bq = w.state.body_q.numpy()
        c_end = bq[b, :3] + R_end.apply(com[b])
        results[name] = dict(case=name, body=body_name, bodies=[model.body_label[x] for x in bodies], mass_kg=m_sys,
                             h0_mm=h0*1e3, counter_top_m=z_top, free_joint_convention_err=convention_err,
                             E0_J=E0, mgh_J=m_sys*G*H, wall_s=wall, partners=sorted(partners_all),
                             tilt_change_deg=float(np.degrees((R_end*R_start.inv()).magnitude())),
                             final_com=c_end.tolist(), lateral_drift_mm=float(np.linalg.norm(c_end[:2] - np.asarray(DROP_XY))*1e3),
                             series=S, contacts_window=C)
        results[name].update(analyse(results[name], cfg.rigid_dt))
        print(json.dumps({k: v for k, v in results[name].items() if k not in ("series", "E0_J")}, indent=1, default=float), flush=True)
    res = dict(experiment="E19 drop onto the counter: analytic free fall, impact, rest, energy",
               fixture="the dropped body's initial pose (100 mm above the counter, zero velocity) is set once before the first tick; "
                       "nothing else touches it", code_at_launch=code_at_launch, rigid_dt=cfg.rigid_dt,
               config=dict(preset=cfg.preset, task=cfg.task, seed=cfg.seed, contact_timeconst=cfg.contact_timeconst,
                           contact_solimp=list(cfg.contact_solimp), impratio=cfg.solver_impratio, noslip=cfg.solver_noslip_iterations),
               reference=dict(h_m=H, g=G, t_contact_s=math.sqrt(2*H/G), v_impact_m_s=math.sqrt(2*G*H)),
               build_s=time.time() - t_build, cases=results)
    if win:                                          # diagnosis run: its own file, never mixed into the report
        (out/f"e19_drop_diag_{'_'.join(results)}_dt{a.dt*1e3:g}ms.json").write_text(json.dumps(res, default=float))
        for name, r in results.items():
            for row in r["contacts_window"]:
                print(row["t_pose"], f"vz={row['vz']:+.4f} E={row['E_mJ']:.2f}mJ",
                      [(c["shapes"][0].replace("scanner_", "")[:10], c["shapes"][1][:11], np.round(c["pos"], 4).tolist(),
                        round(c["dist_mm"], 3), round(c["force_n"][0], 1)) for c in row["contacts"]])
        return
    (out/f"e19_drop_dt{a.dt*1e3:g}ms.json").write_text(json.dumps(res, default=float))


def analyse(r, dt):
    S = r["series"]
    t = np.asarray(S["t"]); h = np.asarray(S["h_low_mm"])*1e-3; vz = np.asarray(S["vz"]); E = np.asarray(S["E"])
    fn = np.asarray(S["c_force_n"]); act = np.asarray(S["c_active"])
    E0 = r["E0_J"]["total"]; mgh = r["mgh_J"]
    h0 = r["h0_mm"]*1e-3
    t_c = math.sqrt(2*h0/G); v_c = math.sqrt(2*G*h0)
    # discrete semi-implicit Euler: first n with g dt^2 n(n+1)/2 >= h0
    n_geom = int(math.ceil((-1 + math.sqrt(1 + 8*h0/(G*dt*dt)))/2))
    k_geom = int(np.argmax(h <= 0.)) if (h <= 0.).any() else None
    k_force = int(np.argmax(fn > 0.)) if (fn > 0.).any() else None
    k_act = int(np.argmax(act > 0)) if (act > 0).any() else None
    ci = [i for i, x in enumerate(S["c_inactive_dist_mm"]) if x is not None]
    out = dict(reference_t_contact_s=t_c, reference_v_impact_m_s=v_c, discrete_first_tick_at_or_below_s=n_geom*dt)
    # the contact list read after tick k was computed from the pose at the start of tick k, i.e. t[k] - dt
    out["t_first_pose_at_or_below_surface_s"] = float(t[k_geom]) if k_geom is not None else None
    out["t_first_contact_force_s"] = float(t[k_force] - dt) if k_force is not None else None
    out["t_first_active_contact_s"] = float(t[k_act] - dt) if k_act is not None else None
    out["t_first_detected_in_envelope_s"] = float(t[ci[0]] - dt) if ci else None
    if k_force is not None:
        kv = k_force - 1                                    # state at t[k_force] - dt, before the contact acts
        out["impact_speed_m_s"] = float(-vz[kv])
        out["impact_speed_rel_err"] = float(-vz[kv]/v_c - 1)
        out["t_contact_rel_err"] = out["t_first_contact_force_s"]/t_c - 1
        ff = slice(1, kv + 1)
        A = np.vstack([t[ff], np.ones(kv)]).T
        out["free_fall_accel_m_s2"] = float(np.linalg.lstsq(A, vz[ff], rcond=None)[0][0])
        # velocity convention check: body_qd COM velocity vs the finite difference of the COM height (semi-implicit Euler)
        zc = np.asarray(S["z_com"])
        out["com_velocity_vs_finite_difference_max_m_s"] = float(np.max(np.abs(np.diff(zc[:kv + 1])/dt - vz[1:kv + 1])))
        after = slice(k_force, None)
        cd = [x for x in S["c_dist_mm"][k_force:] if x is not None]
        out["peak_penetration_mm"] = float(-min(cd)) if cd else None
        out["peak_contact_force_n"] = float(fn[after].max())
        nr = int(round(.2/dt))                             # the last 0.2 s
        out["rest_penetration_mm"] = float(-np.mean([x for x in S["c_dist_mm"][-nr:] if x is not None])) if any(
            x is not None for x in S["c_dist_mm"][-nr:]) else None
        out["rest_h_low_mm"] = float(np.mean(h[-nr:])*1e3)
        # rebound: largest rise of the lowest point after the first impact, above its first-impact minimum; and the
        # largest upward COM velocity after the impact over the impact speed (restitution)
        k_min = k_force + int(np.argmin(h[k_force:k_force + int(.05/dt)]))
        out["rebound_mm"] = float((h[k_min:].max() - h[k_min])*1e3)
        out["rebound_above_surface_mm"] = float(max(0., h[k_min:].max())*1e3)
        out["restitution"] = float(max(0., vz[k_force:].max())/max(1e-9, -vz[kv]))
        moving = (np.asarray(S["v"]) > REST_V) | (np.asarray(S["w"]) > REST_W)
        last = int(np.flatnonzero(moving).max()) if moving.any() else -1
        out["time_to_rest_s"] = float(t[last + 1]) if last + 1 < len(t) else None
        out["time_to_rest_after_contact_s"] = (out["time_to_rest_s"] - out["t_first_contact_force_s"]) if out["time_to_rest_s"] else None
        out["end_speed_mm_s"] = float(S["v"][-1]*1e3); out["end_angular_rad_s"] = float(S["w"][-1])
        # energy: never above E0 (report the largest excess, when, and relative to m g h); free-fall deficit vs 1/2 m g^2 dt t
        dE = E - E0
        kmax = int(np.argmax(dE))
        out["energy"] = dict(E0_J=E0, E_end_J=float(E[-1]), max_excess_J=float(dE[kmax]), max_excess_at_s=float(t[kmax]),
                             max_excess_rel_mgh=float(dE[kmax]/mgh), gain=bool(dE[kmax] > 0.),
                             ticks_above_E0=int((dE > 0).sum()),
                             deficit_at_contact_J=float(-dE[kv]), predicted_free_fall_deficit_J=float(.5*r["mass_kg"]*G*G*dt*t[kv]),
                             dissipated_J=float(E0 - E[-1]), dissipated_rel_mgh=float((E0 - E[-1])/mgh))
        # any rise of E between consecutive ticks after the impact (energy injected by the contact solve)
        dEk = np.diff(E[k_force:])
        out["energy"]["max_tick_increase_after_impact_J"] = float(dEk.max()) if len(dEk) else 0.
        out["energy"]["max_tick_increase_at_s"] = float(t[k_force + 1 + int(np.argmax(dEk))]) if len(dEk) else None
        out["energy"]["sum_of_tick_increases_after_impact_J"] = float(dEk[dEk > 0].sum()) if len(dEk) else 0.
    return out


def soft_contact_1d(dt, h=H, tc=0.004, solimp=(.95, .99, .001, .5, 2.), g=G, T=.30, k_times_d=False):
    """MuJoCo's soft contact reduced to one degree of freedom: a flat face released from rest at height h, integrated
    with MuJoCo's semi-implicit Euler at step dt, the contact detected from the pose at the START of each step (as in
    mj_step), constraint acceleration a = (1 - d) (-g) + d (-b v - k r) (MuJoCo documentation, Computation: soft
    constraints; k_times_d=True uses -b v - k d r instead, a variant that changes the results by < 0.3 %), with
    k = 1/(dmax^2 tc^2), b = 2/(dmax tc), d(r) the solimp curve, and the contact only pushing (a >= -g). dt = 1e-6 s
    gives the continuous limit. Applies to the box and the can landing square (no rotation); not to the scanner."""
    d0, dmax, width, mid, power = solimp
    k, b = 1/(dmax*dmax*tc*tc), 2/(dmax*tc)

    def imp(x):
        y = min(1., x/width)
        if y >= 1.:
            return dmax
        if y < mid:
            return d0 + (dmax - d0)*y**power/mid**(power - 1)
        return dmax - (dmax - d0)*(1 - y)**power/(1 - mid)**(power - 1)
    r, v, n = h, 0., 0
    rmin, vup, hop, v_c, t_c, left = 0., 0., 0., None, None, None
    for n in range(1, int(round(T/dt)) + 1):
        if r < 0:
            d = imp(-r)
            acc = max(-g, (1 - d)*(-g) + d*(-b*v - k*(d if k_times_d else 1.)*r))
            if t_c is None:
                t_c, v_c = (n - 1)*dt, -v                      # the step whose start pose first penetrates
        else:
            acc = -g
        v += acc*dt; r += v*dt
        if t_c is not None:
            rmin = min(rmin, r); vup = max(vup, v); hop = max(hop, r)
            if left is None and r >= 0 and v > 0:
                left = v
    return dict(dt_s=dt, t_first_contact_s=t_c, speed_at_first_contact_m_s=v_c, peak_penetration_mm=-rmin*1e3,
                restitution=vup/v_c, leave_speed_m_s=left, hop_mm=hop*1e3, k_per_s2=k, b_per_s=b, solimp=list(solimp),
                timeconst_s=tc, k_times_d=k_times_d)


def report(a):
    out = Path(a.out)
    runs = {}
    for f in sorted(out.glob("e19_drop_dt*ms.json")):
        d = json.loads(f.read_text())
        runs[d["rigid_dt"]] = d
    table = {}
    for dt, d in sorted(runs.items(), reverse=True):
        for name, r in d["cases"].items():
            table.setdefault(name, {})[f"{dt*1e3:g} ms"] = {k: v for k, v in r.items() if k not in ("series", "contacts_window")}
    d0 = next(iter(runs.values()))
    tc, si = d0["config"]["contact_timeconst"], tuple(d0["config"]["contact_solimp"]) + (.5, 2.)
    ref = soft_contact_1d(1e-6, tc=tc, solimp=si, T=.2)
    ref_variant = soft_contact_1d(1e-6, tc=tc, solimp=si, T=.2, k_times_d=True)
    disc = {f"{dt*1e3:g} ms": soft_contact_1d(dt, tc=tc, solimp=si) for dt in runs}
    for name, per in table.items():
        for dtk, r in per.items():
            if name != "scanner_on_side" and r.get("peak_penetration_mm") is not None:
                m = disc[dtk]
                r["vs_soft_contact_model"] = dict(
                    discrete_model_same_dt=dict(peak_penetration_mm=m["peak_penetration_mm"], restitution=m["restitution"], hop_mm=m["hop_mm"],
                                                t_first_contact_s=m["t_first_contact_s"]),
                    peak_penetration_rel_err_vs_discrete=r["peak_penetration_mm"]/m["peak_penetration_mm"] - 1,
                    restitution_rel_err_vs_discrete=r["restitution"]/m["restitution"] - 1,
                    hop_measured_mm=r["rebound_above_surface_mm"], hop_discrete_mm=m["hop_mm"],
                    peak_penetration_rel_err_vs_continuous=r["peak_penetration_mm"]/ref["peak_penetration_mm"] - 1,
                    restitution_rel_err_vs_continuous=r["restitution"]/ref["restitution"] - 1)
    summary = dict(experiment="E19 drop (summary of e19_drop_dt*ms.json)", reference=next(iter(runs.values()))["reference"],
                   soft_contact_model_flat_face=dict(continuous_limit=ref, continuous_limit_k_times_d_variant=ref_variant,
                                                     discrete_at_run_dt=disc),
                   fixture=next(iter(runs.values()))["fixture"], cases=table,
                   code_at_launch={f"{dt*1e3:g} ms": d["code_at_launch"] for dt, d in runs.items()})
    (out/"e19_drop.json").write_text(json.dumps(summary, indent=1, default=float))
    plot(runs, out/"e19_drop.png")
    print("soft-contact 1-D model, continuous limit:", {k: ref[k] for k in ("peak_penetration_mm", "restitution", "hop_mm")},
          "variant k*d:", {k: ref_variant[k] for k in ("peak_penetration_mm", "restitution", "hop_mm")})
    for k, m in disc.items():
        print(f"  discrete model at {k}:", {x: m[x] for x in ("t_first_contact_s", "peak_penetration_mm", "restitution", "hop_mm")})
    for name, per in table.items():
        for dtk, r in per.items():
            e = r.get("energy", {})
            print(f"{name:18s} {dtk:>6s} t_force={r.get('t_first_contact_force_s')} v_imp={r.get('impact_speed_m_s'):.4f} "
                  f"a={r.get('free_fall_accel_m_s2'):.4f} pen={r.get('peak_penetration_mm')} rest_pen={r.get('rest_penetration_mm')} "
                  f"rebound={r.get('rebound_mm'):.3f}mm e={r.get('restitution'):.3f} t_rest={r.get('time_to_rest_s')} "
                  f"tilt={r['tilt_change_deg']:.2f}deg drift={r['lateral_drift_mm']:.2f}mm partners={r['partners']} "
                  f"maxdE={e.get('max_excess_J'):.3e}J@{e.get('max_excess_at_s')} tickinc={e.get('max_tick_increase_after_impact_J'):.2e}")


def plot(runs, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(next(iter(runs.values()))["cases"])
    fig, ax = plt.subplots(4, len(names), figsize=(6.2*len(names), 15.5))
    styles = {0.001: dict(color="#2a78d6", lw=1.5), 0.0005: dict(color="#eb6834", lw=1.1, ls="--"),
              0.00025: dict(color="#1baf7a", lw=1.1, ls=":")}
    for c, name in enumerate(names):
        for dt, d in sorted(runs.items(), reverse=True):
            r = d["cases"][name]; S = r["series"]
            t = np.asarray(S["t"]); st = styles.get(dt, {})
            h = np.asarray(S["h_low_mm"]); E = np.asarray(S["E"]); E0 = r["E0_J"]["total"]
            lab = f"dt {dt*1e3:g} ms"
            ax[0, c].plot(t, h, label=lab, **st)
            ax[1, c].plot(t, h, label=lab, **st)
            ax[2, c].plot(t, (E - E0)*1e3, label=f"{lab} (E0 = {E0*1e3:.1f} mJ)", **st)
            m = (t >= .147) & (t <= .30)
            ax[3, c].plot(t[m], (E[m] - E[-1])*1e3, label=lab, **st)
        r = next(iter(runs.values()))["cases"][name]
        h0 = r["h0_mm"]*1e-3
        tt = np.linspace(0, math.sqrt(2*h0/G), 200)
        ax[0, c].plot(tt, (h0 - .5*G*tt*tt)*1e3, color="#0b0b0b", ls=":", lw=1., label="analytic free fall h0 - g t^2/2")
        for k in (0, 1):
            ax[k, c].axvline(math.sqrt(2*h0/G), color="#898781", lw=.7)
            ax[k, c].axhline(0, color="#898781", lw=.7)
        ax[0, c].set_xlim(0, .6); ax[0, c].set_ylabel("lowest collision point above counter (mm)")
        ax[0, c].set_title(f"E19 {name}: drop from 100 mm (FIXTURE: initial pose only)\nvertical line: analytic contact time sqrt(2h/g)", fontsize=9)
        ax[1, c].set_xlim(.135, .30); ax[1, c].set_ylim(-3, 2); ax[1, c].set_ylabel("lowest collision point above counter (mm)")
        ax[1, c].set_title(f"{name}: impact, penetration and hop (zoom)", fontsize=9)
        mgh = r["mgh_J"]
        ax[2, c].axhline(0, color="#d03b3b", lw=.9, label="E0: any excursion above it is an energy gain")
        ax[2, c].set_ylabel("E(t) - E0 (mJ)")
        ax[2, c].set_title(f"{name}: mechanical energy KE + PE of the body (m g h = {mgh*1e3:.0f} mJ)", fontsize=9)
        ax[3, c].set_ylabel("E(t) - E(end) (mJ)")
        ax[3, c].set_title(f"{name}: energy from 4 ms after first contact, relative to its final value\n(rises = soft-contact penetration energy returned; never above E0)", fontsize=9)
        for k in range(4):
            ax[k, c].set_xlabel("time after release (s)"); ax[k, c].grid(alpha=.3); ax[k, c].legend(fontsize=7)
            for sp in ("top", "right"):
                ax[k, c].spines[sp].set_visible(False)
        lo = 0.
        for d in runs.values():
            S = d["cases"][name]["series"]; t = np.asarray(S["t"]); E = np.asarray(S["E"])
            m = (t >= .147) & (t <= .30)
            lo = min(lo, float(np.min(E[m] - E[-1])))
        ax[3, c].set_ylim(max(lo*1e3 - 5, -60), None)
    fig.tight_layout(); fig.savefig(path, dpi=105)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dt", type=float, default=0.001)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--cases", default=None, help="comma list of cases to run (default all)")
    ap.add_argument("--seconds", type=float, default=T_RUN)
    ap.add_argument("--contacts-window", default=None,
                    help="t0,t1: also record every contact of the dropped body per tick in this window (diagnosis)")
    a = ap.parse_args()
    report(a) if a.report else run(a)


if __name__ == "__main__":
    main()
