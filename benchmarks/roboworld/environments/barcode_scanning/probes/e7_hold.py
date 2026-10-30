"""E7 -- the robot takes the scanner from its cradle, carries it, holds it still and puts it back (E8 by robot).

ASSISTED only in the choice of motion: a scripted operator (scripted.Operator) sets the right arm's joint targets
from IK waypoints computed on the scanner's seated pose, and commands the Franka Hand's force-mode grasp. No
phantom force or attachment touches the scanner: it is held by fingertip contact alone.

Grips:
  side     pads on the handle flanks 35 mm below the head, the hand approaching from behind the handle;
           the trigger stays free
  squeeze  pads front-to-back under the head, one pad on the trigger blade and one on the back of the handle,
           the hand approaching from the scanner's side: the squeeze itself pulls the trigger above 2.35 N

Sequence: pre-grasp 100 mm off -> approach -> grasp (force per finger) -> lift 70 mm along the cup axis ->
carry 0.17 m back from the shelf and up, and back (0.9 s each way) -> hold still 10 s -> return along the cup axis -> release ->
retreat -> 1 s rest. Logged every 10 ms: scanner pose in the TCP frame (slip mm, pivot deg), finger width,
trigger travel/switch, seat sensor, firmware state, pad-to-pad contact force.

  python -m barcode_scanning.probes.e7_hold --grip head --force 17.5 --out barcode_scanning/evidence/e7_hold
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation


def grasp_frame(env, grip, depth):
    """World TCP position and quaternion for the grip, from the scanner's current (seated) pose."""
    from barcode_scanning.scripted import hand_quat
    w = env.world
    sc = w.scene
    ps, Rs = w.body_pose("scanner")
    hc, hq = sc.handle_frame()
    Rh = Rs @ Rotation.from_quat(hq).as_matrix()
    down = -Rh[:, 2]                                           # along the handle, towards its bottom
    top = ps + Rs @ np.asarray(sc.HANDLE_TOP)
    if grip == "head":
        # top-down across the head's sides (66 mm, the Franka Hand opens 80 mm), pads `depth` below the head top
        # (30 mm, about mid-height: a grasp 12 mm below the top bit the edge and the head rolled out, E7),
        # directly above the scanner's centre of mass (Scene.scanner_com)
        hh = sc.HEAD_HALF
        p = ps + Rs @ np.array([float(sc.scanner_com()[0]), 0., hh[2] - depth])
        approach = -Rs[:, 2]
        q = hand_quat(approach=approach, fingers=Rs[:, 1])
        return p, q, approach, np.asarray(sc.cradle.get("socket_axis", Rh[:, 2]), float)   # lifted out along the socket
    if grip == "side":
        p = top + depth*down
        q = hand_quat(approach=Rh[:, 0], fingers=Rh[:, 1])      # from behind the handle, fingers across the flanks
        approach = Rh[:, 0]
    else:
        p = top + depth*down
        q = hand_quat(approach=-Rh[:, 1], fingers=Rh[:, 0])     # from the scanner's right side, pads front-back
        approach = -Rh[:, 1]
    return p, q, approach, Rh[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grip", choices=("head", "side", "squeeze"), default="head")
    ap.add_argument("--force", type=float, default=17.5, help="grasp force per finger (N)")
    ap.add_argument("--arm", default="right")
    ap.add_argument("--depth", type=float, default=None, help="grasp point below the head along the handle (m)")
    ap.add_argument("--hold", type=float, default=10.)
    ap.add_argument("--out", required=True)
    ap.add_argument("--render", action="store_true", help="attach the render worker (scanner frames if the trigger fires)")
    ap.add_argument("--cradle-yaw", type=float, default=None, help="probe-only override of the stockroom cradle yaw (deg)")
    ap.add_argument("--impratio", type=float, default=None, help="probe-only: MuJoCo friction-to-normal impedance ratio")
    ap.add_argument("--dt", type=float, default=None, help="probe-only: step (default: the environment's)")
    ap.add_argument("--noslip", type=int, default=None, help="probe-only: MuJoCo noslip iterations")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    from barcode_scanning.manifest import launch_stamp
    STAMP = launch_stamp()                # the code as launched (not as at the end of the run)
    if a.cradle_yaw is not None:
        from barcode_scanning.scene import Scene
        Scene.STOCKROOM_CRADLE = (Scene.STOCKROOM_CRADLE[0], a.cradle_yaw)
    from barcode_scanning.config import Config
    from barcode_scanning.env import ScanEnv
    from barcode_scanning.scripted import Operator
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    name = f"e7_{a.grip}_{a.force:g}N" + (f"_yaw{a.cradle_yaw:g}" if a.cradle_yaw is not None else "") + a.tag
    env = ScanEnv(Config(preset="stockroom", task="hold-scanner", **({"rigid_dt": a.dt} if a.dt else {})), output_dir=out/"session")
    if a.impratio is not None:
        env.world.solver.mj_model.opt.impratio = a.impratio
    if a.noslip is not None:
        env.world.solver.mj_model.opt.noslip_iterations = a.noslip
    renderer = None
    if a.render:
        from barcode_scanning.render import export, Renderer
        scene_path, meta = export(env.world, out/"render")
        env.render_meta = meta

        def factory():
            nonlocal renderer
            if renderer is None:
                renderer = Renderer(env.world, out/"render", scene_path)
            return renderer
        env.renderer_factory = factory
    w = env.world
    arm = next(k for k in w.rig.instances if k.startswith(a.arm))
    op = Operator(env, record_every=40, frames_dir=out/f"{name}_imager" if a.render else None)
    depth = a.depth if a.depth is not None else {"side": .035, "squeeze": .012, "head": .030}[a.grip]
    p_g, q_g, approach, axis = grasp_frame(env, a.grip, depth)
    rows = []
    sb = w.scene.bodies["scanner"]
    fingers = {i for i, lab in enumerate(w.model.body_label) if lab.startswith(arm) and "finger" in lab}

    def pad_force():
        m, d = w.solver.mj_model, w.solver.mj_data
        f6 = np.zeros(6); tot = 0.
        sbody = np.asarray(w.scene.b.shape_body)
        for i in range(d.ncon):
            g = d.contact[i].geom
            s = [int(w._geom_shape[x]) if 0 <= x < len(w._geom_shape) else -1 for x in g]
            bodies = [int(sbody[x]) if x >= 0 else -1 for x in s]
            if (bodies[0] in fingers and bodies[1] in (sb, w.scene.bodies["scanner_trigger"])) or \
               (bodies[1] in fingers and bodies[0] in (sb, w.scene.bodies["scanner_trigger"])):
                w._mujoco.mj_contactForce(m, d, i, f6); tot += abs(float(f6[0]))
        return tot

    ref = {}

    def log_row():
        pt, Rt = op.tcp(arm)
        ps, Rs = w.body_pose("scanner")
        rel_p = Rt.inv().apply(ps - pt); rel_R = Rt.inv()*Rotation.from_matrix(Rs)
        row = dict(t=w.sim_time, tick=w.tick, phase=op.phase, rel_p_mm=(rel_p*1e3).tolist(), rel_rotvec=rel_R.as_rotvec().tolist(),
                   scanner_pos=ps.tolist(), tcp=pt.tolist(), width_mm=op.finger_width(arm)*1e3, trigger_mm=w.trigger_travel_m()*1e3,
                   switch=w.trigger.state, seated=w.seated, fw=env.firmware.state, pad_force_n=pad_force(),
                   scanner_speed=float(np.linalg.norm(w.state.body_qd.numpy()[sb, :3])))
        if "p" in ref:
            row["slip_mm"] = float(np.linalg.norm(rel_p - ref["p"])*1e3)
            row["pivot_deg"] = math.degrees((ref["R"].inv()*rel_R).magnitude())
        rows.append(row)

    orig = op._chunk

    def chunk(n=10):
        orig(n)
        log_row()
    op._chunk = chunk
    t0 = time.time()
    result = dict(experiment="E7 hold / E8 reseat by robot", code_at_launch=STAMP, assisted="scripted IK waypoints (Operator); no phantom forces",
                  grip=a.grip, force_per_finger_n=a.force, arm=arm, impratio=float(env.world.solver.mj_model.opt.impratio),
                  noslip_iterations=int(env.world.solver.mj_model.opt.noslip_iterations), cradle=list(__import__("barcode_scanning.scene", fromlist=["Scene"]).Scene.STOCKROOM_CRADLE), depth_m=depth, grasp_tcp=p_g.tolist(), grasp_quat=q_g.tolist())
    try:
        op.phase = "ASSISTED scripted: open hand"
        op.open_hand(arm)
        op.phase = "ASSISTED scripted: to pre-grasp"
        op.move_joints(arm, p_g - .10*approach + (.02*axis if a.grip != "head" else 0.), q_g, seconds=2.0)
        op.phase = "ASSISTED scripted: approach"
        op.move(arm, p_g, q_g, seconds=1.2)
        op.phase = f"ASSISTED scripted: grasp {a.force:g} N/finger"
        op.grasp(arm, a.force, seconds=1.0)
        result["width_after_grasp_mm"] = op.finger_width(arm)*1e3
        pt, Rt = op.tcp(arm)
        ps, Rs = w.body_pose("scanner")
        ref["p"] = Rt.inv().apply(ps - pt); ref["R"] = Rt.inv()*Rotation.from_matrix(Rs)
        op.phase = "ASSISTED scripted: lift out of the cup"
        lift = p_g + .07*axis
        op.move(arm, lift, q_g, seconds=1.2)
        result["seated_after_lift"] = w.seated
        op.phase = "ASSISTED scripted: carry out"
        pt, Rt = op.tcp(arm)
        # back from the shelving and up (the first path, 0.20 m sideways and 0.05 m towards the shelf, swung link 6
        # into the shelf's front beam at 344 N; the FR3 collision reflex stopped it)
        carry = pt + np.array([0., -.15, .08])
        op.move(arm, carry, Rt.as_quat(), seconds=0.9)
        op.phase = "ASSISTED scripted: carry back"
        op.move(arm, lift, q_g, seconds=0.9)
        op.phase = f"ASSISTED scripted: hold still {a.hold:g} s"
        i0 = len(rows)
        op.wait(a.hold)
        hold_rows = rows[i0:]
        from barcode_scanning.scripted import reseat
        op.phase = "ASSISTED scripted: scanner upright over the socket, lower, release 10 mm above its floor"
        reseat(op, arm)
        op.phase = "ASSISTED scripted: retreat"
        op.move(arm, p_g - .10*approach + (.02*axis if a.grip != "head" else 0.), q_g, seconds=1.0)
        op.phase = "rest (hands off)"
        op.wait(1.5)
        result["seated_final"] = w.seated
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if renderer is not None:
            renderer.close()
    held = [r for r in rows if "slip_mm" in r and r["phase"].startswith(("ASSISTED scripted: lift", "ASSISTED scripted: carry", "ASSISTED scripted: hold"))]
    if held:
        result["max_slip_mm"] = max(r["slip_mm"] for r in held)
        result["max_pivot_deg"] = max(r["pivot_deg"] for r in held)
        hr = [r for r in held if r["phase"].startswith("ASSISTED scripted: hold")]
        if hr:
            result["hold"] = dict(slip_start_mm=hr[0]["slip_mm"], slip_end_mm=hr[-1]["slip_mm"], pivot_start_deg=hr[0]["pivot_deg"],
                                  pivot_end_deg=hr[-1]["pivot_deg"], drift_mm_per_s=(hr[-1]["slip_mm"] - hr[0]["slip_mm"])/max(1e-6, hr[-1]["t"] - hr[0]["t"]),
                                  pad_force_mean_n=float(np.mean([r["pad_force_n"] for r in hr])), trigger_mm=hr[-1]["trigger_mm"],
                                  switch=hr[-1]["switch"])
        result["max_scanner_speed_m_s"] = max(r["scanner_speed"] for r in held)
    result["trigger_events"] = w.trigger.events
    result["reads"] = len(env.firmware.reads)
    result["reflexes"] = w.reflexes
    result["fault"] = w.fault
    result["wall_s"] = time.time() - t0
    result["sim_s"] = w.sim_time
    meta = op.save(out, name, extra=dict(e7=result))
    (out/f"{name}.json").write_text(json.dumps(dict(result=result, rows=rows), indent=1, default=float))
    print(json.dumps(result, indent=1, default=float))


if __name__ == "__main__":
    main()
