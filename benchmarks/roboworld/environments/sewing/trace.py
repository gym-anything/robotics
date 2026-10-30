"""Privileged per-tick trace of the sewing cell in GAR's shared format (gym_anything_robotics.physics_trace), served by
/v1/history/{id}/trace, /v1/history/{id}/trace/events and `gar export --include-traces`. Solver state, not sensor data.

One sample per 0.5 ms tick of a traced action (no extra ticks are run and nothing in the world is written). The cloth's
state is large (its vertices live on the GPU), so a sample is a summary built from what the world already holds on the
host after each tick -- no GPU read is added:
  joints   positions, velocities, the commanded targets and the rate-limited servo set points of every robot joint
           (fingers and the G1's Dex3 joints included), in the order of the header's joint_names
  robot    arms inside a protective stop, refused targets, the grip force per hand, the external torque estimate per
           arm joint (FR3/Panda: libfranka's tau_ext_hat as the reflex computed it this tick; the G1 has none), the
           cloth's force on each hand as the arm feels it (the filtered pad wrenches, REACTION_TAU = 1 ms)
  contact  MuJoCo's rigid contact count
  machine  shaft angle, speed (ramped and commanded), mode, needle tip height and position, the needle's state, presser
           and the lifter's stop, the presser foot's height and its load on the work, reverse (switch and feed), the
           cycle, holes, needles engaged in the work, stitches set and formed, interlocks, threads (count, broken,
           highest tension), the trimmed / attached tether, the needle's deflection
  wall     the tick's wall time (rigid, cloth, coupling) as the world's own timers counted it, and the sampler's
Events: machine_io (the control box commands), puncture (a hole in a ply), stitch (formed or skipped, with the reason),
machine_mode, presser, needle_state, machine_fault (needle bent / struck plate / struck robot, hook strike, thread
break), trim, protective_stop, fault (world), capture_error, trace_cost (the sampler's own wall time, at the end).
Samples are buffered and written in one transaction per FLUSH_TICKS ticks or FLUSH_WALL_S of wall time (a reader of a
running action sees them that late; a crash loses at most that buffer, and the summary then shows the missing ticks).
A rigid-only world (Config(cloth=False)) has no machine and no cloth: those parts of a sample are null.

The hook: SewingEnv calls its _record() after every tick of an action (sewing/env.py act); bind() shadows that method on
the instance with one that records the env's history first and then samples, as welding/trace.py does. No physics module
is changed (env.py is part of the start state's physics hash).
"""

import math
import shutil
import time

import numpy as np

LIMIT_BYTES = 512*1024*1024
FLUSH_TICKS = 50                 # samples are written in batches of this many ticks ...
FLUSH_WALL_S = 1.                # ... or after this much wall time, whichever comes first (as welding/trace.py)
DISK_RESERVE = 512*1024**2       # an action is not traced when the trace's disk has less free space than this


def _f(v):
    return float(v)


class Trace:
    def __init__(self, env, path, max_bytes=LIMIT_BYTES):
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100_000, max_bytes//256),
                                       synchronous="NORMAL")
        self.env = None
        self.current = None
        self.bind(env)

    # ------------------------------------------------------------------ attachment
    def bind(self, env):
        """Attach to a SewingEnv (again after a variation replaced it): shadow its per-tick _record on the instance."""
        if self.current is not None:
            raise RuntimeError("Cannot rebind an active action trace")
        self.unbind()
        self.env = env
        inner = env._record

        def record():
            inner()
            if self.current is not None:
                self._on_tick()
        record.sewing_trace = self
        env._record = record
        self._prepare()

    def unbind(self):
        env, self.env = self.env, None
        if env is not None and getattr(env.__dict__.get("_record"), "sewing_trace", None) is self:
            del env._record

    def _prepare(self):
        w = self.env.world
        self.joint_names = list(w.joints)
        self._qi = np.array([w.joints[n]["q"] for n in self.joint_names], np.int64)
        self._di = np.array([w.joints[n]["dof"] for n in self.joint_names], np.int64)
        self.torque_arms = list(w._arm_mj_dofs)
        # the rigid bodies carrying each hand's pads (the cloth's reactions reach the arm there)
        self._hand_bodies = {}
        if w.cloth is not None:
            for i, body, _, _ in w.pads:
                name = w.cloth.shapes[i]["name"]
                for arm in w.hand_arms:
                    if name.startswith(arm + "_"):
                        self._hand_bodies.setdefault(arm, set()).add(int(body))
        self._hand_bodies = {a: sorted(b) for a, b in self._hand_bodies.items()}

    # ------------------------------------------------------------------ action lifecycle
    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < DISK_RESERVE:
            return dict(available=False, reason=f"less than {DISK_RESERVE//1024**2} MiB free for the trace; this "
                                                "action is not traced (it runs)")
        env = self.env
        w = env.world
        key = str(context["action_id"])
        from dataclasses import asdict
        self.store.begin(key, str(context["request_id"]), int(env.steps), float(w.config.dt), dict(
            source="sewing/newton1.5 (sewing/trace.py)", task=env.config.task, seed=env.config.seed,
            robot=env.config.robot, start_state=env.start_state,
            config={k: (list(v) if isinstance(v, tuple) else v) for k, v in asdict(env.config).items()},
            joint_names=self.joint_names, arms=list(w.arms), hand_arms=list(w.hand_arms),
            torque_arms={a: [n for n, _ in w._arm_mj_dofs[a]] for a in self.torque_arms},
            hand_bodies={a: [w.model.body_label[b] for b in bs] for a, bs in self._hand_bodies.items()},
            sampling="Every completed 0.5 ms tick of the action (tick = the episode tick of get_state, after the "
                     "step); no additional physics steps, nothing written to the world, no GPU read added.",
            fields="joints{q, qd, target, servo_target} (float32 arrays in joint_names order; rad, or m for the "
                   "Franka Hand's fingers), robot{stopped_arms, refused_targets, grip_force_n per hand (per finger), "
                   "external_torque_nm per arm (float32, torque_arms order; FR3/Panda only), cloth_force_n per hand "
                   "(world frame, the filtered pad wrenches the arm feels; null without cloth)}, contact{rigid_count}, "
                   "machine{shaft_angle_rad, speed_spm (ramped), speed_cmd_spm, mode, needle_tip_z_m, needle "
                   "(up/down), needle_state, presser, lifter_z_m, foot_z_m, foot_load_n, reverse_switch, reverse_feed, "
                   "cycle, holes, engaged, stitches, formed, interlocks, threads{count, broken, max_tension_n}, "
                   "tether, needle_deflection_m[2]} (null without cloth), wall{rigid_s, cloth_s, coupling_s, sample_s} "
                   "(this tick's), wall_s (wall time since the previous sample: the tick plus the sampler)",
            events="machine_io, puncture, stitch, machine_mode, presser, needle_state, machine_fault, trim, "
                   "protective_stop, fault, capture_error, trace_cost",
            missing="The cloth's vertices, velocities, contacts and constraint forces are not sampled (they live on "
                    "the GPU and are not copied per tick; get_state, checkpoints and the evaluator read them). No "
                    "measured force/torque sensors: contact forces are solver forces; the cloth's force "
                    "on a hand is the coupling's filtered value (1 ms), not the instantaneous pad reaction."))
        self.current = key
        M = w.machine
        self._prev = dict(
            world_fault=w.fault, events=len(w.events), reflexes=len(w.reflexes),
            timing=dict(w.timing),
            mode=M.mode if M else None, presser=M.presser if M else None,
            needle_state=M.needle_state if M else None,
            holes=len(M.holes) if M else 0, stitches=len(M.stitches) if M else 0,
            faults=len(M.faults) if M else 0, log=len(M.log) if M else 0,
            formed=sum(1 for s in M.stitches if s.get("formed")) if M else 0)
        self._wall = self._flushed = time.perf_counter()
        self._buf = []
        self._cost = dict(sample_s=0., store_s=0., ticks=0, errors=0)
        self._first_error = None
        return dict(available=True, id=key, store_path=str(self.store.path),
                    note="Privileged solver trace: joints, servo set points, reflexes, external torques, the cloth's "
                         "force on the hands, the machine (shaft, needle, foot, stitches, threads) every 0.5 ms tick; "
                         "not sensor data; the cloth's vertices are not sampled")

    def _error(self, exc):
        self._cost["errors"] += 1
        if self._first_error is None:
            self._first_error = f"{type(exc).__name__}: {exc}"

    def _on_tick(self):
        t0 = time.perf_counter()
        tick = int(self.env.steps)
        try:
            sample, events = self._sample(tick, t0)
        except Exception as exc:                       # a sampler failure never stops the physics; it is recorded
            self._error(exc)
            sample, events = None, [dict(type="capture_error", tick=tick, error=f"{type(exc).__name__}: {exc}")]
        self._buf.append((tick, sample, events))
        t1 = time.perf_counter()
        self._cost["sample_s"] += t1 - t0
        self._cost["ticks"] += 1
        if len(self._buf) >= FLUSH_TICKS or t1 - self._flushed >= FLUSH_WALL_S:
            self._flush()
        self._wall = time.perf_counter()

    def _flush(self):
        """Write the buffered ticks in one transaction; a store failure is recorded, never raised into the physics."""
        if not self._buf:
            return
        t = time.perf_counter()
        buf, self._buf = self._buf, []
        try:
            self.store.append_batch(self.current, buf)
        except Exception as exc:
            self._error(exc)
        self._flushed = time.perf_counter()
        self._cost["store_s"] += self._flushed - t

    # ------------------------------------------------------------------ the sample
    @staticmethod
    def _tether(M):
        return "trimmed" if M.tether == "trimmed" else ("attached" if M.tether is not None else None)

    def _sample(self, tick, t0):
        env = self.env
        w = env.world
        jq, jqd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()          # the rigid model lives on the host
        servo = w.control.joint_target_q.numpy()
        tm, prev_tm = w.timing, self._prev["timing"]
        timing = {k: _f(tm[k] - prev_tm.get(k, 0.)) for k in tm}
        self._prev["timing"] = dict(tm)
        robot = dict(stopped_arms=sorted(a for a, u in w._reflex_until.items() if w.time < u),
                     refused_targets=int(w.refused_targets),
                     grip_force_n={a: _f(f) for a, f in w.grip_force.items()})
        if self.torque_arms:
            w.external_torque_estimate()                  # the reflex's value of this tick (cached), computed if not
            ext = w._ext_cache[1]
            robot["external_torque_nm"] = {a: np.asarray(ext[a], np.float32) for a in self.torque_arms}
        if self._hand_bodies:
            wr = w._wrench_filtered
            robot["cloth_force_n"] = {a: [_f(x) for x in wr[bs, :3].sum(axis=0)] for a, bs in self._hand_bodies.items()}
        else:
            robot["cloth_force_n"] = None
        sample = dict(
            tick=tick, world_tick=int(w.tick), t_s=tick*w.config.dt, wall_s=t0 - self._wall,
            joints=dict(q=jq[self._qi].astype(np.float32), qd=jqd[self._di].astype(np.float32),
                        target=w.targets[self._qi].astype(np.float32), servo_target=servo[self._qi].astype(np.float32)),
            robot=robot, contact=dict(rigid_count=int(w.solver.mj_data.ncon)),
            machine=self.machine_summary(w.machine), wall=timing)
        events = self._events(tick)
        sample["wall"]["sample_s"] = time.perf_counter() - t0
        return sample, events

    def machine_summary(self, M):
        """The machine's state after a tick (host values the machine keeps; None without cloth)."""
        if M is None:
            return None
        new = M.stitches[self._prev["stitches"]:] if len(M.stitches) >= self._prev["stitches"] else M.stitches
        self._prev["formed"] += sum(1 for s in new if s.get("formed"))
        broken, tmax = 0, 0.
        for t in M.threads:
            if t["broken"]:
                broken += 1
            elif t["tension"] > tmax:
                tmax = t["tension"]
        tip = float(M.needle_tip(M.theta))
        return dict(shaft_angle_rad=_f(M.theta % (2*math.pi)), speed_spm=_f(M.speed_ramped),
                    speed_cmd_spm=_f(M.speed_cmd), mode=M.mode, needle_tip_z_m=tip, needle="down" if tip < 0 else "up",
                    needle_state=M.needle_state, presser=M.presser, lifter_z_m=_f(M.lifter_z), foot_z_m=_f(M.foot_z),
                    foot_load_n=_f(M.foot_N), reverse_switch=bool(M.reverse), reverse_feed=bool(M.reverse_now),
                    cycle=int(M.cycle), holes=len(M.holes), engaged=len(M.engaged), stitches=len(M.stitches),
                    formed=int(self._prev["formed"]), interlocks=len(M.interlocks),
                    threads=dict(count=len(M.threads), broken=broken, max_tension_n=_f(tmax)),
                    tether=self._tether(M), needle_deflection_m=[_f(x) for x in M.last_defl])

    def _events(self, tick):
        w = self.env.world
        M = w.machine
        prev = self._prev
        out = []

        def since(lst, key):
            k = prev[key]
            new = lst[k:] if len(lst) >= k else lst
            prev[key] = len(lst)
            return new
        for e in since(w.events, "events"):
            if "machine_io" in e:
                out.append(dict(type="machine_io", tick=tick, world_tick=int(e.get("tick", w.tick)),
                                outputs={k: (v if isinstance(v, (bool, str)) else float(v))
                                         for k, v in e["machine_io"].items()}))
        for r in since(w.reflexes, "reflexes"):
            out.append(dict(type="protective_stop", tick=tick, world_tick=int(r.get("tick", w.tick)),
                            **{k: _plain(v) for k, v in r.items() if k != "tick"}))
        if M is not None:
            for h in since(M.holes, "holes"):
                out.append(dict(type="puncture", tick=tick, world_tick=int(h.tick), ply=int(h.ply),
                                uv_m=[_f(x) for x in h.uv], cycle=int(h.cycle), edge=bool(h.edge)))
            for s in since(M.stitches, "stitches"):
                out.append(dict(type="stitch", tick=tick, world_tick=int(s.get("tick", w.tick)), cycle=s.get("cycle"),
                                formed=bool(s.get("formed")), reason=s.get("reason"), reverse=bool(s.get("reverse")),
                                holes=len(s.get("holes", ())), needle=s.get("needle")))
            for f in since(M.faults, "faults"):
                out.append(dict(type="machine_fault", tick=tick, **{k: _plain(v) for k, v in f.items() if k != "tick"},
                                world_tick=int(f.get("tick", w.tick))))
            for e in since(M.log, "log"):
                if e.get("event") == "trim":
                    out.append(dict(type="trim", tick=tick, cycle=e.get("cycle")))
            for key, kind in (("mode", "machine_mode"), ("presser", "presser"), ("needle_state", "needle_state")):
                now = getattr(M, key)
                if now != prev[key]:
                    out.append(dict(type=kind, tick=tick, **{"from": prev[key], "to": now}))
                    prev[key] = now
        if w.fault and w.fault != prev["world_fault"]:
            out.append(dict(type="fault", tick=tick, source="world", fault=w.fault))
            prev["world_fault"] = w.fault
        return out

    def finish(self, error=None):
        if self.current is None:
            return None
        key = self.current
        env = self.env
        tick = int(env.steps)
        self._flush()
        c = self._cost
        total = c["sample_s"] + c["store_s"]
        tail = [dict(type="trace_cost", tick=tick, ticks=c["ticks"], sampler_s=round(total, 6),
                     sampler_ms_per_tick=round(1e3*total/c["ticks"], 4) if c["ticks"] else None,
                     sampling_s=round(c["sample_s"], 6), store_s=round(c["store_s"], 6), capture_errors=c["errors"],
                     first_error=self._first_error)]
        self.store.append(key, tick, None, tail)
        self.current = None
        fault = error or env.world.fault
        self.store.finish(key, tick, status="failed" if fault else "completed", capture_errors=c["errors"],
                          error=fault or self._first_error)
        return self.store.summary(key)

    def close(self):
        try:
            self.finish()
        finally:
            self.unbind()
            self.store.close()


def _plain(v):
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.astype(float).tolist()
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, float) and not math.isfinite(v):
        return None
    return v
