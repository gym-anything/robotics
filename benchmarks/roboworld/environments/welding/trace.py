"""Privileged per-physics-step trace of the welding cell in GAR's shared format (gym_anything_robotics.physics_trace),
served by /v1/history/{id}/trace, /v1/history/{id}/trace/events and `gar export --include-traces`. Solver state, not
sensor data.

One sample per 1 ms physics tick of a traced action (no extra ticks are run and nothing in the world is written):
  joints   positions, velocities, the commanded targets and the rate-limited servo set points of every joint of both
           arms (fingers included), in the order of the header's joint_names
  torch    the torch work point's world pose (position, quaternion xyzw; on the wire axis 19 mm beyond the contact
           tip, z along the wire: the Cartesian frame of the torch arm), the contact tip and wire axis in the weld frame
           (x along the joint, y out of the bracket's weld face, z up), the travel and work angles the process uses
  source   the welding power source: arc on, current, terminal and arc voltage, actual wire feed, stick-out, contact tip
           to work (to the metal the arc would reach; None if none within reach), arc length, gas, touch, fault, the
           robot's commanded outputs and set points, stubbing, the air fraction under the nozzle, circuit closed
  contact  the net contact force on each arm's bodies (the gun included) from everything that is not that arm, and
           MuJoCo's contact count; gun_contacts: every (gun shape, other shape) pair in contact with its summed normal
           force (the evaluator's 'no contact' criterion uses the same pairs)
  bond     active, broken, capacity, the load check's interaction ratio and its force and moment, bonded length
  pool     liquid slices, filler deposited / fed / spattered, the metal volume change counted from the thermal cells
           (and the pool's own per-slice bookkeeping beside it), frozen volume, shrinkage debt, unplaced and spilled
           filler, and the energy ledger (input, stored change, losses, residual)
  robot    arms inside a protective stop, refused targets
Events: tool_io (the robot's outputs to the source), arc_strike / arc_extinguish, touch_start / touch_end,
circuit_closed / circuit_open, fault (source or world), protective_stop, bond_made / bond_broken, gun_contact_start /
gun_contact_end, pool_event / process_event, capture_error, trace_cost (the sampler's own wall time, at the end).
Samples are buffered and written in one transaction per FLUSH_TICKS ticks or FLUSH_WALL_S of wall time (a reader of a
running action sees them that late; a crash loses at most that buffer, and the summary then shows the missing ticks).

The energy ledger needs the coupon's stored enthalpy, a sum over 7.2 M cells. The thermal solver's own reduction
(ThermalSolver.stored, one float64 atomic per metal cell) takes 28 ms on the shared RTX 5090 -- about as long as the
tick's thermal substeps -- so the trace computes the same sum (the same per-cell terms, a different summation order)
with a strided reduction kernel of its own, together with the cell-counted metal volume, every tick on which the
thermal solver ran. Each tick's sums and a device copy of the solver's ledger accumulators go to a row of a GPU buffer
that is read back once per flush (waiting for the GPU every tick cost 15-20 ms on the shared GPU; the kernel ~2 ms).
While the coupon has never been heated the solver is skipped exactly and the ledger cannot change: the sample then
carries the values computed at the start of the action (energy.evaluated = false).

The hook: WeldingEnv calls its _record() after every tick of an action (welding/env.py act); bind() shadows that
method on the instance with one that records the env's history first and then samples. No physics module is changed
(env.py is part of the tacked start state's physics hash).
"""

import math
import shutil
import time

import numpy as np

LIMIT_BYTES = 512*1024*1024
FLUSH_TICKS = 50                 # samples are written in batches of this many ticks ...
FLUSH_WALL_S = 1.                # ... or after this much wall time, whichever comes first (one SQLite transaction
                                 # per step cost 0.6 ms: measured)
_STRIDE_THREADS = 32768          # reduction threads (each sums ~220 cells; 2 float64 atomics per thread)
_kernel = None


def _reduction_kernel():
    """Stored enthalpy (J) and metal volume (m3) of the coupon's cells: the terms of ThermalSolver.stored ((H + its
    float32 remainder Hlo) x the float32 cell volume, summed in float64) and of WeldProcess.ledger's metal volume
    (float64 cell volumes)."""
    global _kernel
    if _kernel is not None:
        return _kernel
    import warp as wp

    @wp.kernel
    def k_trace_sums(mat: wp.array(dtype=wp.uint8), H: wp.array(dtype=wp.float32), dx: wp.array(dtype=wp.float32),
                     dy: wp.array(dtype=wp.float32), dz: wp.array(dtype=wp.float32), ny: int, nz: int, n: int,
                     stride: int, out: wp.array(dtype=wp.float64), base: int, Hlo: wp.array(dtype=wp.float32)):
        t = wp.tid()
        e = wp.float64(0.0)
        v = wp.float64(0.0)
        i = t
        while i < n:
            if int(mat[i]) != 0:
                k = i % nz
                j = (i//nz) % ny
                ii = i//(ny*nz)
                e += (wp.float64(H[i]) + wp.float64(Hlo[i]))*wp.float64(dx[ii]*dy[j]*dz[k])
                v += wp.float64(dx[ii])*wp.float64(dy[j])*wp.float64(dz[k])
            i += stride
        wp.atomic_add(out, base, e)
        wp.atomic_add(out, base + 1, v)

    _kernel = k_trace_sums
    return _kernel


def _quat_mul(a, b):
    """Hamilton product of xyzw quaternions."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (aw*bx + ax*bw + ay*bz - az*by, aw*by - ax*bz + ay*bw + az*bx,
            aw*bz + ax*by - ay*bx + az*bw, aw*bw - ax*bx - ay*by - az*bz)


def _quat_rotate(q, v):
    x, y, z, w = q
    t = 2.*np.cross((x, y, z), v)
    return v + w*t + np.cross((x, y, z), t)


def _f(v):
    return float(v)


def _fin(v):
    v = float(v)
    return v if math.isfinite(v) else None


class Trace:
    def __init__(self, env, path, max_bytes=LIMIT_BYTES):
        from gym_anything_robotics.physics_trace import PhysicsTraceStore
        self.store = PhysicsTraceStore(path, max_bytes=max_bytes, max_items=max(100_000, max_bytes//256),
                                       synchronous="NORMAL")
        self.env = None
        self.current = None
        self.selfcheck = None
        self.bind(env)

    # ------------------------------------------------------------------ attachment
    def bind(self, env):
        """Attach to a WeldingEnv (again after a variation replaced it): shadow its per-tick _record on the instance."""
        if self.current is not None:
            raise RuntimeError("Cannot rebind an active action trace")
        self.unbind()
        self.env = env
        inner = env._record

        def record():
            inner()
            if self.current is not None:
                self._on_tick()
        record.welding_trace = self
        env._record = record
        self._prepare()

    def unbind(self):
        env, self.env = self.env, None
        if env is not None and getattr(env.__dict__.get("_record"), "welding_trace", None) is self:
            del env._record

    def _prepare(self):
        w = self.env.world
        self.joint_names = list(w.joints)
        self._qi = np.array([w.joints[n]["q"] for n in self.joint_names], np.int64)
        self._di = np.array([w.joints[n]["dof"] for n in self.joint_names], np.int64)
        self.arms = list(w.arms)
        lab = list(w.model.body_label)
        body_arm = np.full(len(lab) + 1, -1, np.int64)        # index -1 (static shapes) -> no arm
        for i in w._robot_bodies:
            for k, a in enumerate(self.arms):
                if lab[i].startswith(a + "/"):
                    body_arm[i] = k
        self._body_arm = body_arm
        nshape = len(w.model.shape_label)
        gun = np.zeros(nshape + 1, bool)
        gun[list(w.scene.gun_shapes)] = True
        self._gun = gun
        self._shape_label = list(w.model.shape_label)
        # torch work point in the gun body's frame: tool0 in the body, then the work point in tool0 (kinematics.py)
        from scipy.spatial.transform import Rotation
        from .kinematics import torch_tcp_in_tool0
        pt, Rt = w.scene.tool0_in_body
        T = torch_tcp_in_tool0(w.scene.gun_spec)
        self._tcp_t = np.asarray(pt, float) + Rt @ T[:3, 3]
        self._tcp_q = tuple(Rotation.from_matrix(Rt @ T[:3, :3]).as_quat())
        p, q = self._work_point()
        _, _, tcp = w.torch_pose()
        if np.linalg.norm(p - tcp) > 1e-6:
            raise RuntimeError(f"trace work point {p} differs from the world's torch TCP {tcp}")
        self._dev = None
        self._pending = []
        self._v0 = None
        self._ledger_cache = None
        self.selfcheck = None

    # ------------------------------------------------------------------ cheap per-tick reads
    def _work_point(self):
        w = self.env.world
        b = w.state.body_q.numpy()[w.scene.gun_body]
        qb = tuple(float(x) for x in b[3:7])
        p = np.asarray(b[:3], float) + _quat_rotate(qb, self._tcp_t)
        return p, _quat_mul(qb, self._tcp_q)

    def _contacts(self):
        """(net contact force per arm, gun contact pairs, contact count) in one pass over MuJoCo's contacts, with the
        world's own conventions (World.robot_contact_force, World.gun_contacts)."""
        w = self.env.world
        mj = w.solver._mujoco
        m, d = w.solver.mj_model, w.solver.mj_data
        n = int(d.ncon)
        F = np.zeros((len(self.arms), 3))
        pairs = {}
        if n:
            geom = np.asarray(d.contact.geom[:n])
            s = w._g2s[geom]
            b = w._g2b[geom]
            arm = self._body_arm[b]
            gun = self._gun[s]
            rel = ((arm[:, 0] != arm[:, 1]) & ((arm[:, 0] >= 0) | (arm[:, 1] >= 0))) | gun[:, 0] | gun[:, 1]
            idx = np.nonzero(rel)[0]
            if len(idx):
                f6 = np.zeros(6)
                frames = np.asarray(d.contact.frame[:n])
                for i in idx:
                    mj.mj_contactForce(m, d, int(i), f6)
                    a1, a2 = int(arm[i, 0]), int(arm[i, 1])
                    if a1 != a2:
                        fw = frames[i].reshape(3, 3).T @ f6[:3]
                        if a2 >= 0:
                            F[a2] += fw
                        if a1 >= 0:
                            F[a1] -= fw
                    g1, g2 = bool(gun[i, 0]), bool(gun[i, 1])
                    if (g1 or g2) and abs(f6[0]) >= 1e-2:
                        sg, so = (int(s[i, 0]), int(s[i, 1])) if g1 else (int(s[i, 1]), int(s[i, 0]))
                        e = pairs.setdefault((sg, so), [0., 0])
                        e[0] += abs(float(f6[0]))
                        e[1] += 1
        lab = self._shape_label
        gc = [dict(gun=lab[g], other=lab[o] if o >= 0 else "static", force_n=round(f, 3), contacts=k)
              for (g, o), (f, k) in pairs.items()]
        return {a: F[k].tolist() for k, a in enumerate(self.arms)}, gc, n

    # ------------------------------------------------------------------ the ledger's cell sums
    ROW = 10                     # device row per tick: stored enthalpy, metal volume, the solver's 8 ledger accumulators

    def _enqueue_sums(self, row):
        """Launch the reduction of this tick's cells into device row `row` and copy the solver's ledger accumulators
        beside it -- all asynchronous: nothing waits for the GPU here (a wait costs 15-20 ms on the shared GPU; the
        kernel itself about 2 ms). The rows are read back once per flush."""
        import warp as wp
        s = self.env.world.process.s
        g = s.g
        if self._dev is None:
            self._dev = wp.zeros(self.ROW*(FLUSH_TICKS + 1), dtype=wp.float64, device=s.device)
        n = min(g.n, _STRIDE_THREADS)
        wp.launch(_reduction_kernel(), dim=n, inputs=[s.w_mat, s.w_H, s.w_dx, s.w_dy, s.w_dz, g.ny, g.nz, g.n, n,
                                                      self._dev, self.ROW*row, s.w_Hlo], device=s.device)
        wp.copy(self._dev, s.w_ledger, dest_offset=self.ROW*row + 2, src_offset=0, count=8)

    def _ledger_from(self, values, arc_j):
        """The ledger (WeldProcess.ledger's quantities) from a row: stored, metal volume, accumulators."""
        p = self.env.world.process
        if self._v0 is None:
            self._v0 = getattr(p, "_v0", None)
            if self._v0 is None:
                self._v0 = float(np.sum(p.g.volumes()[p.original_metal]))
        stored, vol = float(values[0]), float(values[1])
        inp, loss, sink, evap, dep, rem, det, det_v = (float(x) for x in values[2:10])
        s0 = p.s.stored0
        residual = (stored - s0) - (inp + dep - rem - det - loss - sink - evap)
        return dict(input_j=inp, arc_delivered_j=float(arc_j), stored_change_j=stored - s0, surface_loss_j=loss,
                    sink_loss_j=sink, evaporation_j=evap, deposited_metal_j=dep, removed_metal_j=rem,
                    residual_j=residual, residual_rel=residual/max(inp + dep, 1.),
                    metal_volume_change_m3=vol + det_v - self._v0)

    def _row_now(self):
        self._enqueue_sums(FLUSH_TICKS)
        row = self._dev.numpy()[self.ROW*FLUSH_TICKS:self.ROW*(FLUSH_TICKS + 1)].copy()
        self._dev.zero_()
        return row

    def _ledger_now(self):
        """The ledger now (synchronous: at the start of an action and for the self-check)."""
        out = self._ledger_from(self._row_now(), self.env.world.process.e_input_arc)
        self._ledger_cache = out
        return out

    def _resolve_pending(self):
        """Read the buffered rows (one wait for the GPU per flush) and complete those samples' energy and metal."""
        if not self._pending:
            return
        rows = self._dev.numpy().reshape(-1, self.ROW)
        for k, (sample, arc_j) in enumerate(self._pending):
            led = self._ledger_from(rows[k], arc_j)
            sample["pool"]["metal_volume_change_m3"] = led.pop("metal_volume_change_m3")
            sample["energy"] = dict(led, evaluated=True)
            self._ledger_cache = dict(led, metal_volume_change_m3=sample["pool"]["metal_volume_change_m3"])
        self._pending = []
        self._dev.zero_()

    def _selfcheck(self):
        """Once per binding: the trace's reduction against the process's own ledger (the same quantities)."""
        p = self.env.world.process
        if p is None:
            return None
        ref = p.ledger()
        row = self._row_now()
        # the metal volume's baseline is aligned with the ledger's here: the GPU holds the cell sizes in float32, the
        # host's ledger in float64 (a constant 4e-11 m3 on the 3.1e-4 m3 coupon, measured)
        host_v0 = getattr(p, "_v0", None)
        self._v0 = float(row[1]) - ref["metal_volume_change_m3"]
        mine = self._ledger_from(row, p.e_input_arc)
        self._ledger_cache = mine
        return dict(stored_change_j=[mine["stored_change_j"], ref["stored_change_j"]],
                    residual_rel=[mine["residual_rel"], ref["residual_rel"]],
                    metal_volume_baseline_offset_m3=None if host_v0 is None else self._v0 - host_v0,
                    note="[trace reduction, WeldProcess.ledger()]: the same per-cell terms summed in another order; "
                         "the cell-counted metal volume is aligned with the ledger's at this point (float32 cell "
                         "sizes on the GPU against float64 on the host)")

    # ------------------------------------------------------------------ action lifecycle
    def begin(self, context):
        if self.current:
            raise RuntimeError("Action trace already active")
        if shutil.disk_usage(self.store.path.parent).free < 512*1024**2:
            raise OSError("Trace disk reserve unavailable; no action executed")
        env = self.env
        w = env.world
        key = str(context["action_id"])
        if self.selfcheck is None:
            self.selfcheck = self._selfcheck() or {}
        p = w.process
        self.store.begin(key, str(context["request_id"]), int(env.steps), float(w.config.rigid_dt), dict(
            source="welding/newton1.5 (welding/trace.py)", task=env.config.task, robot=env.config.robot,
            config={k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(env.config).items()},
            joint_names=self.joint_names, arms=self.arms, torch_arm=w.torch_arm,
            gun_shapes=[self._shape_label[i] for i in w.scene.gun_shapes],
            sampling="Every completed 1 ms physics tick of the action (tick = the episode tick of get_state, after "
                     "the step); no additional physics steps, nothing written to the world.",
            fields="joints{q, qd, target, servo_target} (float32 arrays in joint_names order; rad or m for fingers), "
                   "torch{work_point_m, work_point_quat_xyzw (world), contact_tip_weld_m, wire_axis_weld, "
                   "travel_angle_deg, work_angle_deg}, source{arc_on, current_a, voltage_v (terminal), arc_voltage_v, "
                   "wire_feed_m_min (actual), stickout_m, ctwd_m (contact tip to the metal the arc reaches; null if "
                   "none in reach), arc_length_m, gas_on, touch, fault, stubbing, x_air, circuit_closed, commanded{...}}, "
                   "contact{net_force_n per arm (world frame, on the arm and its tool), count}, gun_contacts[], "
                   "bond{active, broken, capacity_n, ratio, force_n, moment_nm, length_m}, pool{liquid_slices, "
                   "deposited_m3, fed_m3, spatter_m3, metal_volume_change_m3 (cells), metal_added_pool_m3 (pool "
                   "bookkeeping), frozen_m3, liquid_expansion_m3, unplaced_m3, spilled_m3, stacked_m3}, energy{input_j, "
                   "arc_delivered_j, stored_change_j, surface_loss_j, sink_loss_j, evaporation_j, deposited_metal_j, "
                   "removed_metal_j, residual_j, residual_rel, evaluated}, robot{stopped_arms, refused_targets}, "
                   "wall_s (wall time since the previous sample: the tick plus the sampler)",
            events="tool_io, arc_strike, arc_extinguish, touch_start, touch_end, circuit_closed, circuit_open, fault, "
                   "protective_stop, bond_made, bond_broken, gun_contact_start, gun_contact_end, pool_event, "
                   "process_event, capture_error, trace_cost",
            ledger="energy and the cell-counted metal volume come from a reduction over all 7.2 M cells on every tick "
                   "on which the thermal solver ran; before the coupon was ever heated the solver is skipped exactly "
                   "and the values are those at the start of the action (energy.evaluated=false)",
            ledger_selfcheck=self.selfcheck,
            missing="No measured force/torque sensors: contact forces are MuJoCo constraint forces. The thermal "
                    "field, the pool's profiles and the cells are not sampled (owner checkpoints hold them)."))
        self.current = key
        st = p.src.st if p is not None else None
        self._prev = dict(
            arc=bool(st.arc) if st else False, touch=bool(st.touch) if st else False,
            fault=st.fault if st else "", circuit=bool(w.circuit_closed), world_fault=w.fault,
            events=len(w.events), reflexes=len(w.reflexes), bond_events=len(w.bond.events),
            gun=set(), pool_events=len(p.pool.events) if p is not None else 0,
            process_events=len(p.events) if p is not None else 0)
        self._wall = self._flushed = time.perf_counter()
        self._buf = []
        self._cost = dict(sample_s=0., store_s=0., ticks=0, ledger_evaluations=0, errors=0)
        self._first_error = None
        self._pending = []
        if p is not None:
            self._ledger_now()
        return dict(available=True, id=key, store_path=str(self.store.path),
                    note="Privileged solver trace: joints, torch work point, the welding source, contact forces, gun "
                         "contacts, bond and pool every 1 ms tick; not sensor data")

    def _error(self, exc):
        self._cost["errors"] += 1
        if self._first_error is None:
            self._first_error = f"{type(exc).__name__}: {exc}"

    def _on_tick(self):
        t0 = time.perf_counter()
        env = self.env
        tick = int(env.steps)
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
        """Write the buffered steps in one transaction; a store failure is recorded, never raised into the physics
        (the summary then shows the missing ticks and the capture errors)."""
        if not self._buf:
            return
        t = time.perf_counter()
        buf, self._buf = self._buf, []
        try:
            self._resolve_pending()
        except Exception as exc:
            self._error(exc)
            self._pending = []
        try:
            self.store.append_batch(self.current, buf)
        except Exception as exc:
            self._error(exc)
        self._flushed = time.perf_counter()
        self._cost["store_s"] += self._flushed - t

    def _sample(self, tick, t0):
        env = self.env
        w = env.world
        p = w.process
        jq, jqd = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
        servo = w.control.joint_target_q.numpy()
        wp_, wq = self._work_point()
        forces, gun, ncon = self._contacts()
        b = w.bond
        sample = dict(
            tick=tick, world_tick=int(w.tick), t_s=tick*w.config.rigid_dt, wall_s=t0 - self._wall,
            joints=dict(q=jq[self._qi].astype(np.float32), qd=jqd[self._di].astype(np.float32),
                        target=w.targets[self._qi].astype(np.float32), servo_target=servo[self._qi].astype(np.float32)),
            torch=dict(work_point_m=[_f(x) for x in wp_], work_point_quat_xyzw=[_f(x) for x in wq]),
            contact=dict(net_force_n=forces, count=ncon), gun_contacts=gun,
            bond=dict(active=bool(b.active), broken=bool(b.broken), capacity_n=_f(b.capacity_n),
                      ratio=_f(b.load["ratio"]), force_n=_f(b.load["force_n"]), moment_nm=_f(b.load["moment_nm"]),
                      length_m=_f(b.length_m)),
            robot=dict(stopped_arms=sorted(a for a, u in w._reflex_until.items() if w.time < u),
                       refused_targets=int(w.refused_targets)))
        if p is not None:
            st, s = p.src.st, p.src.s
            tip = w._tip_prev
            ax = p.axis_last
            sample["torch"].update(
                contact_tip_weld_m=None if tip is None else [_f(x) for x in tip],
                wire_axis_weld=None if ax is None else [_f(x) for x in ax],
                travel_angle_deg=_fin(p.travel_angle), work_angle_deg=None if p.work_angle is None else _fin(p.work_angle))
            sample["source"] = dict(
                arc_on=bool(st.arc), current_a=_f(st.current), voltage_v=_f(st.v_term), arc_voltage_v=_f(st.v_arc),
                wire_feed_m_min=_f(st.wfs_actual*60.), stickout_m=_f(st.l_s), ctwd_m=_fin(st.d_tw),
                arc_length_m=_f(st.l_a), gas_on=bool(st.gas_on), touch=bool(st.touch), fault=st.fault or None,
                stubbing=bool(st.stubbing), x_air=_f(st.x_air), circuit_closed=bool(w.circuit_closed),
                commanded=dict(arc_on=bool(st.on_cmd), touch_sense=bool(st.sensing),
                               wire_inch={1: "forward", -1: "backward", 0: "stop"}.get(st.inch, str(st.inch)),
                               wire_feed_m_min=_f(s.wfs_m_min), voltage_v=_f(s.voltage_v)),
                transfer_mode=st.mode or None)
            pool = p.pool
            sample["pool"] = dict(
                liquid_slices=sum(1 for x in pool.slices if x.liquid), deposited_m3=_f(st.deposited_m3),
                fed_m3=_f(st.fed_m3), spatter_m3=_f(st.spatter_m3), metal_added_pool_m3=_f(pool.dV.sum()),
                frozen_m3=_f(pool.frozen_volume), liquid_expansion_m3=_f(pool.liquid_expansion), unplaced_m3=_f(pool.excess),
                spilled_m3=_f(pool.spilled_m3), stacked_m3=_f(pool.stacked_m3))
            # the thermal solver ran on this tick iff the coupon has ever been heated (process.tick skips it exactly
            # before): only then can the stored enthalpy, the losses or the cells have changed
            if p.ever_hot:
                self._enqueue_sums(len(self._pending))
                self._pending.append((sample, float(p.e_input_arc)))
                self._cost["ledger_evaluations"] += 1
            else:
                led = dict(self._ledger_cache)
                sample["pool"]["metal_volume_change_m3"] = led.pop("metal_volume_change_m3")
                sample["energy"] = dict(led, evaluated=False)
        return sample, self._events(tick, gun)

    def _events(self, tick, gun):
        env = self.env
        w = env.world
        p = w.process
        prev = self._prev
        out = []
        for e in w.events[prev["events"]:] if len(w.events) >= prev["events"] else w.events:
            if "tool_io" in e:
                out.append(dict(type="tool_io", tick=tick, world_tick=int(e.get("tick", w.tick)),
                                outputs={k: (v if isinstance(v, (bool, str)) else float(v)) for k, v in e["tool_io"].items()}))
        prev["events"] = len(w.events)
        if p is not None:
            st = p.src.st
            if bool(st.arc) != prev["arc"]:
                out.append(dict(type="arc_strike" if st.arc else "arc_extinguish", tick=tick, current_a=_f(st.current),
                                ctwd_m=_fin(st.d_tw), stickout_m=_f(st.l_s), fault=st.fault or None))
                prev["arc"] = bool(st.arc)
            if bool(st.touch) != prev["touch"]:
                out.append(dict(type="touch_start" if st.touch else "touch_end", tick=tick, stickout_m=_f(st.l_s),
                                ctwd_m=_fin(st.d_tw)))
                prev["touch"] = bool(st.touch)
            if st.fault != prev["fault"]:
                if st.fault:
                    out.append(dict(type="fault", tick=tick, source="welding power source", fault=st.fault))
                prev["fault"] = st.fault
            for name in ("pool", "process"):
                lst = p.pool.events if name == "pool" else p.events
                k = prev[name + "_events"]
                for e in (lst[k:] if len(lst) >= k else lst):
                    out.append(dict(type=name + "_event", tick=tick, event=_plain(e)))
                prev[name + "_events"] = len(lst)
        if bool(w.circuit_closed) != prev["circuit"]:
            out.append(dict(type="circuit_closed" if w.circuit_closed else "circuit_open", tick=tick))
            prev["circuit"] = bool(w.circuit_closed)
        if w.fault and w.fault != prev["world_fault"]:
            out.append(dict(type="fault", tick=tick, source="world", fault=w.fault))
            prev["world_fault"] = w.fault
        k = prev["reflexes"]
        for r in (w.reflexes[k:] if len(w.reflexes) >= k else w.reflexes):
            out.append(dict(type="protective_stop", tick=tick, world_tick=int(r.get("tick", w.tick)),
                            **{k: _plain(v) for k, v in r.items() if k != "tick"}))
        prev["reflexes"] = len(w.reflexes)
        k = prev["bond_events"]
        for e in (w.bond.events[k:] if len(w.bond.events) >= k else w.bond.events):
            kind = "bond_broken" if e.get("event") == "broken" else "bond_made"
            out.append(dict(type=kind, tick=tick, **{kk: _plain(vv) for kk, vv in e.items() if kk != "tick"},
                            world_tick=int(e.get("tick", w.tick))))
        prev["bond_events"] = len(w.bond.events)
        now = {(g["gun"], g["other"]): g for g in gun}
        for key in now.keys() - prev["gun"]:
            out.append(dict(type="gun_contact_start", tick=tick, **now[key]))
        for key in prev["gun"] - now.keys():
            out.append(dict(type="gun_contact_end", tick=tick, gun=key[0], other=key[1]))
        prev["gun"] = set(now)
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
                     sampling_s=round(c["sample_s"], 6), store_s=round(c["store_s"], 6),
                     ledger_evaluations=c["ledger_evaluations"], capture_errors=c["errors"],
                     first_error=self._first_error)]
        self.store.append(key, tick, None, tail)
        self.current = None
        fault = error or env.world.fault
        status = "failed" if fault else "completed"
        self.store.finish(key, tick, status=status, capture_errors=c["errors"], error=fault or self._first_error)
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
