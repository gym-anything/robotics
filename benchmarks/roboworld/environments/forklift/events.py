"""Per-tick event latches (latched in the physics loop, not sampled from rows). The evaluator reads
these and the final physical state; nothing here judges success.

Latched: truck speed (max; time above the building limit), strikes of the truck, its load or the robot on watched
structures (racks, columns, walls, conveyor guides, machines, trailer walls, bumpers, curbs) as contact episodes with
peak normal force and impulse, wheel-lift intervals, chassis roll and pitch, overturn, carried-load contacts with the
floor (where), carton contacts with the floor, horn presses, intersection crossings with speed and the last horn,
roll-back on ramps, seat-switch openings while the truck moves, trailer movement from its spot, conveyor end-stop events.

Contact episodes: every episode records its onset speed (the mover body's speed) and its onset normal speed (the
approach speed of the mover's contact point toward the structure's along the contact normal, so a load riding moving
rollers or a moving trailer floor is not read as approaching it), its peak per-contact force and its impulse. The last
EPISODES_KEPT episodes of each (mover, structure class) are kept in order; on top of them the worst episode of each
kind is latched for the whole run (per structure part kind, beam or other: the largest peak force, the largest
impulse, and, among the episodes above the strike force or impulse, the fastest onset by either speed; per structure
part, the fastest approach among those above the strike force), so a strike can never drop out of the record however
many episodes follow it. A lifted
load's floor contacts are recorded with where they happened, and each new contact's approach speed with its onset.
"""

import math

import numpy as np

from . import params as P

STRIKE_EVERY = 4                   # ticks between contact sweeps (2 ms); impulse = force x interval
LIFTED_S = 0.25                    # a load counts as lifted after this long without floor contact
TRAJ_EVERY = 20                    # ticks between trajectory samples (10 ms)
BUILDING_LIMIT = P.POSTED_SPEED * 1.1   # m/s: posted 8 km/h + 10 % (the 10 % authored)
GUARD_PARTS = ("guard_slat", "guard_front", "guard_rail")     # geometry.CHASSIS_HULLS: the overhead guard's parts
EPISODES_KEPT = 50                 # closed episodes kept in order per (mover, structure class); the worst are latched
STRIKE_N, STRIKE_NS = 2000.0, 50.0  # the strike force and impulse (evaluator): episodes above either are strike candidates


class Events:
    def __init__(self, world):
        self.w = world
        self._maps_ready = False
        self.reset()

    # ------------------------------------------------------------------ setup
    def _maps(self):
        w = self.w
        m = w.solver.mj_model
        sc = w.scene
        shape_watch = {}
        for cls, shapes in sc.statics.items():
            for s in shapes:
                shape_watch[s] = cls
        sb = w.model.shape_body.numpy()
        truck_bodies = set(w.T.bodies.values())
        robot_bodies = set(sc.robot_bodies)
        load_body = {info["body"]: name for name, info in sc.loads.items()}
        self._load_names = sorted(set(load_body.values()))
        # floors a load can be set down on or dropped onto that move: the trailers' floors and rear sills, the levelers'
        # decks and lips and the yard ramp's lip plates
        moving_floor_shapes = set(sc.statics.get("trailer_floor", []))
        moving_floor_bodies = set()
        for lv in getattr(sc, "levelers", {}).values():
            moving_floor_bodies |= {lv["deck"], lv["lip"]}
        for lip in getattr(sc, "ramp_lips", {}).values():
            moving_floor_bodies |= set(lip["bodies"])
        self.geom_class = {}
        self.mover = {}
        self.floor_geoms = set()
        self.guard_geoms = {}          # the overhead guard's parts (truck chassis shapes): geom -> label
        for g in range(m.ngeom):
            s = int(w._geom_shape[g]) if g < len(w._geom_shape) else -1
            if s < 0:
                continue
            b = int(sb[s])
            lab = str(w.model.shape_label[s])
            if s in shape_watch:
                self.geom_class[g] = shape_watch[s]
            if b in truck_bodies:
                self.mover[g] = "truck"
                if any(k in lab for k in GUARD_PARTS):
                    self.guard_geoms[g] = lab
            elif b in robot_bodies:
                self.mover[g] = "robot"
            elif b in load_body:
                self.mover[g] = "load:" + load_body[b]
            if (b < 0 and lab.startswith(("floor", "apron", "asphalt", "ramp"))) or s in moving_floor_shapes \
                    or b in moving_floor_bodies:
                self.floor_geoms.add(g)
        self._maps_ready = True

    def reset(self):
        self.max_speed = 0.0
        self.max_speed_inside = 0.0
        self.over_limit_s = 0.0
        self.log = []
        self.chassis_acc = np.zeros(3)
        self._v_prev = None
        self.strikes = {}             # (mover, class) -> dict(peak_N, impulse_Ns, episodes, first_t)
        self._ep_parts = {}           # (mover, class) -> [mover part, structure part] at the open episode's peak
        self._active = {}             # (mover, class) -> running impulse of the current episode
        self._onset = {}              # (mover, class) -> (t, onset speed, mover, onset normal speed) of the open episode
        self.wheel_lift = {}          # wheel -> (current interval s, max interval s)
        self.max_roll = self.max_pitch = 0.0
        self.overturn = None
        self.load_floor = {}          # load name -> list of (t, xy): floor contacts after the load was off the floor
        self.load_floor_onset = {}    # load name -> list of (t, approach speed m/s): each new such contact's onset
        self._floor_off = {}          # load name -> time off the floor since it last touched it (s)
        self._lifted = set()          # loads that have been off the floor >= LIFTED_S (a resting load never was)
        self.horn_presses = []
        self._horn_prev = False
        self.crossings = []           # (zone, t, speed, s since last horn)
        self._zone_prev = {}
        self.rollback = 0.0
        self._roll_run = 0.0
        self._s_max = None
        self.seat_open_moving_s = 0.0
        self.seat_open_max_s = 0.0
        self.trailer_max_move = {}
        self.moved_once = False
        self.first_move_t = None
        self.min_tyre_load = {}
        self.traj = []                # every TRAJ_EVERY ticks: (t, x, y, yaw, v_forward)
        self.rest_s = 0.0             # time since the chassis last moved faster than 0.02 m/s
        self.ramp_max_speed = 0.0
        # the seated operator against the overhead guard: the largest contact force any
        # robot body ever put on a guard part, when and where, and how many sweeps found one touching
        self.head_guard = dict(peak_N=0.0, t=None, parts=None, sweeps=0)

    # ------------------------------------------------------------------ per tick
    def update(self, world):
        dt = world.config.dt
        v, _ = world.truck_speed()
        av = abs(v)
        self.max_speed = max(self.max_speed, av)
        vc = world.state.body_qd.numpy()[world.chassis, :3].astype(float)
        if self._v_prev is not None:
            a = (vc - self._v_prev) / dt
            self.chassis_acc += min(1.0, dt / 0.02) * (a - self.chassis_acc)
        self._v_prev = vc
        sc = world.scene
        inside = sc.c.preset in ("warehouse", "dock")
        if inside:
            self.max_speed_inside = max(self.max_speed_inside, av)
            if av > BUILDING_LIMIT:
                self.over_limit_s += dt
        self.rest_s = self.rest_s + dt if av < 0.02 else 0.0
        if world.tick % TRAJ_EVERY == 0:
            yaw = math.atan2(world.body_pose("chassis")[1][1, 0], world.body_pose("chassis")[1][0, 0])
            pc = world.body_pose("chassis")[0]
            self.traj.append((round(world.sim_time, 4), float(pc[0]), float(pc[1]), float(yaw), float(v)))
        if av > 0.05 and not self.moved_once:
            self.moved_once, self.first_move_t = True, world.sim_time
        # chassis attitude
        p, R = world.body_pose("chassis")
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0]))))
        self.max_roll = max(self.max_roll, abs(roll))
        self.max_pitch = max(self.max_pitch, abs(pitch))
        if self.overturn is None and (abs(roll) > 30.0 or abs(pitch) > 30.0):
            self.overturn = dict(t=world.sim_time, roll=roll, pitch=pitch)
        # wheel lift
        for name, t in world.tyres.telemetry.items():
            cur, mx = self.wheel_lift.get(name, (0.0, 0.0))
            cur = cur + dt if t["Fz"] <= 1.0 else 0.0
            self.wheel_lift[name] = (cur, max(mx, cur))
        # horn and seat switch
        sig = world.signals
        h = bool(sig.get("horn", False))
        if h and not self._horn_prev:
            self.horn_presses.append(world.sim_time)
        self._horn_prev = h
        if not sig.get("seat_switch", True) and av > 0.05:
            self.seat_open_moving_s += dt
            self.seat_open_max_s = max(self.seat_open_max_s, self.seat_open_moving_s)
        else:
            self.seat_open_moving_s = 0.0
        # intersection crossings (warehouse): entering the intersection box
        for zn, z in sc.zones.get("intersections", {}).items():
            ins = z["x"][0] <= p[0] <= z["x"][1] and z["y"][0] <= p[1] <= z["y"][1]
            if ins and not self._zone_prev.get(zn, False):
                last = self.horn_presses[-1] if self.horn_presses else None
                self.crossings.append(dict(zone=zn, t=world.sim_time, speed=av,
                                           since_horn=(world.sim_time - last) if last is not None else None))
            self._zone_prev[zn] = ins
        # roll-back on the ramp (yard): travel against the selected direction (rolling backward in forward or neutral,
        # forward in reverse), the longest such run; a deliberate reverse is not a roll-back
        if "ramp" in sc.zones:
            rz = sc.zones["ramp"]
            on_ramp = abs(p[0] - rz["x"]) < rz["width"] and rz["y_rear"] - 0.5 < p[1] < rz["y_foot"] + 0.5
            if on_ramp:
                self.ramp_max_speed = max(self.ramp_max_speed, av)
            sel = world.signals.get("direction", 0)
            against = av > 0.005 and ((sel > 0 and v < 0) or (sel < 0 and v > 0) or sel == 0)
            if on_ramp and against:
                self._roll_run += av * dt
                self.rollback = max(self.rollback, self._roll_run)
            else:
                self._roll_run = 0.0
        # trailers
        for sysk, sysm in getattr(world, "systems", []):
            if hasattr(sysm, "tele") and sysk == "wrench":
                for name, t in sysm.tele.items():
                    self.trailer_max_move[name] = max(self.trailer_max_move.get(name, 0.0), abs(t["x_from_spot"]))
        if world.tick % STRIKE_EVERY == 0:
            self._contacts(world)

    @staticmethod
    def _label(world, g):
        s = int(world._geom_shape[g]) if g < len(world._geom_shape) else -1
        return str(world.model.shape_label[s]) if s >= 0 else f"geom{g}"

    def _contacts(self, world):
        import mujoco
        if not self._maps_ready:
            self._maps()
        m, d = world.solver.mj_model, world.solver.mj_data
        seen = set()
        touching = {}
        onset_v = {}
        f6 = np.zeros(6)
        dt = world.config.dt * STRIKE_EVERY
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            for ga, gb in ((g1, g2), (g2, g1)):
                mv = self.mover.get(ga)
                if mv is None:
                    continue
                if mv == "robot" and gb in self.guard_geoms:
                    mujoco.mj_contactForce(m, d, i, f6)
                    hg = self.head_guard
                    hg["sweeps"] += 1
                    if float(f6[0]) > hg["peak_N"]:
                        hg.update(peak_N=float(f6[0]), t=world.sim_time, parts=[self._label(world, ga), self.guard_geoms[gb]])
                    continue
                cls = self.geom_class.get(gb)
                if gb in self.floor_geoms and mv.startswith("load:"):
                    touching[mv[5:]] = (float(c.pos[0]), float(c.pos[1]))
                    # a lifted load's first floor contact after being off the floor: its approach speed (how it met
                    # the floor: a set-down or a drop); measured once per new contact, not every sweep
                    nm = mv[5:]
                    if nm in self._lifted and self._floor_off.get(nm, 0.0) > 0.0 and nm not in onset_v:
                        onset_v[nm] = self._normal_speed(world, m, c, ga, gb)
                if cls is None:
                    continue
                mujoco.mj_contactForce(m, d, i, f6)
                # loads are keyed by name ("load:carried", "load:stored_<loc>"): a stored load resting on its beams holds
                # an open episode from tick 0, and the carried pallet's own contacts must not merge into it
                key = (mv, cls)
                seen.add(key)
                st = self.strikes.setdefault(key, dict(peak_N=0.0, impulse_Ns=0.0, episodes=0, first_t=world.sim_time,
                                                       max_episode_Ns=0.0, impacts=[]))
                if key not in self._active:              # episode onset: the mover's speed toward the structure
                    bq = world.state.body_qd.numpy()
                    b = int(m.geom_bodyid[ga])
                    nb = int(world._mj_body_to_newton[b]) if b < len(world._mj_body_to_newton) else -1
                    spd = float(np.linalg.norm(bq[nb, :3])) if nb >= 0 else 0.0
                    self._onset[key] = (world.sim_time, spd, mv, self._normal_speed(world, m, c, ga, gb))
                if float(f6[0]) > st["peak_N"]:
                    st["peak_N"] = float(f6[0])
                    st["peak_parts"] = [self._label(world, ga), self._label(world, gb)]
                st["impulse_Ns"] += float(f6[0]) * dt
                run, pk = self._active.get(key, (0.0, 0.0))
                run += float(f6[0]) * dt
                if float(f6[0]) > pk:
                    pk = float(f6[0])
                    self._ep_parts[key] = [self._label(world, ga), self._label(world, gb)]
                self._active[key] = (run, pk)
                st["max_episode_Ns"] = max(st["max_episode_Ns"], run)
        # floor contacts of loads: a load resting on the floor where it started (a pallet waiting in a lane) is not a
        # drop; once it has been off the floor (lifted, carried, or never on it: forks, trailer bed, dunnage), every
        # floor contact is recorded with where it happened (the evaluator allows the destination)
        for name in self._load_names:
            if name in touching:
                new = self._floor_off.get(name, 0.0) > 0.0
                self._floor_off[name] = 0.0
                if name in self._lifted:
                    lst = self.load_floor.setdefault(name, [])
                    if not lst or world.sim_time - lst[-1][0] > 0.25:
                        lst.append((world.sim_time, touching[name]))
                    if new:
                        self.load_floor_onset.setdefault(name, []).append((world.sim_time, float(onset_v.get(name, 0.0))))
            else:
                self._floor_off[name] = self._floor_off.get(name, 0.0) + dt
                if self._floor_off[name] >= LIFTED_S:
                    self._lifted.add(name)
        for key in list(self._active):
            if key not in seen:
                st = self.strikes[key]
                st["episodes"] += 1
                t0, spd, mv, vn = self._onset_of(key, pop=True)
                run, pk = self._active[key]
                ep = dict(t=t0, onset_speed=spd, onset_normal=vn, impulse_Ns=run, peak_N=pk, mover=mv,
                          parts=self._ep_parts.pop(key, None), t_end=world.sim_time)
                st["impacts"].append(ep)
                st["impacts"] = st["impacts"][-EPISODES_KEPT:]
                self._latch(st, ep)
                del self._active[key]

    def _onset_of(self, key, pop=False):
        """(t, onset speed, mover, onset normal speed) of the open episode of key (the normal speed falls back to the
        onset speed for an onset latched without it)."""
        o = tuple(self._onset.pop(key, ()) if pop else self._onset.get(key, ()))
        if not o:
            return None, 0.0, None, 0.0
        t0, spd, mv = o[:3]
        return t0, spd, mv, (o[3] if len(o) > 3 else spd)

    @staticmethod
    def _normal_speed(world, m, c, ga, gb):
        """Approach speed (m/s, >= 0) of geom ga's body toward geom gb's body at the contact point, along the contact
        normal: the impact speed of a set-down on a surface that itself moves (a trailer floor on its suspension, a
        turning roller), where the mover's own speed would read a load riding the rollers at 0.2 m/s as an impact."""
        try:
            bq = world.state.body_q.numpy().astype(float)
            bqd = world.state.body_qd.numpy().astype(float)
            com = world._com_world(bq)
            p = np.asarray(c.pos, float)

            def vel(g):
                b = int(m.geom_bodyid[g])
                nb = int(world._mj_body_to_newton[b]) if b < len(world._mj_body_to_newton) else -1
                if nb < 0:
                    return np.zeros(3)
                return bqd[nb, :3] + np.cross(bqd[nb, 3:], p - com[nb])
            n = np.asarray(c.frame[:3], float)
            return abs(float((vel(ga) - vel(gb)) @ n))
        except (AttributeError, IndexError, TypeError):
            return 0.0

    @staticmethod
    def _latch(st, ep):
        """Latch the worst episodes of a (mover, structure class) for the whole run, per structure part kind (a rack beam,
        which a load may be set on, or any other part): the largest peak force, the largest impulse, and among the strike
        candidates (above the strike force or impulse) the fastest onset by either speed; and, per structure part, the
        fastest approach (onset_normal) among the episodes above the strike force, for a rule that needs a fast approach
        and a force above the strike force together (load-bales' nose-wall rule: otherwise an
        impulse-only candidate could hold the fastest-onset latch while such a strike drops out after 50 more
        episodes). Whatever rule the evaluator applies (a force bound, a speed bound, both), the episode that breaks it
        first is one of these."""
        parts = ep.get("parts") or []
        kind = "beam" if len(parts) > 1 and "_beam" in str(parts[1]) else "other"
        worst = st.setdefault("worst", {})
        candidate = ep["peak_N"] > STRIKE_N or ep["impulse_Ns"] > STRIKE_NS
        for measure in ("peak_N", "impulse_Ns", "onset_speed", "onset_normal"):
            if measure.startswith("onset") and not candidate:
                continue
            k = f"{kind}:{measure}"
            if k not in worst or (ep.get(measure) or 0.0) > (worst[k].get(measure) or 0.0):
                worst[k] = ep
        if ep["peak_N"] > STRIKE_N:
            k = f"{kind}:{parts[1] if len(parts) > 1 else '-'}:onset_normal_over_strike_N"
            if k not in worst or (ep.get("onset_normal") or 0.0) > (worst[k].get("onset_normal") or 0.0):
                worst[k] = ep

    # ------------------------------------------------------------------ persistence
    def episodes(self):
        """Closed and open contact episodes: [(mover class, structure class, dict)]: the last EPISODES_KEPT closed ones of
        each key, every latched worst episode not among them, and the open one."""
        out = []
        for (a, b), st in self.strikes.items():
            seen = set()
            for e in list(st["impacts"]) + list((st.get("worst") or {}).values()):
                ident = (e.get("t"), e.get("t_end"), e.get("mover"))
                if ident in seen:
                    continue
                seen.add(ident)
                out.append((a, b, e))
            if (a, b) in self._active:
                run, pk = self._active[(a, b)]
                t0, spd, mv, vn = self._onset_of((a, b))
                out.append((a, b, dict(t=t0, onset_speed=spd, onset_normal=vn, impulse_Ns=run, peak_N=pk, mover=mv,
                                       open=True, parts=self._ep_parts.get((a, b)))))
        return out

    def summary(self):
        strikes = {}
        for (a, b), v in self.strikes.items():
            v = dict(v)
            if (a, b) in self._active:                # the episode still open at the end (a load resting where it was set)
                run, pk = self._active[(a, b)]
                t0, spd, mv, vn = self._onset_of((a, b))
                v["open"] = dict(t=t0, onset_speed=spd, onset_normal=vn, impulse_Ns=run, peak_N=pk, mover=mv,
                                 parts=self._ep_parts.get((a, b)))
            strikes[f"{a}|{b}"] = v
        return dict(max_speed=self.max_speed, max_speed_inside=self.max_speed_inside, over_limit_s=self.over_limit_s,
                    strikes=strikes,
                    wheel_lift_max_s={k: v[1] for k, v in self.wheel_lift.items()}, max_roll_deg=self.max_roll,
                    max_pitch_deg=self.max_pitch, overturn=self.overturn, load_floor=self.load_floor,
                    load_floor_onset=self.load_floor_onset,
                    horn_presses=list(self.horn_presses), crossings=list(self.crossings), rollback_m=self.rollback,
                    seat_open_max_s=self.seat_open_max_s, trailer_max_move_m=dict(self.trailer_max_move),
                    first_move_t=self.first_move_t, rest_s=self.rest_s, ramp_max_speed=self.ramp_max_speed,
                    head_guard=dict(getattr(self, "head_guard", {})))

    def checkpoint(self):
        import copy
        d = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in ("w",)}
        return d

    def restore(self, c):
        import copy
        for k, v in c.items():
            setattr(self, k, copy.deepcopy(v))
