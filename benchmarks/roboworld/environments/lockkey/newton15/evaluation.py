"""Task evaluator for lockkey/unlock (contract "open-padlock-v1").

Job (task.INSTRUCTION): unlock the padlock with its key from the near (-Y)
stand and swing the shackle open; leave the other key on its stand.

success, read from the state at the time of reading (not latched):
- the padlock was unlocked with its key: the latch has released
  (env.released) and, on the tick it released, the near key was seated in the
  keyway. The reduced latch model releases on the first tick past 30 deg of
  plug turn (env.py), evaluates again on that tick, and never re-engages
  within an episode. Seated: the near key's origin, which lies inside the
  blade about 4.7 mm from the blade collider's tip, at x <= 0.091 m with |y|
  and |z - (bench + 0.10 m)| <= 2 mm. The saved unlocks released at x = 0.0883 m;
  3 mm short of the seat at least one pin is 1.75 mm off the shear line.
  Physically only the correct key, fully seated, gets past 30 deg: wrong
  bitting 4.44 deg, one cut wrong 4.40 deg, half insertion 6.02 deg.
  Binding the key to the release tick, not to "seated at some
  time", keeps that true even if another key could turn the plug.
- the shackle is swung open: the short leg's centre is more than leg radius +
  bore radius = 6.2 mm from its bore axis in plan (shackle.py), first true at
  about 18 deg of swing. Either direction, any contact, key left in or taken
  out. No vertical-clearance term, so a swung toe pressed onto the casing top
  at reading still counts (the swivel record held one at -1.33 mm clearance
  under a 12 N push). The geometry witness `shackle_open` reported by env.py
  is a separate measure and also requires positive clearance.
- the other key rests on its stand: its origin is within 8 mm along the fork
  slot (resting travel is 0 mm toward +X, where the blade's shoulder already
  meets the neck support, and about -7.1 mm toward -X, where the bow neck meets
  the bow-side forks; the flat three-point rest ends near -5.7 mm), 3 mm across
  it (4.2 mm slot, 2.0 mm blade, +/-1.1 mm play; a blade outside the forks is
  at least 5.6 mm off) and 3 mm in height (forks rise 8 mm above the posts),
  tilted at most 20 deg, all relative to its start pose; and it is not gripped
  by both Panda fingers (native contacts), since a key in the hand is not left
  on its stand. Reversed or upside down the key cannot rest in its forks.
  Recorded idle settling is 0.2 mm and 0.13 deg. A nudged or slid key still on
  its stand passes, and so does one tried in the lock, put back and let go; a
  key knocked off, carried away or still held fails until it is back.

score = 1 if success else 0.9 x progress. progress is read from the state at
the time of reading along the chain the lock itself enforces, whatever the
grasp, route or order: 0.10 the near key held off its stand by both Panda
fingers (native contacts); 0.25 -> 0.50 the blade tip 0 -> 27 mm into the
keyway (aligned with the axis, key roll within 15 deg of the plug's, which
rejects an inverted blade; depth is measured from the nominal 30 mm blade end,
so credit starts about 0.6 mm after the collider tip passes the face); 0.50 ->
0.80 seated with the plug at 0 -> 30 deg (inside the seat box the plug angle,
not depth, separates a key that turns from one that binds); 0.80 unlocked as
above, rising to 1.0 as the short leg's offset goes from 1 mm (the unaided
drift after release was 0.81 mm) to the 6.2 mm open line. Withdrawing the key
after the unlock costs nothing. A latch released without the near key seated
scores 0 for the rest of the episode, since success can no longer be reached.

The other key scales that lock progress: x1 resting on its stand; x0.5 off it
(or held) but still on the bench, where the robot can put it back (success
needs it back); x0 once it has left the bench, i.e. it is more than 20 mm
below the bench top or more than 50 mm (about a key length) outside the bench
top's footprint in plan (x -0.55..0.75 m, |y| <= 0.46 m; assembly.py), and
not in the robot's hand. Where a dropped key can come to rest off the bench is
out of the riser-mounted Panda's reach: the floor, the camera-stand base and
the skirting at floor level, the cabinet and its 0.85 m counter about 1.2 m
from its base, and, under the top, the controller shelf at 0.24 m (scene.py).
Ignoring collisions, its tool point cannot get below about 0.43 m
or more than 0.95 m from its shoulder (forward kinematics over its joint
limits). `failure` is true while the other key is off its stand, a breach the
robot can still repair. `unrecoverable` is true when success can no longer be
reached: the other key has left the bench, the near key has before the
unlock, or the latch released without the near key seated. Neither flag
changes success. The weights and the 0.5 factor are authored task choices,
not measurements; no policy campaign has calibrated them.

The diagnostic contract "grip-and-raise-v1" (LEGACY_CONTRACT) latches success at the first tick with key grasp and
insertion, plug > 30 deg, a one-tick hand pose near the short leg and a 5.5 mm shackle rise,
unless the wrong key has ever moved 5 mm. 5.5 mm is half of an 11 mm
spring pop, a check that the lock has popped; in this
environment the spring raises the shackle 17.26 mm with no applied force
within 19 ms of release, so grip-and-raise is not a robot action. Its fields
are reported as named diagnostics and its verdict as `legacy`; neither
enters score/success.
Opening the chest lid is part of neither contract.
"""

import math
import numpy as np
from scipy.spatial.transform import Rotation
from .kinematics import Kinematics
from .shackle import BORE_RADIUS, LEG_RADIUS, geometry_state

CONTRACT = "open-padlock-v1"
SCORE_RULE = "1 if success else 0.9 x progress"
CRITERION = ("Latch released with the near key seated at that tick; now: shackle short leg swung clear of its "
             "bore in plan, other key resting on its stand and not gripped by the robot")
LEGACY_CONTRACT = "grip-and-raise-v1"
LEGACY_CRITERION = "Handle and insert the correct key, turn >30 deg, grip and raise shackle >=5.5 mm; wrong key moves <=5 mm"
RELEASE_DEG = 30.
SEAT_X, AXIS_TOL = .091, .002
KEY_ROLL_TOL_DEG = 15.
OPEN_LATERAL = LEG_RADIUS+BORE_RADIUS
SWING_DEADBAND = .001
STAND_SLIDE, STAND_ACROSS, STAND_HEIGHT, STAND_TILT_DEG = .008, .003, .003, 20.
W_HELD, W_ENTERED, W_SEATED, W_RELEASED = .10, .25, .50, .80
OTHER_KEY_DISTURBED = .5
OFF_BENCH_DROP = .02
BENCH_X, BENCH_HALF_Y, OFF_BENCH_MARGIN = (-.55, .75), .46, .05
GRIP_REACH = .15
FAIL_SCORE_SCALE = .9


class UnlockProgress:
    def __init__(self, env):
        b, c = env.builder, env.config
        self.hand = next(i for i,n in enumerate(b.body_label) if n.endswith("panda_link7"))
        self.fingers = [i for i,n in enumerate(b.body_label) if "finger" in n]
        if len(self.fingers) != 2:
            raise ValueError("Expected two physical Panda fingers")
        self.shape_body = np.asarray(b.shape_body)
        k = Kinematics()
        self.hand_local = k.chain[-2]@k.chain[-1]@np.array([0., 0., .1034, 1.])
        self.origin = np.array([.10, 0., c.table_height+.10])
        # Key origin x at which the blade tip (key frame x = face_x+travel-.030,
        # geometry.key_parts) reaches the keyway face (lock frame x = face_x).
        self.entry_x = self.origin[0]+c.face_x-(c.face_x+c.insertion_travel-.030)
        poses = env.state.body_q.numpy()
        self.start = {n: poses[env.bodies[n]].copy() for n in ("key", "wrong_key")}
        self.initial_key = poses[env.bodies["key"], :3].copy()
        self.initial_wrong = poses[env.bodies["wrong_key"], :3].copy()
        self.initial_shackle = float(poses[env.bodies["shackle"], 2])
        self.initial_gap = np.linalg.norm(poses[self.fingers[0],:3]-poses[self.fingers[1],:3])
        self.previous_key = self.initial_key.copy()
        self.previous_tip = self.tip(poses)
        self.max_wrong_motion = self.max_closure = self.max_angle = self.max_rise = 0.
        self.saw_grasp = self.saw_insert = self.saw_shackle_grip = False
        self.failed = self.succeeded = False
        self.seated_at_release = None

    def tip(self, poses):
        p = poses[self.hand]
        return p[:3]+Rotation.from_quat(p[3:]).apply(self.hand_local[:3])

    def on_stand(self, pose, name):
        start = self.start[name]
        d = pose[:3]-start[:3]
        tilt = (Rotation.from_quat(start[3:]).inv()*Rotation.from_quat(pose[3:])).magnitude()
        return bool(abs(d[0]) <= STAND_SLIDE and abs(d[1]) <= STAND_ACROSS and abs(d[2]) <= STAND_HEIGHT
                    and math.degrees(tilt) <= STAND_TILT_DEG)

    def held(self, env, body):
        """Both Panda fingers in native contact with `body` in the latest collision query."""
        contacts = env.contacts
        n = int(contacts.rigid_contact_count.numpy()[0])
        a = self.shape_body[contacts.rigid_contact_shape0.numpy()[:n]]
        b = self.shape_body[contacts.rigid_contact_shape1.numpy()[:n]]
        touching = set(b[a == body].tolist()) | set(a[b == body].tolist())
        return all(f in touching for f in self.fingers)

    def gripped(self, env, poses, body):
        """held(), with contacts read only when both finger bodies are within GRIP_REACH of `body`."""
        near = all(np.linalg.norm(poses[f, :3]-poses[body, :3]) < GRIP_REACH for f in self.fingers)
        return bool(near and self.held(env, body))

    def off_bench(self, env, poses, body):
        """Dropped below the bench top or beyond its footprint, and not in the robot's hand."""
        p = poses[body]
        beyond = not (BENCH_X[0]-OFF_BENCH_MARGIN <= p[0] <= BENCH_X[1]+OFF_BENCH_MARGIN
                      and abs(p[1]) <= BENCH_HALF_Y+OFF_BENCH_MARGIN)
        below = p[2] < env.config.table_height-OFF_BENCH_DROP
        return bool((below or beyond) and not self.gripped(env, poses, body))

    def legacy(self, env, poses, key, wrong, shackle, plug_angle):
        """The grip-and-raise diagnostic (LEGACY_CONTRACT): its episode history fields and latched verdict."""
        tip = self.tip(poses)
        gap = float(np.linalg.norm(poses[self.fingers[0],:3]-poses[self.fingers[1],:3]))
        self.max_wrong_motion = max(self.max_wrong_motion, float(np.linalg.norm(wrong-self.initial_wrong)))
        self.max_closure = max(self.max_closure, self.initial_gap-gap)
        dk, dt = key-self.previous_key, tip-self.previous_tip
        # The co-motion witness is a 50-micrometre spatial measurement. Checked
        # at every 1920-Hz tick it would demand a 96 mm/s minimum speed, so
        # displacement accumulates until that spatial measurement is available.
        if np.linalg.norm(dk) > .00005:
            self.saw_grasp |= self.max_closure > .015 and np.linalg.norm(dk-dt) < .003
            self.previous_key, self.previous_tip = key.copy(), tip.copy()
        self.saw_insert |= key[0] <= SEAT_X and abs(key[1]) <= AXIS_TOL and abs(key[2]-(env.config.table_height+.10)) <= AXIS_TOL
        self.max_angle = max(self.max_angle, abs(plug_angle))
        self.max_rise = max(self.max_rise, float(shackle[2])-self.initial_shackle)
        point = shackle[:3]+Rotation.from_quat(shackle[3:]).apply([-.0015, .010, .050])
        self.saw_shackle_grip |= self.max_angle > 30. and np.linalg.norm(tip-point) < .010 and .0045 < gap < .014
        if not (self.failed or self.succeeded):
            self.failed = self.max_wrong_motion > .005
            self.succeeded = (not self.failed and self.saw_grasp and self.saw_insert and
                              self.max_angle > 30. and self.saw_shackle_grip and self.max_rise >= .0055)

    def update(self, env, plug_angle):
        poses = env.state.body_q.numpy()
        key_pose, wrong_pose = poses[env.bodies["key"]], poses[env.bodies["wrong_key"]]
        key, wrong = key_pose[:3], wrong_pose[:3]
        shackle = poses[env.bodies["shackle"]]
        self.legacy(env, poses, key, wrong, shackle, plug_angle)
        released = bool(env.released)
        on_axis = abs(key[1]-self.origin[1]) <= AXIS_TOL and abs(key[2]-self.origin[2]) <= AXIS_TOL
        if released and self.seated_at_release is None:
            self.seated_at_release = bool(on_axis and key[0] <= SEAT_X)
        unlocked = released and bool(self.seated_at_release)
        lateral = geometry_state(shackle, self.origin)["shackle_short_leg_lateral_offset_m"]
        roll_error = math.degrees((Rotation.from_euler("x", plug_angle, degrees=True).inv()
                                   *Rotation.from_quat(key_pose[3:])).magnitude())
        aligned = on_axis and roll_error <= KEY_ROLL_TOL_DEG
        depth = max(0., float(self.entry_x-key[0])) if aligned else 0.
        seated = aligned and key[0] <= SEAT_X
        swung_open = unlocked and lateral > OPEN_LATERAL
        wrong_body = env.bodies["wrong_key"]
        other_on_stand = self.on_stand(wrong_pose, "wrong_key") and not self.gripped(env, poses, wrong_body)
        holding = None
        if unlocked:
            swing = float(np.clip((lateral-SWING_DEADBAND)/(OPEN_LATERAL-SWING_DEADBAND), 0., 1.))
            stage, progress = ("open" if swung_open else "released"), W_RELEASED+(1.-W_RELEASED)*swing
        elif released:
            stage, progress = "released_without_its_key", 0.
        elif seated:
            turn = float(np.clip(plug_angle/RELEASE_DEG, 0., 1.))
            stage, progress = "seated", W_SEATED+(W_RELEASED-W_SEATED)*turn
        elif depth > 0.:
            stage, progress = "in_keyway", W_ENTERED+(W_SEATED-W_ENTERED)*min(depth/(self.entry_x-SEAT_X), 1.)
        else:
            holding = not self.on_stand(key_pose, "key") and self.gripped(env, poses, env.bodies["key"])
            stage, progress = ("key_held", W_HELD) if holding else ("none", 0.)
        other_off = not other_on_stand and self.off_bench(env, poses, wrong_body)
        near_off = not released and self.off_bench(env, poses, env.bodies["key"])
        factor = 1. if other_on_stand else 0. if other_off else OTHER_KEY_DISTURBED
        success = bool(swung_open and other_on_stand)
        return {"score": 1. if success else FAIL_SCORE_SCALE*progress*factor, "success": success,
            "failure": not other_on_stand,
            "unrecoverable": bool(other_off or near_off or (released and not unlocked)),
            "progress": progress*factor, "lock_progress": progress, "other_key_factor": factor, "stage": stage,
            "contract": CONTRACT, "score_rule": SCORE_RULE, "criterion": CRITERION,
            "checks": {"near_key_seated_at_release": self.seated_at_release, "padlock_released": released,
                       "shackle_swung_open": bool(swung_open), "other_key_on_its_stand": other_on_stand},
            "off_bench": {"other_key": bool(other_off), "near_key_before_unlock": bool(near_off)},
            "key_tip_depth_m": depth, "key_seated_now": bool(seated), "key_roll_error_deg": roll_error,
            "key_held_by_both_fingers": holding,
            "other_key_displacement_m": float(np.linalg.norm(wrong-self.initial_wrong)),
            "saw_key_grasp": bool(self.saw_grasp), "saw_key_insert": bool(self.saw_insert),
            "saw_shackle_grip": bool(self.saw_shackle_grip), "max_plug_angle_deg": self.max_angle,
            "max_shackle_rise_m": self.max_rise, "wrong_key_motion_m": self.max_wrong_motion,
            "legacy": {"contract": LEGACY_CONTRACT, "criterion": LEGACY_CRITERION,
                       "success": bool(self.succeeded), "failure": bool(self.failed),
                       "score": float(self.succeeded)}}
