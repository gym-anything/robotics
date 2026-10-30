"""USD export of a forklift session and the client of the render-only Isaac worker (render_worker.py).

The stage holds the static appearance; every request carries the poses of all Newton bodies (/Lab/body_<i>), so one
exported stage serves a whole session or recording. Appearance rules: collision geometry is never drawn; the truck is the
NVIDIA Forklift C's own meshes with its own materials (split into the simulated bodies, the seat on its rail setting and
the overhead guard raised as configured: geometry.visual_parts); the robot its URDF visual meshes; the building, dock,
yard, trailers and loads are drawn by forklift.visuals (one module each). This file owns the truck, the robot, the
lights and the cameras. Renders never advance physics (worker reply physics_steps == 0).
"""
from gym_anything_robotics.paths import worker_env

import base64
import json
import math
import os
from pathlib import Path
import secrets
import select
import socket
import subprocess
import threading
import time

import numpy as np

from . import params as P

DIAGNOSTIC_CAMERAS = ("overview", "chase", "cab", "forks", "side", "pedals", "hands", "cab_left", "cab_right")
LAMP_NITS = 1.0e4          # high-bay LED panel luminance: 15,000 lm over a 1.2 x 0.4 m diffuser, L = phi / (pi A) (derived);
                           # on a 6 m grid at 8.4 m the floor gets ~0.039 L = 390 lx (IES: 108 lx active bulky storage, 323 lx
                           # small items and shipping/receiving, 323-538 lx at docks; 15,000 lm is a 150 W high bay; search). The first
                           # value, 4e5, put ~15,000 lx on the floor and saturated every render (E-render calibration)
AMBIENT_NITS = 8.0         # uniform fill (light through doors and translucent roof panels): pi L = 25 lx; the path tracer does the bounce
LAMP_GRID = 6.0
# camera exposure (ISO, f-number, shutter s): indoors EV100 7.9 for ~400 lx; the yard under a partly cloudy midday sky
# (~40 klx) EV100 14.0, as a camera set for each would be
EXPOSURE = {"indoor": (400.0, 2.8, 1 / 120.0), "outdoor": (100.0, 8.0, 1 / 250.0)}
SKY_NITS = 3000.0          # overcast sky luminance (2-8 kcd/m2, CIE overcast; estimate)
SUN_LUX = 12000.0          # direct sun through thin cloud (estimate; used only without the HDR sky)
SKY_HDR = "Skies/Cloudy/kloofendal_48d_partly_cloudy.hdr"     # NVIDIA Skies collection (locked in assets.lock.json)
SKY_HDR_SCALE = 8300.0     # dome multiplier: the HDR's upper hemisphere integrates to a horizontal illuminance of 4.8
                           # in its own units (sun unclipped); x 8,300 = 40 klx, partly cloudy midday (derived)

# linear base colour, roughness, metallic
COLORS = dict(
    # H1-2: black shells and motor housings, brushed-aluminium limb links, white logo (the URDF gives every link one
    # dark grey, 0.1; the split follows the vendor's product photographs)
    h1_dark=((.022, .022, .025), .42, .0), h1_grey=((.50, .51, .53), .48, .9), h1_joint=((.045, .045, .05), .35, .3),
    h1_logo=((.85, .85, .85), .5, .0), h1_pad=((.03, .03, .032), .8, .0),
    rack_upright=((.03, .10, .38), .45, .4), rack_beam=((.75, .22, .02), .4, .35), guard_yellow=((.78, .58, .02), .45, .2),
    wall=((.55, .56, .55), .75, .0), concrete_column=((.42, .42, .40), .85, .0), machine_grey=((.30, .31, .33), .5, .5),
    machine_green=((.08, .22, .12), .45, .35), rubber_black=((.02, .02, .022), .85, .0), seal_foam=((.02, .02, .025), .95, .0),
    pallet_wood=((.42, .30, .17), .85, .0), wrapped=((.55, .45, .30), .25, .0), carton=((.52, .38, .22), .85, .0),
    steel=((.45, .46, .48), .4, .85), steel_roller=((.55, .56, .58), .3, .9), cage_wire=((.35, .36, .38), .5, .8),
    trailer_wall=((.82, .83, .84), .35, .3), oak=((.40, .27, .14), .7, .0), ramp_yellow=((.75, .55, .03), .5, .4),
    pallet_stack=((.40, .29, .17), .85, .0), bale=((.45, .36, .24), .95, .0), dunnage=((.48, .35, .20), .9, .0),
    paint_yellow=((.82, .62, .02), .55, .0), paint_white=((.82, .82, .80), .55, .0), floor_concrete=((.36, .36, .35), .7, .0),
    asphalt=((.07, .07, .075), .9, .0), roof=((.18, .19, .20), .6, .5), coil_steel=((.33, .34, .36), .25, .9),
    trailer_floor=((.36, .24, .13), .75, .0), leveler=((.20, .21, .22), .45, .7), field=((.17, .19, .11), .95, .0),
)


def _q_from_R(R):
    from scipy.spatial.transform import Rotation
    return Rotation.from_matrix(np.asarray(R, float)).as_quat()


def export(world, folder, views=None, mounts=None, movable=None):
    """Write the stage for a built world. Extra cameras: views {name: (eye, target, focal_mm)}
    fixed in world coordinates; mounts {name: (body, eye, target, focal_mm)} riding on a truck body, in truck
    coordinates; movable {name: focal_mm} whose pose each render request sets (a follow camera)."""
    from .usd import Stage
    S = Stage(folder)
    for name, (rgb, rough, metal) in COLORS.items():
        S.color(name, rgb, rough, metal)
    meta = dict(preset=world.config.preset, task=world.config.task, bodies=world.model.body_count, cameras={},
                exposure=EXPOSURE["outdoor" if world.config.preset in ("yard", "lab") else "indoor"])
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        S.xform(f"/Lab/body_{i}", q[i, :3], q[i, 3:])
    _truck(S, world)
    _robot(S, world)
    from . import visuals
    visuals.draw(S, world)
    _lights(S, world)
    _task_lights(S, world)
    _cameras(S, world, meta)
    for name, (eye, target, focal) in (views or {}).items():
        S.camera(f"/Cameras/{name}_cam", eye, target, focal)
        S.alias(name, f"/Cameras/{name}_cam")
        meta["cameras"][name] = dict(resolution=(1280, 720), mount="fixed", role="diagnostic")
    for name, (b, eye, target, focal) in (mounts or {}).items():
        _mount(S, world, name, b, eye, target, focal)
        meta["cameras"][name] = dict(resolution=(1280, 720), mount=name, role="diagnostic")
    for name, focal in (movable or {}).items():
        cam = S.UsdGeom.Camera.Define(S.stage, f"/Cameras/{name}_cam")
        S.place(cam.GetPrim(), (0.0, 0.0, 10.0), (0.0, 0.0, 0.0, 1.0))
        cam.CreateHorizontalApertureAttr(36.0); cam.CreateVerticalApertureAttr(36.0 * 9 / 16)
        cam.CreateFocalLengthAttr(float(focal)); cam.CreateClippingRangeAttr(S.Gf.Vec2f(0.05, 400.0))
        cam.GetPrim().SetCustomDataByKey("observation_role", "diagnostic")
        S.alias(name, f"/Cameras/{name}_cam")
        meta["cameras"][name] = dict(resolution=(1280, 720), mount="follow", role="diagnostic")
    S.save()
    (S.folder / "export.json").write_text(json.dumps(meta, indent=1, default=float))
    return S.folder / "scene.usdc", meta


def frame_state(world):
    """Everything a frame needs: the poses of all bodies."""
    return dict(body_q=world.state.body_q.numpy().astype(float).copy())


# ---------------------------------------------------------------------------------------------------------- truck
def _truck(S, world):
    from .geometry import visual_parts, usd_path
    T = world.T
    asset = str(usd_path())
    mats = {}
    for body, entries in visual_parts().items():
        if body not in T.bodies:
            continue
        b = T.bodies[body]
        anchor0 = np.asarray(T.anchors[body]) - np.asarray(T.shifts.get(body, (0.0, 0.0, 0.0)))
        for k, (label, V, F, UV, mat) in enumerate(entries):
            if mat not in mats:
                name = mat.split("/")[-1]
                prim = S.stage.DefinePrim(f"/Looks/Forklift/{name}", "Material")
                prim.GetReferences().AddReference(asset, mat)
                mats[mat] = S.UsdShade.Material(prim)
            g = S.mesh(f"/Lab/body_{b}/truck_{body}_{k}", V - anchor0, F, uv=UV.reshape(-1, 2), uv_face_varying=True,
                       smooth=True)
            S.bind(g, mats[mat])
    # the horn button (a drawn part on the hub)
    from .truck import HORN, _quat_between
    hb = T.bodies["horn"]
    S.cylinder(f"/Lab/body_{hb}/horn_button", HORN["r"], HORN["thick"], S.mats["rubber_black"],
               p=(0.0, 0.0, 0.0), q=_quat_between((0.0, 0.0, 1.0), T.column_axis))


# ---------------------------------------------------------------------------------------------------------- robot
def _robot(S, world):
    """The robot's visual meshes (collision hulls are never drawn): the humanoid's URDF visuals, or the driving-robot
    kit's drawn parts in the finishes of its catalogue model (a kit shape's label ends in its finish)."""
    import newton
    b = world.scene.b
    if world.rig is None:
        return
    V = int(newton.ShapeFlags.VISIBLE)
    robot = set(world.scene.robot_bodies)
    kit = getattr(world.scene, "kit", None)
    for name, f in ((kit or {}).get("finishes") or {}).items():
        S.color(f"kit_{name}", tuple(f["rgb"]), f["rough"], f["metal"])
    for i in range(b.shape_count):
        body = b.shape_body[i]
        if body not in robot or not int(b.shape_flags[i]) & V:
            continue
        if int(b.shape_flags[i]) & int(newton.ShapeFlags.COLLIDE_SHAPES):
            continue
        src = b.shape_source[i]
        if src is None or not hasattr(src, "vertices"):
            continue
        scale = np.asarray(b.shape_scale[i], float)
        tf = np.asarray(b.shape_transform[i], float)
        g = S.mesh(f"/Lab/body_{body}/vis_{i}", np.asarray(src.vertices) * scale, src.indices, smooth=True)
        S.place(g, tf[:3], tf[3:])
        lab = str(b.body_label[body]).split("/")[-1]
        finish = f"kit_{str(b.shape_label[i]).split(':')[-1]}" if kit is not None else None
        if finish in S.mats:
            mat = finish                                 # the kit's anodised aluminium, black, maker's blue, steel
        elif "logo" in lab:
            mat = "h1_logo"
        elif any(k in lab for k in ("thumb", "index", "middle", "ring", "pinky", "ankle_roll")):
            mat = "h1_pad"                               # finger links and soles
        elif any(lab.endswith(k) for k in ("hip_roll_link", "knee_link", "shoulder_yaw_link", "elbow_link")):
            mat = "h1_grey"                              # thigh, shin, upper arm, forearm
        elif any(k in lab for k in ("hip_", "shoulder_", "wrist_", "ankle_pitch")):
            mat = "h1_joint"                             # motor housings
        else:
            mat = "h1_dark"                              # pelvis, torso and head shells, hand bases
        S.bind(g, S.mats[mat])


def _lights(S, world):
    sc = world.scene
    L = sc.layout
    UsdLux, Gf = S.UsdLux, S.Gf
    if sc.c.preset == "yard" or L is None:
        d = UsdLux.DomeLight.Define(S.stage, "/World/sky")
        try:
            from .assets.install import resolve
            d.CreateTextureFileAttr(str(resolve(SKY_HDR)))
            d.CreateTextureFormatAttr("latlong")
            d.CreateIntensityAttr(SKY_HDR_SCALE)
            return                             # the HDR carries the sun (partly cloudy) and the sky
        except FileNotFoundError:
            d.CreateIntensityAttr(SKY_NITS); d.CreateColorAttr(Gf.Vec3f(0.82, 0.88, 1.0))
        sun = UsdLux.DistantLight.Define(S.stage, "/World/sun")
        sun.CreateIntensityAttr(SUN_LUX); sun.CreateAngleAttr(0.53)
        S.place(sun.GetPrim(), (0, 0, 0), _q_from_R(_look_down(math.radians(50), math.radians(30))))
        return
    x0, x1, y0, y1 = L.extent
    k = 0
    for xx in np.arange(x0 + LAMP_GRID / 2, x1, LAMP_GRID):
        for yy in np.arange(y0 + LAMP_GRID / 2, y1, LAMP_GRID):
            p = (float(xx), float(yy), 8.4)
            r = UsdLux.RectLight.Define(S.stage, f"/World/lamp{k}")
            r.CreateWidthAttr(1.2); r.CreateHeightAttr(0.4); r.CreateIntensityAttr(LAMP_NITS)
            r.CreateColorTemperatureAttr(5000.0); r.CreateEnableColorTemperatureAttr(True)
            S.place(r.GetPrim(), p, (0.0, 0.0, 0.0, 1.0))                 # rect lights emit along -z (down)
            k += 1
    d = UsdLux.DomeLight.Define(S.stage, "/World/ambient")
    d.CreateIntensityAttr(AMBIENT_NITS)


# LED dock light (one per door, on a swing arm beside the door, aimed into the trailer: trailers are unlit inside) and the
# truck's two LED work lights on the front corners of the overhead guard. Luminance from flux, L = phi / (pi A), as for
# the high-bay lamps; the cone (ShapingAPI) clips the emission outside it. Flux: commercial LED dock lights 2,000-3,000
# lm, forklift LED work lights 1,000-2,000 lm (estimates of the class; render only)
# offset: the lamp head from the door jamb (dx), from the dock face (dy, inside) and above the floor; on its swing arm
# it stands inside the door's width, 0.85 m back, so its beam enters the trailer (at the first position, beside the
# jamb against the wall, none of the beam went through the door: it lit the wall beside it)
DOCK_LIGHT = dict(flux=2500.0, radius=0.08, cone_deg=30.0, mount_h=2.6, offset=(-0.20, -0.85, 2.60))
WORK_LIGHT = dict(flux=1500.0, radius=0.05, cone_deg=35.0)


def _spot(S, path, pos, aim, flux, radius, cone_deg, parent_anchor=None):
    """A disk light at pos (stage or parent-local) emitting along `aim` within a cone."""
    from scipy.spatial.transform import Rotation
    a = np.asarray(aim, float); a /= np.linalg.norm(a)
    rot, _ = Rotation.align_vectors([a], [[0.0, 0.0, -1.0]])        # disk lights emit along their -z
    lt = S.UsdLux.DiskLight.Define(S.stage, path)
    lt.CreateRadiusAttr(float(radius))
    lt.CreateIntensityAttr(float(flux / (math.pi * math.pi * radius * radius)))
    lt.CreateColorTemperatureAttr(5000.0); lt.CreateEnableColorTemperatureAttr(True)
    sh = S.UsdLux.ShapingAPI.Apply(lt.GetPrim())
    sh.CreateShapingConeAngleAttr(float(cone_deg)); sh.CreateShapingConeSoftnessAttr(0.3)
    S.place(lt.GetPrim(), pos, rot.as_quat())
    return lt


def _task_lights(S, world):
    sc = world.scene
    T = world.T
    ch = T.bodies["chassis"]
    anchor = np.asarray(T.anchors["chassis"])
    for k, sy in enumerate((1.0, -1.0)):                                # truck work lights (move with the chassis)
        pos = np.array([-0.38, 0.52 * sy, 2.10 + P.GUARD_RAISE]) - anchor
        _spot(S, f"/Lab/body_{ch}/work_light_{k}", pos, (math.cos(math.radians(25)), 0.0, -math.sin(math.radians(25))),
              WORK_LIGHT["flux"], WORK_LIGHT["radius"], WORK_LIGHT["cone_deg"])
    if sc.c.preset == "dock":
        from .facility import DOOR_X
        from .visuals.dockyard import dock_light_source
        for i, x in enumerate(DOOR_X):
            pos, aim = dock_light_source(x)                 # the drawn lamp head and the source share one position
            _spot(S, f"/World/dock_light{i}", pos, aim, DOCK_LIGHT["flux"], DOCK_LIGHT["radius"], DOCK_LIGHT["cone_deg"])


def _look_down(elev, azim):
    from scipy.spatial.transform import Rotation
    return (Rotation.from_euler("z", azim) * Rotation.from_euler("y", math.pi / 2 - elev)).as_matrix()


# ---------------------------------------------------------------------------------------------------------- cameras
def _mount(S, world, name, body, eye, target, focal):
    """A camera riding on a body: a truck body (eye and target in truck coordinates) or any other body, e.g. a robot
    link (eye and target in its own frame); body is its index, a truck body's name or the end of a body label."""
    T = world.T
    if isinstance(body, str):
        if body in T.bodies:
            body = T.bodies[body]
        else:
            labels = [str(x) for x in world.scene.b.body_label]
            body = next(k for k, lb in enumerate(labels) if lb.endswith(body))
    names = [k for k, v in T.bodies.items() if v == body]
    anchor = np.asarray(T.anchors[names[0]]) if names else np.zeros(3)
    S.camera(f"/Lab/body_{body}/cam_{name}", np.asarray(eye) - anchor, np.asarray(target) - anchor, focal)
    S.alias(name, f"/Lab/body_{body}/cam_{name}")


def _cameras(S, world, meta):
    sc = world.scene
    T = world.T
    if world.rig is not None:
        fr = world.operator.handle.frames["head_camera"]
        body = fr["body"]
        p = np.asarray(fr.get("position_m", (0, 0, 0)), float)
        qf = fr.get("orientation_wxyz", (1, 0, 0, 0))
        from scipy.spatial.transform import Rotation
        Rf = Rotation.from_quat([qf[1], qf[2], qf[3], qf[0]])
        # USD camera looks along -z with +y up; the camera link looks along +x with +z up
        Rc = Rf * Rotation.from_matrix(np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]))
        hfov = 69.4
        cam = S.UsdGeom.Camera.Define(S.stage, f"/Lab/body_{body}/head_camera")
        S.place(cam.GetPrim(), p, Rc.as_quat())
        cam.CreateHorizontalApertureAttr(36.0); cam.CreateVerticalApertureAttr(36.0 * 480 / 848)
        cam.CreateFocalLengthAttr(18.0 / math.tan(math.radians(hfov / 2)))
        cam.CreateClippingRangeAttr(S.Gf.Vec2f(0.02, 120.0))
        S.alias("head", f"/Lab/body_{body}/head_camera")
        meta["cameras"]["head"] = dict(resolution=(848, 480), mount=f"{world.operator.handle.instance}/head_camera",
                                       role="policy")
    ch = T.bodies["chassis"]
    # chase: above the counterweight, inside the truck's own footprint and under a trailer's roof, looking forward over
    # the overhead guard. Behind the truck it sat inside the stored loads across a narrow aisle and in the dock wall
    # above a door (4.2 m back, 3.1 m up; still at 2.6 m back), and the films failed on all-black frames
    mounts = {"chase": (ch, (-1.7, 0.25, 2.45), (1.6, 0.0, 0.8), 14.0), "cab": (ch, (-0.2, 1.6, 2.0), (-0.75, 0.0, 1.2), 18.0),
              "forks": (T.bodies["carriage"], (2.6, 1.0, 0.7), (1.2, 0.0, 0.0), 24.0),
              # low on the right of the seat, into the footwell under the forearms (outside the cab the cowl hid the
              # feet; from behind the knee the raised right arm filled the view during pedal work; the dash overhangs
              # the pads from every view, so a film shows the leg working and its caption the pedal travel)
              "pedals": (ch, (-0.70, -0.40, 0.95), (-0.25, 0.05, 0.78), 12.0),
              "hands": (ch, (-0.15, 0.95, 1.85), (-0.40, -0.05, 1.20), 16.0),
              # beside the cab at seat height, through its open sides: the operator from the left (the wheel and knob,
              # the left hand, the parking brake) and from the right (the cowl levers and the right hand)
              "cab_left": (ch, (-0.5, 2.4, 1.5), (-0.6, 0.0, 1.1), 35.0),
              "cab_right": (ch, (-0.5, -2.4, 1.5), (-0.6, 0.0, 1.1), 35.0)}
    for name, (b, eye, tgt, f) in mounts.items():
        _mount(S, world, name, b, eye, tgt, f)
        meta["cameras"][name] = dict(resolution=(1280, 720), mount=name, role="diagnostic")
    x, y, yaw = sc.truck_pose
    c, s = math.cos(yaw), math.sin(yaw)
    ext = sc.layout.extent if sc.layout is not None else (-40.0, 80.0, -30.0, 30.0)
    inside = sc.c.preset in ("warehouse", "dock")

    def clamp(e):
        if not inside:
            return e
        return (min(max(e[0], ext[0] + 0.6), ext[1] - 0.6), min(max(e[1], ext[2] + 0.6), ext[3] - 0.6), e[2])
    ov_eye = clamp((x - 9 * c + 6 * s, y - 9 * s - 6 * c, 6.5))
    S.camera("/Cameras/overview_cam", ov_eye, (x + 2 * c, y + 2 * s, 0.8), 18.0)
    S.alias("overview", "/Cameras/overview_cam")
    S.camera("/Cameras/side_cam", clamp((x + 6 * s, y - 6 * c, 1.6)), (x, y, 1.0), 24.0)
    S.alias("side", "/Cameras/side_cam")
    for name in ("overview", "side"):
        meta["cameras"][name] = dict(resolution=(1280, 720), mount="fixed", role="diagnostic")


# ---------------------------------------------------------------------------------------------------------- worker client
class Renderer:
    """Owns exactly one render-only Isaac worker (render_worker.py)."""

    def __init__(self, folder, scene, *, width=1280, height=720, samples=16, exposure=None, exposure_mode="auto"):
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        if exposure is None:
            mf = Path(folder) / "export.json"
            exposure = json.loads(mf.read_text()).get("exposure", EXPOSURE["indoor"]) if mf.is_file() else EXPOSURE["indoor"]
        self.folder = Path(folder)
        self.closed, self.lock = False, threading.Lock()
        self.width, self.height = width, height
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(600)
        secret = secrets.token_urlsafe(32)
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, FL_RENDER_PORT=str(listener.getsockname()[1]), FL_RENDER_SECRET=secret)
        if env.get("CUDA_VISIBLE_DEVICES") == "":
            env.pop("CUDA_VISIBLE_DEVICES")
        self.log = (self.folder / "worker.log").open("ab")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "forklift.render_worker", "--scene", str(Path(scene).resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples),
                                             "--exposure", ",".join(str(float(v)) for v in exposure),
                                             "--exposure-mode", exposure_mode],
                                            cwd=root, env=worker_env(env), stdout=self.log, stderr=subprocess.STDOUT)
            # wait for the worker's connection, watching the worker: one that dies at startup (an import error) used to
            # leave this waiting the full 600 s
            listener.settimeout(1.0)
            t_end = time.monotonic() + 600
            while True:
                try:
                    self.connection, _ = listener.accept()
                    break
                except socket.timeout:
                    if self.process.poll() is not None:
                        raise RuntimeError("Renderer exited during startup; see " + str(self.folder / "worker.log"))
                    if time.monotonic() > t_end:
                        raise TimeoutError("Renderer did not connect within 600 s")
            self.connection.settimeout(900)
            if not secrets.compare_digest(receive(self.connection).get("secret", ""), secret):
                raise PermissionError("Renderer authentication failed")
            if receive(self.connection).get("phase") != "ready":
                raise RuntimeError("Renderer startup failed")
        except BaseException:
            close_worker(self, graceful=False)
            raise
        finally:
            listener.close()
        watch_worker(self)

    def _call(self, message):
        from whiteboard.ipc import send, receive
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            send(self.connection, message)
            deadline = time.monotonic() + 900
            while not select.select([self.connection], [], [], 1.0)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder / "worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
        if not result.get("ok") or result.get("physics_steps") != 0:
            raise RuntimeError("Invalid render response: " + str(result)[:400])
        return result

    def observe(self, cameras, state, tick, resolution=None, samples=None, resolutions=None, poses=None):
        msg = dict(op="observe", tick=int(tick), body_poses=np.asarray(state["body_q"], float).tolist(), cameras=list(cameras))
        if poses:                                    # movable cameras: {prim path: (x, y, z, qx, qy, qz, qw)}
            msg["poses"] = {k: [float(x) for x in v] for k, v in poses.items()}
        if resolution:
            msg["resolution"] = list(resolution)
        if resolutions:
            msg["resolutions"] = {k: list(v) for k, v in resolutions.items()}
        if samples:
            msg["spp"] = int(samples)
        r = self._call(msg)
        self.last_timings = r.get("timings", {})
        return {k: base64.b64decode(v) for k, v in r["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True
