"""USD export of the actual simulated state + the shared render-only Isaac RTX worker.

The worker (wiping.render_worker) never steps physics: each request carries the
recorded body poses and the label mesh (points, normals). The label is two thin
single-sided sheets offset by its real thickness: the printed facestock (texture from
artwork.py, UV from the label's rest coordinates) and the white adhesive/back face.
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

from .scene import TABLE

COLORS = dict(
    floor=((.30, .31, .30), .75, 0.), laminate=((.62, .60, .55), .45, 0.), laminate_dark=((.25, .25, .24), .6, 0.),
    steel_frame=((.12, .13, .14), .45, .6), pedestal=((.20, .21, .22), .5, .3), wall=((.72, .72, .68), .85, 0.),
    baseboard=((.12, .12, .12), .6, 0.), robot=((.84, .85, .84), .30, 0.), robot_dark=((.07, .075, .08), .4, 0.), camera_dark=((.03, .03, .035), .45, 0.),
    aluminum=((.62, .64, .66), .25, .9), printer_body=((.16, .17, .18), .45, 0.), printer_dark=((.05, .05, .055), .5, 0.),
    printer_screen=((.02, .06, .09), .1, 0.), liner_paper=((.93, .90, .72), .35, 0.), tape=((.55, .42, .27), .08, 0.),
    seam=((.30, .22, .14), .8, 0.), old_label=((.93, .93, .90), .7, 0.), tray=((.10, .11, .12), .55, 0.),
    squeegee_pp=((.10, .35, .75), .35, 0.), pegboard=((.68, .56, .40), .8, 0.), tape_gun=((.75, .10, .08), .4, 0.),
    tape_roll=((.62, .50, .34), .2, 0.), scale=((.2, .2, .22), .5, .1), stainless=((.65, .66, .67), .22, .95),
    bin_grey=((.3, .32, .33), .6, 0.), epoxy_black=((.05, .055, .06), .25, 0.), pom=((.92, .92, .9), .45, 0.),
    pet_clear=((.86, .88, .9), .08, 0.), cap_lavender=((.55, .40, .85), .4, 0.), cap_red=((.75, .08, .08), .4, 0.),
    mfr_label=((.95, .95, .93), .7, 0.), rack_white=((.93, .93, .92), .5, 0.), bag_clear=((.85, .9, .9), .15, 0.),
    sharps_red=((.7, .05, .05), .45, 0.), hdpe_white=((.90, .90, .88), .5, 0.), kraft=((.62, .45, .28), .85, 0.),
    label_back=((.96, .96, .94), .6, 0.), stand_grey=((.62, .63, .64), .45, .15))


def _normals(q, f):
    fn = np.cross(q[f[:, 1]] - q[f[:, 0]], q[f[:, 2]] - q[f[:, 0]])
    vn = np.zeros_like(q)
    for k in range(3):
        np.add.at(vn, f[:, k], fn)
    return vn/np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def label_sheets(world):
    """Front (printed) and back (adhesive) sheet points, offset by +-t/2 along the normal, one vertex
    per triangle CORNER (3T points). A torn (failed) triangle is collapsed to its centroid so the cut
    shows in renders without changing the exported mesh topology."""
    lab = world.label
    alive = getattr(lab, "alive", np.ones(len(lab.tris), bool))
    n = _normals(lab.x, lab.tris[alive] if alive.any() else lab.tris)
    half = .5*lab.stock.thickness
    C = lab.tris.ravel()
    front, back, nc = (lab.x + n*half)[C], (lab.x - n*half)[C], n[C]
    dead = np.repeat(~alive, 3)
    if dead.any():
        cen = np.repeat(lab.x[lab.tris].mean(1), 3, axis=0)
        front[dead] = cen[dead]; back[dead] = cen[dead]
    return front, back, nc


def export(world, folder, art, phantoms=None):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade, Vt
    import newton
    from pencap.newton15.render import pose, mesh, camera
    from whiteboard.usd_scene import bevel_box
    from . import artwork
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(folder/"scene.usdc"))
    UsdGeom.SetStageUpAxis(stage, "Z")
    UsdGeom.SetStageMetersPerUnit(stage, 1.)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage, "/Lab").GetPrim())
    mats = {}

    def material(name, color, rough, metal, texture=None):
        m = UsdShade.Material.Define(stage, "/Materials/"+name)
        s = UsdShade.Shader.Define(stage, str(m.GetPath())+"/surface")
        s.CreateIdAttr("UsdPreviewSurface")
        s.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(rough))
        s.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metal))
        if texture:
            reader = UsdShade.Shader.Define(stage, str(m.GetPath())+"/st")
            reader.CreateIdAttr("UsdPrimvarReader_float2")
            reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
            tex = UsdShade.Shader.Define(stage, str(m.GetPath())+"/tex")
            tex.CreateIdAttr("UsdUVTexture")
            tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(Path(texture).resolve())))
            tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
            tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
                tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3))
        else:
            s.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float, color)))
        m.CreateSurfaceOutput().ConnectToSource(s.ConnectableAPI(), "surface")
        mats[name] = m
        return m

    for name, v in COLORS.items():
        material(name, *v)
    for name, alpha in (("pet_clear", .25), ("bag_clear", .35)):      # clear plastics
        sh = UsdShade.Shader.Get(stage, f"/Materials/{name}/surface")
        sh.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(alpha)
        sh.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.57)
    textures = folder/"textures"
    textures.mkdir(exist_ok=True)
    kraft = artwork.kraft_texture(textures)
    material("kraft", None, .85, 0., texture=kraft)
    pet = "polyester" in world.label.stock.name.lower()        # 3M 7818: matte silver polyester facestock
    material("label_print", None, .42 if pet else .55, .55 if pet else 0., texture=art["path"])
    # printer housing boxes are collision stand-ins; the detailed printer model is in dressing.py
    hidden = ("_housing", "_slot", "_lid_seam", "_display")
    b = world.scene.b
    q = world.state.body_q.numpy()
    for i in range(len(q)):
        pose(UsdGeom.Xform.Define(stage, f"/Lab/body_{i}"), q[i, :3], q[i, 3:])
    # handheld reader rig: an extra posed "body" carrying the verification camera; the evaluator
    # places it in front of the label (optical.py), like a person scanning the applied label
    rig = UsdGeom.Xform.Define(stage, f"/Lab/body_{len(q)}")
    pose(rig, (0., 0., 5.), (0., 0., 0., 1.))
    cam = UsdGeom.Camera.Define(stage, f"/Lab/body_{len(q)}/reader_camera")
    cam.CreateHorizontalApertureAttr(36.); cam.CreateVerticalApertureAttr(20.25)
    cam.CreateFocalLengthAttr(18/math.tan(math.radians(20.)))      # 40 deg HFOV handheld imager
    cam.CreateClippingRangeAttr(Gf.Vec2f(.01, 10.))
    cam.GetPrim().SetCustomDataByKey("observation_role", "diagnostic")
    ra = UsdGeom.Xform.Define(stage, "/Cameras/reader")
    ra.GetPrim().SetCustomDataByKey("mounted_camera_path", str(cam.GetPath()))
    # ASSISTED phantom fixtures (label-solver only): translucent amber proxies, one posed "body" each
    if phantoms:
        m = material("phantom_tool", (1., .62, .12), .4, 0.)
        UsdShade.Shader.Get(stage, "/Materials/phantom_tool/surface").CreateInput(
            "opacity", Sdf.ValueTypeNames.Float).Set(.55)
        for j, (kind, dims, *_) in enumerate(phantoms):
            body = UsdGeom.Xform.Define(stage, f"/Lab/body_{len(q) + 1 + j}")
            pose(body, (0., 0., -5.), (0., 0., 0., 1.))
            if kind == 1:
                from whiteboard.usd_scene import smooth_cylinder
                g = smooth_cylinder(stage, f"/Lab/body_{len(q) + 1 + j}/phantom", float(dims[0]), float(2*dims[1]))
            elif kind == 2:
                g = UsdGeom.Cube.Define(stage, f"/Lab/body_{len(q) + 1 + j}/phantom")
                g.CreateSizeAttr(2.)
                UsdGeom.Xformable(g).AddScaleOp().Set(Gf.Vec3f(*map(float, dims[:3])))
            else:
                continue
            g.GetPrim().SetCustomDataByKey("assisted_phantom", True)
            UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(m)
    for i in range(b.shape_count):
        body, kind, scale = b.shape_body[i], b.shape_type[i], np.asarray(b.shape_scale[i])
        label = b.shape_label[i]
        if label == "floor" or label == "tube_wall" or (label.startswith("printer") and label.endswith(hidden)) \
                or "_hand_box" in label:
            continue
        path = f"/Lab/body_{body}/shape_{i}" if body >= 0 else f"/Lab/static_{i}"
        src = b.shape_source[i]
        if src is not None and hasattr(src, "vertices"):
            g = mesh(stage, path, np.asarray(src.vertices)*scale, src.indices)
        elif kind == newton.GeoType.BOX:
            g = bevel_box(stage, path, 2*scale, min(.0015, float(scale.min())*.2))
        elif kind in (newton.GeoType.CYLINDER, newton.GeoType.CAPSULE):
            g = (UsdGeom.Cylinder if kind == newton.GeoType.CYLINDER else UsdGeom.Capsule).Define(stage, path)
            g.CreateRadiusAttr(float(scale[0])); g.CreateHeightAttr(float(2*scale[1])); g.CreateAxisAttr("Z")
        elif kind == newton.GeoType.SPHERE:
            g = UsdGeom.Sphere.Define(stage, path); g.CreateRadiusAttr(float(scale[0]))
        else:
            continue
        tf = np.asarray(b.shape_transform[i])
        pose(g, tf[:3], tf[3:])
        mname = world.scene.materials.get(i, "robot")
        mat = mats.get(mname, mats["robot"])
        if mname == "robot":
            blabel = b.body_label[body] if body >= 0 else ""
            dark = any(k in blabel for k in ("link7", "finger")) or blabel.endswith("link0")
            mat = mats["robot_dark"] if dark else mats["robot"]
        if mname == "kraft":
            _box_uv(g, 2*scale)
        UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mat)
    # label: printed front + white back sheets
    lab = world.label
    front, back, n = label_sheets(world)
    f = np.arange(3*len(lab.tris)).reshape(-1, 3)          # per-corner vertices (see label_sheets)
    g = mesh(stage, "/Lab/label_front", front, f.ravel(), smooth=False)
    g.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy(n.astype(np.float32))); g.SetNormalsInterpolation("vertex")
    g.CreateDoubleSidedAttr(False)
    g.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([front.min(0), front.max(0)], np.float32)))
    # seen from the printed side with the leading edge (u = MD max) up, image +x runs toward -v
    uv = np.stack([1. - lab.X[:, 1]/lab.width, lab.X[:, 0]/lab.length], 1).astype(np.float32)[lab.tris.ravel()]
    UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "vertex").Set(Vt.Vec2fArray.FromNumpy(uv))
    UsdShade.MaterialBindingAPI.Apply(g.GetPrim()).Bind(mats["label_print"])
    g2 = mesh(stage, "/Lab/label_back", back, f[:, ::-1].ravel(), smooth=False)
    g2.CreateNormalsAttr(Vt.Vec3fArray.FromNumpy((-n).astype(np.float32))); g2.SetNormalsInterpolation("vertex")
    g2.CreateDoubleSidedAttr(False)
    g2.CreateExtentAttr(Vt.Vec3fArray.FromNumpy(np.asarray([back.min(0), back.max(0)], np.float32)))
    UsdShade.MaterialBindingAPI.Apply(g2.GetPrim()).Bind(mats["label_back"])
    # room, lights, props, detailed printer, carton print, camera mount (visual only)
    from . import dressing
    credits = dressing.dress(stage, world, folder, eye_policy=cameras(world)["bench_rgb"][0])
    for name, (eye, target, focal) in cameras(world).items():
        camera(stage, name, eye, target, focal, role="policy" if name == "bench_rgb" else "diagnostic")
    for name, m in world.scene.cameras.items():
        c = UsdGeom.Camera.Define(stage, f"/Lab/body_{m['body']}/{name}")
        c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*m["eye"]), Gf.Vec3d(*m["target"]),
                                                        Gf.Vec3d(*m["up"])).GetInverse())
        c.CreateHorizontalApertureAttr(36.); c.CreateVerticalApertureAttr(20.25)
        c.CreateFocalLengthAttr(18/math.tan(math.radians(m["fov"]/2)))
        c.CreateClippingRangeAttr(Gf.Vec2f(.005, 30.))
        c.GetPrim().SetCustomDataByKey("observation_role", "policy")
        alias = UsdGeom.Xform.Define(stage, "/Cameras/"+name)
        alias.GetPrim().SetCustomDataByKey("mounted_camera_path", str(c.GetPath()))
    stage.GetRootLayer().Save()
    (folder/"export.json").write_text(json.dumps(dict(art=art, tick=world.tick, visual_assets=credits), indent=1))
    return folder/"scene.usdc"


def _box_uv(g, size):
    """Planar UV per face so the kraft texture keeps its physical scale (texture spans 0.4 m)."""
    from pxr import Sdf, Vt
    pts = np.asarray(g.GetPointsAttr().Get())
    idx = np.asarray(g.GetFaceVertexIndicesAttr().Get())
    counts = np.asarray(g.GetFaceVertexCountsAttr().Get())
    uv = []
    k = 0
    for c in counts:
        face = pts[idx[k:k+c]]
        nrm = np.cross(face[1]-face[0], face[2]-face[0])
        ax = int(np.argmax(np.abs(nrm)))
        a, b_ = [i for i in range(3) if i != ax]
        uv.extend(face[:, [a, b_]]/.4)
        k += c
    from pxr import UsdGeom
    UsdGeom.PrimvarsAPI(g).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray, "faceVarying").Set(
        Vt.Vec2fArray.FromNumpy(np.asarray(uv, np.float32)))


def cameras(world):
    """Fixed cameras: (eye, target, focal mm on a 36 mm aperture)."""
    z = TABLE
    preset = world.config.preset
    cams = dict(
        room=((1.55, -2.05, 1.85), (-.05, .15, 1.0), 24.),
        bench_rgb=((.0, -.30, 2.05), (-.02, -.04, .91), 16.),    # policy: overhead ceiling mount over the bench
        printer=((-.02, .22, 1.42), (-.30, -.02, 1.20), 40.),
        printer_wide=((.12, .38, 1.55), (-.30, -.15, 1.22), 30.),
    )
    if preset == "parcel":
        cams.update(carton_top=((.10, -.45, 1.55), (.12, -.02, z+.20), 38.),
                    scanner=((.12, -.10, z+.95), (.12, -.02, z+.20), 28.),     # handheld reader above the top
                    carton_side=((.12, -.75, z+.18), (.12, -.12, z+.10), 32.),
                    closeup=((.32, -.35, z+.40), (.12, -.02, z+.20), 50.))
    elif preset == "specimen":
        # same ceiling mount, narrower lens aimed between printer and tube cradle: the 13 mm tube is ~2x
        # larger than with the 16 mm lens (hard to see there); a lower boom hid the label behind the printer
        cams.update(bench_rgb=((.0, -.30, 2.05), (-.08, -.05, .91), 28.))
        cams.update(tube=((.12, -.20, z+.12), (.10, -.02, z+.02), 60.),
                    scanner=((.10, -.16, z+.02), (.10, -.02, z+.02), 70.),
                    closeup=((.22, -.16, z+.10), (.10, -.02, z+.02), 60.))
    elif world.config.task == "part-id-match":
        # the three plates on the receiving bench; close-up and scanner views on the target plate (diagnostic)
        px, py = world.body_pose(world.scene.target)[0][:2]
        cams.update(part=((.17, -.62, z+.48), (.17, -.03, z+.003), 30.),
                    scanner=((px, py - .15, z+.36), (px, py, z+.003), 32.),
                    closeup=((px + .16, py - .25, z+.20), (px, py, z+.006), 40.))
    else:
        cams.update(part=((.30, -.62, z+.42), (.10, -.07, z+.12), 28.),
                    scanner=((.10, -.40, z+.25), (.10, -.06, z+.12), 45.),
                    # aimed at the canister's labelled side (facing -y, toward the reader); the canister is framed
                    # whole, with its lid
                    closeup=((.30, -.58, z+.36), (.10, -.12, z+.16), 42.))
    return cams


class Renderer:
    """Owns exactly one render-only Isaac worker (wiping.render_worker protocol)."""

    def __init__(self, world, folder, art, *, width=1280, height=720, samples=32, phantoms=None, scene=None):
        """`scene`: a stage this world already exported (export() of the same world, no phantoms). The stage
        holds the static scene only; every observe sends the live body poses and label mesh, so a session
        exports once per world and each per-operation worker loads that file."""
        from gym_anything_robotics.render_lifecycle import close_worker, watch_worker
        from whiteboard.ipc import receive
        self.world, self.folder = world, Path(folder)
        self.closed, self.lock, self.sequence, self.log_position = False, threading.Lock(), 0, 0
        self.phantoms = phantoms or []
        if scene is None or self.phantoms or not Path(scene).is_file():
            scene = export(world, self.folder, art, phantoms=self.phantoms)
        executable = os.environ.get("ISAAC_RENDER_PYTHON")
        if not executable or not Path(executable).is_file():
            raise RuntimeError("ISAAC_RENDER_PYTHON is not configured: set it to the Isaac Sim python interpreter")
        listener = socket.socket(); listener.bind(("127.0.0.1", 0)); listener.listen(1); listener.settimeout(300)
        secret = secrets.token_urlsafe(32)
        env = dict(os.environ, WIPE_RENDER_PORT=str(listener.getsockname()[1]), WIPE_RENDER_SECRET=secret,
                   WIPE_MINIMAL_RENDER="1", WIPE_TEXTURE_STREAMING="1", WIPE_TEXTURE_BUDGET=".05")
        self.log = (self.folder/"worker.log").open("wb")
        self.process = None
        try:
            self.process = subprocess.Popen([executable, "-m", "wiping.render_worker", "--scene", str(scene.resolve()),
                                             "--width", str(width), "--height", str(height), "--samples", str(samples)],
                                            cwd=Path(__file__).resolve().parents[1], env=worker_env(env),
                                            stdout=self.log, stderr=subprocess.STDOUT)
            self.connection, _ = listener.accept(); self.connection.settimeout(300)
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

    def observe(self, cameras=("room",), state=None, reader_pose=None):
        """Render the given recorded state (defaults to the live world). Returns {name: png bytes}."""
        from whiteboard.ipc import send, receive
        from wiping.render_payload import encode
        with self.lock:
            if self.closed:
                raise RuntimeError("Renderer is closed")
            if state is None:
                front, back, n = label_sheets(self.world)
                state = dict(tick=self.world.tick, body_q=self.world.state.body_q.numpy(), front=front, back=back, n=n)
            meshes = {"/Lab/label_front": dict(points=encode(state["front"], "float32"), normals=encode(state["n"], "float32")),
                      "/Lab/label_back": dict(points=encode(state["back"], "float32"), normals=encode(-state["n"], "float32"))}
            self.sequence += 1
            poses = np.asarray(state["body_q"]).tolist()
            if reader_pose is not None or self.phantoms:
                poses.append(list(map(float, reader_pose)) if reader_pose is not None else [0., 0., 5., 0., 0., 0., 1.])
            if self.phantoms:
                tools = state.get("tools")
                for j, ph in enumerate(self.phantoms):
                    kind, dims = ph[0], np.asarray(ph[1], float)
                    s = ph[2] if len(ph) > 2 else j                  # the tool slot this proxy draws
                    if tools is not None and s < len(tools) and tools[s][0] == kind \
                            and np.allclose(np.round(tools[s][1:4], 5), dims):
                        poses.append(list(map(float, tools[s][4:11])))
                    else:
                        poses.append([0., 0., -5., 0., 0., 0., 1.])
            send(self.connection, dict(op="observe", sequence=self.sequence, tick=int(state["tick"]),
                                       body_poses=poses, cameras=list(cameras),
                                       meshes=meshes, material_fields={}, details={}))
            deadline = time.monotonic() + 300
            while not select.select([self.connection], [], [], 1.)[0]:
                if self.process.poll() is not None:
                    raise RuntimeError("Renderer exited before replying; see " + str(self.folder/"worker.log"))
                if time.monotonic() > deadline:
                    raise TimeoutError("Renderer timed out")
            result = receive(self.connection)
            if not result.get("ok") or result.get("physics_steps") != 0:
                raise RuntimeError("Invalid render response: " + str(result)[:400])
            return {k: base64.b64decode(v) for k, v in result["images"].items()}

    def close(self):
        from gym_anything_robotics.render_lifecycle import close_worker
        with self.lock:
            if self.closed:
                return
            close_worker(self)
            self.closed = True


TOOL_SLOTS = 4        # 0-1: phantom tools (applicator, rub finger), 2-3: test-jaw groups


def phantom_tools(world):
    """ASSISTED fixtures that exist only in the label solver, as drawable proxies: (TOOL_SLOTS, 11) rows
    [kind (0 none, 1 cylinder, 2 box), d0, d1, d2, px, py, pz, qx, qy, qz, qw] (cylinder: radius,
    half length; box: half extents)."""
    from scipy.spatial.transform import Rotation
    out = np.zeros((TOOL_SLOTS, 11), np.float32)
    lab = world.label
    scene_cols = set(id(c) for c in world.scene.colliders)
    k = 0
    for c in lab.colliders:
        # phantom tools: the force-controlled applicators (roles "tool" and "applicator")
        if c.role in ("tool", "applicator") and id(c) not in scene_cols and c.shape == "cylinder" and k < 2 \
                and c.p[2] < 4.:                                        # parked tools sit at z = 5 m

            out[k, 0] = 1; out[k, 1:3] = c.size[:2]
            out[k, 4:7] = c.p; out[k, 7:] = Rotation.from_matrix(c.R).as_quat()
            k += 1
    if lab.grip_idx is not None and len(lab.grip_idx):
        gi, tg = np.asarray(lab.grip_idx), np.asarray(lab.grip_target)
        u = lab.X[gi, 0]
        order = np.argsort(u)
        gaps = np.diff(u[order])
        groups = [order] if not len(gaps) or gaps.max() < 3*lab.h else \
            [order[:int(np.argmax(gaps)) + 1], order[int(np.argmax(gaps)) + 1:]]
        for j, g in enumerate(groups[:2]):
            P = tg[g]
            c0 = P.mean(0)
            _, _, Vt = np.linalg.svd(P - c0) if len(P) > 2 else (None, None, np.eye(3))
            ax = Vt[0]; nz = np.cross(ax, [0, 0, 1.]) if abs(ax[2]) < .9 else np.array([1., 0, 0])
            nz /= np.linalg.norm(nz); ny = np.cross(nz, ax)
            R = np.column_stack([ax, ny, nz])
            if np.linalg.det(R) < 0:
                R[:, 2] *= -1
            ext = np.ptp((P - c0) @ ax) if len(P) > 1 else .01
            out[2 + j, 0] = 2; out[2 + j, 1:4] = (ext/2 + .004, .006, .004)
            out[2 + j, 4:7] = c0; out[2 + j, 7:] = Rotation.from_matrix(R).as_quat()
    return out


def snapshot(world):
    """Recordable render state (numpy) of the current tick."""
    front, back, n = label_sheets(world)
    return dict(tick=world.tick, body_q=world.state.body_q.numpy().copy(), front=front.astype(np.float32),
                back=back.astype(np.float32), n=n.astype(np.float32), tools=phantom_tools(world))
