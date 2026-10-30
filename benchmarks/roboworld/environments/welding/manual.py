"""Human control of a live welding GAR session through a small local web page (pattern: barcode_scanning/manual.py;
the 3D view follows whiteboard/ and titrate/newton15/ with the same pinned, locally served three.js).

The page is served by this process, which attaches to an EXISTING `gar run welding ...` session and proxies a small set
of its HTTP API calls. The GAR key stays in this process's memory only: it is never sent to the browser, printed,
logged or written to disk. The browser authenticates to this proxy with a random page token carried in the URL fragment
(fragments are not sent to servers).

Every robot command the page offers is one of the policy interface's own actions -- cartesian_delta, cartesian_pose,
named_joint_target, gripper or tool_io (the robot controller's outputs to the welding power source) -- and the page shows
the exact action JSON that was sent. Physics advances only when a command is sent; nothing steps in the background, and
camera images are rendered only after a command or when asked for. Nothing in the world is moved, lit or fed by a key
directly: the torch moves because the arm's servos move it, the arc burns only where the robot holds the wire to
grounded metal, and the wire moves only through the feeder's inch output.

  python -m welding.manual --session SESSION_ID                 # policy key: robot cameras, source feedback, joints
  python -m welding.manual --session SESSION_ID --role owner    # owner key: adds the evaluator, the privileged state,
                                                                #   checkpoints, the diagnostic render and a 3D view
  python -m welding.manual --url http://127.0.0.1:PORT --key-file F   # F: a bare key, or gar's credential JSON
  GAR_MANUAL_KEY=... python -m welding.manual --url http://127.0.0.1:PORT

It prints one JSON line with the page URL (http://127.0.0.1:PORT/#token=...).
"""

import argparse
import base64
import hashlib
import hmac
import json
import math
import os
import re
import secrets
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import xml.etree.ElementTree as ET
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

STATIC = Path(__file__).parent/"manual_static"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"),
          "/viewer.js": ("viewer.js", "text/javascript; charset=utf-8"),
          "/styles.css": ("styles.css", "text/css; charset=utf-8")}
DEFAULT_PORT = 8796
UPSTREAM_TIMEOUT_S = 1800          # with the arc burning a simulated second costs 55-100 wall seconds
MAX_COMMAND_S = 10.0               # simulated seconds per page command
MAX_STEP_M = 0.10                  # largest Cartesian jog per command
MAX_TURN_DEG = 30.0                # largest rotation jog per command
MAX_TICKS = 10000                  # tool_io / hold ticks per command at the 1 ms tick
PRIVILEGED_STATE = ("tick", "time_s", "generation", "ready", "fault", "start_state", "tool", "process", "bond", "robot",
                    "history", "terminated", "evaluation")


class ProxyError(Exception):
    def __init__(self, status, message, detail=None):
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail


# ---------------------------------------------------------------------------------------------- credentials
def _gar_json(*args):
    """`gar ... --json` parsed in memory (a faulted but live session still prints its JSON)."""
    r = subprocess.run(["gar", *args, "--json"], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise SystemExit(f"`gar {args[0]}` failed (exit {r.returncode}); is the session running?") from None


def resolve_credentials(args):
    """(url, key). The key is returned in memory only."""
    url, key = args.url, None
    if args.session:
        status = _gar_json("status", args.session)
        url = url or status.get("url")
        creds = _gar_json("keys", args.session)
        key = creds.get("privileged" if args.role == "owner" else "policy")
        if not key:
            raise SystemExit(f"session has no {args.role} key (launch it with --policy-key for the policy role)")
    elif args.key_file:
        text = Path(args.key_file).read_text().strip()
        key = json.loads(text).get("privileged" if args.role == "owner" else "policy") if text.startswith("{") else text
    else:
        key = os.environ.get(args.key_env)
    if not url or not key:
        raise SystemExit("Give --session SESSION_ID, or --url plus a key (--key-file or $" + args.key_env + ")")
    return url.rstrip("/"), key


# ---------------------------------------------------------------------------------------------- kinematics
def _transform(xyz=(0., 0., 0.), rpy=(0., 0., 0.)):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", rpy).as_matrix()
    T[:3, 3] = xyz
    return T


def _vec(text, default):
    return np.fromstring(text, sep=" ") if text else np.asarray(default, float)


def _pose7(T):
    return [*map(float, T[:3, 3]), *map(float, Rotation.from_matrix(T[:3, :3]).as_quat())]


class CellKinematics:
    """Forward kinematics of the welding deployments from the same robot assets, mounts and torch frame as the
    backend's own IK (welding/kinematics.py Kinematics, which checks its chains against the simulator's links within
    1 mm when a session starts). Inputs are joint positions only: the robot's proprioception. The TCP comes from
    the very SerialKinematics chain the backend builds; every link's pose (for the 3D view) from a walk of the same
    URDF tree, checked against that chain here."""

    def __init__(self, robot):
        from robots.catalog import for_environment, model
        from robots import assets
        from robots.controllers.kinematics import SerialKinematics
        from welding import layout as LY
        from welding import torch as TO
        from welding.kinematics import HAND_TCP, TORCH_LINK, torch_tcp_in_tool0
        setup = for_environment("welding", robot).to_dict()
        names = list(setup["instances"])
        self.torch_arm = "right" if "right" in names else names[0]      # welding/scene.py Scene.gun
        self.arms = {}
        for name, inst in setup["instances"].items():
            spec = model(inst["model"])
            if spec.id == "g1":
                raise NotImplementedError(
                    "the G1 stands on its own feet: where its torch and hand are depends on where its pelvis is, which "
                    "joint positions alone do not give; the page's TCP readout and 3D view cover the fixed-base arms "
                    "(UR10e, FR3, Panda). Jogs and joint targets work")
            profile = spec.to_dict()["actuation"]["profiles"][inst["controller_profile"]]
            path = assets.resolve(spec.id)
            if spec.id == "fr3" and profile.get("default_tool") == "franka-hand":     # robots/backends/newton.py
                path = path.with_name("fr3_franka_hand.urdf")
            payload = path.read_bytes()
            if spec.id == "panda" and name == self.torch_arm:
                # the torch Panda carries no hand: the backend strips it from panda_arm_hand.urdf at load
                from robots.backends.newton import _panda_without_hand
                payload = _panda_without_hand(path)
            mount = LY.ROBOT_BASES[name]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", math.radians(mount["yaw_deg"])).as_matrix()
            base[:3, 3] = mount["pos"]
            gun = None
            if name == self.torch_arm:
                gun = TO.for_model(spec.id)
                tool0 = TORCH_LINK.get(spec.id, "fr3_link8")
                chain = SerialKinematics(payload, tool0, base=base)
                chain.chain[-1] = ("tool_offset", torch_tcp_in_tool0(gun), None)
                tcp_link, tcp_offset = tool0, torch_tcp_in_tool0(gun)
            else:
                tcp_link, off = HAND_TCP.get(spec.id, ("fr3_hand_tcp", (0., 0., 0.)))
                chain = SerialKinematics(payload, tcp_link, base=base, tip_offset=off)
                tcp_offset = np.eye(4)
                tcp_offset[:3, 3] = off
            joints, links, roots = self._tree(ET.fromstring(payload))
            fingers = [j["name"] for j in joints if j["type"] == "prismatic"]
            self.arms[name] = dict(model=spec.id, urdf=path, base=base, chain=chain,
                                   names=[f"{name}/{j}" for j in chain.names], joints=joints, links=links, roots=roots,
                                   tcp_link=tcp_link, tcp_offset=tcp_offset, gun=gun,
                                   fingers=[f"{name}/{j}" for j in fingers],
                                   limits={f"{name}/{j['name']}": dict(lower=j["lower"], upper=j["upper"],
                                                                       unit="m" if j["type"] == "prismatic" else "rad")
                                           for j in joints if j["type"] in ("revolute", "prismatic")})
        self.hand_arms = [a for a, v in self.arms.items() if v["fingers"]]
        self._check()

    @staticmethod
    def _tree(xml):
        joints, children = [], set()
        for j in xml.findall("joint"):
            o = j.find("origin")
            kind = j.get("type")
            limit = j.find("limit")
            mimic = j.find("mimic")
            joints.append(dict(
                name=j.get("name"), type=kind, parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                origin=_transform(_vec(o.get("xyz") if o is not None else None, (0, 0, 0)),
                                  _vec(o.get("rpy") if o is not None else None, (0, 0, 0))),
                axis=_vec(j.find("axis").get("xyz") if j.find("axis") is not None else None, (1, 0, 0)),
                lower=float(limit.get("lower", "nan")) if limit is not None else None,
                upper=float(limit.get("upper", "nan")) if limit is not None else None,
                mimic=None if mimic is None else (mimic.get("joint"), float(mimic.get("multiplier", "1")),
                                                  float(mimic.get("offset", "0")))))
            children.add(joints[-1]["child"])
        links = {l.get("name"): l for l in xml.findall("link")}
        return joints, links, [n for n in links if n not in children]

    def link_frames(self, arm, positions):
        """World pose (4x4) of every link of `arm` for joint positions {"arm/joint": value} (missing active joints:
        KeyError; mimic joints follow their source when absent)."""
        a = self.arms[arm]
        T = {r: a["base"] for r in a["roots"]}
        pending = list(a["joints"])
        while pending:
            progressed = False
            for j in list(pending):
                if j["parent"] not in T:
                    continue
                M = T[j["parent"]] @ j["origin"]
                if j["type"] in ("revolute", "continuous", "prismatic"):
                    key = f"{arm}/{j['name']}"
                    if key in positions:
                        q = float(positions[key])
                    elif j["mimic"] is not None:
                        src, mult, off = j["mimic"]
                        q = float(positions[f"{arm}/{src}"])*mult + off
                    else:
                        raise KeyError(key)
                    R = np.eye(4)
                    if j["type"] == "prismatic":
                        R[:3, 3] = j["axis"]*q
                    else:
                        R[:3, :3] = Rotation.from_rotvec(j["axis"]*q).as_matrix()
                    M = M @ R
                T[j["child"]] = M
                pending.remove(j)
                progressed = True
            if not progressed:
                raise ValueError(f"{arm}: URDF tree is not connected")
        return T

    def _check(self):
        """The tree walk and the backend's chain must agree (at the middle of every joint range)."""
        for arm, a in self.arms.items():
            q = {f"{arm}/{j['name']}": .5*(j["lower"] + j["upper"]) for j in a["joints"]
                 if j["type"] in ("revolute", "prismatic")}
            T = self.link_frames(arm, q)[a["tcp_link"]] @ a["tcp_offset"]
            C = a["chain"].fk(np.array([q[n] for n in a["names"]]))
            if not np.allclose(T, C, atol=1e-9):
                raise RuntimeError(f"{arm}: URDF tree FK disagrees with the backend's chain")

    def tcp(self, arm, positions):
        a = self.arms[arm]
        return a["chain"].fk(np.array([float(positions[n]) for n in a["names"]]))

    def summary(self, positions):
        out = {}
        for arm, a in self.arms.items():
            try:
                T = self.tcp(arm, positions)
            except (KeyError, ValueError, TypeError):
                continue
            R = Rotation.from_matrix(T[:3, :3])
            q = R.as_quat()
            e = {"position_m": T[:3, 3].tolist(), "quat_wxyz": [float(q[3]), *map(float, q[:3])],
                 "rpy_deg": R.as_euler("xyz", degrees=True).tolist(), "z_axis": T[:3, 2].tolist()}
            if arm == self.torch_arm:
                ax = T[:3, 2]                     # the wire axis toward the work (welding/kinematics.py)
                e["work_deg"] = math.degrees(math.atan2(-ax[2], -ax[1]))
                e["push_deg"] = math.degrees(math.asin(float(np.clip(ax[0], -1, 1))))
            if a["fingers"]:
                try:
                    e["finger_opening_m"] = float(sum(float(positions[f]) for f in a["fingers"]))
                except KeyError:
                    e["finger_opening_m"] = None
            out[arm] = e
        return out

    # -------------------------------------------------------------------------------------- 3D view
    def _mesh_path(self, filename, urdf):
        if filename.startswith("package://"):
            package, rest = filename[len("package://"):].split("/", 1)
            for parent in urdf.parents:
                if parent.name == package:
                    return parent/rest
            raise FileNotFoundError(filename)
        filename = filename[len("file://"):] if filename.startswith("file://") else filename
        p = Path(filename)
        return p if p.is_absolute() else (urdf.parent/p).resolve()

    def scene(self, materials_for):
        """Static description of the robots' collision geometry: per link, its URDF <collision> parts in the link frame
        (meshes content-addressed), and the torch gun's collision shapes in its flange frame (welding/torch.py)."""
        import trimesh
        meshes, links = {}, []
        for arm, a in self.arms.items():
            for lname, link in a["links"].items():
                parts = []
                for c in link.findall("collision"):
                    o = c.find("origin")
                    T = _transform(_vec(o.get("xyz") if o is not None else None, (0, 0, 0)),
                                   _vec(o.get("rpy") if o is not None else None, (0, 0, 0)))
                    g = c.find("geometry")[0]
                    part = {"position": T[:3, 3].tolist(), "quat_xyzw": Rotation.from_matrix(T[:3, :3]).as_quat().tolist()}
                    if g.tag == "mesh":
                        m = trimesh.load(self._mesh_path(g.get("filename"), a["urdf"]), force="mesh")
                        v = np.asarray(m.vertices, np.float32)*_vec(g.get("scale"), (1, 1, 1)).astype(np.float32)
                        f = np.asarray(m.faces, np.int32)
                        key = hashlib.sha256(v.tobytes() + f.tobytes()).hexdigest()[:20]
                        meshes.setdefault(key, {"vertices": np.round(v.reshape(-1), 6).tolist(), "indices": f.reshape(-1).tolist()})
                        part.update(kind="mesh", mesh=key)
                    elif g.tag == "box":
                        part.update(kind="box", size=_vec(g.get("size"), (0, 0, 0)).tolist())
                    elif g.tag == "cylinder":
                        part.update(kind="cylinder", radius=float(g.get("radius")), length=float(g.get("length")))
                    elif g.tag == "sphere":
                        part.update(kind="sphere", radius=float(g.get("radius")))
                    else:
                        continue
                    part["material"] = materials_for(arm, a["model"], lname)
                    parts.append(part)
                if parts:
                    links.append({"id": f"{arm}/{lname}", "parts": parts})
            if a["gun"] is not None:
                parts = []
                for pname, kind, c, R, dims, _mass in a["gun"].parts():
                    part = {"position": np.asarray(c, float).tolist(),
                            "quat_xyzw": Rotation.from_matrix(np.asarray(R, float)).as_quat().tolist(),
                            "material": "gun_" + pname.split("_")[1], "name": pname}
                    if kind == "cylinder":
                        part.update(kind="zcylinder", radius=dims[0], half_height=dims[1])
                    elif kind == "capsule":
                        part.update(kind="zcapsule", radius=dims[0], half_height=dims[1])
                    else:                                  # tapered nozzle: (r_base at -half, r_tip at +half, half)
                        part.update(kind="zfrustum", radius_base=dims[0], radius_tip=dims[1], half_height=dims[2],
                                    sides=a["gun"].nozzle_sides)
                    parts.append(part)
                links.append({"id": f"{arm}/{a['tcp_link']}#gun", "parts": parts})
        return meshes, links

    def poses(self, positions):
        """{link id: [x, y, z, qx, qy, qz, qw]} for every link (and the gun on the torch arm's flange)."""
        out = {}
        for arm, a in self.arms.items():
            try:
                T = self.link_frames(arm, positions)
            except (KeyError, ValueError, TypeError):
                continue
            for lname, M in T.items():
                out[f"{arm}/{lname}"] = _pose7(M)
            if a["gun"] is not None:
                out[f"{arm}/{a['tcp_link']}#gun"] = _pose7(T[a["tcp_link"]])
        return out

    def wire(self, positions, stickout_m):
        """Contact tip and wire tip (world) of the torch from its flange pose and the gun spec (welding/torch.py)."""
        a = self.arms[self.torch_arm]
        T = self.link_frames(self.torch_arm, positions)[a["tcp_link"]]
        g = a["gun"]
        tip = T[:3, :3] @ g.contact_tip + T[:3, 3]
        axis = T[:3, :3] @ g.axis
        return {"contact_tip_m": tip.tolist(), "axis": axis.tolist(),
                "wire_tip_m": (tip + axis*float(stickout_m or 0.)).tolist(),
                "work_point_m": (T @ a["tcp_offset"])[:3, 3].tolist()}


def _robot_material(arm, model_id, link):
    if "finger" in link:
        return "finger"
    if model_id == "ur10e":
        return "ur_joint" if any(k in link for k in ("shoulder", "wrist", "base")) else "ur_link"
    return "fr3_hand" if "hand" in link else "fr3"


def static_scene():
    """The cell's fixed collision geometry from welding/layout.py exactly as welding/scene.py places it: the table top and
    legs, the welding equipment (LY.source_boxes()), the camera mounts (LY.CAMERA_MOUNTS), the screens, and the toggle
    clamps' bases and arms; the clamp pads at their nominal place (their
    slide is not exposed by the API). Mirrored here because /v1/geometry lists only the moving bodies' poses: re-check
    this function when scene.py changes. Also the boxes of the live bodies whose poses /v1/geometry reports."""
    from welding import layout as LY
    T = LY.TABLE
    z, (sx, sy), t = T["top_z"], T["size"], T["top_thickness"]
    items = []

    def box(name, center, size, material, quat=None):
        items.append({"name": name, "kind": "box", "position": list(map(float, center)), "size": list(map(float, size)),
                      "material": material, **({"quat_xyzw": list(map(float, quat))} if quat is not None else {})})

    def zcyl(name, center, radius, half, material):
        items.append({"name": name, "kind": "zcylinder", "position": list(map(float, center)), "radius": float(radius),
                      "half_height": float(half), "material": material})
    box("table_top", (0., 0., z - t/2), (sx, sy, t), "table")
    for x in (-sx/2 + .06, sx/2 - .06):
        for y in (-sy/2 + .06, sy/2 - .06):
            box(f"table_leg_{'p' if x > 0 else 'm'}{'p' if y > 0 else 'm'}", (x, y, (z - t)/2), (T["leg"], T["leg"], z - t),
                "table")
    for name, c, h in LY.source_boxes():              # the welding equipment, read from the layout's own function
        if name == "gas_cylinder":                    # scene.py: a cylinder collider, radius h[0], half height h[2]
            zcyl(name, c, h[0], h[2], "cylinder")
        else:
            box(name, c, 2*np.asarray(h, float), "cart")
    for name, c, h in LY.CAMERA_MOUNTS:
        box(name, c, 2*np.asarray(h, float), "mount")
    for name, c, h, yaw in LY.screen_boxes():         # the same function scene.py places the screens with
        box(name, c, 2*np.asarray(h, float), "screen", LY.yaw_quat(yaw))
    z_plate = float(LY.WELD_ORIGIN[2])
    for c in LY.CLAMPS:
        x, y = c["pad_xy"]
        box(f"{c['name']}_base", (x, y - .085, z + .025), (.044, .06, .05), "clamp")
        box(f"{c['name']}_arm", (x, y - .045, z_plate + .035), (.018, .11, .016), "clamp")
        zcyl(f"{c['name']}_pad", (x, y, z_plate + LY.CLAMP_PAD["height"]/2), LY.CLAMP_PAD["radius"],
             LY.CLAMP_PAD["height"]/2, "pad")
    cameras = {n: {"eye": list(c["eye"]), "target": list(c["target"]), "hfov_deg": c["hfov"], "role": c["role"]}
               for n, c in LY.CAMERAS.items()}
    return items, {"work_clamp": list(LY.WORK_CLAMP["size"])}, cameras


MATERIALS = {   # display colours (linear-ish sRGB, roughness, metalness) for the operator's 3D view only
    "table": ([.30, .31, .33], .6, .5), "cart": ([.18, .27, .40], .7, .1), "clamp": ([.80, .30, .08], .6, .2),
    "pad": ([.08, .08, .08], .9, 0.), "plate": ([.42, .43, .45], .5, .6), "bracket": ([.50, .51, .53], .5, .6),
    "work_clamp": ([.72, .45, .20], .4, .7), "ur_link": ([.66, .70, .74], .5, .3), "ur_joint": ([.20, .40, .62], .5, .2),
    "fr3": ([.88, .89, .90], .6, .1), "fr3_hand": ([.85, .86, .88], .6, .1), "finger": ([.25, .25, .27], .6, .1),
    "gun_mount": ([.12, .12, .13], .6, .1), "gun_body": ([.10, .10, .11], .5, .2), "gun_neck": ([.15, .15, .16], .5, .3),
    "gun_nozzle": ([.78, .52, .26], .35, .8), "wire": ([.85, .60, .30], .3, .9), "cylinder": ([.15, .35, .22], .6, .2),
    "mount": ([.35, .36, .38], .6, .4), "screen": ([.95, .45, .10], .8, 0.)}


# ---------------------------------------------------------------------------------------------- proxy
class Manual:
    def __init__(self, url, key, token):
        self.url, self._key, self.token = url, key, token
        self.busy = threading.Lock()
        self.info = None
        self.kin = None
        self.kin_error = None
        self.positions = None            # {"arm/joint": position} from the latest observation
        self.last_observation = None
        self._scene = None

    # -- upstream
    def upstream(self, method, path, body=None, raw=False, idempotent=False):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": "Bearer " + self._key}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if idempotent:
            headers["Idempotency-Key"] = uuid.uuid4().hex
        req = urllib.request.Request(self.url + path, data=data, method=method, headers=headers)
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_S) as r:
                payload, ctype = r.read(), r.headers.get("Content-Type", "")
        except urllib.error.HTTPError as e:
            text = e.read().decode("utf-8", "replace")
            try:
                parsed = json.loads(text)
                message = parsed.get("error") or text
                committed = parsed.get("committed")
            except (ValueError, AttributeError):
                message, committed = text or e.reason, None
            raise ProxyError(e.code, f"GAR {method} {path.split('?')[0]} -> HTTP {e.code}: {message}",
                             {"committed": committed, "gar_status": e.code})
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            raise ProxyError(502, f"GAR session unreachable: {getattr(e, 'reason', e)}")
        wall = time.perf_counter() - t0
        if raw:
            return payload, ctype, wall
        return json.loads(payload.decode()), wall

    def health(self):
        try:
            with urllib.request.urlopen(self.url + "/v1/health", timeout=10) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            return json.loads(e.read().decode())
        except Exception as e:  # noqa: BLE001 - reported to the page
            return {"ok": False, "ready": False, "error": str(e)}

    @property
    def owner(self):
        return bool(self.info) and self.info["role"] == "owner"

    def connect(self):
        help_, _ = self.upstream("GET", "/v1/help")
        owner = help_.get("role") == "privileged"
        caps = help_.get("effective_capabilities") or {}
        task, _ = self.upstream("GET", "/v1/task")
        robot = caps.get("robot")
        try:
            self.kin = CellKinematics(robot)
        except Exception as e:  # noqa: BLE001 - the page states that TCP and the 3D view are unavailable and why
            self.kin, self.kin_error = None, f"{type(e).__name__}: {e}"
        if owner:
            notes = caps.get("notes") or {}
            state, _ = self.upstream("GET", "/v1/state")
            state = state["state"]
            dt = float(state["config"]["rigid_dt"])
            arms = [a for a in str(notes.get("arms", "")).split(",") if a] or list(self.kin.arms if self.kin else [])
            tool = notes.get("tool_io") or {}
            torch_arm, outputs, feedback = tool.get("arm"), tool.get("outputs") or {}, tool.get("feedback") or []
            limits = {}
            for a in (self.kin.arms.values() if self.kin else ()):
                limits.update(a["limits"])
            joints = {n: dict(limits.get(n, {}), source="robot URDF (the backend's limits)" if n in limits else None)
                      for n in state["joints"]}
            m = re.search(r"<=\s*([0-9.]+)\s*N total", str(notes.get("gripper", "")))
            grip_max = float(m.group(1)) if m else None
            notes_out = {k: notes.get(k) for k in ("task", "clock", "control", "torch_tcp", "gripper", "cameras", "wall_time")}
        else:
            hw = caps.get("hardware") or {}
            dt = float(hw.get("dt_s"))
            arms, torch_arm = list(hw.get("arms") or []), hw.get("torch_arm")
            outputs = hw.get("tool_io") or {}
            feedback = sorted({f.split(".", 1)[1] for f in caps.get("proprioception_fields", []) if f.startswith("tool.")})
            joints = {n: dict(j, source="capabilities.hardware.joints") for n, j in (hw.get("joints") or {}).items()}
            fingers = [j for n, j in joints.items() if "finger" in n]
            grip_max = 2*float(fingers[0]["effort_limit"]) if fingers and fingers[0].get("effort_limit") else None
            notes_out = {"torch_tcp": hw.get("torch_tcp"), "wall_time": hw.get("wall_time")}
        hand_arms = sorted({n.split("/", 1)[0] for n in joints if "finger" in n})
        from welding import params as PR
        sp = PR.v(PR.SCHEDULES)["spray_0p9_90-10"]
        self.info = {
            "role": "owner" if owner else "policy", "robot": robot, "arms": arms, "torch_arm": torch_arm,
            "hand_arms": hand_arms, "cameras": list(caps.get("robot_cameras") or []),
            "instruction": task.get("instruction"), "action_types": sorted(caps.get("action_types") or []),
            "dt_s": dt, "joints": joints, "tool_outputs": outputs, "tool_feedback": feedback,
            "grip_max_n": grip_max, "notes": notes_out,
            "schedule_preset": {"label": "Spray, 0.035 in ER70S-6, 90/10 gas",
                                "wire_feed_m_min": sp["wfs_m_min"], "voltage_v": sp["voltage_v"],
                                "source": "welding/params.py SCHEDULES['spray_0p9_90-10'] (Lincoln C4.200, read)"},
            "tcp_available": self.kin is not None, "tcp_error": self.kin_error,
            "limits": {"max_command_s": MAX_COMMAND_S, "max_step_mm": MAX_STEP_M*1e3, "max_turn_deg": MAX_TURN_DEG,
                       "max_ticks": MAX_TICKS}}
        return self.info

    # -- observations
    def _summarize(self, obs, wall_s):
        prop = obs.get("proprioception") or {}
        joints = prop.get("joints") or {}
        if joints:
            self.positions = {n: v.get("position") for n, v in joints.items()}
        out = {"observation_id": obs.get("observation_id"), "sim_time_s": obs.get("sim_time_s"),
               "sequence": obs.get("sequence"), "cameras": sorted((obs.get("images") or {}).keys()),
               "wall_s": round(wall_s, 3),
               "joints": {n: {k: v.get(k) for k in ("position", "velocity", "target") if k in v} for n, v in joints.items()},
               "tool": prop.get("tool")}
        if self.kin and self.positions:
            out["tcp"] = self.kin.summary(self.positions)
        self.last_observation = out
        return out

    def observe(self):
        body, wall = self.upstream("GET", "/v1/observation")
        return {"observation": self._summarize(body["observation"], wall)}

    def _current_positions(self):
        """Owner: the exact joints now (privileged state). Policy: the latest observation's proprioception."""
        if self.owner:
            state, _ = self.upstream("GET", "/v1/state")
            return {n: v["position"] for n, v in state["state"]["joints"].items()}
        if not self.positions:
            raise ProxyError(409, "No measured joint positions yet: refresh the cameras first (render only, no physics)")
        return dict(self.positions)

    # -- commands (each becomes ONE policy action, or for inching two tool_io actions in one request)
    def _seconds(self, value):
        dt = self.info["dt_s"]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ProxyError(400, "Duration must be a number of seconds")
        ticks = round(value/dt)
        if ticks < 1 or ticks*dt > MAX_COMMAND_S + 1e-9:
            raise ProxyError(400, f"Duration must be between {dt:g} s (one tick) and {MAX_COMMAND_S:g} s")
        if abs(value/dt - ticks) > 1e-6:
            raise ProxyError(400, f"Duration must be a whole number of {dt*1e3:g} ms ticks")
        return round(ticks*dt, 9), ticks

    def _arm(self, arm, hand=False):
        pool = self.info["hand_arms"] if hand else self.info["arms"]
        if arm not in pool:
            raise ProxyError(400, f"arm must be one of {pool}" if pool else "this deployment has no hand")
        return arm

    @staticmethod
    def _number(value, name, lo=-math.inf, hi=math.inf):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not lo <= value <= hi:
            raise ProxyError(400, f"{name} must be a number in [{lo:g}, {hi:g}]")
        return float(value)

    def _need_kin(self):
        if self.kin is None:
            raise ProxyError(409, f"The page's forward kinematics is unavailable ({self.kin_error}); use jogs or joint targets")

    def build(self, req):
        """(label, action payload) for a page request; refuses only malformed requests and page limits -- the robot
        and the source refuse everything else themselves."""
        kind = req.get("kind")
        if kind == "jog":
            arm = self._arm(req.get("arm"))
            axis = req.get("axis")
            if axis not in (0, 1, 2) or req.get("sign") not in (-1, 1):
                raise ProxyError(400, "jog needs axis 0/1/2 and sign +1/-1")
            step = self._number(req.get("step_mm"), "step_mm", 1e-3, MAX_STEP_M*1e3)/1e3
            s, _ = self._seconds(req.get("duration_s"))
            d = [0., 0., 0.]
            d[axis] = req["sign"]*step
            return (f"{arm} {'+-'[req['sign'] < 0]}{'xyz'[axis]} {step*1e3:g} mm in {s:g} s",
                    {"type": "cartesian_delta", "arm": arm, "delta_m": d, "duration_s": s})
        if kind == "rotate":
            self._need_kin()
            arm = self._arm(req.get("arm"))
            axis = req.get("axis")
            if axis not in (0, 1, 2) or req.get("sign") not in (-1, 1):
                raise ProxyError(400, "rotate needs axis 0/1/2 and sign +1/-1")
            deg = self._number(req.get("step_deg"), "step_deg", 1e-3, MAX_TURN_DEG)
            s, _ = self._seconds(req.get("duration_s"))
            T = self.kin.tcp(arm, self._current_positions())
            v = np.zeros(3)
            v[axis] = req["sign"]*math.radians(deg)
            q = (Rotation.from_rotvec(v)*Rotation.from_matrix(T[:3, :3])).as_quat()
            return (f"{arm} turn {'+-'[req['sign'] < 0]}{deg:g} deg about world {'xyz'[axis]} (work point held) in {s:g} s",
                    {"type": "cartesian_pose", "arm": arm, "position_m": T[:3, 3].tolist(),
                     "orientation_wxyz": [float(q[3]), *map(float, q[:3])], "duration_s": s})
        if kind == "goto":
            arm = self._arm(req.get("arm"))
            p = req.get("position_mm")
            if not isinstance(p, list) or len(p) != 3:
                raise ProxyError(400, "position_mm must be [x, y, z]")
            p = [self._number(v, "position_mm", -1e4, 1e4)/1e3 for v in p]
            s, _ = self._seconds(req.get("duration_s"))
            return (f"{arm} to ({', '.join(f'{v*1e3:.1f}' for v in p)}) mm, orientation kept, in {s:g} s",
                    {"type": "cartesian_pose", "arm": arm, "position_m": p, "duration_s": s})
        if kind == "orient":
            self._need_kin()
            arm = self._arm(req.get("arm"))
            if arm != self.info["torch_arm"]:
                raise ProxyError(400, f"Torch angles apply to the torch arm '{self.info['torch_arm']}'")
            work = self._number(req.get("work_deg"), "work_deg", 0., 90.)
            push = self._number(req.get("push_deg"), "push_deg", -45., 45.)
            s, _ = self._seconds(req.get("duration_s"))
            from welding.kinematics import torch_orientation
            T = self.kin.tcp(arm, self._current_positions())
            q = torch_orientation(work, push)
            return (f"torch to {work:g} deg work, {push:g} deg push (world axes), work point held, in {s:g} s",
                    {"type": "cartesian_pose", "arm": arm, "position_m": T[:3, 3].tolist(),
                     "orientation_wxyz": [float(q[3]), *map(float, q[:3])], "duration_s": s})
        if kind == "joint":
            name = req.get("name")
            if name not in self.info["joints"]:
                raise ProxyError(400, f"unknown joint {name!r}")
            value = self._number(req.get("value"), "value")
            _, ticks = self._seconds(req.get("duration_s"))
            return (f"{name} -> {value:g} over {ticks} ticks",
                    {"type": "named_joint_target", "targets_si": {name: value}, "ticks": ticks})
        if kind == "hold":
            _, ticks = self._seconds(req.get("duration_s"))
            return (f"advance {ticks} ticks, every target held",
                    {"type": "named_joint_target", "targets_si": {}, "ticks": ticks})
        if kind == "gripper":
            arm = self._arm(req.get("arm"), hand=True)
            s, _ = self._seconds(req.get("duration_s"))
            fingers = [j for n, j in self.info["joints"].items() if n.startswith(arm + "/") and "finger" in n]
            mode = req.get("mode")
            if mode in ("open", "close"):
                if not fingers or fingers[0].get("upper") is None:
                    raise ProxyError(409, "finger limits unknown")
                w = fingers[0]["upper"] if mode == "open" else fingers[0]["lower"]
                return (f"{arm} hand {mode} (position mode, half gap {w*1e3:g} mm) in {s:g} s",
                        {"type": "gripper", "arm": arm, "half_width_m": float(w), "duration_s": s})
            if mode == "force":
                f = self._number(req.get("effort_n"), "effort_n", 0., 1e4)
                return (f"{arm} hand force grip {f:g} N total in {s:g} s",
                        {"type": "gripper", "arm": arm, "inward_effort_n": f, "duration_s": s})
            raise ProxyError(400, "gripper mode must be open, close or force")
        if kind == "tool":
            outputs = req.get("outputs")
            spec = self.info["tool_outputs"]
            if not isinstance(outputs, dict) or not outputs:
                raise ProxyError(400, "tool needs outputs")
            for k, v in outputs.items():
                if k not in spec:                  # unknown names never reach the source (it would refuse them too)
                    raise ProxyError(400, f"unknown source output {k!r}; outputs: {', '.join(spec)}")
                want = spec[k].get("kind")
                ok = (isinstance(v, bool) if want == "bool" else isinstance(v, str) if want == "choice"
                      else isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v))
                if not ok:
                    raise ProxyError(400, f"{k} must be a {'true/false' if want == 'bool' else want}")
            ticks = req.get("ticks", 1)
            if isinstance(ticks, bool) or not isinstance(ticks, int) or not 1 <= ticks <= MAX_TICKS:
                raise ProxyError(400, f"ticks must be an integer in [1, {MAX_TICKS}]")
            arm = self.info["torch_arm"]
            return (f"source: {json.dumps(outputs)}, {ticks} tick{'s' if ticks > 1 else ''}",
                    {"type": "tool_io", "arm": arm, "outputs": outputs, "ticks": ticks})
        if kind == "inch":
            direction = req.get("direction")
            if direction not in ("forward", "backward"):
                raise ProxyError(400, "inch direction must be forward or backward")
            _, ticks = self._seconds(req.get("duration_s"))
            arm = self.info["torch_arm"]
            return (f"inch the wire {direction} for {ticks} ticks, then stop (1 tick)",
                    {"actions": [{"type": "tool_io", "arm": arm, "outputs": {"wire_inch": direction}, "ticks": ticks},
                                 {"type": "tool_io", "arm": arm, "outputs": {"wire_inch": "stop"}, "ticks": 1}]})
        raise ProxyError(400, f"unknown command kind {kind!r}")

    def _page_hint(self, action):
        """A local reading of a refused source command against the documented ranges (the policy key's refusal text is
        generic by design)."""
        items = action["actions"] if "actions" in action else [action]
        notes = []
        for a in items:
            if a.get("type") != "tool_io":
                continue
            for k, v in a["outputs"].items():
                spec = self.info["tool_outputs"].get(k, {})
                if spec.get("kind") == "number" and spec.get("range"):
                    lo, hi = spec["range"]
                    if not lo <= v <= hi:
                        notes.append(f"{k} = {v:g} is outside the source's range {lo:g}-{hi:g} {spec.get('unit', '')}".strip())
                if spec.get("kind") == "choice" and v not in spec.get("choices", ()):
                    notes.append(f"{k} = {v!r} is not one of {list(spec['choices'])}")
        return "; ".join(notes) or None

    def command(self, req):
        label, action = self.build(req)
        before = self.last_observation
        try:
            body, wall = self.upstream("POST", "/v1/action", action, idempotent=True)
        except ProxyError as e:
            e.detail = dict(e.detail or {}, sent=action, label=label, page_hint=self._page_hint(action),
                            health=self.health())
            raise
        result = {"label": label, "sent": action, "committed": body.get("committed"), "request_id": body.get("request_id"),
                  "wall_s": round(wall, 3), "observation_error": body.get("observation_error")}
        if self.owner:
            info = body.get("info") or {}
            result["info"] = {k: info.get(k) for k in ("requested_ticks", "executed_ticks", "termination", "physics_s",
                                                       "events")}
        if body.get("observation"):
            obs = self._summarize(body["observation"], wall)
            result["observation"] = obs
            if before and before.get("sim_time_s") is not None and obs.get("sim_time_s") is not None:
                dt = self.info["dt_s"]
                result["clock"] = {"before_s": before["sim_time_s"], "after_s": obs["sim_time_s"],
                                   "ticks": int(round((obs["sim_time_s"] - before["sim_time_s"])/dt))}
            if before and before.get("tcp") and obs.get("tcp"):
                result["tcp_delta"] = {}
                for arm, a in obs["tcp"].items():
                    b = before["tcp"].get(arm)
                    if not b:
                        continue
                    rot = (Rotation.from_quat([*a["quat_wxyz"][1:], a["quat_wxyz"][0]]) *
                           Rotation.from_quat([*b["quat_wxyz"][1:], b["quat_wxyz"][0]]).inv()).as_rotvec()
                    d = {"position_mm": [(x - y)*1e3 for x, y in zip(a["position_m"], b["position_m"])],
                         "rotation_deg": np.degrees(rot).tolist()}
                    if a.get("finger_opening_m") is not None and b.get("finger_opening_m") is not None:
                        d["finger_opening_mm"] = (a["finger_opening_m"] - b["finger_opening_m"])*1e3
                    result["tcp_delta"][arm] = d
        return result

    # -- owner
    def _require_owner(self):
        if not self.owner:
            raise ProxyError(403, "Privileged state needs the owner key")

    def owner_state(self, pose=False):
        self._require_owner()
        s = self.upstream("GET", "/v1/state")[0]["state"]
        if s.get("evaluation") is None:          # the backend evaluates on demand (state_evaluation=false)
            s["evaluation"] = self.upstream("GET", "/v1/evaluation")[0]["evaluation"].get("details")
        out = {k: s.get(k) for k in PRIVILEGED_STATE}
        out["events"] = (s.get("events") or [])[-12:]
        out["config"] = {k: (s.get("config") or {}).get(k) for k in ("preset", "task", "seed", "robot", "coupon_offset",
                                                                    "draft_m_s", "work_clamp_on_table")}
        out["joints"] = s.get("joints")
        try:
            session, _ = self.upstream("GET", "/v1/session")
            out["session"] = {k: session.get(k) for k in ("session_id", "paused", "stopping", "clock")}
        except ProxyError:
            out["session"] = None
        if self.kin:
            positions = {n: v["position"] for n, v in s["joints"].items()}
            out["tcp"] = self.kin.summary(positions)
            if pose:
                geo = self.upstream("GET", "/v1/geometry")[0]["geometry"]
                stick = ((s.get("tool") or {}).get("stickout_mm") or 0.)/1e3
                out["pose"] = {"links": self.kin.poses(positions), "bodies": {
                    n: [*b["position_m"], *b["quat_xyzw"]] for n, b in geo.get("bodies", {}).items()},
                    "weld_frame": geo.get("weld_frame"), "coupon": (geo.get("items") or {}).get("coupon"),
                    "wire": self.kin.wire(positions, stick), "tick": s.get("tick"), "time_s": s.get("time_s")}
        return out

    def scene(self):
        self._require_owner()
        if self.kin is None:
            raise ProxyError(409, f"3D view unavailable: {self.kin_error}")
        if self._scene is None:
            meshes, links = self.kin.scene(_robot_material)
            items, sizes, cameras = static_scene()
            geo = self.upstream("GET", "/v1/geometry")[0]["geometry"]
            coupon = (geo.get("items") or {}).get("coupon") or {}
            bodies = {"base_plate": {"kind": "box", "size": coupon.get("base_m"), "material": "plate"},
                      "work_clamp": {"kind": "box", "size": sizes["work_clamp"], "material": "work_clamp"}}
            if coupon.get("web_m"):
                bodies["bracket"] = {"kind": "box", "size": coupon["web_m"], "material": "bracket"}
            self._scene = {"static": items, "bodies": bodies, "links": links, "meshes": meshes, "materials": MATERIALS,
                           "cameras": cameras, "torch_arm": self.kin.torch_arm,
                           "note": ("Collision geometry posed from /v1/state joint positions (forward kinematics of the "
                                    "robots' URDFs, the backend's own chains) and /v1/geometry body poses. The clamp "
                                    "pads are drawn at their nominal place. The weld metal, heat and arc are not drawn: "
                                    "the robot cameras and the diagnostic render show them.")}
        return self._scene

    def checkpoints(self):
        self._require_owner()
        return {"checkpoints": self.upstream("GET", "/v1/checkpoints")[0].get("checkpoints")}

    def save_checkpoint(self, body):
        self._require_owner()
        label = body.get("label")
        if not isinstance(label, str) or not 1 <= len(label) <= 64:
            raise ProxyError(400, "Checkpoint label: 1-64 characters")
        r, wall = self.upstream("POST", "/v1/checkpoints/save", {"label": label})
        return {"checkpoint": r.get("checkpoint"), "wall_s": round(wall, 3)}

    def restore_checkpoint(self, body):
        self._require_owner()
        label = body.get("checkpoint")
        if not isinstance(label, str) or not label:
            raise ProxyError(400, "Choose a checkpoint")
        r, wall = self.upstream("POST", "/v1/checkpoints/restore", {"checkpoint": label})
        out = {"info": r.get("info"), "committed": r.get("committed"), "wall_s": round(wall, 3),
               "observation_error": r.get("observation_error")}
        if r.get("observation"):
            out["observation"] = self._summarize(r["observation"], wall)
        return out

    def render(self):
        self._require_owner()
        data, ctype, wall = self.upstream("GET", "/v1/render", raw=True)
        return data, ctype, wall

    def recover(self):
        self._require_owner()
        body, wall = self.upstream("POST", "/v1/recover", {"acknowledge_uncertain_actions": True})
        result = {"recovered": body.get("recovered"), "previous_fault": body.get("previous_fault"),
                  "health": body.get("health"), "wall_s": round(wall, 3)}
        if body.get("observation"):
            result["observation"] = self._summarize(body["observation"], wall)
        return result


# ---------------------------------------------------------------------------------------------- web
def _vendor():
    """The pinned three.js r180 modules provisioned by whiteboard/web_assets.py (hash-checked; no CDN at runtime)."""
    try:
        from whiteboard.web_assets import FILES, ROOT
    except Exception as e:  # noqa: BLE001
        return {}, f"whiteboard.web_assets unavailable: {e}"
    files, missing = {}, []
    for name, (_source, digest) in FILES.items():
        p = ROOT/name
        if p.is_file() and hashlib.sha256(p.read_bytes()).hexdigest() == digest:
            files[f"/vendor/{name}"] = p
        else:
            missing.append(name)
    return files, (f"three.js modules missing or changed ({', '.join(missing)}): run `python -m whiteboard.web_assets`"
                   if missing else None)


def _csp(index_html):
    """script-src 'self' plus the hash of the page's one inline script (the three.js import map)."""
    m = re.search(r'<script type="importmap">(.*?)</script>', index_html, re.S)
    extra = ""
    if m:
        extra = " 'sha256-" + base64.b64encode(hashlib.sha256(m.group(1).encode()).digest()).decode() + "'"
    return (f"default-src 'self'; script-src 'self'{extra}; img-src 'self' blob: data:; object-src 'none'; "
            "frame-ancestors 'none'; base-uri 'none'; form-action 'none'")


def make_handler(manual, allowed_hosts, vendor):
    csp = _csp((STATIC/"index.html").read_text())

    class Handler(BaseHTTPRequestHandler):
        server_version = "welding-manual"

        def log_message(self, fmt, *args):
            return

        def _headers(self, status, ctype, length, extra=None):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", csp)
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _json(self, value, status=200):
            data = json.dumps(value).encode()
            try:
                self._headers(status, "application/json; charset=utf-8", len(data))
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _bytes(self, data, ctype, extra=None):
            try:
                self._headers(200, ctype, len(data), extra)
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _guard(self, api):
            if self.headers.get("Host", "") not in allowed_hosts:
                self._json({"ok": False, "error": "Host not allowed"}, 403)
                return False
            if api and not hmac.compare_digest(self.headers.get("X-Manual-Token", ""), manual.token):
                self._json({"ok": False, "error": "Missing or wrong page token; open the URL printed by manual.py"}, 401)
                return False
            return True

        def _run(self, fn, exclusive=False):
            if exclusive and not manual.busy.acquire(blocking=False):
                self._json({"ok": False, "error": "A robot request is already in flight; nothing sent", "busy": True}, 429)
                return
            try:
                reply = ({"ok": True, **fn()}, 200)
            except ProxyError as e:
                reply = ({"ok": False, "error": e.message, "detail": e.detail}, e.status)
            except Exception as e:  # noqa: BLE001 - shown to the operator, never contains the key
                reply = ({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            finally:
                if exclusive:
                    manual.busy.release()
            self._json(*reply)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/favicon.ico":
                self._headers(204, "image/x-icon", 0)
                return
            if parsed.path in ASSETS or parsed.path in vendor:
                if not self._guard(False):
                    return
                if parsed.path in ASSETS:
                    name, ctype = ASSETS[parsed.path]
                    data = (STATIC/name).read_bytes()
                else:
                    data, ctype = vendor[parsed.path].read_bytes(), "text/javascript; charset=utf-8"
                self._bytes(data, ctype)
                return
            if not self._guard(True):
                return
            query = urllib.parse.parse_qs(parsed.query)
            if parsed.path == "/api/session":
                self._run(lambda: {"session": manual.info or manual.connect(), "health": manual.health(),
                                   "last_observation": manual.last_observation, "busy": manual.busy.locked(),
                                   "viewer": {"available": manual.owner and bool(vendor) and manual.kin is not None,
                                              "reason": None if manual.owner and vendor and manual.kin else
                                              ("owner key only" if not manual.owner else VENDOR_ERROR or manual.kin_error)}})
            elif parsed.path == "/api/health":
                self._run(lambda: {"health": manual.health(), "busy": manual.busy.locked()})
            elif parsed.path == "/api/owner":
                self._run(lambda: manual.owner_state(pose=query.get("pose", ["0"])[0] == "1"))
            elif parsed.path == "/api/scene":
                self._run(manual.scene)
            elif parsed.path == "/api/checkpoints":
                self._run(manual.checkpoints)
            elif parsed.path == "/api/render":
                if not manual.owner:
                    self._json({"ok": False, "error": "The diagnostic render needs the owner key"}, 403)
                    return
                if not manual.busy.acquire(blocking=False):
                    self._json({"ok": False, "error": "A robot request is already in flight; nothing sent", "busy": True}, 429)
                    return
                try:
                    data, ctype, wall = manual.render()
                except ProxyError as e:
                    self._json({"ok": False, "error": e.message}, e.status)
                    return
                finally:
                    manual.busy.release()
                self._bytes(data, ctype or "image/png", {"X-Wall-Seconds": f"{wall:.3f}"})
            elif parsed.path.startswith("/api/camera/"):
                name = urllib.parse.unquote(parsed.path[len("/api/camera/"):])
                oid = query.get("observation_id", [None])[0]
                if not name.replace("_", "").isalnum() or (oid is not None and not oid.isdigit()):
                    self._json({"ok": False, "error": "bad camera request"}, 400)
                    return
                try:
                    q = "" if oid is None else f"?observation_id={oid}"
                    data, ctype, _ = manual.upstream("GET", f"/v1/cameras/{urllib.parse.quote(name)}{q}", raw=True)
                except ProxyError as e:
                    self._json({"ok": False, "error": e.message}, e.status)
                    return
                self._bytes(data, ctype or "image/png")
            else:
                self._json({"ok": False, "error": "unknown route"}, 404)

        def do_POST(self):
            parsed = urllib.parse.urlparse(self.path)
            if not self._guard(True):
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 <= length <= 100_000:
                    raise ValueError
                body = json.loads(self.rfile.read(length).decode() or "{}") if length else {}
                if not isinstance(body, dict):
                    raise ValueError
            except ValueError:
                self._json({"ok": False, "error": "body must be a JSON object"}, 400)
                return
            if manual.info is None:
                self._json({"ok": False, "error": "not connected"}, 409)
            elif parsed.path == "/api/command":
                self._run(lambda: manual.command(body), exclusive=True)
            elif parsed.path == "/api/observe":
                self._run(manual.observe, exclusive=True)
            elif parsed.path == "/api/checkpoint/save":
                self._run(lambda: manual.save_checkpoint(body), exclusive=True)
            elif parsed.path == "/api/checkpoint/restore":
                self._run(lambda: manual.restore_checkpoint(body), exclusive=True)
            elif parsed.path == "/api/recover":
                self._run(manual.recover, exclusive=True)
            else:
                self._json({"ok": False, "error": "unknown route"}, 404)
    return Handler


VENDOR_ERROR = None


def _interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    global VENDOR_ERROR
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="GAR session id (gar sessions); URL and key are resolved through `gar` in memory")
    ap.add_argument("--role", choices=("policy", "owner"), default="policy",
                    help="Which of the session's keys to use (default policy; owner adds the privileged panels)")
    ap.add_argument("--url", help="GAR session URL (http://127.0.0.1:PORT) when not using --session")
    ap.add_argument("--key-file", help="File with the bare key, or gar's credential JSON (picked by --role)")
    ap.add_argument("--key-env", default="GAR_MANUAL_KEY", help="Environment variable holding the key")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="Local page port on 127.0.0.1 (0 = any free port)")
    args = ap.parse_args(argv)
    # a background launch inherits SIGINT ignored; make both Ctrl-C and SIGTERM stop the server cleanly
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _interrupt)
    url, key = resolve_credentials(args)
    manual = Manual(url, key, secrets.token_urlsafe(24))
    manual.connect()
    vendor, VENDOR_ERROR = _vendor()
    host = "127.0.0.1"
    server = ThreadingHTTPServer((host, args.port), None)
    port = server.server_port
    hosts = {f"{host}:{port}", f"localhost:{port}"}
    server.RequestHandlerClass = make_handler(manual, hosts, vendor)
    server.daemon_threads = True
    print(json.dumps({"url": f"http://{host}:{port}/#token={manual.token}", "role": manual.info["role"],
                      "session_url": url, "pid": os.getpid(), "robot": manual.info["robot"],
                      "tcp_available": manual.info["tcp_available"],
                      "viewer_3d": manual.owner and bool(vendor) and manual.kin is not None}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
