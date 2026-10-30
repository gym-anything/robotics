"""Human keyboard control of a live barcode-scanning GAR session through a small local web page.

The page is served by this process, which attaches to an EXISTING `gar run barcode-scanning ...` session and
proxies a minimal set of its HTTP API calls. The GAR key stays in this process's memory only: it is never sent to
the browser, printed, logged or written to disk. The browser authenticates to this proxy with a random page token
carried in the URL fragment (fragments are not sent to servers, so the token does not reach any request log).

Keys only command the robot (the `keyboard` action of the backend: a held chord -> one Cartesian/finger target, then
physics for the chord's duration). Nothing in the world is moved by a key directly; objects move only through contact
with the robot.

  python -m barcode_scanning.manual --session SESSION_ID --role policy      # key + URL resolved via `gar` in memory
  python -m barcode_scanning.manual --url http://127.0.0.1:PORT --key-file F  # F: bare key, or gar's credential JSON
  GAR_MANUAL_KEY=... python -m barcode_scanning.manual --url http://127.0.0.1:PORT

It prints one JSON line with the page URL (http://127.0.0.1:PORT/#token=...). See barcode_scanning/MANUAL.md.
"""

import argparse
import hmac
import json
import math
import os
from pathlib import Path
import secrets
import signal
import subprocess
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

STATIC = Path(__file__).parent/"manual_static"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/styles.css": ("styles.css", "text/css; charset=utf-8")}
UPSTREAM_TIMEOUT_S = 1800
# Chord validation mirrors the backend's own rejections (newton_barcode_scanning._keyboard). The page sends queued chords
# as ONE multi-action request; if the backend refuses a chord after earlier chords of that request already ran, the
# GAR HTTP service latches a "partially executed" runtime fault that only the OWNER can clear (measured).
# This proxy therefore refuses such chords before anything is sent.
MOTION = set("ADWSRFIKJLUO")
KNOWN = MOTION | {"Q", "E", "H", "G", "B", "P", "T", "ALT", "SHIFT"}
CONFLICTS = (("A", "D"), ("W", "S"), ("R", "F"), ("I", "K"), ("J", "L"), ("U", "O"), ("Q", "E"), ("G", "B"),
             ("P", "T"), ("ALT", "SHIFT"), ("H", "E"))
MAX_CHORD_S = 10.0          # backend limit per chord
MAX_BATCH_S = 10.0          # this GUI's limit per request (sum of chords)
PRIVILEGED_KEYS = ("tick", "time_s", "generation", "fault", "ready", "scanner", "keys_pressed", "host", "reflexes",
                   "refused_targets", "evaluation")


class ProxyError(Exception):
    def __init__(self, status, message, detail=None):
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail


# ---------------------------------------------------------------------------------------------- credentials
def _gar_json(*args):
    """`gar ... --json` parsed in memory. `gar status` exits non-zero for a live but faulted session (its JSON still
    names the URL), so a parseable reply is accepted; the page then shows the fault."""
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
            raise SystemExit(f"session has no {args.role} key")
    elif args.key_file:
        text = Path(args.key_file).read_text().strip()
        if text.startswith("{"):
            data = json.loads(text)
            key = data.get("privileged" if args.role == "owner" else "policy")
        else:
            key = text
    else:
        key = os.environ.get(args.key_env)
    if not url or not key:
        raise SystemExit("Give --session SESSION_ID, or --url plus a key (--key-file or $" + args.key_env + ")")
    return url.rstrip("/"), key


# ---------------------------------------------------------------------------------------------- kinematics
class Kinematics:
    """TCP forward kinematics (and the backend's own IK solve, for refusing unreachable chords before sending them)
    from the robot deployment's URDF and mounts: the same chain barcode_scanning.kinematics.ArmKinematics builds (it
    asserts that chain matches the simulated TCP within 1 mm). Inputs are policy-visible joint positions only."""

    def __init__(self, robot):
        from scipy.spatial.transform import Rotation
        from robots.catalog import for_environment, model
        from robots import assets
        from robots.controllers.kinematics import SerialKinematics
        from barcode_scanning.scene import ARM_MOUNTS
        from barcode_scanning.kinematics import HAND_TCP_OFFSET
        from barcode_scanning import params as P
        self.Rotation = Rotation
        setup = for_environment("barcode-scanning", robot).to_dict()
        self.arms = {}
        for name, inst in setup["instances"].items():
            spec = model(inst["model"])
            profile = spec.to_dict()["actuation"]["profiles"][inst["controller_profile"]]
            if profile.get("default_tool") == "wsg50":
                from robots.attachments.wsg50.geometry import generated_urdf
                payload, _ = generated_urdf(spec.id)
            else:
                path = assets.resolve(spec.id)
                if spec.id == "fr3":
                    path = path.with_name("fr3_franka_hand.urdf")
                payload = path.read_bytes()
            (x, y), yaw = ARM_MOUNTS[inst["mount"]]
            base = np.eye(4)
            base[:3, :3] = Rotation.from_euler("z", yaw).as_matrix()
            base[:3, 3] = (x, y, P.CART_TOP_M)
            tcp = f"{spec.id}_hand_tcp"
            text = payload if isinstance(payload, str) else payload.decode()
            chain = (SerialKinematics(payload, tcp, base=base) if f'name="{tcp}"' in text
                     else SerialKinematics(payload, f"{spec.id}_hand", base=base, tip_offset=(0, 0, HAND_TCP_OFFSET)))
            self.arms[name] = (chain, [f"{name}/{j}" for j in chain.names])

    def q(self, arm, joints):
        chain, names = self.arms[arm]
        return np.array([joints[n]["position"] for n in names], float)

    def tcp(self, arm, joints):
        chain, _ = self.arms[arm]
        T = chain.fk(self.q(arm, joints))
        return T

    def summary(self, joints):
        out = {}
        for arm in self.arms:
            T = self.tcp(arm, joints)
            fingers = [v["position"] for n, v in joints.items() if n.startswith(arm + "/") and "finger" in n]
            out[arm] = {"position_m": T[:3, 3].tolist(),
                        "rpy_deg": self.Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True).tolist(),
                        "rotvec": self.Rotation.from_matrix(T[:3, :3]).as_rotvec().tolist(),
                        "finger_gap_m": float(sum(fingers)) if fingers else None}
        return out

    def check_chords(self, chords, joints, speeds):
        """Replay each chord's target exactly as the backend computes it (FK now + key direction x speed x duration,
        then the same IK solve and tolerances), assuming each earlier chord reaches its target. Returns the index and
        reason of the first chord the robot's IK would refuse, or None."""
        from gym_anything_robotics.keyboard_control import KEY_TRANSLATION, KEY_ROTATION
        R = self.Rotation
        qs = {arm: self.q(arm, joints) for arm in self.arms}
        for i, c in enumerate(chords):
            keys, arm = set(c["keys"]), c["arm"]
            chain, _ = self.arms[arm]

            def direction(table):
                v = np.sum([table[k] for k in keys if k in table] or [np.zeros(3)], axis=0).astype(float)
                n = np.linalg.norm(v)
                return v/n if n > 1e-12 else np.zeros(3)
            move, turn = direction(KEY_TRANSLATION), direction(KEY_ROTATION)
            if not (move.any() or turn.any()):
                continue
            duration = round(c["ticks"]/speeds["rate_hz"]*1000)/1000
            mult = .2 if "ALT" in keys else (6. if "SHIFT" in keys else 1.)
            T = chain.fk(qs[arm])
            position = T[:3, 3] + move*speeds["linear_speed_m_s"]*mult*duration
            rot = R.from_rotvec(turn*speeds["angular_speed_rad_s"]*mult*duration)*R.from_matrix(T[:3, :3])
            seed = np.clip(qs[arm], chain.lower + 1e-6, chain.upper - 1e-6)
            r = chain.solve(position, rot.as_quat(), seed, max_evaluations=300)
            if r["position_error_m"] > 1e-3 or r["orientation_error_rad"] > .0175:
                return i, (f"IK cannot reach this chord's target ({r['position_error_m']*1e3:.1f} mm, "
                           f"{math.degrees(r['orientation_error_rad']):.1f} deg off)")
            qs[arm] = np.asarray(r["joints"], float)
        return None


# ---------------------------------------------------------------------------------------------- proxy
class Manual:
    def __init__(self, url, key, token):
        self.url, self._key, self.token = url, key, token
        self.busy = threading.Lock()
        self.info = None
        self.kin = None
        self.kin_error = None
        self.last_joints = None
        self.last_observation = None

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
                message = json.loads(text).get("error") or text
            except (ValueError, AttributeError):
                message = text or e.reason
            raise ProxyError(e.code, f"GAR {method} {path.split('?')[0]} -> HTTP {e.code}: {message}")
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
        hardware = caps.get("hardware") or {}
        arms = hardware.get("arms")
        notes = None
        if owner:
            full, _ = self.upstream("GET", "/v1/capabilities")
            notes = (full.get("capabilities") or {}).get("notes")
            caps = full.get("capabilities") or caps
        if not arms:
            arms = (notes or {}).get("arms", "left,right").split(",")
        robot = caps.get("robot", "dual-fr3")
        try:
            self.kin = Kinematics(robot)
        except Exception as e:  # noqa: BLE001 - the page states TCP is unavailable and why
            self.kin, self.kin_error = None, f"{type(e).__name__}: {e}"
        self.info = {"role": "owner" if owner else "policy", "robot": robot, "arms": list(arms),
                     "cameras": list(caps.get("robot_cameras") or []), "instruction": task.get("instruction"),
                     "action_types": sorted(caps.get("action_types") or []),
                     "keyboard_note": (notes or {}).get("keyboard"), "notes": notes,
                     "tcp_available": self.kin is not None, "tcp_error": self.kin_error,
                     "dt_s": hardware.get("dt_s") or (.001 if "1 ms" in (notes or {}).get("clock", "") else None)}
        return self.info

    # -- observations
    def _summarize(self, obs, wall_s):
        joints = (obs.get("proprioception") or {}).get("joints") or {}
        if joints:
            self.last_joints = joints
        out = {"observation_id": obs.get("observation_id"), "sim_time_s": obs.get("sim_time_s"),
               "sequence": obs.get("sequence"), "cameras": sorted((obs.get("images") or {}).keys()),
               "wall_s": round(wall_s, 3), "joints": {n: v.get("position") for n, v in joints.items()}}
        if self.kin and joints:
            out["tcp"] = self.kin.summary(joints)
        self.last_observation = out
        return out

    def observe(self):
        body, wall = self.upstream("GET", "/v1/observation")
        return {"observation": self._summarize(body["observation"], wall)}

    def owner_state(self):
        if not self.owner:
            raise ProxyError(403, "Privileged state needs the owner key")
        body, _ = self.upstream("GET", "/v1/state")
        s = body["state"]
        out = {k: s.get(k) for k in PRIVILEGED_KEYS}
        if self.kin:
            out["tcp"] = self.kin.summary(s["joints"])
        session, _ = self.upstream("GET", "/v1/session")
        out["session"] = {k: session.get(k) for k in ("session_id", "paused", "stopping", "clock")}
        return out

    def recover(self):
        if not self.owner:
            raise ProxyError(403, "Only the owner key can recover a faulted session")
        body, wall = self.upstream("POST", "/v1/recover", {"acknowledge_uncertain_actions": True})
        result = {"recovered": body.get("recovered"), "previous_fault": body.get("previous_fault"),
                  "health": body.get("health"), "wall_s": round(wall, 3)}
        if body.get("observation"):
            result["observation"] = self._summarize(body["observation"], wall)
        return result

    # -- keyboard
    def validate(self, request):
        arms = self.info["arms"]
        chords = request.get("chords")
        if not isinstance(chords, list) or not chords:
            raise ProxyError(400, "No chords to send")
        speeds = {}
        for name, lo, hi, default in (("rate_hz", 1., 1000., 30.), ("linear_speed_m_s", 1e-4, .6, .005),
                                      ("angular_speed_rad_s", 1e-4, 1.2, .04), ("gripper_half_speed_m_s", 1e-4, .2, .005)):
            v = request.get(name, default)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or not lo <= v <= hi:
                raise ProxyError(400, f"{name} must be within [{lo}, {hi}]")
            speeds[name] = float(v)
        total, clean = 0., []
        for c in chords:
            keys = [str(k).upper() for k in c.get("keys", [])]
            arm, ticks = c.get("arm"), c.get("ticks")
            if arm not in arms:
                raise ProxyError(400, f"arm must be one of {arms}")
            if not keys or len(set(keys)) != len(keys) or set(keys) - KNOWN:
                raise ProxyError(400, f"Unsupported or duplicate keys {keys}")
            for a, b in CONFLICTS:
                if {a, b} <= set(keys):
                    raise ProxyError(400, f"Conflicting keys {a} and {b}; nothing sent")
            if set(keys) & {"G", "B"} and "H" not in keys:
                raise ProxyError(400, "G/B choose the H force-grip level and need H")
            if isinstance(ticks, bool) or not isinstance(ticks, int) or ticks < 1:
                raise ProxyError(400, "ticks must be a positive integer")
            duration = ticks/speeds["rate_hz"]
            if round(duration*1000) < 1 or duration > MAX_CHORD_S:
                raise ProxyError(400, f"A chord lasts 1 ms .. {MAX_CHORD_S} s")
            total += duration
            clean.append({"keys": keys, "arm": arm, "ticks": ticks})
        if total > MAX_BATCH_S + 1e-9:
            raise ProxyError(400, f"At most {MAX_BATCH_S} s of chords per request")
        return clean, speeds

    def keyboard(self, request):
        chords, speeds = self.validate(request)
        # the owner reads the exact current joints; the policy key has its latest observation's proprioception
        before = self.upstream("GET", "/v1/state")[0]["state"] if self.owner else None
        joints = before["joints"] if before else self.last_joints
        if self.kin and joints:
            refused = self.kin.check_chords(chords, joints, speeds)
            if refused is not None:
                i, why = refused
                raise ProxyError(409, f"Nothing sent: chord {i + 1} of {len(chords)} ({'+'.join(chords[i]['keys'])}, "
                                      f"{chords[i]['arm']} arm) would be refused by the robot: {why}. (A refusal after "
                                      f"earlier chords of the same request ran would latch a session fault.)", {"chord": i})
        actions = [{"type": "keyboard", "keys": c["keys"], "ticks": c["ticks"], "arm": c["arm"], **speeds} for c in chords]
        tcp_before = self.kin.summary(joints) if self.kin and joints else None
        payload = actions[0] if len(actions) == 1 else {"actions": actions}
        try:
            body, wall = self.upstream("POST", "/v1/action", payload, idempotent=True)
        except ProxyError as e:
            e.detail = {"health": self.health()}
            raise
        result = {"committed": body.get("committed"), "request_id": body.get("request_id"), "wall_s": round(wall, 3),
                  "chords": chords, "sim_s": round(sum(c["ticks"]/speeds["rate_hz"] for c in chords), 4),
                  "observation_error": body.get("observation_error")}
        if body.get("observation"):
            result["observation"] = self._summarize(body["observation"], wall)
        if tcp_before and result.get("observation", {}).get("tcp"):
            after = result["observation"]["tcp"]
            result["tcp_delta"] = {
                arm: {"position_mm": [(a - b)*1e3 for a, b in zip(after[arm]["position_m"], tcp_before[arm]["position_m"])],
                      "rotation_deg": np.degrees((self.kin.Rotation.from_rotvec(after[arm]["rotvec"]) *
                                                  self.kin.Rotation.from_rotvec(tcp_before[arm]["rotvec"]).inv()).as_rotvec()).tolist(),
                      "finger_gap_mm": ((after[arm]["finger_gap_m"] or 0) - (tcp_before[arm]["finger_gap_m"] or 0))*1e3}
                for arm in after}
        if self.owner:
            after = self.upstream("GET", "/v1/state")[0]["state"]
            info = body.get("info") or {}
            result["feedback"] = {
                "reflexes": after["reflexes"] - before["reflexes"],
                "reflex_details_last_chord": info.get("reflexes"),
                "refused_targets": after["refused_targets"] - before["refused_targets"],
                "arms_stopped_s": info.get("arms_stopped_s"),
                "scanner_frames": after["scanner"]["frames"] - before["scanner"]["frames"],
                "scanner_reads": after["scanner"]["reads"] - before["scanner"]["reads"],
                "executed_ticks": after["tick"] - before["tick"],
                "source": "owner /v1/state before/after (whole request) + last chord's step info"}
        return result


def make_handler(manual, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        server_version = "barcode-manual"

        def log_message(self, fmt, *args):  # no request log: nothing secret is in URLs, but keep stdout clean
            return

        def _headers(self, status, ctype, length):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; object-src 'none'; "
                                                        "frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()

        def _json(self, value, status=200):
            data = json.dumps(value).encode()
            try:
                self._headers(status, "application/json; charset=utf-8", len(data))
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
                # 429, distinct from a refused chord (409): the page keeps these chords queued and resends them
                self._json({"ok": False, "error": "A robot request is already in flight", "busy": True}, 429)
                return
            try:
                reply = ({"ok": True, **fn()}, 200)
            except ProxyError as e:
                reply = ({"ok": False, "error": e.message, "detail": e.detail}, e.status)
            except Exception as e:  # noqa: BLE001 - shown to the operator, never contains the key
                reply = ({"ok": False, "error": f"{type(e).__name__}: {e}"}, 500)
            finally:
                if exclusive:
                    manual.busy.release()   # before replying, so the page's next request never meets a stale lock
            self._json(*reply)

        def do_GET(self):
            parsed = urllib.parse.urlparse(self.path)
            if parsed.path == "/favicon.ico":
                self._headers(204, "image/x-icon", 0)
                return
            if parsed.path in ASSETS:
                if not self._guard(False):
                    return
                name, ctype = ASSETS[parsed.path]
                data = (STATIC/name).read_bytes()
                self._headers(200, ctype, len(data))
                self.wfile.write(data)
                return
            if not self._guard(True):
                return
            if parsed.path == "/api/session":
                self._run(lambda: {"session": manual.info or manual.connect(), "health": manual.health(),
                                   "last_observation": manual.last_observation, "busy": manual.busy.locked()})
            elif parsed.path == "/api/owner":
                self._run(manual.owner_state)
            elif parsed.path == "/api/health":
                self._run(lambda: {"health": manual.health(), "busy": manual.busy.locked()})
            elif parsed.path.startswith("/api/camera/"):
                name = urllib.parse.unquote(parsed.path[len("/api/camera/"):])
                oid = urllib.parse.parse_qs(parsed.query).get("observation_id", [None])[0]
                if not name.replace("_", "").isalnum() or (oid is not None and not oid.isdigit()):
                    self._json({"ok": False, "error": "bad camera request"}, 400)
                    return
                try:
                    q = "" if oid is None else f"?observation_id={oid}"
                    data, ctype, _ = manual.upstream("GET", f"/v1/cameras/{urllib.parse.quote(name)}{q}", raw=True)
                except ProxyError as e:
                    self._json({"ok": False, "error": e.message}, e.status)
                    return
                self._headers(200, ctype or "image/png", len(data))
                self.wfile.write(data)
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
            elif parsed.path == "/api/keyboard":
                self._run(lambda: manual.keyboard(body), exclusive=True)
            elif parsed.path == "/api/observe":
                self._run(manual.observe, exclusive=True)
            elif parsed.path == "/api/recover":
                self._run(manual.recover, exclusive=True)
            else:
                self._json({"ok": False, "error": "unknown route"}, 404)
    return Handler


def _interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", help="GAR session id (gar sessions); URL and key are resolved through `gar` in memory")
    ap.add_argument("--role", choices=("policy", "owner"), default="policy",
                    help="Which of the session's keys to use (default policy; owner adds privileged panels)")
    ap.add_argument("--url", help="GAR session URL (http://127.0.0.1:PORT) when not using --session")
    ap.add_argument("--key-file", help="File with the bare key, or gar's credential JSON (picked by --role)")
    ap.add_argument("--key-env", default="GAR_MANUAL_KEY", help="Environment variable holding the key")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8794, help="Local page port (0 = any free port)")
    args = ap.parse_args(argv)
    # a background launch inherits SIGINT ignored; make both Ctrl-C and SIGTERM stop the server cleanly
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _interrupt)
    url, key = resolve_credentials(args)
    manual = Manual(url, key, secrets.token_urlsafe(24))
    manual.connect()
    server = ThreadingHTTPServer((args.host, args.port), None)
    port = server.server_port
    hosts = {f"{args.host}:{port}", f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    server.RequestHandlerClass = make_handler(manual, hosts)
    server.daemon_threads = True
    print(json.dumps({"url": f"http://127.0.0.1:{port}/#token={manual.token}", "role": manual.info["role"],
                      "session_url": url, "pid": os.getpid(), "tcp_available": manual.info["tcp_available"]}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
