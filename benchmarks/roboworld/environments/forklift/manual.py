"""Human control of a live forklift GAR session through a small local web page.

The page is served by this process, which attaches to an EXISTING `gar run forklift ...` session and proxies a few of
its HTTP API calls. The GAR key stays in this process's memory only: it is never sent to the browser, printed, logged
or written to disk. The browser authenticates to this proxy with a random page token carried in the URL fragment
(fragments are not sent to servers).

Keys command only the robot, through the policy interface's own action (`named_joint_target`, Unitree LowCmd
channels): a key nudges ONE motor's position target from its measured position, then physics runs for the chosen
duration. Nothing in the world is moved by a key directly; the truck moves only when the H1-2's hands and feet move
its controls by contact.

  python -m forklift.manual --session SESSION_ID                 # policy key: the head camera and the joints
  python -m forklift.manual --session SESSION_ID --role owner    # owner key: adds the evaluator

It prints one JSON line with the page URL (http://127.0.0.1:PORT/#token=...). See forklift/MANUAL.md.
"""

import argparse
import hmac
import json
import math
import os
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
from pathlib import Path

STATIC = Path(__file__).parent/"manual_static"
ASSETS = {"/": ("index.html", "text/html; charset=utf-8"), "/index.html": ("index.html", "text/html; charset=utf-8"),
          "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/styles.css": ("styles.css", "text/css; charset=utf-8")}
UPSTREAM_TIMEOUT_S = 1800               # the backend asks HTTP clients for >= 300 s; a 10 s command can take minutes
MAX_TICKS = 2000*10                     # 10 s of simulated time per page command (0.5 ms ticks; the backend allows 30 s)
MIN_TICKS = 4                           # 2 ms: the backend's shortest command (read from the contract when published)
MAX_DELTA = 1.0                         # rad per nudge
GROUPS = ("left arm", "right arm", "left leg", "right leg", "torso", "left hand fingers", "right hand fingers")


class ProxyError(Exception):
    def __init__(self, status, message, detail=None):
        super().__init__(message)
        self.status, self.message, self.detail = status, message, detail


def body_group(name):
    """The page's motor group from the H1-2 joint name (h1_2/left_knee_joint, h1_2/L_index_proximal_joint, ...)."""
    s = name.split("/")[-1]
    if s.startswith("L_"):
        return "left hand fingers"
    if s.startswith("R_"):
        return "right hand fingers"
    if s.startswith("torso"):
        return "torso"
    for side in ("left", "right"):
        if s.startswith(side + "_"):
            return f"{side} leg" if any(k in s for k in ("hip", "knee", "ankle")) else f"{side} arm"
    return "other"


def _gar_json(*args):
    r = subprocess.run(["gar", *args, "--json"], capture_output=True, text=True, timeout=120)
    try:
        return json.loads(r.stdout)
    except ValueError:
        raise SystemExit(f"`gar {args[0]}` failed (exit {r.returncode}); is the session running?") from None


def resolve_credentials(args):
    """(url, key, contract_key). Keys are returned in memory only. contract_key is the session's policy key, needed
    once with --role owner to read the motor table (the owner contract does not publish it), then dropped."""
    url, key, contract_key = args.url, None, None
    want = "privileged" if args.role == "owner" else "policy"
    if args.session:
        status = _gar_json("status", args.session)
        url = url or status.get("url")
        creds = _gar_json("keys", args.session)
        key, contract_key = creds.get(want), creds.get("policy")
        if not key:
            raise SystemExit(f"session has no {args.role} key")
    elif args.key_file:
        text = Path(args.key_file).read_text().strip()
        if text.startswith("{"):
            creds = json.loads(text)
            key, contract_key = creds.get(want), creds.get("policy")
        else:
            key = text
    else:
        key = os.environ.get(args.key_env)
    if not url or not key:
        raise SystemExit("Give --session SESSION_ID, or --url plus a key (--key-file or $" + args.key_env + ")")
    return url.rstrip("/"), key, contract_key


class Manual:
    def __init__(self, url, key, token, contract_key=None):
        self.url, self._key, self.token = url, key, token
        self._contract_key = contract_key
        self.busy = threading.Lock()
        self.info = None
        self.joints = {}                 # name -> dict(lower, upper, unit, group) from the policy contract
        self.min_ticks = MIN_TICKS
        self.last_positions = None
        self.last_joints = None
        self.last_observation = None

    def upstream(self, method, path, body=None, raw=False, idempotent=False, key=None):
        data = None if body is None else json.dumps(body).encode()
        headers = {"Authorization": "Bearer " + (key or self._key)}
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
        hw = caps.get("hardware")
        if hw is None and self._contract_key:
            body, _ = self.upstream("GET", "/v1/capabilities", key=self._contract_key)
            hw = (body.get("capabilities") or {}).get("hardware")
        self._contract_key = None        # used once; only the role's own key stays in memory
        if not hw or not hw.get("motors"):
            raise SystemExit("The session did not publish its motor table. The owner contract does not include it; "
                             "with --role owner this page reads it with the session's policy key, so launch the "
                             "session with --policy-key (or give --key-file with gar's credential JSON).")
        task, _ = self.upstream("GET", "/v1/task")
        self.joints = {n: dict(lower=j.get("lower"), upper=j.get("upper"), unit=j.get("unit", "radian"),
                               group=body_group(n), velocity_limit=j.get("velocity_limit_rad_s"),
                               effort_limit=j.get("effort_limit_nm"))
                       for n, j in hw["motors"].items()}
        self.min_ticks = max(MIN_TICKS, int(hw.get("min_command_ticks") or MIN_TICKS))
        order = list(GROUPS) + sorted({j["group"] for j in self.joints.values()} - set(GROUPS))
        groups = [dict(name=g, motors=[n for n in self.joints if self.joints[n]["group"] == g]) for g in order]
        groups = [g for g in groups if g["motors"]]
        torso = next((n for n in self.joints if n.endswith("torso_joint")), None)
        self.info = {"role": "owner" if owner else "policy", "robot": caps.get("robot"),
                     "cameras": list(caps.get("robot_cameras") or []), "instruction": task.get("instruction"),
                     "groups": groups, "motors": {n: {k: j[k] for k in ("lower", "upper", "unit", "group")}
                                                  for n, j in self.joints.items()},
                     "torso": torso, "dt_s": float(hw.get("dt_s") or 0.0005), "min_ticks": self.min_ticks,
                     "max_ticks": MAX_TICKS, "max_delta": MAX_DELTA}
        return self.info

    def _summarize(self, obs, wall_s):
        pro = obs.get("proprioception") or {}
        joints = pro.get("joints") or {}
        if joints:
            self.last_positions = {n: v.get("position") for n, v in joints.items()}
            self.last_joints = {n: [v.get("position"), v.get("velocity"), v.get("torque")] for n, v in joints.items()}
        status = pro.get("robot_status") or {}
        out = {"observation_id": obs.get("observation_id"), "sim_time_s": obs.get("sim_time_s"),
               "sequence": obs.get("sequence"), "cameras": sorted((obs.get("images") or {}).keys()),
               "wall_s": round(wall_s, 3), "joints": self.last_joints, "imu": status.get("imu"),
               "camera_frame": status.get("camera_frame")}
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
        out = {"tick": s.get("tick"), "sim_time_s": s.get("time_s"), "finite": s.get("finite"), "fault": s.get("fault"),
               "start_problems": (s.get("start_check") or {}).get("problems")}
        try:
            ev, _ = self.upstream("GET", "/v1/evaluation")
            ev = ev.get("evaluation") or {}
            det = ev.get("details") or {}
            out["evaluation"] = {"available": ev.get("available"), "success": ev.get("success"), "score": ev.get("score"),
                                 "task": det.get("task"),
                                 "criteria": [{k: c.get(k) for k in ("name", "measured", "passed", "unit", "threshold")}
                                              for c in det.get("criteria") or []],
                                 "strikes": (det.get("details") or {}).get("strikes"),
                                 "dropped": (det.get("details") or {}).get("dropped")}
        except ProxyError as e:
            out["evaluation"] = {"available": False, "error": e.message}
        return out

    def _measured(self, name):
        if not self.last_positions or self.last_positions.get(name) is None:
            raise ProxyError(409, "no measured position yet: take a camera image first")
        return float(self.last_positions[name])

    def command(self, req):
        """{"motor": name, "delta": rad} or {"hand": "left"|"right", "grip": "close"|"open"} or neither (hold), plus
        {"ticks": n}. A nudge's target is the motor's measured position + delta; a target outside the joint's limits
        is refused before anything is sent. Every other channel keeps its last value."""
        ticks = req.get("ticks")
        if isinstance(ticks, bool) or not isinstance(ticks, int) or not self.min_ticks <= ticks <= MAX_TICKS:
            raise ProxyError(400, f"ticks must be an integer in [{self.min_ticks}, {MAX_TICKS}]")
        if req.get("motor") is not None and req.get("hand") is not None:
            raise ProxyError(400, "give a motor nudge or a hand command, not both")
        targets = {}
        if req.get("motor") is not None:
            name, delta = req["motor"], req.get("delta", 0.)
            if name not in self.joints:
                raise ProxyError(400, f"unknown motor {name!r}")
            if isinstance(delta, bool) or not isinstance(delta, (int, float)) or not math.isfinite(delta) or \
                    abs(delta) > MAX_DELTA:
                raise ProxyError(400, f"delta must be a finite number within +-{MAX_DELTA:g} rad")
            p = self._measured(name)
            q = p + float(delta)
            lo, hi = self.joints[name]["lower"], self.joints[name]["upper"]
            if lo is not None and hi is not None and not lo <= q <= hi:
                raise ProxyError(400, f"{name}: target {q:.4f} rad is outside its limits [{lo:.4f}, {hi:.4f}] "
                                      f"(measured {p:.4f}); nothing was sent")
            targets[name] = q
        elif req.get("hand") is not None:
            side, grip = req.get("hand"), req.get("grip")
            if side not in ("left", "right") or grip not in ("close", "open"):
                raise ProxyError(400, "hand must be 'left' or 'right' and grip 'close' or 'open'")
            # forklift/scripted.py fingers(): closure 0 = each finger's lower limit (open), 1 = upper (closed); the
            # thumb's rotation (thumb_proximal_yaw) is not a closure and keeps its last target
            for name, j in self.joints.items():
                if j["group"] == f"{side} hand fingers" and not name.endswith("thumb_proximal_yaw_joint"):
                    targets[name] = float(j["upper"] if grip == "close" else j["lower"])
            if not targets:
                raise ProxyError(400, f"the {side} hand has no finger motors")
        body, wall = self.upstream("POST", "/v1/action", {"type": "named_joint_target", "targets_si": targets, "ticks": ticks},
                                   idempotent=True)
        sim_s = ticks*self.info["dt_s"]
        result = {"committed": body.get("committed"), "request_id": body.get("request_id"), "wall_s": round(wall, 3),
                  "targets": targets, "sim_s": sim_s, "wall_per_sim_s": round(wall/sim_s, 2),
                  "observation_error": body.get("observation_error")}
        info = body.get("info") or {}
        if "wall_per_sim" in info:                       # owner responses only: physics alone, without the render
            result["physics_wall_per_sim_s"] = round(float(info["wall_per_sim"]), 2)
            result["executed_ticks"] = info.get("executed_ticks")
        if body.get("observation"):
            result["observation"] = self._summarize(body["observation"], wall)
        return result


def make_handler(manual, allowed_hosts):
    class Handler(BaseHTTPRequestHandler):
        server_version = "forklift-manual"

        def log_message(self, fmt, *args):
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
                    manual.busy.release()
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
            elif parsed.path == "/api/command":
                self._run(lambda: manual.command(body), exclusive=True)
            elif parsed.path == "/api/observe":
                self._run(manual.observe, exclusive=True)
            else:
                self._json({"ok": False, "error": "unknown route"}, 404)
    return Handler


def _interrupt(*_):
    raise KeyboardInterrupt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session")
    ap.add_argument("--role", choices=("policy", "owner"), default="policy")
    ap.add_argument("--url")
    ap.add_argument("--key-file")
    ap.add_argument("--key-env", default="GAR_MANUAL_KEY")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8796, help="local page port (0 = any free port)")
    args = ap.parse_args(argv)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, _interrupt)
    url, key, contract_key = resolve_credentials(args)
    manual = Manual(url, key, secrets.token_urlsafe(24), contract_key)
    del contract_key
    manual.connect()
    server = ThreadingHTTPServer((args.host, args.port), None)
    port = server.server_port
    hosts = {f"{args.host}:{port}", f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}
    server.RequestHandlerClass = make_handler(manual, hosts)
    server.daemon_threads = True
    print(json.dumps({"url": f"http://127.0.0.1:{port}/#token={manual.token}", "role": manual.info["role"],
                      "session_url": url, "pid": os.getpid(), "motors": len(manual.joints)}), flush=True)
    try:
        server.serve_forever(poll_interval=.2)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
