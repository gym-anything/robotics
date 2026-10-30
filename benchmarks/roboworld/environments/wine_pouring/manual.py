"""Human keyboard control of a live wine-pouring GAR session through a small local web page.

Attaches to an EXISTING `gar run wine-pouring ...` session and proxies three calls of its HTTP API: the `keyboard`
action, the observation and the station camera image. The GAR key stays in this process's memory only (never sent to
the browser, printed, logged or written); the browser authenticates to this proxy with a random token in the URL
fragment. Keys only command the robot through the backend's `keyboard` action (the same action a policy sends);
nothing in the world is moved by a key directly. Physics advances only while a key is held (one 0.5 s chord per
request, 15 ticks at 30 Hz), so a held key moves the hand at the chosen speed in simulated time.

  python -m wine_pouring.manual --session SESSION_ID          # url + policy key resolved via `gar` in memory
  python -m wine_pouring.manual --url http://127.0.0.1:PORT --key-file F
Keys: W/S/A/D/R/F move the selected hand along world +y/-y/-x/+x/+z/-z (5 cm/s), I/K J/L U/O rotate it about world
x, y, z (0.3 rad/s: J/L tilts a bottle held in a side grasp), Q/E close/open the fingers, H grips by force (35 N;
with G 10 N, with B 70 N) until E; hold SHIFT for x6, ALT for x0.2; 1/2 select the left/right arm.
"""
import argparse
import json
import secrets
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from barcode_scanning.manual import resolve_credentials

PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>wine-pouring manual</title>
<style>body{font-family:sans-serif;background:#1d1f22;color:#ddd;margin:16px}img{width:960px;border:1px solid #444}
#st{white-space:pre;font-family:monospace;font-size:13px}kbd{background:#333;padding:1px 5px;border-radius:3px}</style></head>
<body><h3>wine-pouring: keyboard control (policy key)</h3>
<p><kbd>W</kbd><kbd>S</kbd><kbd>A</kbd><kbd>D</kbd><kbd>R</kbd><kbd>F</kbd> move the hand,
<kbd>I</kbd><kbd>K</kbd><kbd>J</kbd><kbd>L</kbd><kbd>U</kbd><kbd>O</kbd> rotate it, <kbd>Q</kbd>/<kbd>E</kbd> fingers,
<kbd>H</kbd> force grip (+<kbd>G</kbd> 10 N, +<kbd>B</kbd> 70 N), <kbd>SHIFT</kbd> fast, <kbd>ALT</kbd> slow,
<kbd>1</kbd>/<kbd>2</kbd> left/right arm. Physics runs only while keys are held.</p>
<p id="task"></p><img id="cam"><div id="st">connecting...</div>
<script>
const token = new URLSearchParams(location.hash.slice(1)).get("token");
const names = {Shift:"SHIFT",Alt:"ALT"};
const held = new Set(); let arm = "left", busy = false;
function key(e){ if (names[e.key]) return names[e.key]; const k = e.key.toUpperCase(); return /^[A-Z]$/.test(k) ? k : null; }
async function call(path, body){ const r = await fetch(path, {method: body ? "POST" : "GET",
  headers: {"X-Token": token, "Content-Type": "application/json"}, body: body ? JSON.stringify(body) : undefined}); return r.json(); }
async function camera(){ const b = await (await fetch("/camera", {headers: {"X-Token": token}})).blob();
  document.getElementById("cam").src = URL.createObjectURL(b); }
async function refresh(info){ const o = await call("/observation"); await camera();
  document.getElementById("task").textContent = o.instruction || "";
  const rs = (o.robot_status || {}); document.getElementById("st").textContent = "arm: " + arm + "   sim t = " + o.sim_time_s +
  " s\\narms stopped (s): " + JSON.stringify(Object.fromEntries(Object.entries(rs.arms || {}).map(([a, v]) => [a, v.stopped_s]))) +
  (info ? "\\nlast: " + JSON.stringify(info) : ""); }
async function loop(){ if (busy) return; const keys = [...held].filter(k => !["1","2"].includes(k));
  if (!keys.length) return; busy = true; try { const r = await call("/keyboard", {keys, arm}); await refresh(r.error || r.info); }
  finally { busy = false; } if (held.size) setTimeout(loop, 0); }
document.addEventListener("keydown", e => { if (e.key === "1") arm = "left"; if (e.key === "2") arm = "right";
  const k = key(e); if (!k) return; e.preventDefault(); if (!held.has(k)) { held.add(k); loop(); } });
document.addEventListener("keyup", e => { const k = key(e); if (k) held.delete(k); });
refresh();
</script></body></html>"""


class Manual:
    def __init__(self, url, key):
        self.url, self.key, self.token = url, key, secrets.token_urlsafe(24)

    def upstream(self, method, path, body=None, raw=False):
        req = urllib.request.Request(self.url + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                data = r.read()
                return data if raw else json.loads(data)
        except urllib.error.HTTPError as e:
            return {"error": e.read().decode(errors="replace")[:400], "status": e.code}

    def keyboard(self, request):
        keys = [str(k).upper() for k in request.get("keys", [])][:6]
        arm = request.get("arm") if request.get("arm") in ("left", "right") else "left"
        body = {"type": "keyboard", "keys": keys, "ticks": 15, "rate_hz": 30.0, "arm": arm, "linear_speed_m_s": 0.05,
                "angular_speed_rad_s": 0.3, "gripper_half_speed_m_s": 0.02}
        r = self.upstream("POST", "/v1/action", body)
        return {"error": r["error"]} if "error" in r else {"info": {"committed": r.get("committed"), "keys": keys, "arm": arm}}

    def observation(self):
        o = self.upstream("GET", "/v1/observation")
        obs = o.get("observation", o)
        task = self.upstream("GET", "/v1/task")
        return {"sim_time_s": obs.get("sim_time_s"), "robot_status": (obs.get("proprioception") or {}).get("robot_status"),
                "instruction": (task.get("instruction") or (task.get("task") or {}).get("instruction"))
                if isinstance(task, dict) else None}


def make_handler(manual):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, status, ctype, data):
            self.send_response(status); self.send_header("Content-Type", ctype); self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store"); self.end_headers(); self.wfile.write(data)

        def _ok(self):
            token = self.headers.get("X-Token") or ""
            if not secrets.compare_digest(str(token), manual.token):
                self._send(403, "application/json", b'{"error":"bad token"}')
                return False
            return True

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, "text/html", PAGE.encode())
            if not self._ok():
                return
            if self.path.startswith("/observation"):
                return self._send(200, "application/json", json.dumps(manual.observation()).encode())
            if self.path.startswith("/camera"):
                return self._send(200, "image/png", manual.upstream("GET", "/v1/cameras/station_rgb", raw=True))
            self._send(404, "application/json", b"{}")

        def do_POST(self):
            if not self._ok():
                return
            n = min(int(self.headers.get("Content-Length", 0)), 4096)
            req = json.loads(self.rfile.read(n) or b"{}")
            if self.path == "/keyboard":
                return self._send(200, "application/json", json.dumps(manual.keyboard(req)).encode())
            self._send(404, "application/json", b"{}")
    return H


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--session"); ap.add_argument("--url"); ap.add_argument("--key-file")
    ap.add_argument("--key-env", default="GAR_MANUAL_KEY"); ap.add_argument("--role", default="policy", choices=("policy", "owner"))
    ap.add_argument("--port", type=int, default=0)
    a = ap.parse_args(argv)
    url, key = resolve_credentials(a)
    m = Manual(url, key)
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), make_handler(m))
    print(json.dumps({"page": f"http://127.0.0.1:{srv.server_port}/#token={m.token}", "session_url": url}), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
