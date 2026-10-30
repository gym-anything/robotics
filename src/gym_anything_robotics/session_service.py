"""Owner lifecycle layered on the existing role-safe serialized HTTP service."""

import logging
import threading
from pathlib import Path
from .errors import ActionRejectedError
from .http_server import GARHTTPService, GARHTTPServer, GARHTTPHandler


class SessionService(GARHTTPService):
    def __init__(self, env, *, session_id, history=None):
        super().__init__(env, history=history)
        self.session_id, self.paused, self.stopping = session_id, False, False

    def session(self):
        with self._lock:
            return {"schema": "gar.session-state.v1", "session_id": self.session_id,
                    "paused": self.paused, "stopping": self.stopping,
                    "clock": "command-driven", "health": self.health()}

    def _mutation(self, method, *args, **kwargs):
        with self._lock:
            if self.paused or self.stopping:
                error = ActionRejectedError("Session is paused or stopping; no command executed")
                # The gateway shows a restricted policy only a rejection's `policy_reason` (http_server._error). This
                # one says nothing of the simulator's state: without it a policy cannot tell a paused session from a
                # bad command.
                error.policy_reason = ("The session is paused or stopping; no command was executed. A paused session "
                                       "takes commands again after its owner resumes it.")
                raise error
            return method(*args, **kwargs)

    def action(self, *args, **kwargs):
        return self._mutation(super().action, *args, **kwargs)

    def reset(self, *args, **kwargs):
        return self._mutation(super().reset, *args, **kwargs)

    def restore_checkpoint(self, *args, **kwargs):
        return self._mutation(super().restore_checkpoint, *args, **kwargs)

    def start_variation(self, *args, **kwargs):
        return self._mutation(super().start_variation, *args, **kwargs)

    def help(self, principal=None):
        if principal is None:
            from .access import INTERNAL
            principal = INTERNAL
        result = super().help(principal)
        if not principal.restricted:
            result["routes"].update({
                "GET /v1/session": "Owner identity, clock and command admission state.",
                "POST /v1/session/pause": "Wait for current action; reject subsequent mutations.",
                "POST /v1/session/resume": "Allow commands; does not advance time or reset.",
                "POST /v1/session/stop": "Explicitly end this owned environment process.",
            })
        return result


class SessionHandler(GARHTTPHandler):
    def do_GET(self):
        assets = {"/dashboard": ("index.html", "text/html; charset=utf-8"),
                  "/dashboard/app.js": ("app.js", "text/javascript; charset=utf-8"),
                  "/dashboard/style.css": ("style.css", "text/css; charset=utf-8")}
        if self.path in assets:
            name, mime = assets[self.path]
            content = (Path(__file__).parent/"session_static"/name).read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(content)
            return
        if self.path != "/v1/session":
            return super().do_GET()
        if self._authorized() and self._route_allowed("GET", "/v1/session"):
            self._send_json(self.server.service.session())

    def do_POST(self):
        if not self.path.startswith("/v1/session/"):
            return super().do_POST()
        if not self._authorized() or not self._route_allowed("POST", self.path):
            return
        if self.path not in {"/v1/session/pause", "/v1/session/resume", "/v1/session/stop"}:
            self._error(404, "Unknown lifecycle route")
            return
        try:
            if self._body():
                raise ValueError("Lifecycle operations require an empty object")
            service = self.server.service
            with service._lock:
                if self.path.endswith("/resume") and service.stopping:
                    raise ValueError("Session is stopping")
                service.paused = not self.path.endswith("/resume")
                if self.path.endswith("/stop"):
                    service.stopping = True
                else:
                    try:
                        service.env.session_paused(service.paused)
                    except Exception as error:  # a resource hint never blocks the lifecycle change
                        logging.getLogger(__name__).warning("session_paused hint failed: %s", error)
                self._send_json(service.session())
            if service.stopping:
                threading.Thread(target=self.server.shutdown, daemon=True).start()
        except (TypeError, ValueError) as error:
            self._error(400, error)


class SessionServer(GARHTTPServer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.RequestHandlerClass = SessionHandler
