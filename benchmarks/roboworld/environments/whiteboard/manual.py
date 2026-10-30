"""Local operator dashboard over the same serialized API/physics owner."""

import argparse
import base64
import io
import json
from pathlib import Path
import secrets

from PIL import Image
from .api import Service, ApiError, strict_fields, make_server
from .env import WhiteboardEnv


class ManualService(Service):
    def handle(self, method, route, body, role):
        if route.startswith("/v1/manual/") or route in ("/v1/ik/preview", "/v1/cartesian"):
            if role != "teacher":
                raise ApiError(403, "Operator views and simulator-guarded IK require teacher key")
            with self.lock:
                if method == "GET" and route == "/v1/manual/scene":
                    from .viewer_scene import describe

                    return describe(self.env)
                if method == "GET" and route == "/v1/manual/state":
                    from .viewer_scene import state

                    return {**state(self.env), "episode_id": self.episode}
                if method == "GET" and route == "/v1/manual/ink":
                    buf = io.BytesIO()
                    Image.fromarray(self.env.ink.rgb()).save(buf, format="PNG")
                    return {
                        "png": base64.b64encode(buf.getvalue()).decode(),
                        "revision": self.env.ink.revision,
                    }
                if method == "POST" and route == "/v1/ik/preview":
                    from .ik import HandIK

                    strict_fields(
                        body, ("arm", "position_m", "orientation_wxyz"), ("arm", "position_m")
                    )
                    return HandIK(self.env).plan(
                        body["arm"], body["position_m"], body.get("orientation_wxyz")
                    )
                if method == "POST" and route == "/v1/cartesian":
                    return self._mutation(route, body, role)
                raise ApiError(404, "Unknown operator route")
        return super().handle(method, route, body, role)

    def help(self, role):
        result = super().help(role)
        if role == "teacher":
            result["routes"].update(
                {
                    "GET /v1/manual/scene": "Render-only authored geometry",
                    "GET /v1/manual/state": "Pose update and measured TCP",
                    "POST /v1/ik/preview": {"arm": "left", "position_m": [0, 0, 1]},
                    "POST /v1/cartesian": {
                        "id": "unique",
                        "episode_id": self.episode,
                        "action": {
                            "type": "cartesian_delta",
                            "arm": "left",
                            "delta_m": [0, 0, 0.005],
                            "duration_s": 0.02,
                        },
                    },
                }
            )
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8792)
    parser.add_argument("--output", type=Path, default=Path("whiteboard/output/manual"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    keypath = args.output / "credentials.json"
    if keypath.exists():
        keys = json.loads(keypath.read_text())
    else:
        keys = {"teacher_key": secrets.token_urlsafe(32), "policy_key": secrets.token_urlsafe(32)}
        keypath.write_text(json.dumps(keys, indent=2))
    service = ManualService(WhiteboardEnv(), **keys, recording_dir=args.output / "history")
    server = make_server(service, "127.0.0.1", args.port)
    info = {"url": f"http://127.0.0.1:{server.server_port}/", "credentials": str(keypath.resolve())}
    (args.output / "server.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info), flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        service.env.close()
        service.store.close()


if __name__ == "__main__":
    main()
