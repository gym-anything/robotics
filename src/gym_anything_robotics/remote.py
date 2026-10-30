"""Small attach-only HTTP client. No implicit reset, stop, or motion retries."""

import json
import math
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import Request, HTTPRedirectHandler, build_opener
from uuid import uuid4


class RemoteError(RuntimeError):
    def __init__(self, status, payload):
        self.status, self.payload = status, payload
        super().__init__(f"HTTP {status}: {payload}")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise RemoteError(code, "Redirect refused; credentials stay on the specified endpoint")


class RemoteEnv:
    """Wire-format dictionaries; unlike RoboticsEnv, images are HTTP references."""
    def __init__(self, url, *, api_key, timeout_s=120):
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https") or not parts.hostname or parts.username or parts.password:
            raise ValueError("Use an HTTP(S) URL without embedded credentials")
        if parts.path not in ("", "/") or parts.query or parts.fragment:
            raise ValueError("Use the server origin URL, not an API route")
        if not isinstance(api_key, str) or not api_key or any(c.isspace() for c in api_key):
            raise ValueError("A nonempty bearer key is required")
        if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Positive timeout required")
        self.url, self._key, self.timeout_s = url.rstrip("/"), api_key, timeout_s
        self._opener, self.closed = build_opener(_NoRedirect()), False

    def request(self, path, *, payload=None, idempotency_key=None, binary=False):
        if self.closed:
            raise RuntimeError("Client is detached")
        if not path.startswith("/") or path.startswith("//") or urlsplit(path).netloc:
            raise ValueError("API path must be relative to the configured server")
        headers = {"Authorization": "Bearer " + self._key}
        body = None
        if payload is not None:
            body = json.dumps(payload, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        if idempotency_key is not None:
            headers["Idempotency-Key"] = idempotency_key
        try:
            with self._opener.open(Request(self.url + path, data=body, headers=headers),
                                   timeout=self.timeout_s) as response:
                content = response.read()
        except HTTPError as error:
            raw = error.read()
            try:
                payload = json.loads(raw)
            except ValueError:
                payload = raw.decode(errors="replace")
            raise RemoteError(error.code, payload) from error
        return content if binary else json.loads(content)

    def obs(self):
        return self.request("/v1/observation")["observation"]

    def evaluate(self):
        return self.request("/v1/evaluation")["evaluation"]

    def get_state(self):
        return self.request("/v1/state")["state"]

    def act(self, action, *, idempotency_key=None):
        # One request only. Caller may retain its supplied key for reconciliation.
        return self.request("/v1/action", payload=action,
                            idempotency_key=str(uuid4()) if idempotency_key is None else idempotency_key)

    def reset(self, **options):
        return self.request("/v1/reset", payload=options)

    def session(self):
        return self.request("/v1/session")

    def pause(self):
        return self.request("/v1/session/pause", payload={})

    def resume(self):
        return self.request("/v1/session/resume", payload={})

    def stop(self):
        return self.request("/v1/session/stop", payload={})

    def close(self):
        self.closed = True  # Deliberately no HTTP lifecycle operation.

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def connect(url, *, api_key, timeout_s=120):
    return RemoteEnv(url, api_key=api_key, timeout_s=timeout_s)
