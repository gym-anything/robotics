"""Persistent child ownership, atomic manifests, and explicit attach semantics."""

from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import secrets
import subprocess
import sys
from uuid import uuid4

from .access import APIKeys
from .action_history import ActionHistory
from .launch import LaunchSpec
from .remote import RemoteEnv
from .session_service import SessionServer, SessionService


DEFAULT_CLIENT_TIMEOUT_S = 120.0


def declared_client_timeout(env):
    """The HTTP client timeout the backend declares for its commands (hardware 'client_timeout_s' of its policy
    interface, e.g. welding), or None. Recorded in the session manifest at startup."""
    try:
        value = env.policy_access().hardware.get("client_timeout_s")
    except Exception:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return None
    return float(value)


def client_timeout(manifest, timeout_s=None):
    """An explicit timeout; else the session's declared client timeout; else 120 s (sessions started before the
    declaration was recorded, and backends that declare none)."""
    if timeout_s is not None:
        return timeout_s
    value = manifest.get("client_timeout_s")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        return DEFAULT_CLIENT_TIMEOUT_S
    return float(value)


def write_json(path, value, *, private=False):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(temporary, flags, 0o600 if private else 0o644)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def session_root():
    workspace = Path(os.environ.get("GAR_WORKSPACE", Path.cwd()))
    return Path(os.environ.get("GAR_SESSION_ROOT", workspace/".gar"/"sessions")).resolve()


def _source():
    def git(*args):
        try:
            return subprocess.check_output(["git", "-C", str(Path(__file__).resolve().parents[2]), *args], text=True, timeout=15,
                                           stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "working_tree_changes": git("status", "--porcelain", "--untracked-files=normal")}


def prepare(spec, *, host="127.0.0.1", port=8080, policy=False, task=None):
    if task is not None and task.launch.to_dict() != spec.to_dict():
        raise ValueError("Task and session launch differ; no session created")
    if type(port) is not int or not 0 <= port <= 65535:
        raise ValueError("Port must be an integer in [0, 65535]")
    # Validate keys before creating state; secrets never enter launch/manifests.
    if not isinstance(host, str) or not host:
        raise ValueError("An explicit nonempty listen host is required")
    owner = os.environ.get("GAR_HTTP_API_TOKEN")
    if owner is None:
        owner = secrets.token_urlsafe(32)
    student = os.environ.get("GAR_HTTP_POLICY_API_TOKEN")
    if policy and student is None:
        student = secrets.token_urlsafe(32)
    APIKeys(owner, student)
    identity = uuid4().hex
    directory = session_root()/identity
    directory.mkdir(parents=True, exist_ok=False)
    key_root = Path(os.environ.get("GAR_CREDENTIAL_ROOT",
                    Path.home()/".config"/"gym-anything-robotics"/"credentials"))
    key_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    credential = (key_root/(identity+".json")).resolve()
    write_json(credential, {"privileged": owner, "policy": student}, private=True)
    launch = {"schema": "gar.session-launch.v1", "session_id": identity,
              "spec": spec.to_dict(), "cwd": str(Path.cwd()), "host": host, "port": port,
              "credential_file": str(credential), "source": _source()}
    if task is not None:
        write_json(directory/"task.json", task.to_dict())
    write_json(directory/"launch.json", launch)
    write_json(directory/"session.json", {
        "schema": "gar.session.v1", "session_id": identity, "status": "starting",
        "created_at": datetime.now(timezone.utc).isoformat(), "launch": launch,
        "artifacts": str(directory), "log": str(directory/"server.log")})
    return directory


def start(directory):
    directory = Path(directory).resolve()
    launch = json.loads((directory/"launch.json").read_text())
    options = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        options["start_new_session"] = True
    with (directory/"server.log").open("ab", buffering=0) as log:
        try:
            child = subprocess.Popen(
                [sys.executable, "-m", "gym_anything_robotics.cli", "_serve", str(directory)],
                cwd=launch["cwd"], stdin=subprocess.DEVNULL, stdout=log, stderr=log, **options)
        except OSError as error:
            manifest = json.loads((directory/"session.json").read_text())
            manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
            write_json(directory/"session.json", manifest)
            raise
    # Parent must not overwrite the child's status once it has started.
    return {"session_id": launch["session_id"], "directory": str(directory),
            "status": "starting", "pid": child.pid,
            "note": "Use gar sessions status; process launch is not readiness."}


def serve(directory, *, ready_callback=None):
    directory = Path(directory).resolve()
    launch = json.loads((directory/"launch.json").read_text())
    manifest = json.loads((directory/"session.json").read_text())
    if manifest["status"] != "starting":
        raise ValueError("Session directories cannot be reused; create a new session")
    # A second owner cannot launch another simulator into the same artifacts.
    # Retained permanently: a failed startup requires a new session directory.
    owner_fd = os.open(directory/"owner.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(owner_fd, "w") as owner_file:
        owner_file.write(str(os.getpid()))
    spec = LaunchSpec.from_dict(launch["spec"])
    keys = json.loads(Path(launch["credential_file"]).read_text())
    env = server = history = worker = None
    exit_code = 0
    try:
        # Output is operational storage, not a physics parameter. Explicit paths
        # are honored; otherwise make artifacts discoverable within the session.
        options = spec.to_dict()
        options["backend_options"].setdefault("output_dir", str(directory/"environment"))
        effective = LaunchSpec.from_dict(options)
        manifest["effective_launch"] = effective.to_dict()
        manifest["pid"] = os.getpid()
        write_json(directory/"session.json", manifest)
        env = effective.create()
        history = ActionHistory(directory/"http-history.sqlite3")
        service = SessionService(env, session_id=launch["session_id"], history=history)
        with service.history.span(kind="gateway_start", payload=effective.to_dict(),
                                  request_id=str(uuid4()), role="privileged", allow_missing_before=True):
            pass
        server = SessionServer((launch["host"], launch["port"]), service,
                               api_token=keys["privileged"], policy_api_token=keys["policy"])
        advertised = "127.0.0.1" if launch["host"] == "0.0.0.0" else launch["host"]
        manifest.update(status="running", url=f"http://{advertised}:{server.server_port}",
                        capabilities=env.capabilities.to_dict(), client_timeout_s=declared_client_timeout(env))
        write_json(directory/"session.json", manifest)
        if ready_callback is None:
            print(json.dumps({"ready": True, "session_id": service.session_id, "url": manifest["url"]}), flush=True)
            server.serve_forever(poll_interval=.1)
        else:
            import threading
            worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)
            worker.start()
            ready_callback()
            worker.join()
        manifest["status"] = "stopped"
    except KeyboardInterrupt:
        manifest["status"] = "stopped"
    except Exception as error:
        manifest.update(status="failed", error=f"{type(error).__name__}: {error}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    finally:
        if worker is not None:
            server.shutdown()
            worker.join(timeout=5)
        for operation in ((server.server_close if server else history.close if history else None),
                          (env.close if env else None)):
            if operation:
                try:
                    operation()
                except Exception as error:
                    manifest.update(status="failed", close_error=f"{type(error).__name__}: {error}")
                    exit_code = 1
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(directory/"session.json", manifest)
    return exit_code


def resolve(value):
    direct = Path(value).expanduser()
    if direct.is_dir():
        return direct.resolve()
    if len(value) != 32 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Use a session directory or complete session ID")
    path = session_root()/value
    if not (path/"session.json").is_file():
        raise ValueError(f"Unknown session: {value}")
    return path


def client(directory, *, timeout_s=None):
    directory = resolve(str(directory))
    manifest = json.loads((directory/"session.json").read_text())
    if manifest["status"] != "running":
        raise ValueError(f"Session is {manifest['status']}; inspect {directory/'server.log'}")
    keys = json.loads(Path(manifest["launch"]["credential_file"]).read_text())
    remote = RemoteEnv(manifest["url"], api_key=keys["privileged"], timeout_s=client_timeout(manifest, timeout_s))
    live = remote.session()
    if live["session_id"] != manifest["session_id"]:
        remote.close()
        raise ValueError("Endpoint identity differs from manifest; refusing stale session attachment")
    return remote


def status(directory):
    directory = resolve(str(directory))
    manifest = json.loads((directory/"session.json").read_text())
    result = {"manifest": manifest, "reachable": False}
    if manifest["status"] == "running":
        try:
            with client(directory, timeout_s=3) as remote:
                result.update(reachable=True, live=remote.session())
        except (OSError, ValueError, RuntimeError) as error:
            result["connection_error"] = str(error)
    return result
