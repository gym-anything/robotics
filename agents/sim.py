"""One RoboWorld task episode, started with the gar CLI and read through its HTTP API.

Each episode is its own simulation server with two keys: the owner key reads everything (simulator state, history,
checkpoints, the task evaluation) and the policy key gets only robot commands and onboard observations, as a real
robot would. Agents receive one of the keys; the evaluation is always read with the owner key.
"""

from __future__ import annotations

import json
import subprocess
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path


def gar(*args: str, gar_bin: str = "gar", timeout: float | None = None) -> dict:
    """Run a gar command with JSON output and return its parsed result."""
    out = subprocess.run([gar_bin, *args, "--format", "json"], capture_output=True, text=True, timeout=timeout)
    if out.returncode not in (0, 1) or not out.stdout.strip():
        raise RuntimeError(f"gar {' '.join(args)} failed ({out.returncode}): {out.stderr.strip()[-2000:]}")
    return json.loads(out.stdout)


@dataclass
class Episode:
    environment_id: str
    selection: dict
    folder: Path
    gar_bin: str = "gar"
    startup_timeout_s: float = 3600.0
    session_id: str | None = None
    url: str | None = None
    keys: dict = field(default_factory=dict)

    def start(self) -> "Episode":
        """Resolve the task, start its server with a policy key, and wait until it is ready."""
        self.folder.mkdir(parents=True, exist_ok=True)
        task_file = self.folder / "task.json"
        task_file.unlink(missing_ok=True)
        gar("tasks", "resolve", self.environment_id, "--selector-json", json.dumps(self.selection),
            "--output", str(task_file), gar_bin=self.gar_bin)
        launch = gar("run", "--task-file", str(task_file), "--port", "0", "--policy-key",
                     "--startup-timeout", str(int(self.startup_timeout_s)), gar_bin=self.gar_bin)
        if not launch.get("ready"):
            raise RuntimeError(f"{self.environment_id} did not start: {json.dumps(launch)[:2000]}")
        self.session_id, self.url, self.keys = launch["session_id"], launch["url"], launch["api_keys"]
        (self.folder / "episode.json").write_text(json.dumps(
            {"environment_id": self.environment_id, "selection": self.selection, "session_id": self.session_id,
             "url": self.url}, indent=2))
        return self

    @property
    def owner_key(self) -> str:
        return self.keys["privileged"]

    @property
    def policy_key(self) -> str:
        return self.keys["policy"]

    def get(self, path: str, key: str, timeout: float = 600.0) -> dict:
        request = urllib.request.Request(self.url + path, headers={"Authorization": f"Bearer {key}"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())

    def task(self) -> dict:
        """The task as a robot policy sees it: GET /v1/task with the policy key (instruction and reference images)."""
        return self.get("/v1/task", self.policy_key)

    def evaluate(self) -> dict:
        """The task evaluation of the current state (GET /v1/evaluation with the owner key)."""
        return self.get("/v1/evaluation", self.owner_key)["evaluation"]

    def stop(self) -> None:
        if self.session_id:
            subprocess.run([self.gar_bin, "stop", self.session_id], capture_output=True, text=True)
