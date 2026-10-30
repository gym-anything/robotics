"""Coding-agent CLIs (Claude Code, Codex) run headless on a RoboWorld task.

An agent works in its own folder and reaches the simulation only through the HTTP API. By default it runs in a
throwaway Docker container that mounts nothing but that folder and the agent's own state, so it cannot read the
repository, the session keys of other roles, or the simulator's files. Each agent keeps its CLI state (sessions and
login) in a home folder, so a later turn resumes the same conversation, and copying the home folder forks it.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

IMAGE = "gym-anything-robotics-agent"
DOCKERFILE = Path(__file__).with_name("Dockerfile.agent")


def build_image(image: str = IMAGE) -> None:
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True).returncode != 0:
        subprocess.run(["docker", "build", "-t", image, "-f", str(DOCKERFILE), str(DOCKERFILE.parent)], check=True)


@dataclass
class Agent:
    """One coding-agent conversation.

    cli: "claude-code" or "codex". model: the model name for that CLI. effort: reasoning effort passed to the CLI
    (Codex: model_reasoning_effort; Claude Code: --effort). home: the agent's persistent CLI state. workspace: the folder
    the agent works in, mounted at /work. sandbox: "docker" (default) or "none" to run the CLI directly on this machine.
    """

    cli: str
    model: str
    home: Path
    workspace: Path
    effort: str | None = None
    sandbox: str = "docker"
    image: str = IMAGE
    extra_args: tuple[str, ...] = ()
    session_id: str | None = None
    turns: int = field(default=0)

    def __post_init__(self):
        if self.cli not in ("claude-code", "codex"):
            raise ValueError(f"unknown agent CLI {self.cli!r}; use claude-code or codex")
        self.home, self.workspace = Path(self.home).resolve(), Path(self.workspace).resolve()
        for d in (self.home, self.workspace, self.home / "logs"):
            d.mkdir(parents=True, exist_ok=True)
        if self.cli == "codex":
            auth = Path(os.environ.get("CODEX_HOME", Path.home() / ".codex")) / "auth.json"
            if not (self.home / "auth.json").exists() and auth.is_file() and not os.environ.get("OPENAI_API_KEY"):
                shutil.copy2(auth, self.home / "auth.json")

    def fork(self, home: Path, workspace: Path) -> "Agent":
        """A new agent that continues this conversation independently (a copy of its CLI state)."""
        if Path(home).exists():
            raise FileExistsError(home)
        shutil.copytree(self.home, home, ignore=shutil.ignore_patterns("logs"))
        return Agent(self.cli, self.model, Path(home), Path(workspace), self.effort, self.sandbox, self.image,
                     self.extra_args, self.session_id)

    def workspace_path(self, name: str) -> str:
        """A file in the agent's workspace, as the agent sees it."""
        return f"/work/{name}" if self.sandbox == "docker" else str(self.workspace / name)

    # ---------------------------------------------------------------------------------------------- CLI commands
    def _command(self, prompt_file: str) -> list[str]:
        if self.cli == "claude-code":
            cmd = ["claude", "-p", "--verbose", "--output-format", "stream-json",
                   "--permission-mode", "bypassPermissions", "--model", self.model]
            if self.effort:
                cmd += ["--effort", self.effort]
            if self.session_id:
                cmd += ["--resume", self.session_id]
            return ["sh", "-c", shlex.join(cmd + list(self.extra_args)) + f" < {shlex.quote(prompt_file)}"]
        cmd = ["codex", "exec"] + (["resume", self.session_id] if self.session_id else [])
        cmd += ["--dangerously-bypass-approvals-and-sandbox", "--skip-git-repo-check", "--json", "--model", self.model]
        if self.effort:
            cmd += ["--config", f"model_reasoning_effort={json.dumps(self.effort)}"]
        return ["sh", "-c", shlex.join(cmd + list(self.extra_args)) + f' -- "$(cat {shlex.quote(prompt_file)})"']

    def _env(self, home: str) -> dict[str, str]:
        env = {"CLAUDE_CONFIG_DIR": f"{home}/claude", "CODEX_HOME": home, "IS_SANDBOX": "1",
               "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1", "PATH": "/work/.bin:/usr/local/bin:/usr/bin:/bin"}
        for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "CLAUDE_CODE_OAUTH_TOKEN", "OPENAI_API_KEY",
                     "OPENAI_BASE_URL"):
            if os.environ.get(name):
                env[name] = os.environ[name]
        return env

    def send(self, prompt: str, *, timeout_s: float) -> dict:
        """Run one turn: a new conversation, or the next message in this one. Stops the CLI at the timeout."""
        self.turns += 1
        prompt_path = self.home / "logs" / f"prompt-{self.turns}.txt"
        prompt_path.write_text(prompt)
        log_path = self.home / "logs" / f"turn-{self.turns}.jsonl"
        started = time.time()
        with log_path.open("w") as log:
            if self.sandbox == "docker":
                name = f"gar-agent-{uuid.uuid4().hex[:12]}"
                env = self._env("/home/agent")
                docker = ["docker", "run", "--rm", "--name", name, "--network", "host", "-w", "/work",
                          "-v", f"{self.workspace}:/work", "-v", f"{self.home}:/home/agent",
                          "-v", f"{prompt_path}:/tmp/prompt.txt:ro", "-e", "HOME=/home/agent"]
                for k, v in env.items():
                    docker += ["-e", f"{k}={v}"]
                proc = subprocess.Popen(docker + [self.image] + self._command("/tmp/prompt.txt"),
                                        stdout=log, stderr=subprocess.STDOUT)
            else:
                name = None
                env = {**os.environ, **self._env(str(self.home)),
                       "PATH": f"{self.workspace / '.bin'}{os.pathsep}{os.environ.get('PATH', '')}"}
                proc = subprocess.Popen(self._command(str(prompt_path)), cwd=self.workspace, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
            try:
                code = proc.wait(timeout=max(1.0, timeout_s))
                timed_out = False
            except subprocess.TimeoutExpired:
                timed_out = True
                if name:
                    subprocess.run(["docker", "kill", name], capture_output=True)
                proc.kill()
                code = proc.wait()
        self.session_id = self._session_from_log(log_path) or self.session_id
        return {"turn": self.turns, "exit_code": code, "timed_out": timed_out, "seconds": time.time() - started,
                "session_id": self.session_id, "log": str(log_path)}

    def _session_from_log(self, path: Path) -> str | None:
        found = None
        for line in path.read_text(errors="replace").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            found = event.get("session_id") or event.get("thread_id") or found
        return found
