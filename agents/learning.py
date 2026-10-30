"""Frontier models learning in simulation: in-context rounds, a privileged teacher and coach, and self-transfer.

    python -m agents.learning --task football --agent claude-code --model claude-opus-5-5 --effort xhigh

Settings (all run three rounds of the same task by default, six hours each):
  solo           the agent sees only what a real robot sees (policy key) and keeps its context across rounds.
  teacher        the agent has privileged access (owner key): simulator state, history, checkpoints, evaluation.
  coached        a student with the policy key can run `ask_coach "question"`. The coach is the teacher after its
                 first round; it inspects the student's episode with the owner key and replies with hints only.
  self-transfer  the teacher after its first round continues with the policy key only, on new starting layouts.

Every round after the first is a new episode of the same task with a different seed (tasks without seeds restart from
the same state). Each round is scored with the task evaluation when the agent ends its session or the time is up.
"""

from __future__ import annotations

import argparse
import json
import stat
import threading
import time
from pathlib import Path

from .harness import Agent, build_image
from .prompts import (ASK_COACH, COACH, NEXT_ROUND, NEXT_SEED, NO_MORE_HINTS, POLICY, STUDENT_EXTRA, TEACHER_EXTRA,
                      TRANSFER, hours)
from .sim import Episode

SETTINGS = ("solo", "teacher", "coached", "self-transfer")
NO_SEED = {"football"}  # one fixed task: every round starts from the identical state


def selection_for(task: dict, seed: int | None) -> dict:
    base = dict(task.get("selection", {}))
    return base if seed is None or task["environment_id"] in NO_SEED else {**base, "seed": seed}


class Condition:
    """One setting of one task: an agent, its current episode, and its scored rounds."""

    def __init__(self, name: str, task: dict, agent: Agent, folder: Path, round_s: float, privileged: bool):
        self.name, self.task, self.agent, self.folder = name, task, agent, folder
        self.round_s, self.privileged = round_s, privileged
        self.episode: Episode | None = None
        self.rounds: list[dict] = []

    def key(self) -> str:
        return self.episode.owner_key if self.privileged else self.episode.policy_key

    def new_episode(self, number: int, seed: int | None) -> None:
        if self.episode is not None:
            self.episode.stop()
        self.episode = Episode(self.task["environment_id"], selection_for(self.task, seed),
                               self.folder / f"episode-{number}").start()

    def deadline(self) -> str:
        return time.strftime("%Y-%m-%d %H:%M", time.gmtime(time.time() + self.round_s))

    def play(self, number: int, prompt: str) -> dict:
        turn = self.agent.send(prompt, timeout_s=self.round_s)
        record = {"round": number, "seed": self.episode.selection.get("seed"), "agent": turn}
        try:
            record["evaluation"] = self.episode.evaluate()
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
        self.rounds.append(record)
        (self.folder / "rounds.json").write_text(json.dumps(self.rounds, indent=2))
        return record

    def score(self):
        return (self.rounds[-1].get("evaluation") or {}).get("score") if self.rounds else None

    def first_round(self, extra: str) -> dict:
        self.new_episode(1, 0)
        prompt = POLICY.format(url=self.episode.url, key=self.key(), task=json.dumps(self.episode.task(), indent=2),
                               hours=hours(self.round_s), deadline=self.deadline(), extra=extra)
        return self.play(1, prompt)

    def next_round(self, number: int, rounds: int, seed: int | None, template: str | None = None) -> dict:
        previous = self.score()
        self.new_episode(number, seed)
        fixed = self.task["environment_id"] in NO_SEED
        template = template or (NEXT_ROUND if fixed else NEXT_SEED)
        prompt = template.format(round=number, rounds=rounds, score=previous, url=self.episode.url, key=self.key(),
                                 hours=hours(self.round_s), deadline=self.deadline())
        return self.play(number, prompt)

    def close(self) -> None:
        if self.episode is not None:
            self.episode.stop()


class CoachRelay(threading.Thread):
    """Answers the student's `ask_coach` questions with the coach agent, up to max_hints answers."""

    def __init__(self, student: Condition, coach: Agent, max_hints: int, timeout_s: float):
        super().__init__(daemon=True)
        self.student, self.coach, self.max_hints, self.timeout_s = student, coach, max_hints, timeout_s
        self.hints, self.stopped = [], threading.Event()
        self.box = student.agent.workspace / ".coach"
        self.box.mkdir(parents=True, exist_ok=True)
        bin_dir = student.agent.workspace / ".bin"
        bin_dir.mkdir(exist_ok=True)
        script = bin_dir / "ask_coach"
        script.write_text(ASK_COACH)
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    def run(self):
        while not self.stopped.is_set():
            for question in sorted(self.box.glob("q-*.txt"), key=lambda p: int(p.stem[2:])):
                n = question.stem[2:]
                answer = self.box / f"a-{n}.txt"
                if not answer.exists():
                    answer.write_text(self.answer(n, question.read_text()))
            self.stopped.wait(2)

    def answer(self, n: str, question: str) -> str:
        if len(self.hints) >= self.max_hints:
            return NO_MORE_HINTS
        episode = self.student.episode
        reply = f"answer-{n}.txt"
        prompt = COACH.format(url=episode.url, key=episode.owner_key, question=question,
                              answer=self.coach.workspace_path(reply))
        turn = self.coach.send(prompt, timeout_s=self.timeout_s)
        path = self.coach.workspace / reply
        text = path.read_text() if path.exists() else "The coach did not write a reply."
        self.hints.append({"question": question, "answer": text, "round": len(self.student.rounds) + 1, "coach": turn})
        (self.student.folder / "hints.json").write_text(json.dumps(self.hints, indent=2))
        return text


def run_task(task: dict, settings: list[str], agent_args: dict, folder: Path, round_s: float, rounds: int,
             max_hints: int) -> dict:
    folder.mkdir(parents=True, exist_ok=True)

    def agent(name: str) -> Agent:
        return Agent(home=folder / name / "agent-home", workspace=folder / name / "workspace", **agent_args)

    conditions: dict[str, Condition] = {}
    try:
        need_teacher = bool({"teacher", "coached", "self-transfer"} & set(settings))
        if need_teacher:
            teacher = Condition("teacher", task, agent("teacher"), folder / "teacher", round_s, privileged=True)
            conditions["teacher"] = teacher
            teacher.first_round(TEACHER_EXTRA)
            teacher.close()
            # the coach and the self-transfer agent both start from the teacher's context after its first round
            snapshots = {name: teacher.agent.fork(folder / name / "agent-home", folder / name / "workspace")
                         for name in ("coach", "self-transfer") if name.replace("coach", "coached") in settings}
            if "teacher" in settings:
                for number in range(2, rounds + 1):
                    teacher.next_round(number, rounds, number - 1)
                teacher.close()
        if "solo" in settings:
            solo = conditions["solo"] = Condition("solo", task, agent("solo"), folder / "solo", round_s, False)
            solo.first_round("")
            for number in range(2, rounds + 1):
                solo.next_round(number, rounds, number - 1)
            solo.close()
        if "coached" in settings:
            student = Condition("coached", task, agent("student"), folder / "coached", round_s, False)
            conditions["coached"] = student
            relay = CoachRelay(student, snapshots["coach"], max_hints, timeout_s=2 * 3600)
            relay.start()
            try:
                student.first_round(STUDENT_EXTRA)
                for number in range(2, rounds + 1):
                    student.next_round(number, rounds, number - 1)
            finally:
                relay.stopped.set()
                student.close()
        if "self-transfer" in settings:
            transfer = Condition("self-transfer", task, snapshots["self-transfer"], folder / "self-transfer", round_s,
                                 privileged=False)
            conditions["self-transfer"] = transfer
            transfer.next_round(1, rounds, 5, template=TRANSFER)
            for number in range(2, rounds + 1):
                transfer.next_round(number, rounds, 4 + number)
            transfer.close()
    finally:
        for c in conditions.values():
            c.close()
    result = {"environment_id": task["environment_id"], "selection": task.get("selection", {}),
              "settings": {name: [(r.get("evaluation") or {}).get("score") for r in c.rounds]
                           for name, c in conditions.items() if name in settings}}
    (folder / "learning.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv=None):
    from .evaluate import load_split
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test")
    p.add_argument("--task", action="append", required=True, help="environment ID from the split (repeatable)")
    p.add_argument("--settings", nargs="+", choices=SETTINGS, default=list(SETTINGS))
    p.add_argument("--agent", choices=("claude-code", "codex"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--effort")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--hours", type=float, default=6.0, help="time budget per round (default: 6)")
    p.add_argument("--max-hints", type=int, default=10)
    p.add_argument("--sandbox", choices=("docker", "none"), default="docker")
    p.add_argument("--output", type=Path, default=Path("runs") / time.strftime("learning-%Y%m%d-%H%M%S"))
    args = p.parse_args(argv)
    if args.sandbox == "docker":
        build_image()
    tasks = {t["environment_id"]: t for t in load_split(args.split)}
    agent_args = dict(cli=args.agent, model=args.model, effort=args.effort, sandbox=args.sandbox)
    for name in args.task:
        result = run_task(tasks[name], args.settings, agent_args, args.output / name, args.hours * 3600,
                          args.rounds, args.max_hints)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
