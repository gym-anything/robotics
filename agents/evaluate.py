"""Evaluate a coding agent on a RoboWorld split.

    python -m agents.evaluate --split test --agent claude-code --model claude-opus-5-5 --effort xhigh

For each task the agent gets the task description, the simulation's URL and the restricted policy key, and a time
budget (three hours by default). When the agent ends its session or the time is up, the task is scored on the state
of the simulation with the owner key. Results go to <output>/<environment>/result.json and <output>/results.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from .harness import Agent, build_image
from .prompts import POLICY, hours
from .sim import Episode

SPLITS = Path(__file__).resolve().parents[1] / "benchmarks" / "roboworld" / "splits"


def load_split(name: str) -> list[dict]:
    path = Path(name) if Path(name).is_file() else SPLITS / f"{name}.json"
    return json.loads(path.read_text())["tasks"]


def run_task(task: dict, agent_args: dict, folder: Path, budget_s: float) -> dict:
    episode = Episode(task["environment_id"], task.get("selection", {}), folder / "episode")
    result = {"environment_id": task["environment_id"], "selection": task.get("selection", {})}
    try:
        episode.start()
        agent = Agent(home=folder / "agent-home", workspace=folder / "workspace", **agent_args)
        start = time.time()
        prompt = POLICY.format(url=episode.url, key=episode.policy_key, task=json.dumps(episode.task(), indent=2),
                               hours=hours(budget_s), deadline=time.strftime("%Y-%m-%d %H:%M", time.gmtime(start + budget_s)),
                               extra="")
        result["agent"] = agent.send(prompt, timeout_s=budget_s)
        result["evaluation"] = episode.evaluate()
    except Exception as error:  # a failed task is recorded and the run moves on
        result["error"] = f"{type(error).__name__}: {error}"
    finally:
        episode.stop()
    (folder / "result.json").write_text(json.dumps(result, indent=2))
    return result


def summarize(results: list[dict]) -> dict:
    scored = [r["evaluation"] for r in results if r.get("evaluation", {}).get("available")]
    n = len(results)
    return {"tasks": n, "scored": len(scored),
            "success_rate": sum(bool(e["success"]) for e in scored) / n if n else None,
            "average_reward": sum(float(e["score"]) for e in scored) / n if n else None}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split", default="test", help="split name in benchmarks/roboworld/splits, or a split file")
    p.add_argument("--tasks", nargs="*", help="only these environment IDs")
    p.add_argument("--agent", choices=("claude-code", "codex"), required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--effort", help="reasoning effort, e.g. xhigh")
    p.add_argument("--hours", type=float, default=3.0, help="time budget per task (default: 3)")
    p.add_argument("--sandbox", choices=("docker", "none"), default="docker")
    p.add_argument("--output", type=Path, default=Path("runs") / time.strftime("eval-%Y%m%d-%H%M%S"))
    args = p.parse_args(argv)
    if args.sandbox == "docker":
        build_image()
    tasks = [t for t in load_split(args.split) if not args.tasks or t["environment_id"] in args.tasks]
    agent_args = dict(cli=args.agent, model=args.model, effort=args.effort, sandbox=args.sandbox)
    results = []
    for task in tasks:
        folder = args.output / task["environment_id"]
        result = run_task(task, agent_args, folder, args.hours * 3600)
        results.append(result)
        e = result.get("evaluation") or {}
        print(f"{task['environment_id']}: success={e.get('success')} score={e.get('score')}"
              + (f" error={result['error']}" if "error" in result else ""), flush=True)
    summary = summarize(results)
    (args.output / "results.json").write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
