# Coding agents on RoboWorld

This folder runs coding agents (Claude Code or Codex) on RoboWorld tasks, as in the paper. Each task episode is its own
simulation server started with `gar run`. The agent controls the robot only through the HTTP API.

| File | What it does |
| --- | --- |
| [`sim.py`](sim.py) | Starts and stops one task episode and reads its task description and evaluation. |
| [`harness.py`](harness.py) | Runs Claude Code or Codex headless, resumes a conversation, and forks it. |
| [`prompts.py`](prompts.py) | The prompts the agents receive. |
| [`evaluate.py`](evaluate.py) | Evaluates an agent on a split. |
| [`learning.py`](learning.py) | Runs the learning settings: repeated attempts, teacher, coached student and self-transfer. |

## Setup

Install the repository first (see the [top-level README](../README.md)) so that `gar` is on your `PATH`. Each agent
gets its own fresh CLI state, so give it credentials through the environment:

- Claude Code: set `ANTHROPIC_API_KEY`, or run `claude setup-token` and set `CLAUDE_CODE_OAUTH_TOKEN`.
- Codex: set `OPENAI_API_KEY`, or log in once with `codex login`. The harness copies `~/.codex/auth.json` into each
  agent's state.

By default each agent runs in a Docker container built from [`Dockerfile.agent`](Dockerfile.agent). The container
mounts only the agent's own workspace and CLI state, so the agent cannot read this repository, the simulator files or
any key it was not given. It uses the host network to reach the simulation. `--sandbox none` runs the CLI directly on the host
instead, which is only safe on a machine you are willing to give the agent.

## Evaluate

```bash
python -m agents.evaluate --split test --agent claude-code --model claude-opus-5-5 --effort xhigh
python -m agents.evaluate --split test --agent codex --model gpt-6-astra --effort xhigh --tasks welding laundry
```

For each task the agent gets the task description, the simulation URL, the policy key and a time budget of three hours
(`--hours`). When the agent ends its session or the time is up, the task is scored with the owner key. Results go to
`runs/eval-<time>/<environment>/result.json` and the summary to `runs/eval-<time>/results.json`. The summary reports
the success rate and the average reward over all tasks, with failed runs counted as zero.

## Learning in simulation

```bash
python -m agents.learning --task football --agent claude-code --model claude-opus-5-5 --effort xhigh
```

Every setting plays three rounds of the same task (`--rounds`), six hours each (`--hours`). Each new round is a new
episode with a different seed. Football has a single fixed task, so its rounds restart from the same state.

- **solo**: the agent has the policy key only and keeps its context across rounds.
- **teacher**: the agent has the owner key, so it can read simulator state and history, use checkpoints and read the
  evaluation.
- **coached**: a student with the policy key can run `ask_coach "question"`. The coach is the teacher after its first
  round. It inspects the student's episode with the owner key and answers with hints, up to 10 per task
  (`--max-hints`).
- **self-transfer**: the teacher after its first round continues with the policy key only, on new seeds.

`--settings` runs a subset. Per-round scores go to `runs/learning-<time>/<task>/learning.json`, and each setting's
folder keeps its rounds, episodes and the agent's logs.
