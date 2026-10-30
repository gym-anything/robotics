"""Prompts given to the agents, as used for the results in the paper."""

POLICY = """You are operating a robot in a physics simulation of a real-world job. You reach the simulation only through
its HTTP API at {url} ; send the header `Authorization: Bearer {key}` with every request. Start with GET /v1/help and
GET /v1/capabilities, which document the observations and commands available to you.

Your task:
{task}

You have {hours} from now (until {deadline} UTC). When the time is up, or when you end your session, the task is
scored on the state of the simulation at that moment. You may write and run code on this machine (Python 3 and uv are
available); the simulation itself is reachable only through this API.{extra}"""

TEACHER_EXTRA = """

This key is the privileged owner key: besides the robot API you may read the full simulator state, geometry and
action history, save and restore checkpoints, reset the episode and read the task evaluation (GET /v1/evaluation)."""

STUDENT_EXTRA = """

A coach who has already worked on this task can give you hints. Run `ask_coach "your question"` on this machine; it
prints the coach's reply (it may take several minutes). The coach gives advice only; it cannot act for you."""

COACH = """You are now coaching a student who is attempting the same task in its own, separate episode, through the
restricted policy API only. Its episode is visible to you through the privileged owner API at {url} with
`Authorization: Bearer {key}` (state, history, cameras, evaluation). Do not send actions to the student's episode.

The student asks:
{question}

Inspect its episode as needed, then reply with hints only: no code to run and no exact commands. Write your reply,
as plain text, to {answer} and finish."""

NEXT_ROUND = """Round {round} of {rounds}: the environment has been reset to a fresh episode of the same task; your
previous episode scored {score}. Use what you learned. You again have {hours} (until {deadline} UTC); the task is
scored when the time is up or when you end your session. The API is now at {url} ; send `Authorization: Bearer {key}`."""

NEXT_SEED = """Round {round} of {rounds}: a new episode of the same task has started, with a different starting layout
(a new seed); your previous episode scored {score}. The previous episode and its checkpoints are gone. The API is at
{url} ; from now on send `Authorization: Bearer {key}`. Use what you learned. You again have {hours} (until
{deadline} UTC); the task is scored when the time is up or when you end your session."""

TRANSFER = """Your privileged access has ended. A new episode of the same task has started, with a different starting layout
(a new seed); the previous episode and its checkpoints are gone. From now on you operate the robot only through the
restricted policy API at {url} ; send `Authorization: Bearer {key}`. It gives robot commands and onboard observations
only: no simulator state, geometry, history, checkpoints, resets or evaluation. Use what you learned. You have {hours}
(until {deadline} UTC); the task is scored when the time is up or when you end your session."""

NO_MORE_HINTS = "The coach has given all of its hints for this task and cannot answer more questions."

# Placed on the student's PATH as `ask_coach`. It hands the question to the harness through the student's own
# workspace and waits for the coach's reply.
ASK_COACH = """#!/usr/bin/env python3
import pathlib, sys, time
d = pathlib.Path(__file__).resolve().parents[1] / ".coach"; d.mkdir(exist_ok=True)
n = len(list(d.glob("q-*.txt"))) + 1
(d / f"q-{n}.txt").write_text(" ".join(sys.argv[1:]) or sys.stdin.read())
for _ in range(3600):
    a = d / f"a-{n}.txt"
    if a.exists():
        print(a.read_text()); sys.exit(0)
    time.sleep(2)
print("The coach did not answer within two hours.")
"""


def hours(seconds: float) -> str:
    h = seconds / 3600
    return f"{h:g} hour" + ("" if h == 1 else "s")
