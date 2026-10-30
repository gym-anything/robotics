# Creating and auditing simulations

This folder holds the creation-audit loop used to build the RoboWorld simulations. A creation agent builds a
simulation, and four audit agents check it and write reports that the creation agent then fixes. The auditor launcher
follows the auditor half of [Gym-Anything's creation_audit/method.py](https://github.com/cmu-l3/gym-anything/blob/1a2b4d94114da9733b6544d3275728c2a2ae84a0/extras/research/software_as_env/creation_audit/method.py)
(MIT, see [UPSTREAM_LICENSE.txt](UPSTREAM_LICENSE.txt)).

Both commands run a coding-agent CLI with full access to the workspace, so run them on a machine you are willing to give
the agent. Claude Code is the default backend (`--backend cc`, model `claude-opus-5-5`, effort `xhigh`). Codex is the
other backend (`--backend codex`), and it uses the model in your Codex config. `--model` overrides the model.

## Create a simulation

```bash
gar create --action-group PATH_TO_ACTION_GROUP --env-dir benchmarks/roboworld/environments/NEW_ENV
```

The action group is a file or folder that describes the job: its actions and the capabilities they need. The creator
runs three turns in one agent session:

1. [Research](memory/env_creation_notes/research_prompt.md): the agent studies the job and designs the simulation and
   writes `research.md`.
2. [Build](memory/env_creation_notes/build_prompt.md): sent only after the research succeeds, with the research still in
   context.
3. [Blind nudge](memory/env_creation_notes/nudge_prompt.md): a short request to keep improving the simulation, sent
   after the build succeeds.

Prompts, logs and the session ID of each turn go to `artifacts/creations/<environment>-<time>-<id>/`.

## Audit a simulation

```bash
gar audit --env-dir benchmarks/roboworld/environments/NEW_ENV
gar audit --env-dir benchmarks/roboworld/environments/NEW_ENV --auditor evidence
```

All four auditors run by default. Repeat `--auditor` to pick some of them.

- [Qualitative](memory/qualitative_prompt.md): task-relevant physics, scene realism, relevance to the real job and
  grounded parameters.
- [Evidence](memory/evidence_prompt.md): demonstrations of the dynamics and of each interaction.
- [Infrastructure](memory/infrastructure_prompt.md): speed, completeness of the API and CLI, and reuse of the shared
  robot templates.
- [Hack](memory/hack_prompt.md): tries legal actions that break the simulation, and records how to reproduce them.

Each auditor gets a fresh session, explores the repository, then reads its prompt in the same session and writes
`audit.md`. Reports, prompts and logs go to `artifacts/audits/<environment>-<time>-<id>/`. Exit code zero means every
report was written, not that the simulation passed. Each call times out after two hours by default (`--timeout-sec`).

The evidence auditor can also send videos to Gemini (`gemini-3.8-flash`) with a [video tool](memory/video_tool.md). It
needs `pip install -e '.[creation-audit]'` and `GEMINI_API_KEY` or `GOOGLE_API_KEY`. Without them the auditor is told
that the tool is not available.
