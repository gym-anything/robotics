"""Instructions without a privileged key-code answer or an autonomous policy.

INSTRUCTION is the job. The catalogue (benchmark.json) serves it unchanged and
the policy's /v1/task text starts with it. Scoring tolerances and their
provenance live in evaluation.py, not in the task text.
"""

INSTRUCTION = (
    "Unlock the padlock on the chest with its key from the near (-Y) stand and swing the shackle open. "
    "Leave the other key on its stand. The chest lid does not need to be opened."
)

DESCRIPTION = INSTRUCTION + (
    " Operate the Panda through joint, gripper or Cartesian commands. The lock and chest are fixed to the workbench. "
    "The external bench_rgb camera and joint encoders are the policy's observations. "
    "The shared action interface names this single arm 'left'; there is no right arm."
)
