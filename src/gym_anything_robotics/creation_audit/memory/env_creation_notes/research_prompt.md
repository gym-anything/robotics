# Research and design a robot environment

Alright, deeply research the action group at {action_group} and design a
high-quality robot environment for it. The proposed environment directory is
{env_dir}. Focus on understanding the activity and producing a thorough,
concrete design, not implementing an environment or solving a robot task.

Read the complete group: every action, its robot and physics capabilities, and
its grounding, not just its title. Explore relevant seed environments in this
repository, their reports and actual visual evidence. Read the relevant notes
in `docs/lessons/`, including `docs/lessons/deposit-cans-v1.md`. Use
`docs/robot-environment-quality-verbatim.md` for the user's quality standard.
Paths here are relative to the shared checkout.

Research how the activity actually happens. Look at real settings, equipment,
object placement, robot hardware, dimensions, materials and physical mechanisms.
Find the best digitally available basis for the parameters and record the
sources and justified estimates. Physical calibration is not expected.

Develop a coherent, realistic and challenging scenario as you research. Think
through the interactions that matter, equations, units, geometry, numerical
choices and likely runtime cost. We need physical realism to the degree required
to represent a meaningful task, not every possible phenomenon. Consult
`docs/unified-contract.md`, `docs/unified-cli.md` and `robots/README.md` to
understand the available foundation and how the design fits the project.

Propose tasks that represent real activities or test nuanced robotic capabilities,
with diversity in capabilities, difficulty and interaction with components.
Use minor setup changes where possible; do not force every action in the group
into one contrived scene. Explain how assisted experiments could demonstrate
the necessary physical interactions without requiring an autonomous solution.

Write down your research, sources, concrete scene design, physical model,
parameter basis and proposed evidence. Take the time to think this through.
Save your research to: {research_file}
