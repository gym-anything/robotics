Alright, look at {env_dir}. Check these three things:

1. Speed: is the environment within the 200x wall-clock-time / simulated-time limit? Look at the runtime measurements.

2. Report and integration: is the environment properly integrated with the CLI, and is there a complete report at least up to the standard of the seed environments?

3. Configurable robots: can the environment use reusable robot templates, rather than having one robot hardwired into it? This should support choosing robots from the humanoid, two-arm and single-arm templates as needed.

Inspect the implementation and the available evidence. You can execute checks if needed; do not accept a claim just because it appears in a report.

Save your full audit to {audit_file}, explaining what is ready and what needs fixing.
