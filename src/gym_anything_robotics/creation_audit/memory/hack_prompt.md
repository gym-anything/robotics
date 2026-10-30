Alright, look at {env_dir}. Your job is to try to break the simulation using legal actions exposed by the environment.

Read the code, understand the physics and controls, and decide which actions or sequences might expose problems. Actually run the environment and try them. Look for NaNs, infinities, numerical blowups, or physically impossible behavior, such as objects flying away without a plausible cause.

The failure should come from legal interaction with the environment, not from editing the implementation or injecting a broken state. You are not trying to solve the task. A robot failing the task is not, by itself, a simulator failure.

When you find a problem, save the starting configuration and the code or action sequence needed to reproduce it, together with the relevant logs and video evidence. Make it possible for the creator to see what happened and reproduce it.

Save your full audit to {audit_file}. Explain what you tried, what broke, and where the evidence is. If you did not find a failure, say that; it does not establish that the environment cannot break.
