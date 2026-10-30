"""Opt-in adapter for the whiteboard task set. The native backend and mechanics are not changed."""
from dataclasses import replace
from .newton_whiteboard import NewtonWhiteboardBackend
from ..types import BackendStep
from whiteboard.task_set.catalogue import CatalogueTask, CONTRACT, preset_config, source_identity
from whiteboard.task_set.presets import apply_start, tool_poses
from whiteboard.task_set.mechanical import read_state
from whiteboard.task_set.verifier import evaluate


class NewtonWhiteboardTaskSetBackend(NewtonWhiteboardBackend):
    def __init__(self, *, catalogue_task, task_contract, robot="g1", config=None, task=None,
                 cameras=("g1_rgb",), renderer_options=None, output_dir=None, recording=True,
                 variation_id=None, **config_options):
        from whiteboard.config import Config
        if task_contract != CONTRACT:
            raise ValueError("Unsupported whiteboard task contract")
        if task is not None or variation_id is not None:
            raise ValueError("Named task presets cannot combine with native task/variation overrides")
        if config is not None and config_options:
            raise ValueError("Pass Config or individual options, not both")
        selected, selected_task = preset_config(catalogue_task, config or Config(**config_options))
        self.task_source = source_identity()
        super().__init__(robot=robot, config=selected, task=selected_task, cameras=cameras,
                         renderer_options=renderer_options, output_dir=output_dir, recording=False)
        try:
            apply_start(self.env, selected_task)
            if recording:
                from whiteboard.storage import Store
                self.store = Store(self.output / "history")
                self.env.trace.attach(self.store)
                self._command_sequence = max((int(k) for k in self.store.keys("command")), default=0)
            self._record_task()
        except Exception:
            self.close()
            raise

    def _record_task(self):
        if isinstance(self.env.task, CatalogueTask):
            self.env.trace.event("task_set_identity", 0, task_id=self.env.task.task_id,
                                 contract=self.env.task.contract, source=self.task_source,
                                 start="constructed dry pigment and free tool poses; no integration",
                                 start_ink=self.env.ink.metadata(), deposition_time_s=-600.,
                                 tool_poses={n: p.tolist() for n, p in tool_poses(self.config, self.env.task).items()})

    def reset(self, *, seed=None, options=None):
        if options or not isinstance(self.env.task, CatalogueTask):
            # Preserve native reset validation/coercion and variation selection verbatim.
            return super().reset(seed=seed, options=options)
        if seed is not None and seed != self.config.seed:
            from whiteboard.env import WhiteboardEnv
            config = replace(self.config, seed=seed)
            candidate = WhiteboardEnv(config, task=self.env.task)
            try:
                apply_start(candidate, candidate.task)
            except Exception:
                candidate.close()
                raise
            self.env.close()
            self.config, self.env = config, candidate
            if self.store is not None:
                self.env.trace.attach(self.store)
        else:
            self.env.reset()
        self._score_cache = None
        self._record_task()
        return self.get_state(), {"seed": self.config.seed, "continuity": "reset"}

    def evaluate(self):
        if not isinstance(self.env.task, CatalogueTask):
            return super().evaluate()
        if not self.env.healthy:
            raise RuntimeError("Cannot evaluate a faulted whiteboard")
        with self.env._lock:
            state = None if self.env.task.name in ("write-text", "write-in-colour", "draw-table") else read_state(self.env)
            result = evaluate(self.env.task, self.env.ink, self.config, state)
            result["task_source_sha256"] = self.task_source["sha256"]
            return result

    def act(self, action):
        if not isinstance(self.env.task, CatalogueTask):
            return super().act(action)
        # No ink-only cache: poses/contact/release can change without an ink revision.
        before = self.evaluate()["score"]
        info = self._checked(action, "teacher")
        result = self.evaluate()
        return BackendStep(self.get_state(), info, result["score"] - before, result["success"],
                           self.env.tick >= self.config.episode_ticks)

    def policy_access(self):
        access = super().policy_access()
        return replace(access, task_images=self._task_images) if isinstance(self.env.task, CatalogueTask) else access
