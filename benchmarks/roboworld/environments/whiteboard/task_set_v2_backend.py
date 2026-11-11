"""Backend for a launch selecting whiteboard-tasks-v2 (task_contract="whiteboard-tasks-v2").

The released task-set adapter (gym_anything_robotics/backends/newton_whiteboard_task_set.py) builds this class only for
that contract (its __new__); every other launch is that adapter or the native backend, as released. Construction
mirrors the adapter with the v2 task (task_set_v2.TaskV2) and the v2 source identity; evaluation is the released
verifier through task_set_v2.evaluate. Reset, the authored start, the teacher step, the trace record and the policy's
reference image are the adapter's, unchanged. The native backend and mechanics are not changed.
"""
from gym_anything_robotics.backends.newton_whiteboard import NewtonWhiteboardBackend
from gym_anything_robotics.backends.newton_whiteboard_task_set import NewtonWhiteboardTaskSetBackend
from whiteboard.task_set.mechanical import read_state
from whiteboard.task_set.presets import apply_start
from whiteboard.task_set_v2 import CONTRACT, TaskV2, evaluate, preset_config, source_identity


class NewtonWhiteboardTaskSetV2Backend(NewtonWhiteboardTaskSetBackend):
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
        NewtonWhiteboardBackend.__init__(self, robot=robot, config=selected, task=selected_task, cameras=cameras,
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

    def evaluate(self):
        if not isinstance(self.env.task, TaskV2):
            return super().evaluate()
        if not self.env.healthy:
            raise RuntimeError("Cannot evaluate a faulted whiteboard")
        with self.env._lock:
            state = None if self.env.task.name in ("write-text", "write-in-colour", "draw-table") else read_state(self.env)
            result = evaluate(self.env.task, self.env.ink, self.config, state)
            result["task_source_sha256"] = self.task_source["sha256"]
            return result
