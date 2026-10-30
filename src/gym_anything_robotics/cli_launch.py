"""Friendly selectors layered on exact, environment-owned launch recipes."""
import json
from .launch import LaunchSpec
from . import tasks


def selection(args, environment):
    values = json.loads(args.selector_json) if getattr(args, "selector_json", None) else {}
    if not isinstance(values, dict):
        raise ValueError("Task selection must be a JSON object")
    descriptor = tasks.catalog(environment)
    def add(key, value):
        if key in values:
            raise ValueError(f"Selector '{key}' supplied twice; no value overwritten")
        values[key] = value
    for item in getattr(args, "select", []) or []:
        key, separator, value = item.partition("=")
        if not separator or not key or not value:
            raise ValueError("Use --select KEY=VALUE; for example --select garment=shirt")
        try:
            value = json.loads(value)
        except ValueError:
            pass
        add(key, value)
    if getattr(args, "task", None):
        key = descriptor.get("cli_task_selector")
        if key:
            choices = descriptor.get("selector_choices", {}).get(key, [])
            if choices and args.task not in choices:
                raise ValueError(f"Unknown task '{args.task}'. Choose: {', '.join(choices)}. See 'gar tasks {environment}'.")
            add(key, args.task)
        elif args.task not in (descriptor.get("default_task_id"), (descriptor.get("default_task_id") or "").split("/")[-1]):
            raise ValueError(f"Use 'gar tasks {environment}' to find this environment's task selectors.")
    if getattr(args, "variation", None) is not None:
        keys = [k for k in ("variation_id", "episode_index") if k in descriptor["selectors"]]
        if len(keys) != 1:
            raise ValueError(f"{environment} does not have a numeric variation selector; see 'gar tasks {environment}'.")
        add(keys[0], args.variation)
    return values


def selected(args):
    return bool(getattr(args, "task", None) or getattr(args, "select", None)
                or getattr(args, "selector_json", None) or getattr(args, "variation", None) is not None)


def launch(args):
    if args.task_file:
        competing = (args.config, args.environment, args.backend_options_json, args.robot, args.backend, args.device)
        if any(v is not None for v in competing) or selected(args):
            raise ValueError("--task-file cannot be combined with competing launch options")
        task = tasks.TaskSpec.load(args.task_file)
        return task.launch, task
    if args.config:
        if any(v is not None for v in (args.environment, args.backend_options_json, args.robot, args.backend, args.device)) or selected(args):
            raise ValueError("--config cannot be combined with competing launch options")
        return LaunchSpec.load(args.config), None
    if args.environment is None:
        raise ValueError("Choose an environment: 'gar list', then 'gar run ENV'.")
    options = json.loads(args.backend_options_json) if args.backend_options_json is not None else {}
    if not isinstance(options, dict):
        raise ValueError("Backend options must be a JSON object")
    if args.device is not None:
        if "device" in options or isinstance(options.get("config"), dict) and "device" in options["config"]:
            raise ValueError("Device was supplied twice; no value overwritten")
        options["device"] = args.device
    spec = LaunchSpec(args.environment, backend="newton" if args.backend is None else args.backend,
                      robot=args.robot, backend_options=options)
    if selected(args):
        canonical = tasks.catalog(args.environment)["environment_id"]
        base = LaunchSpec(canonical, backend=spec.backend, robot=spec.robot, backend_options=spec.backend_options)
        task = tasks.resolve(canonical, selection(args, canonical), base=base)
        return task.launch, task
    return spec, None
