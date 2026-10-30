"""Robot-only handles into an environment-owned native simulation."""

from dataclasses import dataclass, field
import math


@dataclass
class RobotRuntime:
    instance: str
    model_id: str
    definition_fingerprint: str
    joints: dict
    groups: dict
    frames: dict
    body_ids: tuple
    shape_ids: tuple
    source: dict
    model: object = field(default=None, repr=False)
    metadata: dict = field(default_factory=dict)

    def bind(self, model):
        # Resolve final native labels; fixed-link collapse and composition may
        # change offsets. Never infer velocity addresses from coordinates.
        labels = list(model.joint_label)
        qs = model.joint_q_start.numpy()
        vs = model.joint_qd_start.numpy()
        for name, entry in self.joints.items():
            matches = [
                i for i, label in enumerate(labels) if label == entry["native_label"]
            ]
            if len(matches) != 1:
                raise ValueError(f"Ambiguous/missing finalized joint: {name}")
            entry["q"] = int(qs[matches[0]])
            entry["dof"] = int(vs[matches[0]])
        bodies = {label: i for i, label in enumerate(model.body_label)}
        self.body_ids = tuple(bodies[label] for label in self.metadata["body_labels"])
        for frame in self.frames.values():
            if frame["body_label"] is not None:
                frame["body"] = bodies[frame["body_label"]]
        for name, entry in self.joints.items():
            if (
                entry["q"] >= model.joint_coord_count
                or entry["dof"] >= model.joint_dof_count
            ):
                raise ValueError(f"Robot joint mapping outside finalized model: {name}")
        if any(i >= model.body_count for i in self.body_ids):
            raise ValueError("Robot body mapping outside model")
        for index, label in zip(self.shape_ids, self.metadata.get("shape_labels", ())):
            if index >= model.shape_count or model.shape_label[index] != label:
                raise ValueError(
                    "Finalized shape layout changed; explicit remapping is required"
                )
        self.model = model
        return self

    def validate(self, targets):
        if not isinstance(targets, dict):
            raise TypeError("Joint targets must be a dictionary")
        for name, value in targets.items():
            if name not in self.joints:
                raise ValueError(f"Unknown robot joint: {name}")
            spec = self.joints[name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"Finite SI joint target required: {name}")
            if not spec["lower"] <= value <= spec["upper"]:
                raise ValueError(f"Joint limit exceeded: {name}")
        return dict(targets)

    def write_controls(self, targets, control):
        validated = self.validate(targets)
        values = control.joint_target_q.numpy()
        for name, value in validated.items():
            values[self.joints[name]["q"]] = value
        control.joint_target_q.assign(values)

    def observe(self, state):
        q, qd = state.joint_q.numpy(), state.joint_qd.numpy()
        return {
            "joints": {
                name: {"position": float(q[x["q"]]), "velocity": float(qd[x["dof"]])}
                for name, x in self.joints.items()
            }
        }

    def capabilities(self):
        return {
            "model": self.model_id,
            "instance": self.instance,
            "joints": {
                n: {
                    k: v
                    for k, v in x.items()
                    if k in ("unit", "lower", "upper", "velocity_limit", "effort_limit")
                }
                for n, x in self.joints.items()
            },
            "groups": self.groups,
            "frames": list(self.frames),
            "controller_profile": self.metadata.get("controller_profile"),
            "parameters": dict(self.metadata.get("parameters", {})),
            "asset_sha256": self.source.get("sha256"),
            "definition_fingerprint": self.definition_fingerprint,
        }

    def frame_pose(self, name, state):
        """Native robot frame query for internal control/teacher use, not policy truth."""
        from scipy.spatial.transform import Rotation

        frame = self.frames[name]
        if frame["body"] == -1:
            return {key: frame[key] for key in ("position_m", "orientation_wxyz")}
        physical = state.body_q.numpy()[frame["body"]]
        r = Rotation.from_quat(physical[3:])
        q = frame.get("orientation_wxyz", [1, 0, 0, 0])
        local = Rotation.from_quat([*q[1:], q[0]])
        rotation = (r * local).as_quat()
        return {
            "position_m": (
                physical[:3] + r.apply(frame.get("position_m", [0, 0, 0]))
            ).tolist(),
            "orientation_wxyz": [float(rotation[3]), *rotation[:3].tolist()],
        }


@dataclass
class RobotRig:
    setup: object
    instances: dict

    def bind(self, model):
        for robot in self.instances.values():
            robot.bind(model)
        return self

    @property
    def joints(self):
        return {
            n: spec
            for robot in self.instances.values()
            for n, spec in robot.joints.items()
        }

    def validate(self, targets):
        if not isinstance(targets, dict):
            raise TypeError("Targets must be a dictionary")
        unknown = set(targets) - self.joints.keys()
        if unknown:
            raise ValueError(f"Unknown robot joints: {sorted(unknown)}")
        for robot in self.instances.values():
            robot.validate({k: v for k, v in targets.items() if k in robot.joints})
        return dict(targets)

    def write_controls(self, targets, control):
        validated = self.validate(
            targets
        )  # All instances validate before any mutation.
        values = control.joint_target_q.numpy()
        for name, value in validated.items():
            values[self.joints[name]["q"]] = value
        control.joint_target_q.assign(values)

    def observe(self, state):
        return {
            "joints": {
                n: value
                for robot in self.instances.values()
                for n, value in robot.observe(state)["joints"].items()
            }
        }

    def capabilities(self):
        return {
            "setup_id": self.setup.id,
            "setup_fingerprint": self.setup.fingerprint,
            "instances": {
                name: robot.capabilities() for name, robot in self.instances.items()
            },
        }

    def frame_pose(self, role, state):
        """Resolve a setup role or instance/frame without changing state."""
        data = self.setup.to_dict()
        target = data.get("roles", {}).get(role, role)
        owner, separator, frame = target.partition("/")
        if not separator:
            raise ValueError("Expected a role alias or instance/frame")
        if owner in data.get("attachments", {}):
            attachment = data["attachments"][owner]
            if frame != "tcp" or attachment["model"].split("@")[0] not in (
                "wsg50",
                "franka-hand",
            ):
                raise ValueError("This attachment does not expose that runtime frame")
            owner = attachment["parent"].split("/")[0]
        if owner not in self.instances:
            raise ValueError("Unknown robot instance")
        return self.instances[owner].frame_pose(frame, state)
