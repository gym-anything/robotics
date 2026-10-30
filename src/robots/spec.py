"""Small immutable JSON specifications; importing this module never loads physics."""

from dataclasses import dataclass
import hashlib
import json
import math
import re

NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _name(value, label):
    if not isinstance(value, str) or not NAME.fullmatch(value):
        raise ValueError(f"{label} must be a nonempty simple name")
    return value


def _object(value, label):
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    return value


def _keys(value, required, allowed, label):
    _object(value, label)
    missing, unknown = set(required) - value.keys(), value.keys() - set(allowed)
    if missing or unknown:
        raise ValueError(
            f"{label}: missing {sorted(missing)}; unknown {sorted(unknown)}"
        )


def _encoded(value):
    # JSON serialization both rejects non-finite values and detaches caller state.
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def pose(value):
    _keys(value, (), ("position_m", "orientation_wxyz"), "pose")
    p = value.get("position_m", [0.0, 0.0, 0.0])
    q = value.get("orientation_wxyz", [1.0, 0.0, 0.0, 0.0])
    for v, n in ((p, 3), (q, 4)):
        if (
            not isinstance(v, (tuple, list))
            or len(v) != n
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not math.isfinite(x)
                for x in v
            )
        ):
            raise ValueError("Pose requires finite metric xyz and wxyz quaternion")
    if abs(sum(x * x for x in q) - 1.0) > 1e-6:
        raise ValueError("orientation_wxyz must have unit norm")
    return {"position_m": list(p), "orientation_wxyz": list(q)}


@dataclass(frozen=True)
class RobotSpec:
    _json: str

    @classmethod
    def from_dict(cls, value):
        fields = (
            "schema",
            "id",
            "revision",
            "asset",
            "root",
            "groups",
            "frames",
            "actuation",
            "mechanisms",
            "effectors",
            "sensors",
            "contact",
            "poses",
            "sources",
            "support",
            "description",
        )
        _keys(value, ("schema", "id", "revision", "asset", "root"), fields, "robot")
        if value["schema"] != "gar.robot.v1":
            raise ValueError("Unsupported robot schema")
        _name(value["id"], "robot id")
        if type(value["revision"]) is not int or value["revision"] < 1:
            raise ValueError("Positive integer revision required")
        _keys(
            value["asset"],
            ("format", "resolver", "path"),
            ("format", "resolver", "path", "lock"),
            "asset",
        )
        if value["asset"]["format"] not in ("urdf", "mjcf", "authored"):
            raise ValueError("Unsupported asset format")
        _keys(value["root"], ("frame", "mounting"), ("frame", "mounting"), "root")
        if value["root"]["mounting"] not in ("fixed", "floating", "attachment"):
            raise ValueError("Invalid mounting mode")
        for group, entry in value.get("groups", {}).items():
            _name(group, "group")
            _keys(entry, ("joints",), ("joints", "role"), "group")
            names = entry["joints"]
            if (
                not isinstance(names, list)
                or not names
                or any(not isinstance(n, str) or not n for n in names)
                or len(names) != len(set(names))
            ):
                raise ValueError("Group joints must be unique nonempty joint names")
        for name, frame in value.get("frames", {}).items():
            _name(name, "frame")
            _keys(
                frame,
                ("parent",),
                ("parent", "position_m", "orientation_wxyz"),
                "frame",
            )
            if not isinstance(frame["parent"], str) or not frame["parent"]:
                raise ValueError("Frame parent required")
            pose({k: v for k, v in frame.items() if k != "parent"})
        return cls(_encoded(value))

    def to_dict(self):
        return json.loads(self._json)

    @property
    def id(self):
        return self.to_dict()["id"]

    @property
    def revision(self):
        return self.to_dict()["revision"]

    @property
    def fingerprint(self):
        return hashlib.sha256(self._json.encode()).hexdigest()


@dataclass(frozen=True)
class RobotSetup:
    _json: str

    @classmethod
    def from_dict(cls, value):
        _keys(
            value,
            ("schema", "id", "instances"),
            ("schema", "id", "instances", "attachments", "roles", "description"),
            "setup",
        )
        if value["schema"] != "gar.robot-setup.v1":
            raise ValueError("Unsupported robot setup schema")
        _name(value["id"], "setup id")
        instances = _object(value["instances"], "instances")
        if not instances:
            raise ValueError("At least one robot instance required")
        for name, item in instances.items():
            _name(name, "instance")
            _keys(
                item,
                ("model", "controller_profile"),
                (
                    "model",
                    "controller_profile",
                    "mount",
                    "pose",
                    "initial_pose",
                    "parameters",
                ),
                "instance",
            )
            if not isinstance(item["model"], str) or not item["model"]:
                raise ValueError("Model reference required")
            _name(item["controller_profile"], "controller profile")
            if ("mount" in item) == ("pose" in item):
                raise ValueError("Specify exactly one mount slot or pose per instance")
            if "pose" in item:
                pose(item["pose"])
            if "mount" in item:
                _name(item["mount"], "mount")
            _object(item.get("parameters", {}), "parameters")
        attachments = _object(value.get("attachments", {}), "attachments")
        if set(attachments) & set(instances):
            raise ValueError("Instance and attachment names collide")
        parents = {}
        for name, item in attachments.items():
            _name(name, "attachment")
            _keys(
                item,
                ("model", "parent", "mount_profile"),
                ("model", "parent", "mount_profile"),
                "attachment",
            )
            if not isinstance(item["parent"], str) or "/" not in item["parent"]:
                raise ValueError("Attachment parent must identify instance/frame")
            owner = item["parent"].split("/")[0]
            if owner not in instances and owner not in attachments:
                raise ValueError("Unknown attachment parent")
            parents[name] = owner
        for name in parents:
            seen = set()
            node = name
            while node in parents:
                if node in seen:
                    raise ValueError("Cyclic attachment graph")
                seen.add(node)
                node = parents[node]
        for role, target in _object(value.get("roles", {}), "roles").items():
            _name(role, "role")
            if (
                not isinstance(target, str)
                or "/" not in target
                or target.split("/")[0] not in set(instances) | set(attachments)
            ):
                raise ValueError("Role must reference a declared component/frame")
        return cls(_encoded(value))

    def to_dict(self):
        return json.loads(self._json)

    @property
    def id(self):
        return self.to_dict()["id"]

    @property
    def fingerprint(self):
        return hashlib.sha256(self._json.encode()).hexdigest()
