"""Sewing environment wrapper (the GAR backend's engine): the coupled world, the task's start state, checkpoints,
actions (joint targets, the machine's control box), histories for the evaluator, and the state dictionary.

Start states (disclosed initial conditions):
  under_foot  the two cut plies laid stacked and aligned from just above the feed-dog teeth, their far ends 8 mm past
              the needle and their right edges 10 mm right of it, with the foot lowered on them (what an operator
              does holding the plies until the foot is down); the part nearest the operator hangs over the table's
              front edge, as on a real machine table, and drapes during the hands-off settle. Plain seam, baste,
              pivot.
  tray        the stack lies on the table left of the machine with the foot down (A0 load).
The robots start at their 'sewing' ready pose (hands down, 100 mm above the table in front of the work). The machine
is idle, needle up, threads trimmed.

Preparation (disclosed): the laid shape is brought to rest by dynamic relaxation (the whole world steps;
the cloth's velocities are zeroed at kinetic-energy peaks) for RELAX_S, then the cell settles hands-off and undamped
for SETTLE_S. The draped overhang cannot be laid exactly on the discrete mesh (a 4 mm bend over 7.5 mm triangles
stores ~0.3 mJ of chordal compression), and without relaxation its release set the overhang swinging for seconds.
The prepared state is cached per configuration and physics source under ~/.cache/gar/sewing/ (or
$GAR_WORKSPACE/.gar/sewing-cache/), as welding does.
"""
from gym_anything_robotics.paths import package_dir

import copy
import hashlib
import json
import os
import pickle
import threading
from pathlib import Path
from dataclasses import asdict

import numpy as np

from .world import World, Config
from .task import Evaluator, INSTRUCTIONS, TIME_LIMIT_S

RELAX_S = 1.0
SETTLE_S = .25
SOURCE_FILES = ("bench.py", "cloth.py", "env.py", "g1.py", "kinematics.py", "layout.py", "machine.py", "mesh.py",
                "params.py", "scene.py", "variants.py", "world.py")


def _cache_dir():
    ws = os.environ.get("GAR_WORKSPACE")
    base = Path(ws)/".gar"/"sewing-cache" if ws else Path.home()/".cache"/"gar"/"sewing"
    base.mkdir(parents=True, exist_ok=True)
    return base


ROBOT_BUILDER_FILES = ("__init__.py", "assets.py", "deployment.py", "runtime.py", "spec.py", "backends/newton.py",
                       "controllers/kinematics.py")


def _code_key(src):
    """A Python module as the code it runs: its syntax tree with the docstrings, the comments and the formatting
    gone, and every raised exception's message removed (the exception's type stays), serialised the same way on every
    Python version (fields that are empty or None are left out). Another environment rewording an error, a docstring
    or a comment in the shared robot code is not a change of sewing's world. A functional change anywhere in the module still moves it, whether or
    not sewing's rigs execute that path: the key is conservative there."""
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            b = node.body
            if b and isinstance(b[0], ast.Expr) and isinstance(b[0].value, ast.Constant) and isinstance(b[0].value.value, str):
                node.body = b[1:] or [ast.Pass()]
        if isinstance(node, ast.Raise) and isinstance(node.exc, ast.Call):
            node.exc.args, node.exc.keywords = [], []

    def ser(n):
        if isinstance(n, ast.AST):
            parts = []
            for f in n._fields:
                v = getattr(n, f, None)
                if v is None or v == [] or f in ("type_comment", "type_params", "kind"):
                    continue
                parts.append(f"{f}={ser(v)}")
            return f"{type(n).__name__}({','.join(parts)})"
        if isinstance(n, list):
            return "[" + ",".join(ser(x) for x in n) + "]"
        return repr(n)
    return ser(tree).encode()


# keys of a robot model or attachment that build nothing: named frames are lookups on bodies the model already has
# (another environment can add frames to the shared G1), sources record provenance
SPEC_METADATA = ("frames", "sources")


def hash_inputs(root=None):
    """What sewing's world is built from, as (label, bytes) pairs: this package's physics source; sewing's four rigs;
    of each robot model and attachment those rigs name, the model file without the other environments' controller
    profiles and poses and without its metadata (SPEC_METADATA), and its asset lock (the digest of every file the
    world loads; `robots/assets.py` refuses a file that differs); the shared code that builds and drives them, as the
    code it runs (_code_key); the catalogue's code and sewing's own entry of its table. A robot another environment adds
    to the catalogue, a profile or frame it adds to a shared model, or a reworded message is not an input."""
    import ast
    here = Path(root)/"sewing" if root else Path(__file__).resolve().parent
    rob = Path(root)/"robots" if root else package_dir("robots")
    out = [(f"sewing/{f}", (here/f).read_bytes()) for f in SOURCE_FILES if (here/f).exists()]
    models, attachments, profiles = set(), set(), {"sewing"}
    for f in sorted(rob.glob("rigs/sewing-*.json")):
        out.append((f"robots/rigs/{f.name}", f.read_bytes()))
        rig = json.loads(f.read_text())
        for inst in rig.get("instances", {}).values():
            models.add(inst["model"].split("@")[0])
            profiles.add(inst.get("controller_profile", "sewing"))
        for att in rig.get("attachments", {}).values():
            attachments.add(att["model"].split("@")[0])
    for kind, names in (("models", models), ("attachments", attachments)):
        for name in sorted(names):
            d = rob/kind/name.replace("-", "_")
            d = d if d.is_dir() else rob/kind/name
            spec = json.loads((d/"robot.json").read_text())
            for key in SPEC_METADATA:
                spec.pop(key, None)
            for key in ("poses",):
                if isinstance(spec.get(key), dict):
                    spec[key] = {k: v for k, v in spec[key].items() if k in profiles}
            prof = spec.get("actuation", {}).get("profiles")
            if isinstance(prof, dict):
                spec["actuation"]["profiles"] = {k: v for k, v in prof.items() if k in profiles}
            out.append((f"robots/{kind}/{name}/robot.json", json.dumps(spec, sort_keys=True).encode()))
            if (d/"assets.lock.json").exists():
                out.append((f"robots/{kind}/{name}/assets.lock.json", (d/"assets.lock.json").read_bytes()))
    for f in ROBOT_BUILDER_FILES:
        if (rob/f).exists():
            out.append((f"robots/{f} (code)", _code_key((rob/f).read_text())))
    cat = rob/"catalog.py"
    if cat.exists():
        # the catalogue's code without its table of environments, and sewing's own entry of that table
        tree = ast.parse(cat.read_text())
        for node in list(tree.body):
            if isinstance(node, ast.Assign) and any(getattr(t, "id", None) == "ENVIRONMENTS" for t in node.targets):
                entry = next((v for k, v in zip(node.value.keys, node.value.values)
                              if isinstance(k, ast.Constant) and k.value == "sewing"), None)
                out.append(("robots/catalog.py:ENVIRONMENTS['sewing']",
                            _code_key(ast.unparse(entry)) if entry is not None else b""))
                tree.body.remove(node)
                break
        out.append(("robots/catalog.py (code)", _code_key(ast.unparse(tree))))
    return out


def _hash_sources(root=None):
    """The physics hash of the repository at root (default this one)."""
    h = hashlib.sha256()
    for label, data in hash_inputs(root):
        h.update(label.encode())
        h.update(data)
    return h.hexdigest()[:16]


# taken once at import: a process keeps the key of the code it runs (as welding does)
_PHYSICS_HASH = _hash_sources()


def physics_hash():
    return _PHYSICS_HASH


def start_path(config):
    c = hashlib.sha256(json.dumps(asdict(config), sort_keys=True, default=str).encode()).hexdigest()[:12]
    return _cache_dir()/f"start_{_PHYSICS_HASH}_{c}.pkl"


def _prune_cache(keep_other=2):
    """Keep the current physics source's start states and those of the newest `keep_other` older sources."""
    groups = {}
    for p in _cache_dir().glob("start_*.pkl"):
        groups.setdefault(p.name.split("_")[1], []).append(p)
    old = sorted((h for h in groups if h != _PHYSICS_HASH), key=lambda h: -max(p.stat().st_mtime for p in groups[h]))
    for h in old[keep_other:]:
        for p in groups[h]:
            p.unlink(missing_ok=True)


class SewingEnv:
    def __init__(self, config=None, *, output_dir=None):
        self.config = c = config or Config()
        self.lock = threading.RLock()
        self.output = output_dir
        self.world = World(c)
        self.generation = 0
        self.steps = 0
        self.start_state = c.start
        self.assisted_log = []
        self._make_start()
        self._reset_histories()
        self.initial = self.checkpoint()
        self.evaluator = Evaluator(self)

    # ------------------------------------------------------------------ start state
    def _make_start(self, use_cache=True):
        w = self.world
        if w.cloth is None:
            return
        path = start_path(self.config)
        log = None
        if use_cache and path.exists():
            try:
                with open(path, "rb") as f:
                    cached = pickle.load(f)
                w.restore(cached["world"])
                log = dict(cached["log"], cache=str(path), cached=True)
            except Exception as exc:                 # a stale or truncated cache entry is rebuilt, never trusted
                log = None
                self.cache_error = f"{type(exc).__name__}: {exc}"
        if log is None:
            log = w.relax(RELAX_S)
            w.settle(SETTLE_S)
            log.update(settle_s=SETTLE_S, cloth_ke_j=w.cloth.kinetic_energy(), physics=_PHYSICS_HASH, cached=False)
            if use_cache:
                _prune_cache()
                tmp = path.with_suffix(f".{os.getpid()}.tmp")          # concurrent preparers never share a file
                with open(tmp, "wb") as f:
                    pickle.dump(dict(world=w.snapshot(), log=log), f)
                tmp.replace(path)
        self.start_log = log
        self.assisted_log.append(dict(initial_condition=f"start {self.config.start}: plies laid flat just above the "
                                                        "feed-dog teeth" + (" with the foot lowered on them" if
                                                        self.config.start == "under_foot" else "") +
                                                        f"; prepared by dynamic relaxation ({RELAX_S} s, cloth "
                                                        f"velocities zeroed at kinetic-energy peaks) and {SETTLE_S} s "
                                                        "undamped hands-off settle", preparation=log))
        w.events, w.reflexes, w._reflex_until = [], [], {}
        w.timing = {k: 0. for k in w.timing}

    # ------------------------------------------------------------------ histories
    def _reset_histories(self):
        self.steps = 0
        self.history = dict(reflex_while_sewing=0, max_finger_force_n=0., sewing_s=0., max_speed_spm=0.)

    def _record(self):
        w = self.world
        if w.machine is None:
            return
        M = w.machine
        if M.mode != "idle":
            self.history["sewing_s"] += w.config.dt
            self.history["max_speed_spm"] = max(self.history["max_speed_spm"], float(M.speed_ramped))
            # the reflexes recorded in this tick (a record carries the tick it was made in; counting records of the
            # previous tick too counts each reflex twice)
            for r in reversed(w.reflexes):
                if r["tick"] != w.tick:
                    break
                self.history["reflex_while_sewing"] += 1
        if self.steps % 20 == 0:
            for a in w.hand_arms:
                self.history["max_finger_force_n"] = max(self.history["max_finger_force_n"],
                                                         float(np.linalg.norm(w.finger_force(a))))

    @property
    def sim_time(self):
        return self.steps*self.config.dt

    # ------------------------------------------------------------------ checkpoints
    def checkpoint(self):
        return dict(world=self.world.snapshot(), steps=self.steps, history=copy.deepcopy(self.history))

    def restore(self, snap):
        self.world.restore(snap["world"])
        self.steps = snap["steps"]
        self.history = copy.deepcopy(snap["history"])

    def reset(self):
        with self.lock:
            self.restore(self.initial)
            self.generation += 1
            return self.get_state()

    # ------------------------------------------------------------------ actions
    @property
    def joints(self):
        return self.world.joints

    def act(self, targets, *, ticks, state=True):
        if type(ticks) is not int or ticks <= 0:
            raise ValueError("ticks must be a positive integer")
        with self.lock:
            w = self.world
            if w.fault:
                raise RuntimeError(w.fault)
            if targets:
                w.set_targets(targets)
            for _ in range(ticks):
                w.step()
                self.steps += 1
                self._record()
            return self.get_state() if state else None

    def machine_io(self, outputs):
        with self.lock:
            self.world.machine_io(**outputs)

    # ------------------------------------------------------------------ state
    def machine_feedback(self):
        """What the machine's control box reports (its status: shaft angle, speed, needle position, presser, reverse,
        mode, fault, stitch counter)."""
        M = self.world.machine
        return None if M is None else M.status()

    def get_state(self, evaluate=True):
        with self.lock:
            w = self.world
            jq, jv = w.state.joint_q.numpy(), w.state.joint_qd.numpy()
            tau = w.external_torque_estimate()
            ev = (self.evaluator.evaluate() if (evaluate and not w.fault and w.machine is not None)
                  else dict(success=False, score=0., checks={}))
            M = w.machine
            return {
                "tick": self.steps, "generation": self.generation, "time_s": self.sim_time,
                "ready": w.fault is None, "fault": w.fault,
                "config": {k: v for k, v in asdict(self.config).items()},
                "instruction": INSTRUCTIONS.get(self.config.task),
                "start_state": self.start_state,
                "joints": {n: {"position": float(jq[s["q"]]), "velocity": float(jv[s["dof"]]),
                               "target": float(w.targets[s["q"]]),
                               **({"external_torque": tau[n]} if n in tau else {})} for n, s in w.joints.items()},
                "machine": self.machine_feedback(),
                "seam": None if M is None else dict(
                    holes=len(M.holes), stitches=len(M.stitches),
                    formed=sum(1 for s in M.stitches if s.get("formed")),
                    faults=list(M.faults[-5:]), needle=M.needle_state, foot_z_mm=float(M.foot_z*1e3),
                    foot_load_n=float(M.foot_N), threads="trimmed" if M.tether == "trimmed" else "attached"),
                "robot": {"reflexes": list(w.reflexes[-10:]), "refused_targets": int(w.refused_targets),
                          "stopped_arms": sorted(a for a, u in w._reflex_until.items() if w.time < u),
                          "cloth_force_n": {a: [float(v) for v in w.finger_force(a)] for a in w.hand_arms}
                          if w.cloth is not None else {}},
                "history": dict(self.history),
                "events": list(w.events[-20:]),
                "terminated": self.sim_time >= TIME_LIMIT_S.get(self.config.task, 1e9),
                "evaluation": ev,
            }

    def close(self):
        pass
