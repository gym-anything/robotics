"""Record actual render states without starting Isaac or rerunning actions later.

This is privileged component evidence, not a simulator checkpoint or a policy action.
Each mesh field is losslessly stored once by its content hash. A frame is published
only after all its blobs exist. Incomplete recordings remain explicit prefixes.
"""
import base64
import gzip
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_new(path, data):
    with Path(path).open("xb") as f:
        f.write(data)


class StateWriter:
    """Append-only frames and content-addressed arrays in a fresh local directory."""
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        (self.folder/"blobs").mkdir()
        (self.folder/"frames").mkdir()
        self.count, self.last_tick, self.last_time = 0, None, None
        self.entries = []
        self.known = set()

    def append(self, state, *, tick, sim_time_s, material_clock_s, cameras, lines):
        if type(tick) is not int or tick < 0 or not math.isfinite(sim_time_s):
            raise ValueError("invalid recorded tick/time")
        if self.last_tick is not None and (tick <= self.last_tick or sim_time_s <= self.last_time):
            raise ValueError("recording cannot splice a reset, restore or duplicate tick")
        poses = np.asarray(state["body_q"], dtype=np.float32)
        if poses.ndim != 2 or poses.shape[1] != 7 or not np.isfinite(poses).all():
            raise ValueError("invalid recorded body poses")
        meshes = {}
        for name, mesh in state["meshes"].items():
            if mesh.get("same"):
                raise ValueError("recording requires complete meshes, not worker-local reuse markers")
            refs = {}
            for key, value in mesh.items():
                data = base64.b64decode(value, validate=True)
                digest = sha(data)
                if digest not in self.known:
                    write_new(self.folder/"blobs"/(digest+".gz"), gzip.compress(data, compresslevel=1, mtime=0))
                    self.known.add(digest)
                refs[key] = digest
            meshes[name] = refs
        item = dict(index=self.count, tick=tick, sim_time_s=float(sim_time_s), material_clock_s=float(material_clock_s),
                    body_q=poses.tolist(), meshes=meshes, cameras=list(cameras), lines=list(lines))
        data = canonical(item)
        name = f"frames/{self.count:06d}.json"
        write_new(self.folder/name, data)
        self.entries.append(dict(file=name, sha256=sha(data)))
        self.count += 1
        self.last_tick, self.last_time = tick, sim_time_s


class RecordedFilm:
    """Capture a continuous world's exact scene inputs at a fixed tick cadence.

    Unlike Film, construction and capture never create a render worker. close()
    publishes a recording manifest, not a completed film or task-success claim.
    """
    def __init__(self, env, folder, cameras, *, title, assistance, views=None, labels=None,
                 states_per_s=2, width=720, height=405, samples=8):
        from .render import export
        from .probes.common import code_stamp
        interval = 1/(states_per_s*.0005)
        if not math.isfinite(interval) or interval < 1 or abs(interval-round(interval)) > 1e-7:
            raise ValueError("recording cadence must be an integer number of 0.5 ms ticks")
        self.env, self.world = env, env.world
        self.writer = StateWriter(folder)
        self.folder = self.writer.folder
        self.title, self.assistance, self.note = title, assistance, ""
        self.cameras, self.labels = list(cameras), labels or {}
        self.interval = int(round(interval))
        self.next_tick = self.world.tick + self.interval
        self.closed, self.failure = False, None
        self.started = time.perf_counter()
        scene, meta = export(self.world, self.folder/"scene", views)
        assets = {str(p.relative_to(self.folder)): sha(p.read_bytes()) for p in sorted((self.folder/"scene").rglob("*")) if p.is_file()}
        pkg = Path(__file__).parent
        self.manifest = dict(schema="concrete.render-recording.v1", complete=False, code=code_stamp(),
                             renderer_source={n:sha((pkg/n).read_bytes()) for n in ("render.py", "render_worker.py")},
                             scene=str(scene.relative_to(self.folder)), assets=assets, cameras=self.cameras,
                             labels=self.labels, width=width, height=height, samples=samples, interval_ticks=self.interval,
                             tick_s=.0005, title=title, assistance=assistance, scene_meta=meta,
                             scope="Actual ordered render states; no simulator replay, checkpoint or autonomous-task claim")
        self._publish()
        self.capture()
        self.world.paste_recorders.append(self)

    def _publish(self):
        doc = dict(self.manifest, frames=self.writer.entries, blobs=len(self.writer.known), failure=self.failure,
                   wall_s=time.perf_counter()-self.started)
        temp = self.folder/"recording.tmp"
        temp.write_bytes(canonical(doc))
        temp.replace(self.folder/"recording.json")

    def __call__(self, world):
        if world.tick >= self.next_tick:
            if world.tick != self.next_tick:
                raise RuntimeError("recording missed its scheduled tick")
            self.capture()
            self.next_tick += self.interval

    def capture(self):
        from .render import frame
        from .video import Film
        if self.failure:
            raise RuntimeError(self.failure)
        try:
            w = self.world
            self.writer.append(frame(w), tick=int(w.tick), sim_time_s=w.sim_time,
                               material_clock_s=self.env.material_clock(), cameras=self.cameras,
                               lines=[f"{self.title} | {self.assistance}",
                                      f"sim {w.sim_time:.3f} s | material clock {self.env.material_clock()/60:.3f} min | {self.note}",
                                      Film.evaluation_line(self)])
            self._publish()
        except Exception as exc:
            self.failure = f"{type(exc).__name__}: {exc}"
            self._publish()
            raise

    def close(self, *, error=None):
        if self.closed:
            return self.folder/"recording.json"
        self.closed = True
        if self in self.world.paste_recorders:
            self.world.paste_recorders.remove(self)
        if error:
            self.failure = str(error)
        if not self.failure and self.world.tick != self.writer.last_tick:
            self.capture()
        self.manifest["complete"] = not bool(self.failure)
        from .probes.common import code_stamp
        self.manifest["code_at_close"] = code_stamp()
        self._publish()
        return self.folder/"recording.json"


def read_recording(folder):
    """Verify every state and asset before starting a renderer; return complete ordered frames."""
    folder = Path(folder)
    doc = json.loads((folder/"recording.json").read_text())
    if doc.get("schema") != "concrete.render-recording.v1" or not doc.get("complete") or doc.get("failure"):
        raise ValueError("recording is incomplete; do not publish it as a complete film")
    for name, digest in doc["assets"].items():
        if sha((folder/name).read_bytes()) != digest:
            raise ValueError(f"scene asset hash mismatch: {name}")
    frames, previous, verified = [], None, set()
    for i, entry in enumerate(doc["frames"]):
        if entry["file"] != f"frames/{i:06d}.json":
            raise ValueError("missing or reordered frame")
        raw = (folder/entry["file"]).read_bytes()
        if sha(raw) != entry["sha256"]:
            raise ValueError("frame hash mismatch")
        f = json.loads(raw)
        if f["index"] != i:
            raise ValueError("frame index mismatch")
        if previous is not None:
            delta = f["tick"] - previous["tick"]
            if delta != doc["interval_ticks"] and not (i == len(doc["frames"])-1 and 0 < delta < doc["interval_ticks"]):
                raise ValueError("missing interior tick or restored segment")
            if not math.isclose(f["sim_time_s"]-previous["sim_time_s"], delta*doc["tick_s"], abs_tol=1e-7):
                raise ValueError("recorded clock disagrees with ticks")
        for fields in f["meshes"].values():
            for digest in fields.values():
                if digest not in verified:
                    if sha(gzip.decompress((folder/"blobs"/(digest+".gz")).read_bytes())) != digest:
                        raise ValueError("mesh blob hash mismatch")
                    verified.add(digest)
        frames.append(f)
        previous = f
    if len(frames) < 2:
        raise ValueError("a film needs at least two recorded states")
    return doc, frames


def load_frame(folder, item):
    meshes = {name:{key:base64.b64encode(gzip.decompress((Path(folder)/"blobs"/(digest+".gz")).read_bytes())).decode()
                   for key,digest in fields.items()} for name,fields in item["meshes"].items()}
    return dict(body_q=np.asarray(item["body_q"], dtype=np.float32), meshes=meshes)
