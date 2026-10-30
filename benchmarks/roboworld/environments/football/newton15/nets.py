"""Cable net construction; migration back-panel matches retained baseline."""

from pathlib import Path
from dataclasses import replace
import os
import hashlib
import json
from football.football_specs import FieldSpec
from football.net_model import CableNetPatch, CableNetPrior
from football.net_state_cache import settle_or_load
from football.net_vectorized import VectorizedCableNetPatch


def build_nets(layout):
    if layout == "none":
        return {}, {}
    f = FieldSpec()
    if layout == "enclosure":
        return enclosure(f)
    back = CableNetPatch.from_corners(
        bottom_left_world_m=(f.goal_depth_m, -f.half_goal_width_m, 0.03),
        bottom_right_world_m=(f.goal_depth_m, f.half_goal_width_m, 0.03),
        top_left_world_m=(0.82, -f.half_goal_width_m, f.goal_inside_height_m),
        top_right_world_m=(0.82, f.half_goal_width_m, f.goal_inside_height_m),
        prior=CableNetPrior(),
    )
    cache = Path(__file__).resolve().parents[1] / "data/p5_back_net_settled.json"
    info = settle_or_load(back, cache, duration_s=0.20, physics_hz=24_000)
    return {"back": VectorizedCableNetPatch(back)}, {"back": info}


def enclosure(f):
    """Four physical panels attached to the goal's perimeter support rails.

    100 mm mesh is a documented football-net scale (Harrod 4G Portagoal).
    Other constitutive priors are retained, not attributed to that product.
    """
    y, z, d = f.half_goal_width_m, f.goal_inside_height_m, f.goal_depth_m
    corners = {
        "back": ((d, -y, 0.03), (d, y, 0.03), (0.82, -y, z), (0.82, y, z)),
        "left": ((0, -y, 0.03), (d, -y, 0.03), (0, -y, z), (0.82, -y, z)),
        "right": ((d, y, 0.03), (0, y, 0.03), (0.82, y, z), (0, y, z)),
        "roof": ((0, -y, z), (0, y, z), (0.82, -y, z), (0.82, y, z)),
    }
    cache = (
        Path(os.environ.get("FOOTBALL_CACHE", Path.home() / ".cache/football-newton15"))
        / "nets"
    )
    cache.mkdir(parents=True, exist_ok=True)
    nets, info = {}, {}
    for name, c in corners.items():
        prior = replace(CableNetPrior(), mesh_spacing_m=0.10)
        source = CableNetPatch.from_corners(
            bottom_left_world_m=c[0],
            bottom_right_world_m=c[1],
            top_left_world_m=c[2],
            top_right_world_m=c[3],
            prior=prior,
        )
        net = VectorizedCableNetPatch(source)
        signature = hashlib.sha256(
            json.dumps(
                {
                    "corners": c,
                    "prior": vars(prior),
                    "duration": 0.2,
                    "hz": 24000,
                    "schema": 1,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()
        file = cache / (signature + ".npz")
        if file.exists():
            import numpy as np

            with np.load(file, allow_pickle=False) as a:
                net.position[:] = a["position"]
                net.velocity[:] = a["velocity"]
        else:
            for _ in range(4800):
                net.step(ball_center_world_m=None, dt_s=1 / 24000)
            import numpy as np

            np.savez_compressed(file, position=net.position, velocity=net.velocity)
        net.sync_nodes()
        nets[name] = net
        info[name] = {
            "signature": signature,
            "initial_settling_s": 0.2,
            "hz": 24000,
            "cache": str(file),
            "nodes": len(net.nodes),
            "edges": len(net.edges),
        }
    return nets, info
