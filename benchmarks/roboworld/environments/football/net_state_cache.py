"""Reproducible cache for expensive, deterministically settled cable states."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

from .net_model import CableNetPatch


def settling_signature(
    net: CableNetPatch,
    *,
    duration_s: float,
    physics_hz: int,
) -> tuple[str, dict[str, Any]]:
    if duration_s <= 0.0 or physics_hz <= 0:
        raise ValueError("net settling duration/rate must be positive")
    specification = {
        "schema": "football-cable-net-settled-state-v1",
        "prior": asdict(net.prior),
        "surface_corners_world_m": net.surface_corners_world_m,
        "node_count": len(net.nodes),
        "edge_count": len(net.edges),
        "duration_s": duration_s,
        "physics_hz": physics_hz,
        "integrator": "CableNetPatch.step semi-implicit Euler",
    }
    canonical = json.dumps(
        specification, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest(), specification


def settle_or_load(
    net: CableNetPatch,
    path: str | Path,
    *,
    duration_s: float,
    physics_hz: int,
) -> dict[str, Any]:
    destination = Path(path)
    signature, specification = settling_signature(
        net, duration_s=duration_s, physics_hz=physics_hz
    )
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        states = payload.get("nodes", [])
        if payload.get("signature_sha256") == signature and len(states) == len(net.nodes):
            for node, state in zip(net.nodes, states):
                node.position_m = tuple(float(value) for value in state["position_m"])
                node.velocity_m_s = tuple(float(value) for value in state["velocity_m_s"])
            return {
                "source": "validated deterministic cache",
                "path": str(destination),
                "signature_sha256": signature,
                "settling_steps": int(round(duration_s * physics_hz)),
            }

    steps = int(round(duration_s * physics_hz))
    for _ in range(steps):
        net.step(ball_center_world_m=None, dt_s=1.0 / physics_hz)
    payload = {
        **specification,
        "signature_sha256": signature,
        "nodes": [
            {
                "position_m": node.position_m,
                "velocity_m_s": node.velocity_m_s,
            }
            for node in net.nodes
        ],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "source": "computed and cached",
        "path": str(destination),
        "signature_sha256": signature,
        "settling_steps": steps,
    }
