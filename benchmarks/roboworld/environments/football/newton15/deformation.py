"""Retain the legacy contact-owned visual shell mode; never alter rigid state."""

from dataclasses import asdict
from football.ball_visual_deformation import (
    ContactCompressionMode,
    compute_ball_visual_deformation,
)


def shell_state(world):
    modes = []
    foot = world.last.get("foot", {})
    if foot.get("compression_m", 0.0) > 0 and world.foot:
        geometry = world.foot.current_geometry()
        modes.append(
            ContactCompressionMode(
                "foot",
                foot["compression_m"],
                geometry.normal_foot_to_ball_world,
                geometry.compression_m,
            )
        )
    pitch = world.last.get("pitch", {})
    if pitch.get("compression_m", 0.0) > 0:
        modes.append(
            ContactCompressionMode(
                "pitch",
                pitch["compression_m"],
                (0.0, 0.0, 1.0),
                max(0.0, world.ball.radius_m - float(world.poses[world.ball_id, 2])),
            )
        )
    q = world.poses[world.ball_id, 3:]
    result = compute_ball_visual_deformation(
        ball_radius_m=world.ball.radius_m,
        volume_coupling=world.foot.model.prior.volume_coupling if world.foot else 0.22,
        ball_orientation_world_wxyz=tuple(float(v) for v in (q[3], *q[:3])),
        modes=modes,
    )
    return asdict(result)
