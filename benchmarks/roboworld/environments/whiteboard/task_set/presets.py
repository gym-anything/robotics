"""Constructed pigment and free-body start poses; no mechanics or policy actions."""
import numpy as np
from scipy.spatial.transform import Rotation

from whiteboard.config import COLORS
from whiteboard.geometry import tools
from .catalogue import raster

OPEN_COLORS = ("black", "red", "blue")
CAP_SEATED_M = .0555
CAP_START_OFFSET_M = .075


def tool_poses(config, task):
    poses = {}
    for i, color in enumerate(config.layout.marker_order):
        # Two rows on the existing 80 mm tray, 35 mm apart; no added holder.
        x = -.40 + (i % 3) * .22
        y = config.board.face_y + (.019 if i < 3 else .054)
        z = config.board.bottom - .015 + .011
        quat = Rotation.from_euler("y", np.pi / 2).as_quat()
        poses[f"marker_{color}"] = np.r_[x, y, z, quat]
        offset = CAP_SEATED_M + (CAP_START_OFFSET_M if task.name == "recap-and-tidy" and color in OPEN_COLORS else 0)
        poses[f"cap_{color}"] = np.r_[x + offset, y, z, quat]
    eraser = next(obj for obj in tools(config) if obj.name == "eraser")
    poses["eraser"] = np.r_[eraser.position, eraser.rotation]
    return poses


def seed_board(ink, task):
    """Pre-existing dry writing, not a call to the marker's deposit law."""
    if ink.tiles or ink.total_deposited_ul or ink.total_removed_ul:
        raise ValueError("Task props must be applied to an untouched ink surface")
    channels = raster(task.strokes(start=True), ink.board, ink.width, ink.height)
    for ty in range((ink.height + 127) // 128):
        for tx in range((ink.width + 127) // 128):
            block = channels[ty*128:(ty+1)*128, tx*128:(tx+1)*128]
            if not (block < len(COLORS)).any():
                continue
            h, w = block.shape
            pigment = np.zeros((128, 128, ink.subcells**2, len(COLORS)), np.float32)
            for c in range(len(COLORS)):
                pigment[:h, :w, :, c] = (block == c)[..., None] * .20
            ink.tiles[tx, ty] = pigment, np.full((128, 128, ink.subcells**2), -600., np.float32)
            ink.dirty.add((tx, ty))
    ink.total_deposited_ul = ink.pigment_volume_ul()
    if ink.tiles:
        ink.revision += 1
    return {"constructed": True, "deposition_time_s": -600., "pigment_ul": ink.total_deposited_ul,
            "marker_reservoirs_consumed": False}


def apply_start(env, task):
    """Only called while constructing an episode, before any integration/action."""
    if env.tick != 0 or env.action_id != 0:
        raise ValueError("Task start presets are tick-zero construction only")
    q = env.state_0.joint_q.numpy().copy()
    qd = env.state_0.joint_qd.numpy().copy()
    for name, pose in tool_poses(env.config, task).items():
        body = env.body_ids[name]
        joint = next(j for j, child in enumerate(env.builder.joint_child) if child == body)
        qs, ds = env.builder.joint_q_start[joint], env.builder.joint_qd_start[joint]
        q[qs:qs+7], qd[ds:ds+6] = pose, 0.
    env.state_0.joint_q.assign(q)
    env.state_0.joint_qd.assign(qd)
    env.newton.eval_fk(env.model, env.state_0.joint_q, env.state_0.joint_qd, env.state_0)
    # Normal reset/restore synchronises the native solver on the next explicit step.
    env._needs_sync = True
    env._previous_contacts.clear()
    env._pad_contact_state = None
    # Rigid translations leave seated caps' constitutive state unchanged.
    # Only separated caps lose their former contact warm start/deformation.
    if task.name == "recap-and-tidy":
        for color in OPEN_COLORS:
            env.deformations.pop(f"cap_{color}", None)
    record = seed_board(env.ink, task)
    env._initial = env.snapshot()
    env.trace.event("task_start_props", 0, task_id=task.task_id, contract=task.contract,
                    pigment=record, tools={k: v.tolist() for k, v in tool_poses(env.config, task).items()})
    return record
