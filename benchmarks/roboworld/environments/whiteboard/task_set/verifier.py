"""Deterministic final-state pigment judgements; no history/action milestones.

Each preset's verdict is its verify(task, ink, config, state) (tasks/<name>/verifier.py), which evaluate calls. Shared
here: the visible pigment and the target rasters (board), the body-state requirement and tool return of the tasks that
use tools (require_state, tool_return), the verdict dict (result), and the judgements two presets share: lettering in
one colour (write-text, write-in-colour) and erasing around kept writing (erase-board, erase-except-boxed).
"""
from functools import lru_cache
import numpy as np
from scipy.ndimage import maximum_filter

from whiteboard.config import COLORS
from whiteboard.bridge_verifier import _tiles, _crop, FAIL_SCORE_SCALE
from whiteboard.tasks import task_verifier
from .catalogue import raster
from .mechanical import mechanics

AREA_ALLOWANCE = 1e-4


def clip(value):
    return float(np.clip(value, 0, 1))


def result(task, checks, progress, **detail):
    checks = {k: bool(v) for k, v in checks.items()}
    success = all(checks.values())
    progress = 1. if success else clip(progress)
    return dict(task_id=task.task_id, contract=task.contract, revision=task.contract,
                score_rule="1 if success else 0.9 x progress", score=1. if success else FAIL_SCORE_SCALE * progress,
                success=success, progress=progress, checks=checks, **detail)


def visible(ink):
    return _crop(_tiles(ink), (0, 0, ink.width, ink.height))


@lru_cache(maxsize=16)
def targets(task, board, width, height):
    final = raster(task.strokes(), board, width, height)
    start = raster(task.strokes(start=True), board, width, height)
    return start, final


def f1(target, actual, ink):
    # Graded visibility, no thresholded count of checks. Keep the native 2 mm window.
    size = 2 * max(1, round(.002 * ink.width / ink.board.writing_width)) + 1
    if not target.any():
        return 0.
    precision = float((actual * maximum_filter(target, size=size)).sum() / max(float(actual.sum()), 1e-12))
    recall = float((maximum_filter(actual, size=size) * target).sum() / target.sum())
    return 2 * precision * recall / max(precision + recall, 1e-12)


def evaluate(task, ink, config, state=None):
    return task_verifier(task.name).verify(task, ink, config, state)


def board(task, ink):
    """Visible colour and weight per texel, the start and final target rasters, and a texel's area (m^2)."""
    colors, weight = visible(ink)
    start, target = targets(task, ink.board, ink.width, ink.height)
    texel = ink.board.writing_width * ink.board.writing_height / (ink.width * ink.height)
    return colors, weight, start, target, texel


def require_state(state):
    if state is None:
        raise ValueError("Task requires current tool poses, velocities, robot contacts and pelvis height")


def tool_return(config, state, names):
    """Standing and tools-returned checks, the tool factor and the tray readings of the named tools."""
    checks, return_quality, tray = mechanics(config, state, names)
    # Returning a tool refines completed board work; untouched tool placement earns nothing.
    tool_factor = .9 + .1 * return_quality
    return checks, tool_factor, tray


def lettering(task, ink, colour):
    """The target lettering in one marker colour (write-text: black, write-in-colour: green); no body state."""
    colors, weight, start, target, texel = board(task, ink)
    channel = tuple(COLORS).index(colour)
    actual = weight * (colors == channel)
    overlap = f1(target == channel, actual, ink)
    other = float((weight * (colors != channel)).sum()) * texel
    clean = clip(1 - other / max(float((target == channel).sum()) * texel, 1e-12))
    return result(task, {"lettering": overlap >= .85, "other_colors_clean": other <= AREA_ALLOWANCE},
                  overlap * clean, factors={"f1": overlap, "color_cleanliness": clean}, other_color_m2=other)


def erasing(task, ink, config, state):
    """The start writing outside the target erased, the target's writing kept, and the wiper returned."""
    colors, weight, start, target, texel = board(task, ink)
    require_state(state)
    checks, tool_factor, tray = tool_return(config, state, ("eraser",))
    keep = target < len(COLORS)
    erase = (start < len(COLORS)) & ~keep
    residual = float((weight * ~keep).sum()) * texel
    erased = clip(1 - residual / max(float(erase.sum()) * texel, 1e-12))
    retention = float((weight * (colors == target) * keep).sum() / max(keep.sum(), 1)) if keep.any() else 1.
    # Endpoint rule: replacement with identical pigment is fine; no action-history ban.
    added = float((weight * ((start == 255) | (colors != start))).sum()) * texel
    checks.update(erased=residual <= AREA_ALLOWANCE, no_added_marks=added <= AREA_ALLOWANCE,
                  kept=retention >= .90)
    progress = erased * min(1., retention/.9) * tool_factor
    return result(task, checks, progress, factors={"erased": erased, "retention": retention,
                  "tool_return": tool_factor}, residual_m2=residual, added_m2=added, tray=tray)
