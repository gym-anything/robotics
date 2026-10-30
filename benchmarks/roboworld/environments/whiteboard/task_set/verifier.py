"""Deterministic final-state pigment judgements; no history/action milestones."""
from functools import lru_cache
import numpy as np
from scipy.ndimage import maximum_filter, distance_transform_edt

from whiteboard.config import COLORS
from whiteboard.bridge_verifier import _tiles, _crop, FAIL_SCORE_SCALE
from .catalogue import TablePlacement, raster
from .mechanical import mechanics, recap

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
    if task.name == "draw-table":
        from .table_verifier import evaluate_table
        answer = evaluate_table(TablePlacement(), ink)
        answer.update(task_id=task.task_id, contract=task.contract + "/draw-table")
        return answer
    colors, weight = visible(ink)
    start, target = targets(task, ink.board, ink.width, ink.height)
    texel = ink.board.writing_width * ink.board.writing_height / (ink.width * ink.height)
    if task.name in ("write-text", "write-in-colour"):
        channel = tuple(COLORS).index("green" if task.name == "write-in-colour" else "black")
        actual = weight * (colors == channel)
        overlap = f1(target == channel, actual, ink)
        other = float((weight * (colors != channel)).sum()) * texel
        clean = clip(1 - other / max(float((target == channel).sum()) * texel, 1e-12))
        return result(task, {"lettering": overlap >= .85, "other_colors_clean": other <= AREA_ALLOWANCE},
                      overlap * clean, factors={"f1": overlap, "color_cleanliness": clean}, other_color_m2=other)
    if state is None:
        raise ValueError("Task requires current tool poses, velocities, robot contacts and pelvis height")
    if task.name == "recap-and-tidy":
        checks, progress, detail = recap(config, task, state)
        area = float(weight.sum()) * texel
        checks["board_clean"] = area <= AREA_ALLOWANCE
        clean = clip(1 - max(0., area - AREA_ALLOWANCE) / .001)
        return result(task, checks, progress * clean, visible_m2=area, **detail)
    names = ("eraser", "marker_black") if task.name == "correct-misspelling" else ("eraser",)
    checks, return_quality, tray = mechanics(config, state, names)
    # Returning a tool refines completed board work; untouched tool placement earns nothing.
    tool_factor = .9 + .1 * return_quality
    if task.name in ("erase-board", "erase-except-boxed"):
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
    # SCIENSE -> SCIENCE: sixth glyph. Each neighbour gets its own retention gate.
    start_strokes, final_strokes = task.strokes(True), task.strokes(False)
    wrong = raster(tuple(s for s in start_strokes if s[0] == "letter 5"), ink.board, ink.width, ink.height) == 0
    corrected = raster(tuple(s for s in final_strokes if s[0] == "letter 5"), ink.board, ink.width, ink.height) == 0
    # Non-shared wrong ink measured outside the corrected letter's 2 mm band.
    sampling = (ink.board.writing_height / ink.height, ink.board.writing_width / ink.width)
    allowed = distance_transform_edt(~corrected, sampling=sampling) <= .002
    unwanted = wrong & ~allowed
    residual = float((weight * unwanted).sum()) * texel
    removed = clip(1 - residual / max(float(unwanted.sum()) * texel, 1e-12))
    # A glyph-local crop excludes neighbouring letters from its precision denominator.
    cell = distance_transform_edt(~(wrong | corrected), sampling=sampling) <= .012
    overlap = f1(corrected, weight * (colors == 0) * cell, ink)
    # Start already shares strokes with the desired glyph: credit only its improvement.
    f0 = f1(corrected, ((start == 0) & cell).astype(float), ink)
    improved = clip((overlap - f0) / max(1 - f0, 1e-12))
    retention = {}
    for i in (0, 1, 2, 3, 4, 6):
        mask = raster(tuple(s for s in final_strokes if s[0] == f"letter {i}"), ink.board, ink.width, ink.height) == 0
        retention[str(i)] = float((weight * (colors == 0) * mask).sum() / max(mask.sum(), 1))
    keep_factor = min(1., min(retention.values()) / .9)
    word_band = distance_transform_edt(target != 0, sampling=sampling) <= .002
    stray = float((weight * ((colors != 0) | ~word_band)).sum()) * texel
    cleanliness = clip(1 - max(0., stray - AREA_ALLOWANCE) / max(float((target == 0).sum()) * texel, 1e-12))
    checks.update(wrong_letter_removed=residual <= .5e-4, corrected_letter=overlap >= .85,
                  neighbours_kept=all(v >= .9 for v in retention.values()), rest_clean=stray <= AREA_ALLOWANCE)
    return result(task, checks, (.5 * removed + .5 * improved) * keep_factor * cleanliness * tool_factor,
                  factors={"removed": removed, "glyph_f1": overlap, "initial_glyph_f1": f0,
                           "improved_glyph": improved, "keep": keep_factor, "cleanliness": cleanliness,
                           "tool_return": tool_factor}, retention=retention, residual_m2=residual, stray_m2=stray, tray=tray)
