"""Outcome of correct-misspelling: SCIENSE corrected to SCIENCE, the wrong sixth letter gone and a black C in its
place, the other letters kept and the rest of the board clean; the wiper and the black marker resting in the tray and
the robot standing clear of them.

verify(task, ink, config, state) gives the verdict of a correct-misspelling preset (task_set/verifier.py's evaluate):
ink is the board's InkSurface, config the episode's Config, state the body state the task-set backends read
(mechanical.read_state: tool poses and velocities, robot contacts, pelvis height; None raises ValueError). It returns
a dict with score, success, progress, checks (wrong_letter_removed, corrected_letter, neighbours_kept, rest_clean,
standing, returned), factors, the neighbours' retention and the tray readings.
"""
from scipy.ndimage import distance_transform_edt

from whiteboard.task_set.catalogue import raster
from whiteboard.task_set.verifier import AREA_ALLOWANCE, board, clip, f1, require_state, result, tool_return


def verify(task, ink, config, state=None):
    colors, weight, start, target, texel = board(task, ink)
    require_state(state)
    checks, tool_factor, tray = tool_return(config, state, ("eraser", "marker_black"))
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
