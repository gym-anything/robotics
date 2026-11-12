"""Reference-image tasks and private image-space scoring; no action trajectories."""

from dataclasses import dataclass
from functools import lru_cache
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import maximum_filter
from gym_anything_robotics import task_folders
from .config import COLORS
from .stroke_font import GLYPHS

BRIDGE_INSTRUCTION = (
    "Copy the supplied colour-coded diagram of the Golden Gate Bridge onto the whiteboard, using the "
    "markers in the tray: orange towers, red cables and hangers, black roadway and blue water. The "
    "reference image shows the whole writing surface as seen from the front; draw the diagram where "
    "it appears and at the size shown."
)
PACKAGE = "whiteboard"


def task_setup(name):
    """The benchmark task's setup.py (tasks/<name>/), or None. A native task's folder is named by its kind (bridge), a
    task-set preset's by its name; the text kind is not a benchmark task."""
    return task_folders.setup(PACKAGE, name) if task_folders.exists(PACKAGE, name) else None


def task_verifier(name):
    """The benchmark task's verifier.py (tasks/<name>/), or None (as task_setup)."""
    return task_folders.verifier(PACKAGE, name) if task_folders.exists(PACKAGE, name) else None


@lru_cache(maxsize=4)
def reference_png(task, board, width, height):
    """PNG bytes of the public picture supplied with the task (Task.reference_view)."""
    import io
    from .ink import InkSurface
    from .config import MaterialPriors

    buf = io.BytesIO()
    Image.fromarray(task.reference_view(InkSurface(board, width, height, MaterialPriors()))).save(buf, format="PNG")
    return buf.getvalue()


@lru_cache(maxsize=8)
def reference_channels(specification, board, width, height):
    """Cache only immutable public targets, never actual pigment or a task score.

    Eight byte maps bound the default-resolution cache to 64 MiB. A changed
    task, board or image size has a different key; callers cannot mutate maps.
    """
    from .ink import InkSurface
    from .config import MaterialPriors

    reference = Task(*specification).reference(InkSurface(board, width, height, MaterialPriors()))
    channel_map = np.full((height, width), 255, np.uint8)
    for channel, rgb in enumerate(COLORS.values()):
        color = np.asarray([round(v * 255) for v in rgb], np.uint8)
        channel_map[np.all(reference == color, axis=2)] = channel
    channel_map.flags.writeable = False
    return channel_map


@dataclass(frozen=True)
class Task:
    kind: str = "text"
    text: str = "HELLO"
    center_x: float = 0.0
    center_z: float = 1.38
    width_m: float = 0.70

    def __post_init__(self):
        if self.kind not in ("text", "bridge"):
            raise ValueError("Task kind must be text or bridge")
        if any(not math.isfinite(v) for v in (self.center_x, self.center_z, self.width_m)):
            raise ValueError("Task placement must be finite")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("Reference must contain visible characters")
        if not self.text or len(self.text) > 80 or not 0 < self.width_m <= 1.4:
            raise ValueError("Invalid reference text or width")
        if self.kind == "text" and any(char not in GLYPHS for char in self.text):
            raise ValueError(
                "Reference text supports A-Z, 0-9, spaces, +, -, . and !; case is not silently changed"
            )

    @property
    def description(self):
        if self.kind == "text":
            return f"Use the black marker to write {self.text!r} on the whiteboard, matching the supplied reference image. Leave other board areas clean."
        return BRIDGE_INSTRUCTION

    @property
    def reference_role(self):
        role = "Public desired drawing, not current board state"
        return role + (": the whole writing surface as seen by a person facing the board" if self.kind == "bridge" else "")

    def reference_view(self, ink):
        """The picture handed to whoever does the job.

        Bridge: the board as seen from the front (texture columns grow with world
        +x, which is the viewer's left; its setup.py). Text kind: unchanged texture orientation.
        """
        own = task_setup(self.kind)
        if own is not None:
            return own.reference_view(self, ink)
        return self.reference(ink)

    def validate_board(self, board):
        own = task_setup(self.kind)                 # the bridge's diagram height (its setup.py)
        half_height = (
            own.HALF_HEIGHT if own is not None else 0.5 / (len(self.text) * 1.1 - 0.3)
        ) * self.width_m + 0.005
        if (
            abs(self.center_x) + self.width_m * 0.5 + 0.005 > board.writing_width / 2
            or abs(self.center_z - board.center_z) + half_height > board.writing_height / 2
        ):
            raise ValueError("Task reference lies outside the writable board")

    def reference(self, ink):
        image = Image.new("RGB", (ink.width, ink.height), (246, 246, 246))
        draw = ImageDraw.Draw(image)

        def point(x, y):
            return tuple(
                map(
                    float,
                    ink.uv(
                        self.center_x + x * self.width_m,
                        self.center_z + y * self.width_m,
                    ),
                )
            )

        palette = {c: tuple(round(v * 255) for v in rgb) for c, rgb in COLORS.items()}
        line = max(1, round(0.003 * ink.width / ink.board.writing_width))
        if self.kind == "text":
            # Single-stroke lettering avoids asking a 2–5 mm marker to fill
            # the broad strokes of an enlarged typographic font.
            span = len(self.text) * 1.1 - 0.3
            for index, char in enumerate(self.text):
                for path in GLYPHS[char]:
                    pixels = [
                        point((index * 1.1 + x) / span - 0.5, (y - 0.5) / span) for x, y in path
                    ]
                    draw.line(pixels, fill=palette["black"], width=line, joint="curve")
        else:
            task_setup(self.kind).diagram(draw, point, palette, line)     # the bridge (its setup.py)
        return np.asarray(image)

    def evaluate(self, ink):
        """Bridge: registered-copy verifier (its verifier.py, tasks/bridge/). Text: overlap scorer below."""
        own = task_verifier(self.kind)
        if own is not None:
            return own.verify(self, ink)
        return self.overlap_evaluation(ink)

    def overlap_evaluation(self, ink):
        """Colour-aware overlap scorer: the text kind's verifier."""
        # Compare actual deposited pigment with nearest reference colour; allow
        # 2 mm geometric tolerance. This is an explicit benchmark convention.
        specification = (self.kind, self.text, self.center_x, self.center_z, self.width_m)
        target_channel = reference_channels(specification, ink.board, ink.width, ink.height)
        size = 2 * max(1, round(0.002 * ink.width / ink.board.writing_width)) + 1
        results = {}
        for channel, color in enumerate(COLORS):
            target = target_channel == channel
            actual = np.zeros((ink.height, ink.width), bool)
            for (tx, ty), (pigment, _) in ink.tiles.items():
                h, w = min(128, ink.height - ty * 128), min(128, ink.width - tx * 128)
                actual[ty * 128 : ty * 128 + h, tx * 128 : tx * 128 + w] = (
                    pigment[:h, :w, :, channel].mean(axis=2) > 0.15
                )
            has_target, has_actual = target.any(), actual.any()
            if not has_target and not has_actual:
                continue
            if not has_target or not has_actual:
                precision = recall = 0.0
            else:
                precision = float(
                    (actual & maximum_filter(target, size=size)).sum() / actual.sum()
                )
                recall = float(
                    (target & maximum_filter(actual, size=size)).sum() / target.sum()
                )
            results[color] = {
                "precision": precision,
                "recall": recall,
                "f1": 2 * precision * recall / max(1e-12, precision + recall),
            }
        score = float(np.mean([v["f1"] for v in results.values()])) if results else 0.0
        return {
            "score": score,
            "success": bool(results) and all(v["f1"] >= 0.85 for v in results.values()),
            "colors": results,
            "metric": "color-aware reference overlap, 2 mm tolerance; not semantic OCR",
        }
