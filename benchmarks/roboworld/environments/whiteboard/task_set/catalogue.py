"""Eight explicit selections. Native classroom-v1 selections resolve through whiteboard.benchmark."""
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json

import numpy as np
from PIL import Image, ImageDraw

from whiteboard.config import COLORS, Config
from whiteboard.stroke_font import GLYPHS

CONTRACT = "whiteboard-tasks-v1"
NAMES = ("erase-board", "write-text", "write-in-colour", "erase-except-boxed",
         "draw-table", "recap-and-tidy", "correct-misspelling", "bridge")
DIFFICULTY = dict(zip(NAMES, ("entry", "entry", "medium", "medium", "medium", "hard", "hard", "hardest")))
# One fixed, reviewable authored start per new task. Native variation selectors remain separate.
SELECTIONS = tuple({"task": name, "contract": CONTRACT} for name in NAMES[:-1]) + ({"variation_id": 0},)
INSTRUCTIONS = {
    "erase-board": "Wipe the whiteboard clean and leave the wiper resting in the tray.",
    "write-text": "Write HELLO on the whiteboard with the black marker, where the supplied reference image shows it and at the size shown. Leave the rest of the board clean.",
    "write-in-colour": "Write WELCOME on the whiteboard in green, where the reference image shows it and at the size shown. Leave the rest of the board clean.",
    "erase-except-boxed": "Erase everything on the board except the homework inside the blue box. Keep the box and its contents as they are, and leave the wiper resting in the tray.",
    "draw-table": "Draw the blank table from the reference image on the whiteboard in blue, where it is shown and at the size shown: the outer frame, the header line and the column lines.",
    "recap-and-tidy": "Put the caps back on the three open markers and leave all six capped markers resting in the tray. Leave the board clean.",
    "correct-misspelling": "Correct the misspelt word on the board to match the reference image, with the corrected letter in black and the other letters left as they are. Leave the marker and wiper resting in the tray.",
}


def table_strokes():
    """Same tuple format as bridge_strokes; metres divided by a 0.60 m width."""
    return tuple((name, "blue", weight, points) for name, weight, points in (
        ("top", "thick", ((-.5, .3), (.5, .3))),
        ("bottom", "thick", ((-.5, -.3), (.5, -.3))),
        ("left", "thick", ((-.5, -.3), (-.5, .3))),
        ("right", "thick", ((.5, -.3), (.5, .3))),
        ("header", "normal", ((-.5, .2), (.5, .2))),
        ("column one", "normal", ((-1/6, -.3), (-1/6, .3))),
        ("column two", "normal", ((1/6, -.3), (1/6, .3))),
    ))


def text_strokes(text, center_x, center_z, width, color="black"):
    span = len(text) * 1.1 - .3
    # World +x is viewer LEFT. Mirror glyph geometry as well as letter order;
    # merely flipping the public image would leave mirror writing on the board.
    return tuple((f"letter {i}", color, "normal", tuple(
        (center_x - ((i * 1.1 + x) / span - .5) * width,
         center_z + (y - .5) / span * width) for x, y in path))
        for i, char in enumerate(text) for path in GLYPHS[char])


def world_table(task):
    return tuple((n, c, w, tuple((task.center_x + x * task.width_m, task.center_z + z * task.width_m)
                               for x, z in p)) for n, c, w, p in table_strokes())


@dataclass(frozen=True)
class CatalogueTask:
    name: str
    contract: str = CONTRACT
    center_x: float = -.05
    center_z: float = 1.28
    width_m: float = .70
    kind: str = "task_set"

    def __post_init__(self):
        if self.name not in INSTRUCTIONS or self.contract != CONTRACT:
            raise ValueError("Unknown whiteboard task/contract")
        if self.kind != "task_set" or (self.center_x, self.center_z, self.width_m) != (-.05, 1.28, .70):
            raise ValueError("These task presets have fixed authored placements")

    @property
    def description(self):
        return INSTRUCTIONS[self.name]

    @property
    def reference_role(self):
        return "Desired final board, whole writing surface as seen from the front"

    @property
    def task_id(self):
        return f"whiteboard/{self.contract}/{self.name}"

    def strokes(self, start=False):
        if self.name == "write-text":
            return () if start else text_strokes("HELLO", self.center_x, self.center_z, .60)
        if self.name == "write-in-colour":
            return () if start else text_strokes("WELCOME", self.center_x, self.center_z, .70, "green")
        if self.name == "draw-table":
            return () if start else world_table(TablePlacement())
        if self.name == "correct-misspelling":
            return text_strokes("SCIENSE" if start else "SCIENCE", self.center_x, self.center_z, .70)
        if self.name == "erase-board":
            return (text_strokes("HELLO", -.05, 1.38, .60) +
                    text_strokes("12345", -.05, 1.15, .60, "blue")) if start else ()
        if self.name == "erase-except-boxed":
            box = (("box", "blue", "normal", ((-.40, 1.12), (-.10, 1.12), (-.10, 1.40), (-.40, 1.40), (-.40, 1.12))),)
            kept = box + tuple(s for word, z in (("HW", 1.34), ("A+B", 1.26), ("123", 1.18))
                               for s in text_strokes(word, -.25, z, .20))
            outside = text_strokes("WORK", .18, 1.34, .38) + text_strokes("1234", .18, 1.18, .38, "red")
            return kept + outside if start else kept
        return ()

    def validate_board(self, board):
        for start in (False, True):
            for _, _, weight, points in self.strokes(start):
                for x, z in points:
                    if abs(x) + .003 > board.writing_width / 2 or not board.bottom + board.trim + .003 <= z <= board.bottom + board.height - board.trim - .003:
                        raise ValueError("Task prop outside board")
                    if abs(x + .05) + .003 > .5 or not .955 <= z - .003 < z + .003 <= 1.48:
                        raise ValueError("Task prop outside documented assisted reach band")

    def reference(self, ink):
        channels = raster(self.strokes(), ink.board, ink.width, ink.height)
        rgb = np.full((ink.height, ink.width, 3), 246, np.uint8)
        for i, values in enumerate(COLORS.values()):
            rgb[channels == i] = [round(v * 255) for v in values]
        return rgb

    def reference_view(self, ink):
        return np.ascontiguousarray(self.reference(ink)[:, ::-1])

    def evaluate(self, ink):
        # Mechanical end states require the adapter's current body/contact state.
        # Fail explicitly rather than awarding an incomplete ink-only success.
        raise RuntimeError("Task-set evaluation requires its GAR task adapter")


@dataclass(frozen=True)
class TablePlacement:
    center_x: float = -.05
    center_z: float = 1.28
    width_m: float = .60


@lru_cache(maxsize=16)
def raster(strokes, board, width, height):
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    for _, color, weight, points in strokes:
        pixels = [((x / board.writing_width + .5) * width,
                   (.5 - (z - board.center_z) / board.writing_height) * height) for x, z in points]
        line = max(1, round((.006 if weight == "thick" else .003) * width / board.writing_width))
        draw.line(pixels, fill=tuple(COLORS).index(color), width=line, joint="curve")
    result = np.asarray(image)
    result.flags.writeable = False
    return result


def preset_config(name, base=None):
    task = CatalogueTask(name)
    base = base or Config()
    task.validate_board(base.board)
    # Keep the frozen layout's valid envelope. Body-only tool placements are applied separately.
    layout = replace(base.layout, robot_x=-.05, eraser_x=.35)
    return replace(base, layout=layout), task


def resolve(selection):
    from whiteboard.benchmark import resolve as legacy
    if "task" not in selection and "contract" not in selection:
        return legacy(selection)
    if set(selection) - {"task", "contract"}:
        raise ValueError("Named presets do not combine with native variation selectors")
    name = selection.get("task")
    if name == "bridge":
        if "contract" in selection:
            raise ValueError("Bridge keeps its native contract; select variation_id 0")
        return legacy({"variation_id": 0})
    if selection.get("contract") != CONTRACT:
        raise ValueError(f"Named tasks require explicit contract {CONTRACT}")
    task = CatalogueTask(name)
    return dict(task_id=task.task_id, description=task.description,
                backend_options={"catalogue_task": name, "task_contract": CONTRACT},
                selection={"task": name, "contract": CONTRACT, "difficulty": DIFFICULTY[name],
                           "start": "constructed task props, no simulated preparation"}, split=None)


def source_identity():
    """Additional task identity, without changing frozen mechanics/provenance inputs."""
    from gym_anything_robotics.paths import package_dir
    board, core = package_dir("whiteboard"), package_dir("gym_anything_robotics")
    paths = {f"whiteboard/task_set/{p.name}": p for p in (board / "task_set").glob("*.py")}
    paths.update({"gym_anything_robotics/backends/newton_whiteboard_task_set.py": core / "backends/newton_whiteboard_task_set.py",
                  "gym_anything_robotics/launch.py": core / "launch.py", "whiteboard/benchmark.json": board / "benchmark.json"})
    files = {name: hashlib.sha256(p.read_bytes()).hexdigest() for name, p in sorted(paths.items())}
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}
