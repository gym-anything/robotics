"""Start of library-checkin: three to five returned books stacked on the library desk, each with its library label on
the back cover or inside the front cover (that book lies front cover up, so the cover has to be opened), the empty
returns bin beside them and the scanner in its cradle.

setup(scene) stocks the scene (scene.Scene calls it after the library desk's fixtures) and returns nothing."""
import math

from barcode_scanning import params as P
from barcode_scanning.catalog import SKUS
from barcode_scanning.stock import _open_box, _qmul, library_item_number


def setup(scene):
    from barcode_scanning.scene import quat_axis_angle
    rng = scene.rng
    z = scene.fixtures["counter"]["top"]
    _open_box(scene, "returns_bin", (.34, .24, z), (.40, .32, .16), "tote_grey", wall=.004, mass=.9)
    books = ["K12", "K13", "K12", "K13", "K12"][:3 + scene.c.seed % 3]
    zz = z
    from barcode_scanning.artwork import Symbol, render_symbol
    for k, sku_id in enumerate(books):
        s = SKUS[sku_id]
        w, d, h = s.dims
        # where the library put its item label (seeded): on the back cover near the top, or inside the front cover;
        # returned books come back in either orientation -- the ones labelled inside lie
        # front cover up here, so the cover has to be opened to reach the label
        inside = (k + scene.c.seed) % 3 == 1
        yaw = float(rng.normal(0, .08)) + (math.pi if k % 2 else 0.)
        flip = -math.pi/2 if inside else math.pi/2          # -90: front (-y) up; +90: back (+y) up
        q = _qmul(quat_axis_angle((0, 0, 1), yaw), quat_axis_angle((1, 0, 0), flip))
        name = f"book_{k}"
        body, cover = scene.add_book(name, sku_id, (-.30 + float(rng.normal(0, .01)), .24, zz + d/2 + .0005), q)
        item_no = library_item_number(1234, 5000 + 37*k + scene.c.seed)
        scene.items[name].update(container="stack", library_id=item_no, label_inside_cover=inside)
        sym = Symbol("Codabar", f"A{item_no}B", .25, bar_height_mm=8.5, hri=True)
        _, w_mm, h_mm = render_symbol(sym, 2)
        size = (w_mm/1e3 + .004, h_mm/1e3 + .0055)
        if inside:
            # inner face of the front board (cover frame: hinge at the origin, board along +x, inner face +y)
            centre = (w - .012 - w_mm/2e3, P.book_cover(SKUS[sku_id])[0]/2, h/2 - .018 - h_mm/2e3)
            scene.label(f"{name}_library", cover, sym, centre, (0, 1, 0), (0, 0, 1), size, "matte", kind="library",
                        item=name, text=item_no)
        else:
            centre = (w/2 - .012 - w_mm/2e3, d/2, h/2 - .018 - h_mm/2e3)
            scene.label(f"{name}_library", body, sym, centre, (0, 1, 0), (0, 0, 1), size, "film", kind="library",
                        item=name, text=item_no)
        zz += d + .001
    scene.task_meta = dict(truth={})
