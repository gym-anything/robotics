"""gift-bags-tasks-v2 start states: task_set_presets.py with the render-check fixes (open-flaps contact stock,
bag-candle's worked creases). Task-only placements and start states. Never edit materials, crease laws, contact,
solver or robot settings."""
import numpy as np


def lid_offset(half_y, clearance, wall):
    # The seed-0 robot demo carried the lid 170 mm toward the front. Larger boxes need more space.
    return max(.17, 2*half_y + clearance + wall + .035)


def prepare(spec, task):
    if task == "open-flaps":
        from . import carton as CT, scene as S
        # Same cached carton, same aged/cached hinge state; only the seam-tape prop is absent.
        st = CT.settled(spec.surface_z, S.CARTON_XY)
        tape = next(s for s in spec.sheets if s.name == "seam_tape")
        # The tape is the receiving world's only sheet, and IPCWorld takes the world's one contact caliper from its
        # thickest sheet stock (the tape's two-ply tab, carton.caliper()). Without the strip the world has no sheet,
        # so name that same stock as the caliper stock, as the unwrapped home preset does (SceneSpec.contact_stock):
        # every contact distance and the barrier stiffness stay exactly open-carton's. Without it the world refuses
        # to build ("a contact world needs a sheet or a contact_stock for its caliper").
        spec.contact_stock = max(tape.mats, key=lambda m: m.t)
        spec.sheets = [s for s in spec.sheets if s is not tape]
        spec.ipc_setup = [lambda ipc, world: CT.install_hinges(ipc, st)]
        spec.geometry["task_start"] = dict(task=task, seam_tape=False, hinge_state="sealed carton cache, unchanged",
                                           contact_stock=spec.contact_stock.name)
    elif task == "put-lid-on":
        bodies = {b.name: b for b in spec.bodies}
        base, lid, g = bodies["gift_base"], bodies["gift_lid"], spec.geometry["box"]
        origin = base.p - base.R @ base.render["com_offset"]
        offset = lid_offset(g["hy"], g["clearance"], g["wall"])
        lid.p = origin + base.R @ np.array([0., -offset, 0.]) + lid.R @ lid.render["com_offset"]
        low = float((np.asarray(lid.verts) @ lid.R.T + lid.p)[:, 2].min())
        lid.p[2] += spec.surface_z + .00005 - low
        spec.geometry["task_start"] = dict(task=task, lid_forward_m=offset)
    elif task == "bag-candle":
        from .ipcworld import contact_caliper
        bag = spec.sheets[0]
        centre = np.asarray(spec.geometry["mat"]["center"], float)
        cal = contact_caliper(bag.mats[0])
        X = np.asarray(bag.X2, float)
        # The spec's start (demos/robot_bag_lift.stand_open, probes/bag_stand.py "worked"): the tote stands in its open
        # manufactured shape with its creases worked open, i.e. the crease lines' plastic rest is the open shape's
        # (shipping-state "hinge_state" hook dropped, crease_rest cleared), the state of a bag a clerk has opened. The
        # B3 candle release this task rests on was made on such a tote; with the factory memory kept, the mouth only
        # reached 0.861 against 1.000 worked (REPORT §6.7). Materials, crease laws and the membrane/slack history are
        # unchanged.
        bag.x0 = X + centre + np.array([0., 0., cal["rest_sheet_rigid"] + 5e-6 - X[:, 2].min()])
        bag.crease_rest = {}
        dropped = ("hinge_state", "handles")
        spec.ipc_setup = [f for f in spec.ipc_setup if getattr(f, "gift_bags_hook", None) not in dropped]
        spec.rods = []
        spec.bodies = [b for b in spec.bodies if not b.name.startswith("knot_")]
        board = next(b for b in spec.bodies if b.name == "board_0")
        hz = spec.geometry["board"]["half"][2]
        board.p = centre + np.array([0., 0., 2*cal["rest_sheet_rigid"] + hz + 1e-5])
        board.R = np.eye(3)
        spec.geometry["bag"]["handles"] = False
        spec.geometry["task_start"] = dict(task=task, crease_memory="worked open (spec; stand_open)", handles=False,
                                            board="board_0 on the bottom panel", cached=False)
