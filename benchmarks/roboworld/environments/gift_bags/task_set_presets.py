"""Task-only placements. Never edit material, crease memory, contact or robot settings."""
import numpy as np


def lid_offset(half_y, clearance, wall):
    # The seed-0 robot demo carried the lid 170 mm toward the front. Larger boxes need more space.
    return max(.17, 2*half_y + clearance + wall + .035)


def prepare(spec, task):
    if task == "open-flaps":
        from . import carton as CT, scene as S
        # Same cached carton, same aged/cached hinge state; only the seam-tape prop is absent.
        st = CT.settled(spec.surface_z, S.CARTON_XY)
        spec.sheets = [s for s in spec.sheets if s.name != "seam_tape"]
        spec.ipc_setup = [lambda ipc, world: CT.install_hinges(ipc, st)]
        spec.geometry["task_start"] = dict(task=task, seam_tape=False, hinge_state="sealed carton cache, unchanged")
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
        # Stand the bag on the mat with its factory memory: preserve hinge_state, membrane_set AND crease_rest.
        bag.x0 = X + centre + np.array([0., 0., cal["rest_sheet_rigid"] + 5e-6 - X[:, 2].min()])
        spec.ipc_setup = [f for f in spec.ipc_setup if getattr(f, "gift_bags_hook", None) != "handles"]
        spec.rods = []
        spec.bodies = [b for b in spec.bodies if not b.name.startswith("knot_")]
        board = next(b for b in spec.bodies if b.name == "board_0")
        hz = spec.geometry["board"]["half"][2]
        board.p = centre + np.array([0., 0., 2*cal["rest_sheet_rigid"] + hz + 1e-5])
        board.R = np.eye(3)
        spec.geometry["bag"]["handles"] = False
        spec.geometry["task_start"] = dict(task=task, crease_memory="factory, unchanged", handles=False,
                                            board="board_0 on the bottom panel", cached=False)
