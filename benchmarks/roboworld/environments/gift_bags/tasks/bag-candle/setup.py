"""Start of bag-candle: a gift bag standing open on the counter mat with a bottom board on its bottom panel, the
customer's boxed candle on the counter in front of the mat (bag-purchase's world, without the bag's rope handles).

setup(spec, contract) changes the counter preset's scene spec (scene.SceneSpec of bag-purchase's world) in place
before the contact world is built. The task-set env adapters call it through task_set_presets.prepare
(gift-bags-tasks-v1: start_v1, the tote with its factory crease memory) and task_set_v2_presets.prepare
(gift-bags-tasks-v2: start_v2, the tote with its creases worked open)."""
import numpy as np


def setup(spec, contract):
    from gift_bags.task_set_catalog import CONTRACT as V1
    from gift_bags.task_set_v2_catalog import CONTRACT as V2
    {V1: start_v1, V2: start_v2}[contract](spec, "bag-candle")


def start_v1(spec, task):
    from gift_bags.ipcworld import contact_caliper
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


def start_v2(spec, task):
    from gift_bags.ipcworld import contact_caliper
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
