"""Start of open-flaps: the supply carton on the receiving table with its four top flaps closed but not taped:
open-carton's sealed carton (the same cached settle, the same aged hinges) without its seam tape.

setup(spec, contract) changes the receiving preset's scene spec (scene.SceneSpec of open-carton's world) in place
before the contact world is built. The task-set env adapters call it through task_set_presets.prepare
(gift-bags-tasks-v1: start_v1) and task_set_v2_presets.prepare (gift-bags-tasks-v2: start_v2, which also names the
tape's stock as the world's contact stock so that the world builds without a sheet)."""


def setup(spec, contract):
    from gift_bags.task_set_catalog import CONTRACT as V1
    from gift_bags.task_set_v2_catalog import CONTRACT as V2
    {V1: start_v1, V2: start_v2}[contract](spec, "open-flaps")


def start_v1(spec, task):
    from gift_bags import carton as CT, scene as S
    # Same cached carton, same aged/cached hinge state; only the seam-tape prop is absent.
    st = CT.settled(spec.surface_z, S.CARTON_XY)
    spec.sheets = [s for s in spec.sheets if s.name != "seam_tape"]
    spec.ipc_setup = [lambda ipc, world: CT.install_hinges(ipc, st)]
    spec.geometry["task_start"] = dict(task=task, seam_tape=False, hinge_state="sealed carton cache, unchanged")


def start_v2(spec, task):
    from gift_bags import carton as CT, scene as S
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
