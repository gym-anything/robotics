"""Start of part-id-hdpe: the receiving bench (scene.part) with the 4 L round screw-lid HDPE canister standing on it
(canister_offset: within +-20 mm of its place, seed 0 centred), and the thermal-transfer printer presenting the 2 x 4 in
polyester ID label it has just printed, carrying the canister's part number and the seed's serial number (payload).

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). part-id-hdpe starts from the printer's presented label as the scene
builds it: setup constructs nothing and returns {}. place_items(scene, z) puts the canister on the bench (scene.part
calls it while building the scene, z the bench top); payload(seed) is the barcode payload printed for the seed
(layouts.payload calls it)."""
from label_application.layouts import _rng, _serial

PAYLOAD = "NG-3310-0007-000912"


def payload(seed):
    if seed == 0:
        return PAYLOAD
    return f"NG-3310-0007-{_serial(seed, 7):06d}"


def canister_offset(seed):
    """Canister position offset on the bench (m). Seed 0: none."""
    return (0., 0.) if seed == 0 else tuple(float(v) for v in _rng(seed, 6).uniform(-.02, .02, 2))


def place_items(scene, z):
    """The canister on the bench, at its seeded offset (scene.part)."""
    from label_application import materials as M
    from label_application.labelsim import Collider
    dx, dy = canister_offset(scene.c.seed)
    body = scene.free_body("part", (.10 + dx, -.08 + dy, z+.125))
    scene.cyl_hull("jug_wall", (0, 0, 0), .075, .125, "hdpe_white", body=body, mu=.3, density=950.)
    scene.set_mass(body, 3.2)      # 4 L round screw-lid HDPE canister with contents (authored)
    scene.colliders.append(Collider("part", "cylinder", (.075, .125), M.SURFACES["hdpe"], body=body))
    scene.items["part"] = dict(body=body, kind="hdpe_jug", radius=.075, height=.25)


def setup(world):
    return {}
