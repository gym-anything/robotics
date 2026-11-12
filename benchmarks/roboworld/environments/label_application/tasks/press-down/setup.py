"""Start of press-down: the shipping pack bench with the sealed carton at its seeded place and turn, and the 4 x 6 in
shipping label already laid on the carton's top by a co-worker: flat, adhesive down and untensioned, on the flap half
toward the robots inside the carrier zone, centre within +-30 mm along the carton and +-5 mm across the zone's middle,
turned up to 2 deg (laid_label). It tacks on under its own physics during the settle and nothing holds it afterwards; it
carries the seed's tracking number (payload).

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). setup lays the label on the carton (lay_label_on_carton) and returns
{}. laid_label(seed, carton_W) is the seeded place, which scene.parcel keeps as task_geometry["laid_label"];
payload(seed) is the barcode payload printed for the seed (layouts.payload calls it)."""
import math
import numpy as np
from label_application.layouts import IN, _rng

PAYLOAD = "961201950155501234"


def payload(seed):
    if seed == 0:
        return PAYLOAD
    r = _rng(seed, 1)
    return "9612019" + "".join(str(d) for d in r.integers(0, 10, 11))           # 18-digit tracking number


def laid_label(seed, carton_W=12*IN, tape_half=.024, margin=5e-3, label_w=4*IN):
    """Where the co-worker laid the 4 x 6 in label on the carton top for press-down (carton coordinates): on the
    flap half toward the robots, inside the carrier zone, centre within +-30 mm along the carton and +-5 mm across
    the zone's middle, turned up to 2 deg."""
    r = _rng(seed, 4)
    zone = (tape_half + carton_W/2 - margin)/2
    return dict(centre=(float(r.uniform(-.03, .03)), float(zone + r.uniform(-.005, .005))),
                yaw=float(math.radians(r.uniform(-2., 2.))))


def lay_label_on_carton(world):
    """press-down START STATE (constructed, not a robot action): the label as a co-worker left it -- laid flat,
    adhesive down and untensioned, on the carton top at the seeded position inside the carrier zone of the flap
    half toward the robots (laid_label), touching, so it tacks on under its own physics during the
    settle. Nothing holds it afterwards. Same placement as the assisted demos' fixture (demos/assisted_apply.place)."""
    lab = world.label
    g = world.scene.task_geometry["carton"]
    lay = world.scene.task_geometry["laid_label"]
    p, R = world.body_pose("carton")
    (uc, vc), yaw = lay["centre"], lay["yaw"]
    c, s_ = np.cos(yaw), np.sin(yaw)
    md = R @ np.array([c, s_, 0.])                  # label length (6 in) along the carton
    cd = R @ np.array([-s_, c, 0.])
    centre = p + R @ np.array([uc, vc, g["H"]/2])
    n = np.cross(md, cd)
    X = lab.X
    lab.x = (centre[None, :] + (X[:, 0:1] - lab.length/2)*md[None, :] + (X[:, 1:2] - lab.width/2)*cd[None, :]
             + n[None, :]*(.5*lab.stock.thickness))
    lab.v[:] = 0.
    lab.bond[:] = -1; lab.phi[:] = 0.; lab.damage[:] = 0.; lab.gas[:] = 0.
    lab.theta_rest[:] = 0.
    world.label_history.clear()


def setup(world):
    lay_label_on_carton(world)
    return {}
