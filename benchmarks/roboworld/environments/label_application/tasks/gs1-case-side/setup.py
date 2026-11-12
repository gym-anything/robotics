"""Start of gs1-case-side: the shipping pack bench with the sealed carton at its seeded place and turn, set further from
the robots than for the carton-top tasks (CARTON_Y), and the desktop printer presenting the 4 x 6 in GS1 logistic label
it has just printed (GS1_LABEL: env.make_art prints the GS1 layout), carrying the seed's SSCC with its GS1 check digit
(payload).

setup(world) gets the task's world (world.World), built and not yet settled, and returns the start information LabelEnv
keeps as start_info (env.prepare_start calls it). gs1-case-side starts from the printer's presented label as the scene
builds it: setup constructs nothing and returns {}. CARTON_Y is read by scene.parcel; payload(seed) is the barcode
payload printed for the seed (layouts.payload calls it); WRIST_CAMERAS: the backend turns the wrist cameras on by
default."""
from label_application.layouts import _rng, gs1_check_digit

# seed 0: the SSCC's GS1 mod-10 check digit is 4
PAYLOAD = "00106141410000004154"
CARTON_Y = -.12         # m: the carton's place across the bench (scene.parcel; the other carton tasks' is -0.06)
GS1_LABEL = True        # the printer prints the GS1 logistic label (artwork.shipping_label gs1)
# wrist cameras on by default (backend): the long side faces (the robot-facing one is not visible overhead at all)
WRIST_CAMERAS = True


def payload(seed):
    if seed == 0:
        return PAYLOAD
    r = _rng(seed, 1)
    data = "1" + "0614141" + "".join(str(d) for d in r.integers(0, 10, 9))      # extension + GS1 prefix + serial
    return "00" + data + gs1_check_digit(data)                                    # AI (00) + SSCC-18


def setup(world):
    return {}
