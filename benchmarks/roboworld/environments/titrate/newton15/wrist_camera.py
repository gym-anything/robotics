"""Intel RealSense D405 on the Panda's wrist: policy camera panda_wrist_rgb.

The case and bracket are real mass and collision geometry on panda_link7, with the geometry, masses and optical pose
of the Franka-wrist D405 in the other RoboWorld simulations (drilling/scene.py _wrist_camera). The arm always carries
them, whichever cameras a session observes: laboratory.build_laboratory installs them after every other shape of the
laboratory, so every body, joint and other shape keeps its index.
"""

MOUNT_FRAME = "link7_d405_envelope"
HORIZONTAL_FOV_DEG = 87.
NATIVE_RESOLUTION_PX = (1280, 720)
# (part, centre in the wrist-link frame, full size, mass kg, render material): the D405's published 42 x 42 x 23 mm
# envelope and 60 g, and a 10 g aluminium bracket (engineering estimate); the shapes' own inertia applies.
PARTS = (("camera_case", (.075, 0., .07), (.042, .042, .023), .060, "camera_dark"),
         ("camera_bracket", (.05, 0., .046), (.050, .026, .006), .010, "aluminum"))
# Optical centre 1.5 mm in front of the case, looking along the wrist link's +z with image up along +y.
OPTICS = {"eye": (.075, -.012, .083), "target": (.075, -.012, .283), "up": (0., 1., 0.)}


def camera_name(arm):
    return f"{arm}_wrist_rgb"


def install(lab):
    """Add the D405 case and bracket to the wrist link of every robot arm (none in a laboratory without the robot);
    record the cameras' mounts as lab.camera_mounts (name: body, eye, target and up in the body frame, horizontal
    FOV)."""
    mounts = {}
    for arm, handle in getattr(lab.builder, "_gar_robots", {}).items():
        body = handle.frames["wrist"]["body"]
        for part, centre, size, mass, material in PARTS:
            lab.box(f"{arm}_{part}", centre, size, material, body=body, density=mass/(size[0]*size[1]*size[2]))
        mounts[camera_name(arm)] = dict(body=body, **OPTICS, horizontal_fov_deg=HORIZONTAL_FOV_DEG)
    lab.camera_mounts = mounts
    return mounts
