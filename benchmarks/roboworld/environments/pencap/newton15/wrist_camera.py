"""Intel RealSense D405 wrist camera on a Panda's link7, built only for a session that selects it.

Used by pencap, lockkey and laundry. The case, the bracket and the optical frame are the Franka wrist mount of the
other RoboWorld simulations (drilling/scene.py _wrist_camera): the D405's published 42 x 42 x 23 mm envelope and 60 g
beside the flange on a 10 g aluminium bracket (an estimate), both collision geometry whose mass and inertia join
link7, and an ideal RGB pinhole at the case front looking along link7 +z. A backend constructs its environment inside
`selected`; the Panda builder then calls `install` for each arm, which adds nothing to an arm whose camera the session
does not select, so such an arm is built exactly as without this module.
"""

from contextlib import contextmanager
from contextvars import ContextVar
import math

NAMES = ("left_wrist_rgb", "right_wrist_rgb")
HFOV_DEG = 87.
NATIVE_RESOLUTION_PX = (1280, 720)
MOUNT_FRAME = "link7_d405_envelope"
# Optical centre, a point on the optical axis and the image up direction in link7.
EYE, TARGET, UP = (.075, -.012, .083), (.075, -.012, .283), (0., 1., 0.)
# Label suffix, centre and half extents in link7 (m), density (kg/m^3) for the part's mass, render material.
PARTS = (("camera_case", (.075, 0., .07), (.021, .021, .0115), .060/(.042*.042*.023), "robot_dark"),
         ("camera_bracket", (.05, 0., .046), (.025, .013, .003), .010/(.050*.026*.006), "aluminum"))

_selected = ContextVar("wrist_cameras", default=None)


@contextmanager
def selected(cameras):
    """Environments built inside this block carry the wrist cameras `cameras`, {robot instance: camera name}."""
    token = _selected.set(dict(cameras))
    try:
        yield
    finally:
        _selected.reset(token)


def install(builder, arm, cfg, materials):
    """Add the D405 case and bracket to link7 of the robot instance `arm` (built by robots.backends.newton.add_robot)
    when the session selects its wrist camera, with the contact settings `cfg`; record each part's render material in
    `materials` and the camera's link7 body in builder.wrist_cameras. Add nothing otherwise."""
    name = (_selected.get() or {}).get(arm)
    if name is None:
        return
    import warp as wp
    body = builder._gar_robots[arm].frames["wrist"]["body"]
    for label, center, half, density, material in PARTS:
        part = cfg.copy()
        part.density = density
        shape = builder.add_shape_box(body, xform=wp.transform(center, wp.quat_identity()), hx=half[0], hy=half[1],
                                      hz=half[2], cfg=part, label=f"{arm}_{label}")
        materials[shape] = material
    if not hasattr(builder, "wrist_cameras"):
        builder.wrist_cameras = {}
    builder.wrist_cameras[name] = body


def export(stage, builder):
    """USD cameras of the installed wrist cameras: each under its link7 body prim (/Lab/body_<i>, posed from the
    simulated state) and a /Cameras/<name> entry naming it (mounted_camera_path), which the render workers resolve."""
    from pxr import Gf, UsdGeom
    for name, body in getattr(builder, "wrist_cameras", {}).items():
        path = f"/Lab/body_{body}/{name}"
        camera = UsdGeom.Camera.Define(stage, path)
        camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*EYE), Gf.Vec3d(*TARGET),
                                                            Gf.Vec3d(*UP)).GetInverse())
        camera.CreateHorizontalApertureAttr(36.)
        camera.CreateVerticalApertureAttr(20.25)
        camera.CreateFocalLengthAttr(18./math.tan(math.radians(HFOV_DEG/2)))
        camera.CreateClippingRangeAttr(Gf.Vec2f(.001, 30.))
        camera.GetPrim().SetCustomDataByKey("observation_role", "policy")
        UsdGeom.Xform.Define(stage, "/Cameras/"+name).GetPrim().SetCustomDataByKey("mounted_camera_path", path)


def spec(name, width, height):
    """The policy CameraSpec of wrist camera `name` rendered at width x height."""
    from gym_anything_robotics.types import CameraSpec
    focal = .018/math.tan(math.radians(HFOV_DEG/2))
    return CameraSpec(name, MOUNT_FRAME, width, height, *NATIVE_RESOLUTION_PX, (.036, .036*height/width),
                      (focal, focal))


def note(names):
    """Capability note for the session's wrist cameras `names`."""
    return (f"{', '.join(names)}: Intel RealSense D405 on its Panda's link7 beside the Franka Hand "
            f"({MOUNT_FRAME}), {HFOV_DEG:g} deg HFOV, native {NATIVE_RESOLUTION_PX[0]}x{NATIVE_RESOLUTION_PX[1]}, "
            "ideal pinhole RGB; no depth or noise model. A selected wrist camera adds the D405's 60 g case and its "
            "10 g bracket to that arm as collision geometry and mass; an arm whose wrist camera is not selected is "
            "built without them")
