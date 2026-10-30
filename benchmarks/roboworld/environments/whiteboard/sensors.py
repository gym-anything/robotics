"""Hardware-referenced optical pose. Ideal RGB first; sensor noise is not fabricated."""

from scipy.spatial.transform import Rotation


def optical_pose(env):
    labels = env.model.body_label
    i = next(i for i, name in enumerate(labels) if name.split("/")[-1] == "torso_link")
    q = env.state_0.body_q.numpy()[i]
    torso = Rotation.from_quat(q[3:])
    mount = Rotation.from_euler("xyz", (0, 0.8307767239493009, 0))
    eye = q[:3] + torso.apply((0.0576235, 0.01753, 0.42987))
    # URDF camera-link convention +X forward, +Z up. Optical calibration offset
    # from d435_link to the RGB lens is not published in this asset; zero-offset
    # approximation is a declared model limit, not called exact calibration.
    forward = (torso * mount).apply([1, 0, 0])
    up = (torso * mount).apply([0, 0, 1])
    return eye, eye + forward, up
