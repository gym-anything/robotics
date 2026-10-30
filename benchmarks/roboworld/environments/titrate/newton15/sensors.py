"""One metric camera frame for physical mount, RGB projection and API metadata."""

import numpy as np
from scipy.spatial.transform import Rotation


def camera_frame(eye, target):
    forward = np.asarray(target, float)-np.asarray(eye, float)
    if not np.isfinite(forward).all() or np.linalg.norm(forward) < 1e-9:
        raise ValueError("Distinct finite camera eye/target required")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, [0., 0., 1.])
    if np.linalg.norm(right) < 1e-8:
        raise ValueError("Camera optical axis is parallel to configured up vector")
    right /= np.linalg.norm(right)
    return Rotation.from_matrix(np.column_stack((right, np.cross(right, forward), -forward)))


def camera_metadata(config, width, height):
    frame = camera_frame(config.camera_eye, config.camera_target)
    return {"name": "bench_rgb", "kind": "mounted_external_rgb", "width": width,
            "height": height, "focal_length_mm": 28., "sensor_width_mm": 36.,
            "sensor_height_mm": 36.*height/width, "position_m": config.camera_eye,
            "quaternion_xyzw": frame.as_quat().tolist(), "camera_axes": "+X right, +Y up, -Z optical",
            "intrinsic_matrix": [[width*28/36, 0., width/2],
                                 [0., width*28/36, height/2], [0., 0., 1.]]}
