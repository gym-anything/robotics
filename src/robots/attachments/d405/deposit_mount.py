"""Collision-bearing D405-envelope RGB accessories, not task-facing magic eyes."""

import warp as wp


def install(a):
    b = a.builder
    a.camera_mounts = {}
    for arm in b._gar_robots:
        handle = b._gar_robots[arm]
        body = handle.frames["wrist"]["body"]
        first_shape = b.shape_count
        # Outside the146mm-wide WSG base.60g published envelope;10g bracket
        # mass is an engineering estimate. Actual added shapes contribute inertia.
        a.box(
            arm + "_camera_case",
            (0.112, 0, 0.140),
            (0.021, 0.021, 0.0115),
            body=body,
            material="robot_dark",
            cfg=a.shape_config(density=0.060 / (0.042 * 0.042 * 0.023)),
        )
        a.box(
            arm + "_camera_bracket",
            (0.076, 0, 0.108),
            (0.036, 0.013, 0.003),
            body=body,
            material="aluminum",
            cfg=a.shape_config(density=0.010 / (0.072 * 0.026 * 0.006)),
        )
        for y in (-0.012, 0.012):
            shape = b.add_shape_cylinder(
                body,
                xform=wp.transform((0.112, y, 0.152), wp.quat_identity()),
                radius=0.004,
                half_height=0.0005,
                cfg=a.shape_config(density=0.0),
                label=arm + "_camera_lens",
            )
            a.materials[shape] = "robot_dark"
        handle.shape_ids += tuple(range(first_shape, b.shape_count))
        handle.metadata["shape_labels"] += list(b.shape_label[first_shape:])
        a.camera_mounts[arm + "_wrist_rgb"] = dict(
            body=body,
            eye=[0.112, -0.012, 0.153],
            target=[0.112, -0.012, 0.353],
            up=[0, 1, 0],
            horizontal_fov_deg=87.0,
            published_source="https://www.realsenseai.com/products/stereo-depth-camera-d405/",
            sensor_scope="Ideal RGB pinhole; no simulated depth/noise or ground-truth labels",
        )
