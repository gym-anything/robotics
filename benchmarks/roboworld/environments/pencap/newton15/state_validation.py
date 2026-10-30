"""Fail closed on numerical runaway; never clamp, repair, or reset physics.

These deliberately broad tripwires are NOT joint/actuator limits and are not
used as evidence of realistic behavior. Physical fixtures assert much tighter
pose, energy, settling, and contact requirements independently.
"""

import numpy as np


def validate_state(model, state, tick):
    arrays = {name: getattr(state, name).numpy()
              for name in ("body_q", "body_qd", "joint_q", "joint_qd")}
    for name, values in arrays.items():
        bad = np.argwhere(~np.isfinite(values))
        if len(bad):
            index = tuple(bad[0])
            body = model.body_label[index[0]] if name.startswith("body_") else "joint"
            raise FloatingPointError(f"Non-finite {name} for {body} at attempted tick {tick}, index={index}")
    # Avoid overflow in the diagnostic norm itself for enormous finite float32
    # candidates. A warning from the checker must not obscure the original body.
    q, v = arrays["body_q"].astype(float), arrays["body_qd"].astype(float)
    checks = (("position_m", np.max(np.abs(q[:, :3]), axis=1), 100.),
              ("linear_speed_m_s", np.linalg.norm(v[:, :3], axis=1), 100.),
              ("angular_speed_rad_s", np.linalg.norm(v[:, 3:], axis=1), 10000.))
    for metric, values, bound in checks:
        bad = np.flatnonzero(values > bound)
        if len(bad):
            i = int(bad[0])
            raise FloatingPointError(f"Numerical runaway for {model.body_label[i]} at attempted tick {tick}: "
                f"{metric}={float(values[i]):.9g} exceeds diagnostic tripwire {bound}; "
                f"candidate_pose={q[i].tolist()}, candidate_velocity={v[i].tolist()}")
