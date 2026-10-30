# Adapted from MuJoCo-Warp forward.py's implicit integration stages.
# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Narrow compatibility correction for MuJoCo-Warp 3.11 inverse-dynamics sign.

RNE returns the derivative of bias force c(v), which is SUBTRACTED from the
equations of motion: M*a = applied - c(v). The implicit matrix therefore needs
M - h*D(applied) + h*D(c), not minus h*D(c). The shipped IMPLICIT path uses
flg_subtract=True. Its measured first free-body update matches that wrong sign.

This subclass changes that one sign in the native implicit path. No package
files or global functions are patched; other environments are unaffected.
"""

from importlib.metadata import version
import newton
import warp as wp
from mujoco_warp._src import derivative, forward, smooth
from mujoco_warp._src.types import IntegratorType


class CorrectedImplicitSolver(newton.solvers.SolverMuJoCo):
    def __init__(self, *args, **kwargs):
        if newton.__version__ != "1.5.0" or version("mujoco-warp") != "3.11.0":
            raise RuntimeError("Re-audit the implicit RNE sign correction for this engine version")
        super().__init__(*args, **kwargs)
        if self.mjw_model.opt.integrator != IntegratorType.IMPLICIT:
            raise ValueError("Sign correction requires the full implicit integrator")
        # Newton imports the URDF's finger2 mimic as a MuJoCo equality, whose
        # compliant default allows independent rail motion on impact. The real
        # single-drive linkage is rigid.
        response = (2.0 * self.mj_model.opt.timestep, 1.0)
        if self.mjc_eq_to_newton_mimic is not None:
            mimic_joints = self.mjc_eq_to_newton_mimic.numpy()[0]
            selected = [i for i, joint in enumerate(mimic_joints) if joint >= 0]
            self.mj_model.eq_solref[selected] = response
            self.mj_model.eq_solimp[selected, :2] = .9999
            solref = self.mjw_model.eq_solref.numpy()
            solref[:, selected] = response
            self.mjw_model.eq_solref.assign(solref)
            impedance = self.mjw_model.eq_solimp.numpy()
            impedance[:, selected, :2] = .9999
            self.mjw_model.eq_solimp.assign(impedance)
        labels_by_dof = {
            int(dof): label
            for label, dof in zip(
                self.model.joint_label, self.model.joint_qd_start.numpy()
            )
        }
        joint_map = self.mjc_jnt_to_newton_dof.numpy()[0]
        selected_limits = [
            i
            for i, dof in enumerate(joint_map)
            if labels_by_dof.get(int(dof), "").endswith(
                ("panda_finger_joint1", "panda_finger_joint2")
            )
        ]
        # The predictive margin belongs only to the audited upper stop; move
        # the internal lower endpoint down by that margin so q=0 remains the
        # unchanged physical lower stop.
        ranges = self.mj_model.jnt_range
        ranges[selected_limits, 0] -= self.mj_model.jnt_margin[selected_limits]
        self.mj_model.jnt_range[:] = ranges
        warped_ranges = self.mjw_model.jnt_range.numpy()
        warped_ranges[:, selected_limits, 0] -= self.mjw_model.jnt_margin.numpy()[
            :, selected_limits
        ]
        self.mjw_model.jnt_range.assign(warped_ranges)
        # Use the fastest stable native response at the hard rail endpoints.
        self.mj_model.jnt_solref[selected_limits] = response
        limit_ref = self.mjw_model.jnt_solref.numpy()
        limit_ref[:, selected_limits] = response
        self.mjw_model.jnt_solref.assign(limit_ref)

    def _mujoco_warp_step(self):
        m, d = self.mjw_model, self.mjw_data
        # Same stages as upstream forward.step -> forward.implicit(IMPLICIT).
        forward.forward(m, d)
        qH = wp.empty(d.M.shape, dtype=float)
        derivative.deriv_smooth_vel(m, d, qH)
        wp.launch(forward._map_m2d, dim=(d.nworld, m.nD),
            inputs=[m.mapM2D, qH], outputs=[d.qLU])
        derivative.deriv_rne_vel(m, d, d.qLU, flg_subtract=False)
        qacc = wp.empty((d.nworld, m.nv), dtype=float)
        smooth.factor_solve_lu(m, d, d.qLU, qacc, d.efc.Ma)
        forward._advance(m, d, qacc)
