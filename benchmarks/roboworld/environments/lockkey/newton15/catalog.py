"""Privileged geometry with the active, not merely stored, contact parameters."""

from pencap.newton15.catalog import Catalog as NativeGeometryCatalog


class Catalog(NativeGeometryCatalog):
    def __init__(self, env):
        super().__init__(env)
        refs = env.model.mujoco.solref.numpy()
        impedance = env.model.mujoco.geom_solimp.numpy()
        priority = env.model.mujoco.geom_priority.numpy()
        for item in self.objects.values():
            i = int(item["id"])
            item.pop("stiffness_n_m")
            item.pop("damping_n_s_m")
            item["normal_contact_model"] = {"kind": "native_mujoco_solref",
                "time_constant_s": float(refs[i, 0]),
                "damping_ratio": float(refs[i, 1]),
                "impedance_parameters": impedance[i].tolist(),
                "pair_mixing_priority": int(priority[i]),
                "note": "Numerical rigid-contact regularization, not measured material springs"}

    def index(self):
        result = super().index()
        result.pop("pen_frame")
        result["lock_frame"] = "+X points out of the keyway; insertion is -X; pins slide +Z; plug turns about X"
        return result
