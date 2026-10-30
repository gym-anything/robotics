"""Exact teacher-only geometry and force-model ownership."""
from pencap.newton15.catalog import Catalog as GeometryCatalog

class Catalog(GeometryCatalog):
    def __init__(self,env):
        super().__init__(env)
        self.inventory=env.assembly.cable_inventory
        rigid=set(env.assembly.rigid_bodies);cables=set(env.assembly.cable_bodies)
        for item in self.objects.values():
            item["electrode"]=env.assembly.electrodes.get(int(item["id"]))
            body=item["body_id"]
            if body<0:
                owner="static boundary shared by native contact solvers"
            elif body in rigid:
                owner="MuJoCo rigid articulation"
            elif body in cables:
                owner="VBD cable/mechanism body"
            else:raise ValueError("Geometry body has no physical solver owner: "+str(body))
            item["solver_ownership"]=owner
            item["contact_model"]=owner+"; cross-owner rigid/cable contacts use ADMM"
            item["parameters_note"]="Authored per-shape values; solver pair mixing applies"

    def index(self):
        result=super().index()
        result.pop("pen_frame")
        result["connector_frames"]="Socket +Z points out; insertion is local -Z. Exact world frames are in shape descriptors."
        result["cable_inventory"]=self.inventory
        return result
