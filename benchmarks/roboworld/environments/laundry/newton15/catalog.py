"""Privileged rigid geometry plus the full constructed deformable mesh."""
import hashlib
from pencap.newton15.catalog import Catalog as RigidCatalog


class Catalog(RigidCatalog):
    def __init__(self, env):
        super().__init__(env)
        top = env.sim.topology
        x, triangles = env.state.particle_q.numpy(), env.sim.inputs.physical_triangles
        asset = hashlib.sha256(x.tobytes()+triangles.tobytes()).hexdigest()
        self.assets[asset] = dict(asset_id=asset, vertices_local_m=x.tolist(), triangles=triangles.tolist(),
            apply_shape_scale=False, coordinates="initial world; current vertices are in /v1/state garment.positions_m",
            particle_mass_kg=top.physical_vertex_mass_kg.tolist(), particle_radius_m=top.physical_vertex_radius_m.tolist(),
            panel_ids=list(top.panel_ids), triangle_panel_index=top.triangle_panel_index.tolist(),
            material_coordinates_m=top.panel_material_coordinates_m.tolist(), panel_triangles=top.panel_triangles.tolist(),
            material_to_physical_vertex=top.material_to_physical_vertex.tolist(),
            material_profile_ids=list(top.material_profile_ids),
            triangle_material_index=top.triangle_material_index.tolist(),
            seams={k:v.tolist() for k, v in top.seam_physical_edges.items()})
        self.objects["garment"] = dict(id="garment", name=env.sim.construction.id, type="DEFORMABLE_SEWN_CLOTH",
            body_id=None, asset_id=asset, visible=True, mass_kg=env.sim.inputs.total_mass_kg)

    def index(self):
        result = super().index()
        result.pop("pen_frame", None)
        result["cloth_frame"] = "World XYZ, +Z up; vertices deform, so no single rigid garment pose"
        return result
