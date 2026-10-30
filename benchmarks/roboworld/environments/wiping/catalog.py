"""Exact native geometry plus deformable topology; privileged only."""
from pencap.newton15.catalog import Catalog as RigidCatalog


class Catalog(RigidCatalog):
    def __init__(self,env):
        super().__init__(env);self.env=env
        for a in env.assembly.cloths:
            self.objects[a['name']]=dict(id=a['name'],name=a['name'],type='DEFORMABLE_CLOTH',
                first_particle=a['first'],particle_count=a['count'],triangles=a['triangles'].tolist(),
                material_uv_m=a['uv'].tolist(),dry_mass_kg=env.config.cloth_dry_mass,
                contact_radius_m=env.config.cloth_radius,collision='native VBD edge/face/vertex and self contact',
                parameters=dict(shear_n_m=25.,area_n_m=120.,bending_n_m=2e-5),
                current_vertices='privileged state cloths, or action trace particle_q')
        for item in self.objects.values():
            item['cloth_contact_pair']=dict(stiffness_n_m=env.config.contact_ke,
                damping_n_s_m=env.config.contact_kd,friction='geometric mean of cloth and shape friction')

    def index(self):
        value=super().index();value.pop('pen_frame',None)
        value['inventory']=self.env.assembly.objects
        value['soil_model']='contact-conditioned conservative thin film and porous-cloth reservoirs'
        return value
