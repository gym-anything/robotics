"""Privileged, exact collision/topology catalogue; never policy observations."""
from dataclasses import asdict
from pencap.newton15.catalog import Catalog as RigidCatalog

class Catalog(RigidCatalog):
    def __init__(self,env):
        super().__init__(env);self.env=env
        c=env.config;b=env.builder
        for a in env.assembly.surfaces:
            first=a['first_triangle'];last=first+len(a['triangles'])
            item=dict(id=a['name'],name=a['name'],type='DEFORMABLE_THIN_SHELL',
                material='aluminium' if a['kind']=='can' else 'estimated LLDPE film',
                first_particle=a['first'],particle_count=a['count'],
                triangles=a['triangles'].tolist(),rest_vertices_world_m=a['rest_vertices'].tolist(),
                mass_kg=a['mass_kg'],particle_contact_radius_m=c.contact_radius,
                self_contact_distance_m=2*c.contact_radius,
                elastic_face_parameters=b.tri_materials[first:last],
                elastic_face_parameter_units=['shear_N_m','area_N_m','damping','unused','unused'],
                current_vertices='Privileged state surfaces or action trace particle_q',
                plasticity='Yield-bounded rest-curvature history' if a['kind']=='can' and c.plasticity else 'None')
            if a['kind']=='can':
                item.update(target=a['target'],initial_source=a['origin'],can_type=a['can_type'])
            else:
                item.update(open_mouth_vertices=a['mouth'].tolist(),
                    retention='Installed finite-compliance bag locks with equal/opposite frame reactions')
            self.objects[a['name']]=item

    def index(self):
        result=super().index();result.pop('pen_frame',None)
        result.update(inventory=self.env.assembly.objects,config=asdict(self.env.config),
            physical_scope='Dry opened cans, deformable bag, modest shell denting; no liquids, puncture or fracture',
            source='Actual model geometry and material data; not a camera-derived estimate')
        return result
