"""Privileged simulator-authored geometry index for GAR owner diagnostics."""

import hashlib
import numpy as np


class Catalog:
    def __init__(self,env):
        import newton
        self.objects={}
        self.assets={}
        b,m=env.builder,env.model
        for i in range(b.shape_count):
            source=b.shape_source[i]
            asset=None
            if source is not None and hasattr(source,"vertices"):
                v=np.asarray(source.vertices)
                f=np.asarray(source.indices).reshape(-1,3)
                asset=hashlib.sha256(v.tobytes()+f.tobytes()).hexdigest()
                self.assets[asset]={"asset_id":asset,"vertices_local_m":v.tolist(),
                                    "triangles":f.tolist(),"apply_shape_scale":True}
            self.objects[str(i)]={"id":str(i),"name":b.shape_label[i],
                "body_id":b.shape_body[i],"type":newton.GeoType(b.shape_type[i]).name,
                "pose_local_xyzw":np.asarray(b.shape_transform[i]).tolist(),
                "scale":np.asarray(b.shape_scale[i]).tolist(),"asset_id":asset,
                "visible":bool(b.shape_flags[i]&int(newton.ShapeFlags.VISIBLE)),
                "collides_with_rigids":bool(b.shape_flags[i]&int(newton.ShapeFlags.COLLIDE_SHAPES)),
                "friction":b.shape_material_mu[i],
                "stiffness_n_m":b.shape_material_ke[i],
                "damping_n_s_m":b.shape_material_kd[i]}
        self.bodies=[{"id":i,"name":name,"mass_kg":float(m.body_mass.numpy()[i]),
            "com_local_m":m.body_com.numpy()[i].tolist(),
            "inertia_local_kg_m2":m.body_inertia.numpy()[i].tolist()}
            for i,name in enumerate(m.body_label)]

    def index(self):
        return {"units":"metres, kilograms, seconds",
            "quaternion_order":"xyzw","bodies":self.bodies,
            "geometries":list(self.objects.values()),
            "assets":[{"id":k,"vertex_count":len(v["vertices_local_m"]),
                "triangle_count":len(v["triangles"])} for k,v in self.assets.items()],
            "size_convention":"BOX half extents; CYLINDER radius and half length along local Z; MESH scale"}
