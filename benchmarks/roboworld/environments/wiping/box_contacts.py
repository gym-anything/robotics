"""Opt-in Newton1.5 native analytic BOX/BOX manifold routing.

The retained startup poses expose millimetre generic MPR witnesses for boxes
whose independently checked overlap is tens of micrometres. Use Newton's
existing analytic box routine for actual boxes only. All other pairs keep the
original factory, support map, postprocessing, writer and launch geometry.
No installed module or global is patched; this pipeline gets a local factory.
"""
import hashlib
import importlib.metadata
from pathlib import Path
from types import FunctionType
from typing import Any
from functools import partial
import numpy as np
import warp as wp
from newton._src.geometry import narrow_phase as np_native
from newton._src.geometry.collision_core import create_find_contacts as native_find
from newton._src.geometry.collision_primitive import collide_box_box
from newton._src.geometry.contact_data import ContactData
from newton._src.geometry.support_function import GenericShapeData
from newton._src.geometry.types import GeoType
from newton._src.sim.collide import write_contact,ContactWriterData,_shape_collide_mask
from .stable_box_manifold import collide_box_box_stable


@wp.func
def write_native_analytic_box_contact(data:ContactData,writer:ContactWriterData,index:int):
    # Distinct name gives the native kernel factory a distinct Warp module.
    write_contact(data,writer,index)


@wp.func
def write_stable_analytic_box_contact(data:ContactData,writer:ContactWriterData,index:int):
    write_contact(data,writer,index)


def create_find_contacts(writer_func:Any,support_func:Any=None,post_process_contact:Any=None,*,stable=False):
    original=native_find(writer_func,support_func=support_func,post_process_contact=post_process_contact)
    box_query=collide_box_box_stable if stable else collide_box_box

    @wp.func
    def find(pos_a:wp.vec3,pos_b:wp.vec3,quat_a:wp.quat,quat_b:wp.quat,
             shape_data_a:GenericShapeData,shape_data_b:GenericShapeData,
             plane_a:bool,plane_b:bool,radius_a:float,radius_b:float,
             gap:float,shape_a:int,shape_b:int,margin_a:float,margin_b:float,writer:Any):
        if shape_data_a.shape_type==GeoType.BOX and shape_data_b.shape_type==GeoType.BOX:
            distances,points,normals=box_query(pos_a,wp.quat_to_matrix(quat_a),shape_data_a.scale,
                pos_b,wp.quat_to_matrix(quat_b),shape_data_b.scale,gap+margin_a+margin_b)
            for i in range(8):
                if distances[i]<1.0e6:
                    data=ContactData()
                    data.contact_point_center=points[i]
                    data.contact_normal_a_to_b=normals[i]
                    data.contact_distance=distances[i]
                    data.shape_a=shape_a;data.shape_b=shape_b
                    data.margin_a=margin_a;data.margin_b=margin_b
                    data.gap_sum=gap;data.sort_sub_key=i
                    writer_func(data,writer,-1)
        else:
            original(pos_a,pos_b,quat_a,quat_b,shape_data_a,shape_data_b,plane_a,plane_b,
                     radius_a,radius_b,gap,shape_a,shape_b,margin_a,margin_b,writer)
    return find


def install(pipeline,model,*,stable=False):
    if importlib.metadata.version('newton')!='1.5.0':
        raise RuntimeError('Analytic box routing requires audited Newton1.5.0')
    factory=np_native.create_narrow_phase_kernel_gjk_mpr
    local_globals=dict(factory.__globals__,create_find_contacts=partial(create_find_contacts,stable=stable))
    local_factory=FunctionType(factory.__code__,local_globals,factory.__name__,factory.__defaults__,factory.__closure__)
    local_factory.__kwdefaults__=factory.__kwdefaults__
    types=model.shape_type.numpy()
    colliding=set(types[_shape_collide_mask(model,len(types))].tolist())
    lean=not bool(colliding & {int(GeoType.CAPSULE),int(GeoType.ELLIPSOID),int(GeoType.CYLINDER),int(GeoType.CONE)})
    kwargs=dict(support_func=np_native.support_map_lean,post_process_contact=np_native.post_process_minkowski_only) if lean else {}
    expected='support_map_lean_post_process_minkowski_only' if lean else 'default_default'
    old=pipeline.narrow_phase.narrow_phase_kernel.module.name
    if not old.endswith(expected) or 'write_contact_' not in old:
        raise RuntimeError(f'Unexpected native kernel variant: {old}')
    pipeline.narrow_phase.narrow_phase_kernel=local_factory(
        pipeline.narrow_phase.external_aabb,write_stable_analytic_box_contact if stable else write_native_analytic_box_contact,**kwargs)
    root=Path(np_native.__file__).parent
    pipeline.wiping_box_contact_manifest=dict(newton='1.5.0',previous_kernel=old,lean=lean,stable_face_ties=stable,
        stable_source_sha256=hashlib.sha256(Path(__file__).with_name('stable_box_manifold.py').read_bytes()).hexdigest() if stable else None,
        native_sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                        ('narrow_phase.py','collision_core.py','collision_primitive.py','contact_data.py')},
        scope='Only actual BOX/BOX pairs; native analytic SAT/manifold, unchanged native contact writer.')
    return pipeline.wiping_box_contact_manifest
