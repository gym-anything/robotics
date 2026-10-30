"""Apply recorded geometry and material fields; no simulator in this process."""
import numpy as np
from laundry.newton15.details import attribute_values
from .render_payload import decode


def apply_recorded_state(stage,request):
    from pxr import Gf,Sdf,Vt
    # Resolve every authored attribute before ONE change notice. Previously
    # points changed in the geometry helper before their topology/primvars;
    # Hydra observed inconsistent array lengths on changing runoff/film meshes.
    values=attribute_values(request.get('details',{}))
    for i,q in enumerate(request['body_poses']):
        if not np.isfinite(q).all():raise ValueError('Non-finite body pose')
        path=f'/Lab/body_{i}'
        values[path+'.xformOp:translate']=Gf.Vec3d(*q[:3])
        values[path+'.xformOp:orient']=Gf.Quatf(q[6],Gf.Vec3f(*q[3:6]))
    for path,data in request['meshes'].items():
        points=decode(data['points'],'float32',3);normal=decode(data['normals'],'float32',3)
        if not np.isfinite(points).all() or not np.isfinite(normal).all():
            raise ValueError('Non-finite recorded mesh')
        values[path+'.points']=Vt.Vec3fArray.FromNumpy(points)
        values[path+'.normals']=Vt.Vec3fArray.FromNumpy(normal)
        values[path+'.extent']=Vt.Vec3fArray.FromNumpy(np.asarray([points.min(0),points.max(0)],np.float32))
    layer=stage.GetRootLayer()
    updates=[(layer.GetAttributeAtPath(path),value) for path,value in values.items()]
    if 'film_groups' in request:
        from .film_render import prepare,FIELDS
        grouped={}
        for body,fields in request['film_groups'].items():
            if set(fields)!=set(FIELDS):raise ValueError('Invalid paged film fields')
            grouped[body]={key:decode(value,'float32',3 if key in ('points','normals','color') else None)
                           for key,value in fields.items()}
        updates.extend(prepare(stage,grouped))
    if request.get('runoff'):
        data=request['runoff'];points=decode(data['points'],'float32',3)
        widths=decode(data['widths'],'float32');color=decode(data['color'],'float32',3)
        if len(widths)!=len(points) or len(color)!=len(points) or np.any(widths<0):raise ValueError('Invalid runoff optics')
        from .runoff_render import prepare
        updates.extend(prepare(stage,dict(points=points,widths=widths,color=color)))
    for path,data in request['meshes'].items():
        if 'triangles' in data:
            if not path.startswith('/Lab/') or '/film_' not in path:raise ValueError('Only film topology varies')
            triangles=decode(data['triangles'],'int32',3)
            points=decode(data['points'],'float32',3)
            if np.any(triangles<0) or np.any(triangles>=len(points)):raise ValueError('Invalid film indices')
            updates.append((layer.GetAttributeAtPath(path+'.faceVertexIndices'),Vt.IntArray.FromNumpy(triangles.ravel())))
            updates.append((layer.GetAttributeAtPath(path+'.faceVertexCounts'),Vt.IntArray.FromNumpy(np.full(len(triangles),3,np.int32))))
            updates.append((layer.GetAttributeAtPath(path+'.normals'),Vt.Vec3fArray.FromNumpy(decode(data['normals'],'float32',3))))
    for path,fields in request.get('material_fields',{}).items():
        if not path.startswith('/Lab/') or set(fields)-{'color','opacity','roughness','metallic','coat_weight','food_coverage'}:
            raise ValueError('Invalid material field')
        color=decode(fields['color'],'float32',3)
        if np.any(color<0) or np.any(color>1):raise ValueError('Invalid color')
        updates.append((layer.GetAttributeAtPath(path+'.primvars:displayColor'),Vt.Vec3fArray.FromNumpy(color)))
        for field in ('opacity','roughness','metallic','coat_weight','food_coverage'):
            if field in fields:
                value=decode(fields[field],'float32')
                if len(value)!=len(color) or not np.isfinite(value).all() or np.any(value<0) or np.any(value>1):
                    raise ValueError('Invalid material scalar: '+field)
                name='displayOpacity' if field=='opacity' else field
                updates.append((layer.GetAttributeAtPath(path+'.primvars:'+name),Vt.FloatArray.FromNumpy(value)))
    if any(not spec for spec,_ in updates):raise ValueError('Missing material field attribute')
    with Sdf.ChangeBlock():
        for spec,value in updates:spec.default=value
