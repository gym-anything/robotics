"""Transport faces derived from the physical kitchen collision geometry."""
import numpy as np
from scipy.spatial.transform import Rotation


def kitchen_faces(assembly,Surface):
    import newton
    b=assembly.builder;out=[]
    # Expose real box faces, not arbitrary overlapping UV patches. Thin-wall
    # pan interiors/exteriors and scraper faces move in their body's frame.
    names=('prep_tray','clean_staging_tray','collection_pan','scraper','used_cloth_bin',
           'waste_bin','tool_trolley_top','blender_motor','worktop','backsplash','clean_cloth_tray')
    for shape,label in enumerate(b.shape_label):
        if not label.startswith(names) or b.shape_type[shape]!=newton.GeoType.BOX:continue
        half=np.asarray(b.shape_scale[shape]);pose=np.asarray(b.shape_transform[shape])
        rot=Rotation.from_quat(pose[3:]);body=b.shape_body[shape]
        for axis in range(3):
            i,j=(axis+1)%3,(axis+2)%3
            for sign in (-1,1):
                if label=='worktop' and (axis,sign) in ((2,1),(0,-1)):continue
                if label=='backsplash' and (axis,sign)==(0,-1):continue
                normal=np.eye(3)[axis]*sign
                u=np.eye(3)[i];v=np.eye(3)[j]*sign
                origin=normal*half[axis]-u*half[i]-v*half[j]
                out.append(Surface(f'{label}_face_{axis}_{sign}',shape,body,
                    tuple(rot.apply(origin)+pose[:3]),tuple(rot.apply(u)),tuple(rot.apply(v)),
                    (2*half[i],2*half[j]),protected=label.startswith('clean_staging_tray'),
                    material=assembly.materials[shape]))
    floor=next(i for i,label in enumerate(b.shape_label) if label=='floor')
    for name,obj in assembly.objects.items():
        if not isinstance(obj,dict):continue
        for patch in obj.get('planar_patches',[]):
            if 'triangle' in patch:
                out.append(Surface.from_triangle(name+'_'+patch['name'],obj['bottom_shape'],obj['body'],patch['triangle'],
                    protected=name=='clean_staging_tray',material=assembly.materials[obj['bottom_shape']]))
                continue
            out.append(Surface(name+'_'+patch['name'],obj['bottom_shape'],obj['body'],
                tuple(patch['origin']),tuple(patch['u']),tuple(patch['v']),tuple(patch['size']),
                protected=name=='clean_staging_tray',material=assembly.materials[obj['bottom_shape']]))
    out.append(Surface('floor',floor,-1,(-3.5,-3.5,0),(1,0,0),(0,1,0),(7.,7.),
                       resolution=.025,protected=True,material='floor'))
    return out
