"""Read-only USD laboratory adapter for volumetric light transport.

All scene geometry, body transforms, cameras, labels, and material subsets come
from the exported USD. No Newton state or scene arrangement is changed here.
MDL and Disney microfacet lobes are not numerically identical; mappings below
preserve the authored inputs and expose this renderer difference explicitly.
"""
import numpy as np
from pxr import Usd, UsdGeom, UsdShade, UsdLux
from .volume_transport import runtime


def rgb(value):
    return dict(type='rgb', value=list(map(float, value)))


def get(shader, name, default=None):
    attr = shader.GetInput(name)
    value = attr.Get() if attr else None
    return default if value is None else value


def material(mat):
    mi = runtime()
    name = mat.GetPrim().GetName()
    if name in ('water', 'water_interface'):
        return mi.load_dict(dict(type='dielectric', int_ior=1.333, ext_ior=1.))
    if name == 'glass':
        shader = UsdShade.Shader(mat.GetPrim().GetChild('rtx_surface'))
        return mi.load_dict(dict(type='roughdielectric', distribution='ggx',
            alpha=max(float(get(shader,'frosting_roughness',.015))**2,1e-6),
            int_ior=float(get(shader,'glass_ior',1.474)),ext_ior=1.))
    surface = mat.ComputeSurfaceSource()[0]
    if not surface:
        raise ValueError(f'Missing portable material: {mat.GetPath()}')
    color = get(surface,'diffuseColor',(.5,.5,.5))
    roughness = float(get(surface,'roughness',.5))
    metallic = float(get(surface,'metallic',0.))
    mdl = mat.ComputeSurfaceSource('mdl')[0]
    texture = get(mdl,'diffuse_texture') if mdl else None
    if texture:
        from PIL import Image
        path = texture.resolvedPath
        if not path:
            raise ValueError(f'Unresolved material texture: {texture}')
        pixels = np.asarray(Image.open(path).convert('RGB'),np.float32)/255.
        pixels = np.where(pixels<=.04045,pixels/12.92,((pixels+.055)/1.055)**2.4)
        saturation = float(get(mdl,'albedo_desaturation',0.))
        luminance = pixels@np.array([.2126,.7152,.0722])
        pixels = pixels*(1-saturation)+luminance[...,None]*saturation
        pixels = np.clip(pixels*float(get(mdl,'albedo_brightness',1.))+float(get(mdl,'albedo_add',0.)),0,1)
        color = dict(type='bitmap',bitmap=mi.Bitmap(pixels.astype(np.float32)),raw=True,
            to_uv=mi.ScalarTransform3f().scale(list(get(mdl,'texture_scale',(1.,1.)))))
        roughness = float(get(mdl,'reflection_roughness_constant',roughness))
    else:
        color = rgb(color)
    return mi.load_dict(dict(type='principled',base_color=color,roughness=roughness,metallic=metallic))


def triangles(prim):
    """Preserve polygon vertices/UVs; triangulate without changing positions."""
    m = UsdGeom.Mesh(prim)
    v = np.asarray(m.GetPointsAttr().Get(),np.float32)
    counts = np.asarray(m.GetFaceVertexCountsAttr().Get(),int)
    indices = np.asarray(m.GetFaceVertexIndicesAttr().Get(),int)
    uvprim = UsdGeom.PrimvarsAPI(m).GetPrimvar('st')
    uv = np.asarray(uvprim.ComputeFlattened(),np.float32) if uvprim and uvprim.HasValue() else None
    normals = np.asarray(m.GetNormalsAttr().Get(),np.float32) if m.GetNormalsAttr().HasValue() else None
    if normals is not None and m.GetNormalsInterpolation() != 'vertex':
        normals = None
    if uv is not None and uvprim.GetInterpolation() == 'faceVarying':
        v = v[indices]
        normals = normals[indices] if normals is not None else None
        indices = np.arange(len(indices))
    faces, ids, offset = [], [], 0
    for i,n in enumerate(counts):
        for j in range(1,n-1):
            faces.append((indices[offset],indices[offset+j],indices[offset+j+1]))
            ids.append(i)
        offset += n
    f = np.asarray(faces,np.uint32).reshape(-1,3)
    if m.GetOrientationAttr().Get() == 'leftHanded':
        f = f[:,::-1]
    return v,f,uv,normals,np.asarray(ids)


def make_mesh(name, vertices, faces, bsdf, transform, uv=None, normals=None, interior=None):
    mi = runtime()
    props = mi.Properties()
    props['bsdf'] = bsdf
    if interior is not None:
        props['interior'] = interior
    v = np.asarray(vertices,np.float32)
    v = v@transform[:3,:3].T+transform[:3,3]
    n = None
    if normals is not None:
        n = normals@np.linalg.inv(transform[:3,:3])
        n /= np.maximum(np.linalg.norm(n,axis=1,keepdims=True),1e-20)
    mesh = mi.Mesh(name,len(v),len(faces),props,has_vertex_normals=n is not None,has_vertex_texcoords=uv is not None)
    p = mi.traverse(mesh)
    p['vertex_positions'] = v.astype(np.float32).ravel()
    p['faces'] = np.asarray(faces,np.uint32).ravel()
    if n is not None:
        p['vertex_normals'] = n.astype(np.float32).ravel()
    if uv is not None:
        uv = uv.copy()
        uv[:,1] = 1.-uv[:,1]  # USD bottom-origin UV vs image top-origin.
        p['vertex_texcoords'] = uv.ravel()
    p.update()
    return mesh


def primitive_mesh(prim):
    import trimesh
    kind = prim.GetTypeName()
    if kind == 'Cylinder':
        p = UsdGeom.Cylinder(prim)
        m = trimesh.creation.cylinder(float(p.GetRadiusAttr().Get()),float(p.GetHeightAttr().Get()),sections=64)
    elif kind == 'Capsule':
        p = UsdGeom.Capsule(prim)
        m = trimesh.creation.capsule(height=float(p.GetHeightAttr().Get()),radius=float(p.GetRadiusAttr().Get()),count=[16,32])
    elif kind == 'Sphere':
        p = UsdGeom.Sphere(prim)
        return trimesh.creation.icosphere(subdivisions=3,radius=float(p.GetRadiusAttr().Get()))
    elif kind == 'BasisCurves':
        p = UsdGeom.BasisCurves(prim)
        if p.GetTypeAttr().Get() != 'linear' or p.GetWidthsInterpolation() != 'constant':
            raise ValueError('Only authored linear constant-width markings supported')
        points = np.asarray(p.GetPointsAttr().Get())
        radius = float(p.GetWidthsAttr().Get()[0])/2
        parts,offset = [],0
        for n in p.GetCurveVertexCountsAttr().Get():
            for a,b in zip(points[offset:offset+n-1],points[offset+1:offset+n]):
                if np.linalg.norm(b-a)>1e-10:
                    parts.append(trimesh.creation.cylinder(radius,segment=[a,b],sections=6))
            offset += n
        return trimesh.util.concatenate(parts)
    else:
        raise ValueError(f'Unsupported visible primitive {kind}: {prim.GetPath()}')
    axis = p.GetAxisAttr().Get()
    if axis != 'Z':
        m.apply_transform(trimesh.geometry.align_vectors([0,0,1], [1,0,0] if axis=='X' else [0,1,0]))
    return m


class LaboratoryScene:
    def __init__(self, stage):
        if UsdGeom.GetStageMetersPerUnit(stage) != 1.:
            raise ValueError('Volume transport requires world metres')
        self.stage, self.mesh_cache, self.materials = stage, {}, {}
        self.mi = runtime()
        self.prims = []
        for prim in stage.Traverse():
            if not prim.IsA(UsdGeom.Gprim):
                continue
            if prim.GetName() == 'indicator_absorption':
                continue  # Obsolete RTX render proxy, never a physical object.
            if UsdGeom.Imageable(prim).ComputeVisibility() == 'invisible':
                continue
            if prim.IsA(UsdGeom.Mesh):
                data = triangles(prim)
            else:
                m = primitive_mesh(prim)
                data = (m.vertices,m.faces,None,m.vertex_normals,np.arange(len(m.faces)))
            self.mesh_cache[str(prim.GetPath())] = data
            self.prims.append(prim)
        for prim in stage.Traverse():
            if prim.IsA(UsdShade.Material):
                self.materials[str(prim.GetPath())] = material(UsdShade.Material(prim))

    def build(self, camera_name, width, height, liquid, field):
        from .volume_transport import medium
        mi, stage = self.mi,self.stage
        cam = UsdGeom.Camera(stage.GetPrimAtPath('/Cameras/'+camera_name))
        if not cam:
            raise ValueError('Unknown camera '+camera_name)
        cache = UsdGeom.XformCache()
        # USD camera looks down -Z, Mitsuba looks down +Z with opposite X.
        cm = np.asarray(cache.GetLocalToWorldTransform(cam.GetPrim())).T@np.diag([-1,1,-1,1])
        scene = dict(type='scene',integrator=dict(type='volpathmis',max_depth=12),
            sensor=dict(type='perspective',fov_axis='x',
                fov=float(np.degrees(2*np.arctan(cam.GetHorizontalApertureAttr().Get()/(2*cam.GetFocalLengthAttr().Get())))),
                to_world=mi.ScalarTransform4f(cm),near_clip=float(cam.GetClippingRangeAttr().Get()[0]),
                far_clip=float(cam.GetClippingRangeAttr().Get()[1]),
                film=dict(type='hdrfilm',width=width,height=height,rfilter=dict(type='box'))))
        liquid_medium = medium(field)
        for prim in self.prims:
            name = str(prim.GetPath())
            v,f,uv,n,face_ids = self.mesh_cache[name]
            if name == '/Lab/liquid' and liquid is not None:
                import trimesh
                v,f,uv = liquid.vertices,liquid.triangles,None
                n = trimesh.Trimesh(v,f,process=False).vertex_normals
                face_ids = np.arange(len(f))
            if not len(f):
                continue
            mat,_ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
            groups = [(np.ones(len(f),bool),mat)]
            subsets = UsdGeom.Subset.GetGeomSubsets(UsdGeom.Imageable(prim),'face','materialBind')
            if subsets:
                groups,assigned = [],np.zeros(len(f),bool)
                for subset in subsets:
                    selected = np.isin(face_ids,subset.GetIndicesAttr().Get())
                    bound,_ = UsdShade.MaterialBindingAPI(subset).ComputeBoundMaterial()
                    groups.append((selected,bound))
                    assigned |= selected
                if not assigned.all():
                    groups.append((~assigned,mat))
            transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
            for i,(selected,mat) in enumerate(groups):
                key = str(mat.GetPath())
                water = mat.GetPrim().GetName() in ('water','water_interface')
                scene[name+f'_{i}'] = make_mesh(name,v,f[selected],self.materials[key],transform,uv,n,
                    interior=liquid_medium if water else None)
        for prim in stage.Traverse():
            if prim.IsA(UsdLux.RectLight):
                light = UsdLux.RectLight(prim)
                transform = np.asarray(cache.GetLocalToWorldTransform(prim)).T
                transform = transform@np.diag([light.GetWidthAttr().Get()/2,light.GetHeightAttr().Get()/2,-1.,1.])
                intensity = float(light.GetIntensityAttr().Get())*2**float(light.GetExposureAttr().Get())
                # USD light intensity is luminance; use a fixed shared unit
                # conversion, never a chemistry-dependent exposure adjustment.
                color = np.array(light.GetColorAttr().Get())*intensity
                scene[str(prim.GetPath())] = dict(type='rectangle',to_world=mi.ScalarTransform4f(transform),
                    emitter=dict(type='area',radiance=rgb(color)))
            elif prim.IsA(UsdLux.DomeLight):
                light = UsdLux.DomeLight(prim)
                asset = light.GetTextureFileAttr().Get()
                intensity = float(light.GetIntensityAttr().Get())*2**float(light.GetExposureAttr().Get())
                if asset and asset.resolvedPath:
                    scene[str(prim.GetPath())] = dict(type='envmap',filename=asset.resolvedPath,scale=intensity)
                else:
                    scene[str(prim.GetPath())] = dict(type='constant',radiance=rgb(np.array(light.GetColorAttr().Get())*intensity))
        return mi.load_dict(scene)
