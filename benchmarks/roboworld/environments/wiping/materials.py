"""Measured-scale scan microstructure; state-driven soil never baked into RGB."""
from pathlib import Path
import os
from functools import lru_cache
import numpy as np

ROOT=Path(__file__).parent/'assets'

def polycarbonate(stage,material):
    from pxr import Sdf,UsdShade
    shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/rtx_polycarbonate')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(Path(__file__).with_name('polycarbonate.mdl'))),'mdl')
    shader.SetSourceAssetSubIdentifier('polycarbonate','mdl')
    for output in (material.CreateSurfaceOutput('mdl'),material.CreateVolumeOutput('mdl')):
        output.ConnectToSource(shader.ConnectableAPI(),'out')

def state_surface(stage,material,constant=None):
    from pxr import Sdf,UsdShade,Gf
    shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/rtx_state')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(Path(__file__).with_name('state_surface.mdl'))),'mdl')
    shader.SetSourceAssetSubIdentifier('state_surface','mdl')
    shader.CreateInput('brushed_finish',Sdf.ValueTypeNames.Bool).Set(
        os.environ.get('WIPE_BRUSHED_SURFACES')=='1')
    shader.CreateInput('layered_coat',Sdf.ValueTypeNames.Bool).Set(
        os.environ.get('WIPE_LAYERED_COAT')=='1')
    shader.CreateInput('residue_detail',Sdf.ValueTypeNames.Bool).Set(
        os.environ.get('WIPE_RESIDUE_DETAIL')=='1' and
        (os.environ.get('WIPE_SPECIES_OPTICS')=='1' or os.environ.get('WIPE_LAYERED_COAT')=='1'))
    if constant is not None:
        color,roughness,metallic=constant
        shader.CreateInput('surface_color',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*map(float,color)))
        shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(roughness)
        shader.CreateInput('metallic',Sdf.ValueTypeNames.Float).Set(metallic)
        shader.CreateInput('use_state',Sdf.ValueTypeNames.Bool).Set(False)
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')

@lru_cache(maxsize=8)
def albedo_brightness(path):
    from PIL import Image
    rgb=np.asarray(Image.open(path).convert('RGB'),dtype=float)/255.
    linear=np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)
    return 1./float(np.mean(linear@np.asarray((.2126,.7152,.0722))))

def textile(stage,material,side_m=.4):
    from pxr import Sdf,UsdShade
    shader=UsdShade.Shader.Define(stage,str(material.GetPath())+'/rtx_textile')
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(str(Path(__file__).with_name('microfiber.mdl'))),'mdl')
    shader.SetSourceAssetSubIdentifier('microfiber','mdl')
    for name,field in (('albedo','Diffuse'),('normal_map','nor_gl'),('roughness_map','Rough')):
        parameter=shader.CreateInput(name,Sdf.ValueTypeNames.Asset)
        parameter.Set(Sdf.AssetPath(str(ROOT/('terry_cloth_'+field+'.jpg'))))
        parameter.GetAttr().SetColorSpace('sRGB' if name=='albedo' else 'raw')
    shader.CreateInput('brightness',Sdf.ValueTypeNames.Float).Set(albedo_brightness(ROOT/'terry_cloth_Diffuse.jpg'))
    shader.CreateInput('side_m',Sdf.ValueTypeNames.Float).Set(side_m)
    material.CreateSurfaceOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    material.CreateDisplacementOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')
    material.CreateVolumeOutput('mdl').ConnectToSource(shader.ConnectableAPI(),'out')

def scanned(stage,material,asset,*,diffuse=True,normal_strength=1.):
    from pxr import Sdf,UsdShade,Gf
    path=str(material.GetPath());surface=UsdShade.Shader.Get(stage,path+'/surface')
    uv=UsdShade.Shader.Define(stage,path+'/scan_uv');uv.CreateIdAttr('UsdPrimvarReader_float2')
    uv.CreateInput('varname',Sdf.ValueTypeNames.Token).Set('st')
    uv.CreateOutput('result',Sdf.ValueTypeNames.Float2)
    for field in (('Diffuse',) if diffuse else ())+('nor_gl','Rough'):
        p=ROOT/(asset+'_'+field+'.jpg')
        if not p.is_file():raise FileNotFoundError(f'Bundled scene asset missing: {p}')
        t=UsdShade.Shader.Define(stage,path+'/scan_'+field);t.CreateIdAttr('UsdUVTexture')
        t.CreateInput('file',Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(p)))
        t.CreateInput('sourceColorSpace',Sdf.ValueTypeNames.Token).Set('sRGB' if field=='Diffuse' else 'raw')
        t.CreateInput('st',Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),'result')
        for axis in ('S','T'):t.CreateInput('wrap'+axis,Sdf.ValueTypeNames.Token).Set('repeat')
        t.CreateOutput('rgb',Sdf.ValueTypeNames.Float3);t.CreateOutput('r',Sdf.ValueTypeNames.Float)
        if field=='nor_gl':
            t.CreateInput('scale',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2*normal_strength,2*normal_strength,2.,1.))
            t.CreateInput('bias',Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-normal_strength,-normal_strength,-1.,0.))
            surface.CreateInput('normal',Sdf.ValueTypeNames.Normal3f).ConnectToSource(t.ConnectableAPI(),'rgb')
        elif field=='Rough':surface.GetInput('roughness').ConnectToSource(t.ConnectableAPI(),'r')
        else:surface.GetInput('diffuseColor').ConnectToSource(t.ConnectableAPI(),'rgb')

def planar_uv(mesh,axes,period):
    """World/mesh-local metric mapping, not one stretched texture per object."""
    from pxr import UsdGeom,Sdf,Vt
    p=np.asarray(mesh.GetPointsAttr().Get(),np.float32)
    uv=p[:,axes]/np.asarray(period,np.float32)
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar('st',Sdf.ValueTypeNames.TexCoord2fArray,UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv))
