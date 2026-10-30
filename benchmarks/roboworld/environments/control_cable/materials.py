"""Portable USD PreviewSurface texture bindings used by the workcell renderer."""
from pxr import Gf,Sdf,UsdShade

def bind_floor(stage,material,maps,tile_m=3.):
    surface=UsdShade.Shader(stage.GetPrimAtPath(str(material.GetPath())+"/surface"))
    uv=UsdShade.Shader.Define(stage,str(material.GetPath())+"/uv")
    uv.CreateIdAttr("UsdPrimvarReader_float2")
    uv.CreateInput("varname",Sdf.ValueTypeNames.Token).Set("st")
    scale=UsdShade.Shader.Define(stage,str(material.GetPath())+"/uv_scale")
    scale.CreateIdAttr("UsdTransform2d")
    scale.CreateInput("in",Sdf.ValueTypeNames.Float2).ConnectToSource(uv.ConnectableAPI(),"result")
    scale.CreateInput("scale",Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(.5/tile_m))
    for kind,destination in (("diff","diffuseColor"),("rough","roughness"),("nor_gl","normal")):
        tex=UsdShade.Shader.Define(stage,str(material.GetPath())+"/"+kind)
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file",Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/"+maps[kind].name))
        tex.CreateInput("sourceColorSpace",Sdf.ValueTypeNames.Token).Set("sRGB" if kind=="diff" else "raw")
        tex.CreateInput("wrapS",Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT",Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("st",Sdf.ValueTypeNames.Float2).ConnectToSource(scale.ConnectableAPI(),"result")
        if kind=="diff":
            # Neutral grey epoxy tint, retaining the scanned surface variation.
            tex.CreateInput("scale",Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(.74,.95,1.33,1.))
        if kind=="nor_gl":
            tex.CreateInput("scale",Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(2,2,2,1))
            tex.CreateInput("bias",Sdf.ValueTypeNames.Float4).Set(Gf.Vec4f(-1,-1,-1,0))
        typ=Sdf.ValueTypeNames.Float if kind=="rough" else Sdf.ValueTypeNames.Color3f
        surface.CreateInput(destination,typ).ConnectToSource(tex.ConnectableAPI(),"r" if kind=="rough" else "rgb")
