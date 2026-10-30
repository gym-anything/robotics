"""Deterministic laboratory label artwork and metric cylindrical application.

Artwork is authored for this simulated reagent inventory, not a manufacturer's
product photograph or a measured optical spectrum. Labels move with the vessel.
"""

from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pxr import Gf, Sdf, UsdGeom, UsdShade, Vt


def reagent_label(folder, name, formula, description, concentration):
    image = Image.new("RGB", (800, 720), (244, 244, 236))
    d = ImageDraw.Draw(image)
    navy = (22, 49, 72)
    d.rectangle((0, 0, 800, 102), fill=navy)
    d.text((36, 27), "VOLUMETRIC SOLUTION", font=ImageFont.load_default(size=42), fill="white")
    d.text((40, 139), formula, font=ImageFont.load_default(size=102), fill=navy)
    d.text((42, 265), description, font=ImageFont.load_default(size=40), fill=(25, 27, 29))
    d.text((42, 345), f"{concentration:.3f} mol/L", font=ImageFont.load_default(size=58), fill=(12, 15, 18))
    d.line((40, 438, 760, 438), fill=(90, 93, 91), width=2)
    for y, text in ((474, "Reference temperature: 20 C"), (546, "Aqueous laboratory reagent"),
                    (613, "Keep container closed after use")):
        d.text((42, y), text, font=ImageFont.load_default(size=31), fill=(42, 44, 46))
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/(name+".png")
    image.save(path)
    return path


def cylinder_label(stage, path, radius, z0, z1, arc_width, texture):
    angles = np.linspace(-arc_width/(2*radius), arc_width/(2*radius), 65)
    vertices = np.array([(radius*np.sin(t), -radius*np.cos(t), z) for z in (z0, z1) for t in angles])
    n = len(angles)
    faces = np.array([(i, i+1, i+1+n, i+n) for i in range(n-1)])
    surface = UsdGeom.Mesh.Define(stage, path)
    surface.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(np.float32)))
    surface.CreateFaceVertexCountsAttr([4]*len(faces))
    surface.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(faces.astype(np.int32).ravel()))
    surface.CreateSubdivisionSchemeAttr("none")
    uv = np.array([(i/(n-1), z) for z in (0., 1.) for i in range(n)], np.float32)
    UsdGeom.PrimvarsAPI(surface).CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex).Set(Vt.Vec2fArray.FromNumpy(uv))
    mat = UsdShade.Material.Define(stage, path+"/material")
    shader = UsdShade.Shader.Define(stage, path+"/material/mdl")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath("OmniPBR.mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier("OmniPBR", "mdl")
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/"+texture.name))
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1.))
    shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(.65)
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
    mat.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    # Portable preview equivalent for USD consumers without MDL.
    preview = UsdShade.Shader.Define(stage, path+"/material/preview")
    preview.CreateIdAttr("UsdPreviewSurface")
    preview.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(.65)
    tex = UsdShade.Shader.Define(stage, path+"/material/texture")
    tex.CreateIdAttr("UsdUVTexture")
    tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath("textures/"+texture.name))
    tex.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")
    reader = UsdShade.Shader.Define(stage, path+"/material/uv")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    preview.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    mat.CreateSurfaceOutput().ConnectToSource(preview.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI.Apply(surface.GetPrim()).Bind(mat)
