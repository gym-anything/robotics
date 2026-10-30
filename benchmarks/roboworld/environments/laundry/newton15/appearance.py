"""Textile finish in material coordinates; no changes to physical state."""
import numpy as np
from pathlib import Path


def textile(stage, path, panel):
    from pxr import Gf, Sdf, UsdShade
    from PIL import Image
    from ..garment_showcase_scene import _textile_material, _TEXTILE_TEXTURE_STEMS, ASSETS
    material = _textile_material(stage, path, panel)
    shader = UsdShade.Shader.Get(stage, path+"/Shader")
    albedo = ASSETS/(_TEXTILE_TEXTURE_STEMS[panel.material_profile]+"_albedo.png")
    # Retain the authored garment dye. Use the existing procedural textile's
    # luminance modulation, not its unrelated catalog swatch colour.
    rgb = np.asarray(Image.open(albedo).convert("RGB"), dtype=float)/255.
    linear = np.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055)**2.4)
    luminance = float(np.mean(linear@np.array([.2126, .7152, .0722])))
    shader.CreateInput("diffuse_texture", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(albedo.resolve().as_posix()))
    shader.CreateInput("albedo_desaturation", Sdf.ValueTypeNames.Float).Set(1.)
    shader.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(1/luminance)
    shader.CreateInput("diffuse_tint", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*panel.display_color_rgb))
    shader.CreateInput("normalmap_texture", Sdf.ValueTypeNames.Asset).GetAttr().SetColorSpace("raw")
    shader.CreateInput("reflectionroughness_texture", Sdf.ValueTypeNames.Asset).GetAttr().SetColorSpace("raw")
    shader.CreateInput("flip_tangent_v", Sdf.ValueTypeNames.Bool).Set(False)
    if panel.material_profile == "terry_cotton_prior":
        # A UV shader gives straight woven boundaries. Assigning whole rendered
        # triangles to bands produced a visibly serrated edge and was removed.
        shader.SetSourceAsset(Sdf.AssetPath((Path(__file__).parent/"terry_finish.mdl").resolve().as_posix()), "mdl")
        shader.SetSourceAssetSubIdentifier("LaundryTextile", "mdl")
        for old in list(shader.GetInputs()):
            shader.GetPrim().RemoveProperty(old.GetFullName())
        inputs = {"dye": (Sdf.ValueTypeNames.Color3f, Gf.Vec3f(*panel.display_color_rgb)),
            "albedo": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath(albedo.resolve().as_posix())),
            "normalmap": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath((ASSETS/"terry_cotton_normal.png").resolve().as_posix())),
            "roughmap": (Sdf.ValueTypeNames.Asset, Sdf.AssetPath((ASSETS/"terry_cotton_roughness.png").resolve().as_posix())),
            "brightness": (Sdf.ValueTypeNames.Float, 1/luminance),
            "woven_borders": (Sdf.ValueTypeNames.Bool, True)}
        for name, (kind, value) in inputs.items():
            shader.CreateInput(name, kind).Set(value)
        for name in ("normalmap", "roughmap"):
            shader.GetInput(name).GetAttr().SetColorSpace("raw")
    return material
