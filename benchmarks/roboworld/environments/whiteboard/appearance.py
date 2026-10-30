"""Renderer-only PBR material bindings, separate from dynamics and evaluation."""

from pxr import Gf, Sdf, UsdShade


def mdl_surface(stage, material, module):
    shader = UsdShade.Shader.Define(stage, str(material.GetPath()) + "/rtx_surface")
    shader.CreateImplementationSourceAttr(UsdShade.Tokens.sourceAsset)
    shader.SetSourceAsset(Sdf.AssetPath(module + ".mdl"), "mdl")
    shader.SetSourceAssetSubIdentifier(module, "mdl")
    material.CreateSurfaceOutput("mdl").ConnectToSource(shader.ConnectableAPI(), "out")
    return shader


def bind_pbr(stage, material, maps, tile_m, *, coated=False, roughness=0.35):
    shader = mdl_surface(stage, material, "OmniPBR")
    # Authored box UVs use two cycles/metre. Undo that to use the asset's
    # published physical tile size, independent of object dimensions.
    scale = 0.5 / tile_m
    shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(scale, scale))
    shader.CreateInput("project_uvw", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("diffuse_color_constant", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(1))
    for key, input_name in (
        ("diff", "diffuse_texture"),
        ("rough", "reflectionroughness_texture"),
        ("nor_gl", "normalmap_texture"),
    ):
        shader.CreateInput(input_name, Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath("textures/" + maps[key].name)
        )
    shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(1.0)
    # A finished classroom surface is sealed/levelled; raw veneer pore relief
    # should not produce deep, wet-looking streaks through its clear finish.
    shader.CreateInput("bump_factor", Sdf.ValueTypeNames.Float).Set(0.08 if coated else 1.0)
    shader.CreateInput("metallic_constant", Sdf.ValueTypeNames.Float).Set(0)
    if coated:
        shader.CreateInput("reflection_roughness_texture_influence", Sdf.ValueTypeNames.Float).Set(
            0.25
        )
        shader.CreateInput("reflection_roughness_constant", Sdf.ValueTypeNames.Float).Set(roughness)


def bind_glass(stage, material):
    shader = mdl_surface(stage, material, "OmniGlass")
    shader.CreateInput("glass_color", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.96, 0.99, 1.0))
    shader.CreateInput("glass_ior", Sdf.ValueTypeNames.Float).Set(1.50)
    shader.CreateInput("thin_walled", Sdf.ValueTypeNames.Bool).Set(False)
    shader.CreateInput("frosting_roughness", Sdf.ValueTypeNames.Float).Set(0.015)
