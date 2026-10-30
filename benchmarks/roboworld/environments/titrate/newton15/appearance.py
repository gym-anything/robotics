"""Pinned laboratory finish assets; rendering only, separate from dynamics."""

import hashlib
from pathlib import Path
import shutil
from pxr import Sdf, UsdShade


def daylight_asset(output):
    from whiteboard.visual_assets import asset, ASSETS
    source = asset("kiara_5_noon", "hdri")
    target = Path(output)/"textures"/source.name
    if not target.exists():
        shutil.copyfile(source, target)
    if hashlib.sha256(target.read_bytes()).hexdigest() != ASSETS["kiara_5_noon"]["hdri"]:
        raise ValueError("Daylight HDRI hash mismatch")
    return "textures/"+target.name


def apply_room_finishes(stage, materials, output):
    from whiteboard.visual_assets import ASSETS, asset
    from whiteboard.appearance import bind_pbr

    folder = Path(output)/"textures"
    folder.mkdir(parents=True, exist_ok=True)
    maps = {}
    for kind in ("diff", "rough", "nor_gl"):
        source = asset("beige_wall_001", kind)
        target = folder/source.name
        if not target.exists():
            shutil.copyfile(source, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != ASSETS["beige_wall_001"][kind]:
            raise ValueError("Laboratory finish asset hash mismatch")
        maps[kind] = target
    bind_pbr(stage, materials["wall"], maps, 3., coated=True, roughness=.75)
    # Sealed painted walls use low geometric relief, not raw outdoor stucco.
    shader = stage.GetPrimAtPath(str(materials["wall"].GetPath())+"/rtx_surface")
    shader.GetAttribute("inputs:bump_factor").Set(.05)
    # Authored off-white sealed laboratory paint. Reuse the photographed
    # finish's spatial variation, not its original tan pigment/albedo.
    paint = UsdShade.Shader(shader)
    paint.CreateInput("albedo_desaturation", Sdf.ValueTypeNames.Float).Set(.8)
    paint.CreateInput("albedo_brightness", Sdf.ValueTypeNames.Float).Set(.28)
    paint.CreateInput("albedo_add", Sdf.ValueTypeNames.Float).Set(.62)
    return {"beige_wall_001": {k: ASSETS["beige_wall_001"][k] for k in maps},
            "license": "https://polyhaven.com/license", "wall_tile_m": 3., "wall_bump_factor": .05,
            "paint_albedo_remap": {"desaturation": .8, "brightness": .28, "add": .62}}
