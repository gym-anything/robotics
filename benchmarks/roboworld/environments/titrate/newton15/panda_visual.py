"""Restore source DAE face materials without replacing Newton robot geometry.

Newton retains DAE groups, but a previous brightness threshold misclassified
their sRGB-converted dark materials as white. Match source triangle centroids
exactly and bind source material colors; retain all Newton vertices, poses,
inertial properties and collision shapes unchanged.
"""

from functools import lru_cache
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial import cKDTree


@lru_cache(maxsize=1)
def catalog():
    import trimesh
    from .assets import resolve_panda

    path = resolve_panda()
    def resolve(name):
        prefix = "package://franka_description/"
        if name.startswith(prefix):
            return (path.parent.parent/name[len(prefix):]).resolve()
        return (path.parent/name).resolve()

    files = sorted({resolve(x.attrib["filename"])
                    for x in ET.parse(path).findall(".//visual/geometry/mesh")})
    result = []
    for file in files:
        scene = trimesh.load_scene(file)
        centers, colors = [], []
        for node in scene.graph.nodes_geometry:
            transform, name = scene.graph[node]
            part = scene.geometry[name]
            local = part.copy()
            local.apply_transform(transform)
            material = getattr(part.visual, "material", None)
            color = getattr(material, "baseColorFactor", None)
            if color is None:
                color = getattr(material, "diffuse", None)
            if color is None:
                raise ValueError(f"Missing source DAE material for {file}: {name}")
            color = np.asarray(color, float)[:3]
            if color.max() > 1:
                color /= 255.
            centers.append(local.triangles_center)
            colors.append(np.tile(color, (len(local.faces), 1)))
        c = np.concatenate(centers)
        result.append((len(c), cKDTree(c), np.concatenate(colors), str(file)))
    return result


def triangle_colors(vertices, faces):
    v, f = np.asarray(vertices), np.asarray(faces).reshape(-1, 3)
    centers = v[f].mean(axis=1)
    for count, tree, colors, _ in catalog():
        if count < len(f):
            continue
        distance, ids = tree.query(centers)
        if distance.max() < 1e-7:
            return colors[ids]
    return None


def bind(stage, usd_mesh, vertices, faces, materials):
    from pxr import UsdGeom, UsdShade

    colors = triangle_colors(vertices, faces)
    if colors is None:
        return False
    # These source meshes use white shells, dark housings, and red decals.
    ids = np.where(colors.mean(axis=1) > .5, 0, 1)
    red = (colors[:, 0] > .6) & (colors[:, 1] < .2)
    ids[red] = 2
    for index, name in enumerate(("robot_white", "robot_dark", "robot_red")):
        faces_here = np.flatnonzero(ids == index)
        if not len(faces_here):
            continue
        subset = UsdGeom.Subset.Define(stage, str(usd_mesh.GetPath())+"/"+name)
        subset.CreateElementTypeAttr("face")
        subset.CreateFamilyNameAttr("materialBind")
        subset.CreateIndicesAttr(faces_here.tolist())
        UsdShade.MaterialBindingAPI.Apply(subset.GetPrim()).Bind(materials[name])
    UsdGeom.Subset.SetFamilyType(usd_mesh, "materialBind", "partition")
    return True
