"""Physically mounted wrist RGB accessories and camera-frame exports."""
import math
import numpy as np
import warp as wp
from pxr import Gf,UsdGeom

def install(assembly):
    b=assembly.builder
    assembly.camera_mounts={}
    for arm in ("left","right"):
        matches=[i for i,n in enumerate(b.body_label) if n.startswith(arm+"/") and n.endswith("fr3_link7")]
        if len(matches)!=1:raise ValueError("Expected one FR3 flange link")
        body=matches[0]
        # D405 published external envelope and 60 g mass. Mount bracket is a
        # declared 10 g aluminium estimate; native added-shape inertia applies.
        assembly.box(arm+"_camera_case",(.075,0,.07),(.021,.021,.0115),body=body,
            material="robot_dark",cfg=assembly.shape_config(density=.060/(.042*.042*.023)))
        assembly.box(arm+"_camera_bracket",(.05,0,.046),(.025,.013,.003),body=body,
            material="aluminum",cfg=assembly.shape_config(density=.010/(.050*.026*.006)))
        for y in (-.012,.012):
            shape=b.add_shape_cylinder(body,xform=wp.transform((.075,y,.082),wp.quat_identity()),
                radius=.004,half_height=.0005,cfg=assembly.shape_config(density=0.),label=arm+"_camera_lens")
            assembly.materials[shape]="robot_dark"
        assembly.camera_mounts[arm+"_wrist_rgb"]=dict(body=body,eye=[.075,-.012,.083],
            target=[.075,-.012,.283],up=[0,1,0],horizontal_fov_deg=87.,
            published_source="https://www.realsenseai.com/products/stereo-depth-camera-d405/")

def export(stage,assembly):
    for name,mount in getattr(assembly,"camera_mounts",{}).items():
        path=f"/Lab/body_{mount['body']}/{name}"
        c=UsdGeom.Camera.Define(stage,path)
        c.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*mount["eye"]),
            Gf.Vec3d(*mount["target"]),Gf.Vec3d(*mount["up"])).GetInverse())
        c.CreateHorizontalApertureAttr(36.)
        c.CreateVerticalApertureAttr(20.25)
        c.CreateFocalLengthAttr(18/math.tan(math.radians(mount["horizontal_fov_deg"]/2)))
        c.CreateClippingRangeAttr(Gf.Vec2f(.001,30.))
        c.GetPrim().SetCustomDataByKey("observation_role","policy")
        alias=UsdGeom.Xform.Define(stage,"/Cameras/"+name)
        alias.GetPrim().SetCustomDataByKey("mounted_camera_path",path)
