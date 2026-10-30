"""FR3 FK/IK from the exact runtime URDF; returns servo targets, never moves bodies."""
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

def transform(xyz=(0,0,0),rpy=(0,0,0)):
    t=np.eye(4);t[:3,:3]=Rotation.from_euler("xyz",rpy).as_matrix();t[:3,3]=xyz
    return t

class Kinematics:
    def __init__(self,assembly):
        source=assembly.sources["fr3_urdf"]
        with open(source["path"],"rb") as f:data=f.read()
        if hashlib.sha256(data).hexdigest()!=source["sha256"]:raise ValueError("Robot URDF changed since assembly")
        tree=ET.fromstring(data)
        by_child={j.find("child").attrib["link"]:j for j in tree.findall("joint")}
        node="fr3_hand_tcp";chain=[]
        while node in by_child:
            j=by_child[node];chain.append(j);node=j.find("parent").attrib["link"]
        self.chain=[];self.lower=[];self.upper=[]
        for j in reversed(chain):
            kind=j.attrib["type"];origin=j.find("origin")
            xyz=np.fromstring(origin.get("xyz","0 0 0"),sep=" ") if origin is not None else np.zeros(3)
            rpy=np.fromstring(origin.get("rpy","0 0 0"),sep=" ") if origin is not None else np.zeros(3)
            axis=np.fromstring(j.find("axis").get("xyz"),sep=" ") if kind=="revolute" else None
            if kind not in ("fixed","revolute"):raise ValueError(f"Unexpected TCP chain joint {kind}")
            if kind=="revolute":
                limit=j.find("limit");self.lower.append(float(limit.get("lower")));self.upper.append(float(limit.get("upper")))
            self.chain.append((j.attrib["name"],transform(xyz,rpy),axis))
        if len(self.lower)!=7:raise ValueError("Expected the standard seven-axis FR3")
        self.lower=np.asarray(self.lower);self.upper=np.asarray(self.upper)
        self.bases={arm:transform(base) for arm,base in assembly.robot_bases.items()}
        self.source=dict(source)

    def fk(self,arm,q,*,through=None):
        q=np.asarray(q,float)
        if q.shape!=(7,) or not np.isfinite(q).all():raise ValueError("Seven finite FR3 joint positions required")
        t=self.bases[arm].copy();i=0
        for name,origin,axis in self.chain:
            t=t@origin
            if axis is not None:
                r=np.eye(4);r[:3,:3]=Rotation.from_rotvec(axis*q[i]).as_matrix();t=t@r;i+=1
            if name==through:break
        return t

    def solve(self,arm,position,orientation_xyzw,seed,*,max_evaluations=160):
        p=np.asarray(position,float);r=np.asarray(orientation_xyzw,float);seed=np.asarray(seed,float)
        if p.shape!=(3,) or not np.isfinite(p).all():raise ValueError("Finite xyz target required")
        if r.shape!=(4,) or not np.isfinite(r).all() or abs(np.linalg.norm(r)-1)>1e-5:raise ValueError("Unit xyzw quaternion required")
        if seed.shape!=(7,) or not np.isfinite(seed).all():raise ValueError("Finite seven-joint seed required")
        if np.any(seed<self.lower) or np.any(seed>self.upper):raise ValueError("Seed exceeds manufacturer joint bounds")
        target=Rotation.from_quat(r)
        def residual(q):
            t=self.fk(arm,q)
            angle=(target*Rotation.from_matrix(t[:3,:3]).inv()).as_rotvec()
            return np.r_[t[:3,3]-p,.12*angle,1e-5*(q-seed)]
        fit=least_squares(residual,seed,bounds=(self.lower,self.upper),max_nfev=max_evaluations,
            ftol=1e-10,xtol=1e-10,gtol=1e-10)
        final=self.fk(arm,fit.x)
        error=float(np.linalg.norm(final[:3,3]-p))
        angular=float((target*Rotation.from_matrix(final[:3,:3]).inv()).magnitude())
        return dict(joints=fit.x.tolist(),position_error_m=error,orientation_error_rad=angular,
            converged=bool(error<.001 and angular<.01745329252),evaluations=int(fit.nfev),
            note="Bounded kinematic target only; not a collision-free path or physical move")
