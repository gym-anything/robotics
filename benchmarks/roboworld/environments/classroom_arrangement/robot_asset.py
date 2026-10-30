"""Build the locked Mobile FR3 Duo robot description for the shared robot catalog (run once per machine).

Source: Franka Robotics `franka_description` (Apache-2.0) at commit 7aeeddc449edf8d62b594f9e36a81da53e7796f9
(2026-08-17), `robots/mobile_fr3_duo_v0_2/mobile_fr3_duo_v0_2.urdf.xacro` expanded with xacro 2.1.0 and
robot_types ['tmrv0_2', 'fr3', 'fr3'] (TMR v0.2 base, Franka Spine v0.1, FR3 Duo mount v0.3, Franka head v0.2, two
FR3 arms with Franka Hands). The expanded URDF is then edited, each edit recorded in SOURCE.json:

  * mesh URIs package://franka_description/... -> ../meshes/... (the package is self-contained);
  * FR3 joint position limits are kept as franka_description has them: they are the current Franka limits (FCI
    "Robot and interface specifications" and the June 2026 FR3 datasheet R02212 2.5: A1 +-166, A2 +-105, A4 -176..-7,
    A6 25..265 deg). The catalog FR3 (robots/models/fr3, Newton asset bundle) carries the older, narrower set
    (+-2.7437 rad on A1, ...); link inertials and joint frames are identical;
  * masses reconciled to the Mobile FR3 Duo manual R04245 (TMR 130 kg, Spine 45 kg): a battery/electronics block of
    11.46 kg fixed in the TMR chassis, 19.07 kg of drive/screw/guide mass at the Spine column's centre of mass, and a
    4.0 kg Spine carriage (the URDF's 0.01 kg is a placeholder);
  * Spine lift: effort 670 N (2*pi*2.37 N m / 0.020 m * 0.9 ball-screw efficiency), velocity 0.3 m/s (manual); the
    URDF's 100 N / 0.1 m/s cannot lift the 46 kg it carries;
  * ArgoDrive AD-100.12 (ebm-papst datasheet): drive wheel torque 28.4 N m (braking maximum; acceleration torque 16 N m
    is enforced by the base controller), 40 rad/s (2 m/s nominal); steering 30 N m (estimate), pi rad/s (180 deg/s);
    the URDF's Gazebo <dynamics> on the TMR joints are removed (the controller owns drive damping); passive casters and
    rocker keep their URDF damping (0.003);
  * `mobile_fr3_single.urdf`: the same robot with the left FR3 and hand removed (an authored reduced configuration).

Usage:  python -m classroom_arrangement.robot_asset --franka-description PATH [--xacro-path DIR] [--out ~/.cache/gar/robots]
PATH must be a franka_description checkout at the commit above; xacro needs PyYAML (the Isaac runtime has it).
"""

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET

from gym_anything_robotics.paths import package_dir

COMMIT = "7aeeddc449edf8d62b594f9e36a81da53e7796f9"
XACRO = "robots/mobile_fr3_duo_v0_2/mobile_fr3_duo_v0_2.urdf.xacro"
ROBOT_TYPES = "['tmrv0_2','fr3','fr3']"
FR3_LIMITS = {}   # franka_description limits kept (current Franka specification; see the module docstring)
TMR_MASS_KG, SPINE_MASS_KG = 130.0, 45.0          # Mobile FR3 Duo manual R04245 (D1)
CARRIAGE_KG = 4.0                                  # estimate: linear carriage + mounting plate
SPINE_FORCE_N = 2*math.pi*2.37/0.020*0.9           # 670 N
DRIVE = dict(effort=28.4, velocity=40.0)           # AD-100.12: 28.4 N m braking max; 2 m/s / 0.05 m
STEER = dict(effort=30.0, velocity=math.pi)        # 180 deg/s; torque estimate
TMR_LINKS = ("base_inertia", "caster_front_left_fixed_link", "caster_front_left_steering_link", "caster_front_left_link",
             "argo_drive_front_fixed_link", "argo_drive_front_steering_link", "argo_drive_front_link", "rocker_arm_link",
             "caster_rear_right_fixed_link", "caster_rear_right_steering_link", "caster_rear_right_link",
             "argo_drive_rear_fixed_link", "argo_drive_rear_steering_link", "argo_drive_rear_link")


def box_inertia(m, x, y, z):
    return dict(ixx=m*(y*y + z*z)/12, iyy=m*(x*x + z*z)/12, izz=m*(x*x + y*y)/12, ixy=0., ixz=0., iyz=0.)


def add_mass_link(robot, name, parent, mass, xyz, size):
    link = ET.SubElement(robot, "link", name=name)
    inertial = ET.SubElement(link, "inertial")
    ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(inertial, "mass", value=f"{mass:.6f}")
    ET.SubElement(inertial, "inertia", **{k: f"{v:.9f}" for k, v in box_inertia(mass, *size).items()})
    joint = ET.SubElement(robot, "joint", name=name + "_joint", type="fixed")
    ET.SubElement(joint, "parent", link=parent)
    ET.SubElement(joint, "child", link=name)
    ET.SubElement(joint, "origin", xyz=" ".join(f"{v:.6f}" for v in xyz), rpy="0 0 0")


def mass_of(robot, names):
    out = 0.
    for link in robot.findall("link"):
        if link.get("name") in names and link.find("inertial") is not None:
            out += float(link.find("inertial/mass").get("value"))
    return out


def expand(src, xacro_path):
    """xacro on a copy of the checkout with $(find franka_description) resolved (no ROS installation needed)."""
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="fd-xacro-"))/"franka_description"
    shutil.copytree(src, work, ignore=shutil.ignore_patterns(".git"))
    for p in work.rglob("*.xacro"):
        t = p.read_text()
        if "$(find franka_description)" in t:
            p.write_text(t.replace("$(find franka_description)", str(work)))
    out = work.parent/"expanded.urdf"
    code = ("import sys, xacro; sys.argv = ['xacro', %r, %r, '-o', %r]; xacro.main()"
            % (str(work/XACRO), f"robot_types:={ROBOT_TYPES}", str(out)))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (xacro_path, os.environ.get("PYTHONPATH")) if p))
    subprocess.run([os.environ.get("XACRO_PYTHON", sys.executable), "-c", code], check=True, env=env)
    return out.read_text(), work


def edit(text):
    robot = ET.fromstring(text)
    log = []
    used = set()
    for el in robot.iter("mesh"):
        fn = el.get("filename")
        rel = fn.replace("package://franka_description/", "")
        used.add(rel)
        el.set("filename", "../" + rel)
    log.append(f"{len(used)} mesh files relocated to ../meshes")
    for j in robot.findall("joint"):
        name = j.get("name")
        lim = j.find("limit")
        for side in ("left", "right"):
            pre = f"{side}_fr3_"
            if name.startswith(pre) and name[len(pre):] in FR3_LIMITS:
                lo, hi = FR3_LIMITS[name[len(pre):]]
                lim.set("lower", str(lo)); lim.set("upper", str(hi))
        if name == "franka_spine_vertical_joint":
            lim.set("effort", f"{SPINE_FORCE_N:.1f}"); lim.set("velocity", "0.3")
            if j.find("dynamics") is not None:
                j.remove(j.find("dynamics"))
        if name in ("tmrv0_2_joint_1", "tmrv0_2_joint_3"):
            lim.set("effort", str(DRIVE["effort"])); lim.set("velocity", str(DRIVE["velocity"]))
            j.remove(j.find("dynamics"))
        if name in ("tmrv0_2_joint_0", "tmrv0_2_joint_2"):
            lim.set("effort", str(STEER["effort"])); lim.set("velocity", f"{STEER['velocity']:.6f}")
            j.remove(j.find("dynamics"))
    log.append("FR3 joint limits as franka_description (current Franka specification); spine 670 N / 0.3 m/s; ArgoDrive "
               "28.4 N m / 40 rad/s, steering 30 N m / pi rad/s; TMR and spine Gazebo dynamics removed")
    for link in robot.findall("link"):
        # a link with colliders but no <inertial> (base_link: its mass sits on base_inertia) would get the collider
        # volume x Newton's default density as mass (+143 kg for the chassis hull, measured); give it a negligible one
        if link.find("collision") is not None and link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "mass", value="0.001")
            ET.SubElement(inertial, "inertia", ixx="1e-6", iyy="1e-6", izz="1e-6", ixy="0", ixz="0", iyz="0")
            log.append(f"{link.get('name')}: explicit 1 g inertial (collider density mass suppressed)")
    tmr = mass_of(robot, TMR_LINKS)
    add_mass_link(robot, "tmr_battery_block", "base_link", TMR_MASS_KG - tmr, (0., 0., 0.12), (0.45, 0.35, 0.15))
    spine = [l for l in robot.findall("link") if l.get("name") == "franka_spine"][0]
    com = [float(v) for v in spine.find("inertial/origin").get("xyz").split()]
    column = float(spine.find("inertial/mass").get("value"))
    carriage = [l for l in robot.findall("link") if l.get("name") == "franka_spine_mounting_point"][0]
    carriage.find("inertial/mass").set("value", f"{CARRIAGE_KG:.6f}")
    for k, v in box_inertia(CARRIAGE_KG, 0.12, 0.30, 0.20).items():
        carriage.find("inertial/inertia").set(k, f"{v:.9f}")
    extra = SPINE_MASS_KG - column - CARRIAGE_KG
    add_mass_link(robot, "spine_drive_block", "franka_spine", extra, com, (0.16, 0.20, 0.90))
    log.append(f"masses: TMR {tmr:.2f} -> {TMR_MASS_KG} kg (+{TMR_MASS_KG - tmr:.2f} kg block at z 0.12 m); spine column "
               f"{column:.3f} + carriage 0.01 -> {SPINE_MASS_KG} kg (carriage {CARRIAGE_KG} kg, +{extra:.2f} kg at the column COM)")
    return robot, used, log


def single_arm(robot):
    r = copy.deepcopy(robot)
    drop_links = {l.get("name") for l in r.findall("link") if l.get("name").startswith(("left_fr3_", "left_base"))}
    for l in list(r.findall("link")):
        if l.get("name") in drop_links:
            r.remove(l)
    for j in list(r.findall("joint")):
        if j.find("child").get("link") in drop_links:
            r.remove(j)
    return r


def write(robot, path):
    ET.indent(robot, space="  ")
    path.write_text('<?xml version="1.0"?>\n<!-- generated by classroom_arrangement/robot_asset.py; see SOURCE.json -->\n'
                    + ET.tostring(robot, encoding="unicode") + "\n")


def lock(directory, files, source):
    entries = {}
    for rel in sorted(files):
        data = (directory/rel).read_bytes()
        entries[rel] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))
    return {"source": source, "license": "Apache-2.0 (franka_description LICENSE, included)", "files": entries}


def build(src, out, xacro_path):
    src, out = Path(src).resolve(), Path(out).expanduser().resolve()
    commit = subprocess.run(["git", "-C", str(src), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    if commit != COMMIT:
        raise SystemExit(f"franka_description at {commit or '?'}; expected {COMMIT}")
    text, work = expand(src, xacro_path)
    robot, used, log = edit(text)
    repo = package_dir("robots")/"models"
    results = {}
    for model, tree, urdf in (("mobile-fr3-duo", robot, "mobile_fr3_duo.urdf"),
                              ("mobile-fr3-single", single_arm(robot), "mobile_fr3_single.urdf")):
        d = out/model
        if d.exists():
            shutil.rmtree(d)
        (d/"urdf").mkdir(parents=True)
        meshes = sorted({el.get("filename")[3:] for el in tree.iter("mesh")})
        for rel in meshes:
            (d/rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src/rel, d/rel)
        shutil.copy2(src/"LICENSE", d/"LICENSE")
        write(tree, d/"urdf"/urdf)
        meta = dict(source="https://github.com/frankarobotics/franka_description", commit=COMMIT, xacro=XACRO,
                    robot_types=ROBOT_TYPES, xacro_version="2.1.0", edits=log + (["left FR3 arm and hand removed"]
                                                                                 if model.endswith("single") else []))
        (d/"SOURCE.json").write_text(json.dumps(meta, indent=1) + "\n")
        files = ["LICENSE", "SOURCE.json", f"urdf/{urdf}"] + meshes
        (repo/model).mkdir(parents=True, exist_ok=True)
        (repo/model/"assets.lock.json").write_text(json.dumps(lock(d, files, meta["source"] + "@" + COMMIT), indent=1) + "\n")
        results[model] = dict(directory=str(d), files=len(files))
    shutil.rmtree(work.parent, ignore_errors=True)
    return results, log


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--franka-description", required=True)
    p.add_argument("--xacro-path", default=None, help="directory containing the xacro package (pip --target)")
    p.add_argument("--out", default=os.environ.get("GAR_ROBOT_ASSETS", "~/.cache/gar/robots"))
    a = p.parse_args()
    print(json.dumps(build(a.franka_description, a.out, a.xacro_path), indent=1))


if __name__ == "__main__":
    main()
