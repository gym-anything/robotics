"""Install the locked UR10e asset package (the welding environment's torch arm).

Source: Universal Robots' ROS 2 description, BSD-3-Clause, pinned commit COMMIT. The URDF is the xacro expansion of
urdf/ur.urdf.xacro with ur_type:=ur10e name:=ur10e (committed beside this script as source/ur10e.xacro.urdf, so
installing needs no ROS tooling), with two deterministic edits applied here: the 'world' link and its fixed
'base_joint' are removed (the environment mounts the arm), and 'package://ur_description/' mesh URIs become paths
relative to the URDF. The meshes and the license are taken from the archive of the same commit and every file is
checked against assets.lock.json.

Usage: python -m robots.models.ur10e.install [--dest DIR] [--archive PATH_TO_TARBALL]
Default destination: ~/.cache/gar/robots/ur10e (robots.assets resolver), or $GAR_ROBOT_ASSETS/ur10e.
"""

import argparse
import io
import os
import re
import sys
import tarfile
import urllib.request
from pathlib import Path

COMMIT = "6662e15f32c23c12ece57d0050aee9a716e4fc41"
URL = f"https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/archive/{COMMIT}.tar.gz"
HERE = Path(__file__).resolve().parent


def processed_urdf():
    text = (HERE/"source"/"ur10e.xacro.urdf").read_text()
    text = re.sub(r'\s*<link name="world"/>', "", text, count=1)
    text, n = re.subn(r'\s*<joint name="base_joint" type="fixed">.*?</joint>', "", text, count=1, flags=re.S)
    if n != 1:
        raise RuntimeError("source URDF layout changed: no world base_joint")
    return text.replace("package://ur_description/", "../")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None)
    ap.add_argument("--archive", default=None, help="local copy of the pinned GitHub archive")
    a = ap.parse_args()
    if a.dest:
        dest = Path(a.dest)
    elif os.environ.get("GAR_ROBOT_ASSETS"):
        dest = Path(os.environ["GAR_ROBOT_ASSETS"]).expanduser()/"ur10e"
    else:
        dest = Path.home()/".cache/gar/robots/ur10e"
    data = Path(a.archive).read_bytes() if a.archive else urllib.request.urlopen(URL, timeout=120).read()
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        for m in tar.getmembers():
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not m.isfile():
                continue
            rel = parts[1]
            if rel == "LICENSE" or rel.startswith("meshes/ur10e/"):
                out = dest/rel
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(tar.extractfile(m).read())
    (dest/"urdf").mkdir(parents=True, exist_ok=True)
    (dest/"urdf"/"ur10e.urdf").write_text(processed_urdf())
    sys.path.insert(0, str(HERE.parents[2]))
    from robots.assets import verify
    verify("ur10e", dest)
    print(f"ur10e assets installed and verified in {dest}")


if __name__ == "__main__":
    main()
