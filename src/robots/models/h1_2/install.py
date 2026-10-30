"""Install the locked Unitree H1-2 asset package (the forklift environment's seated operator).

Source: unitreerobotics/unitree_ros, BSD-3-Clause, pinned commit COMMIT: robots/h1_2_description/h1_2.urdf (H1-2 with
Inspire RH56 hands: 27 body joints, 12 hand joints per hand of which 6 are actuated and 6 mimic) and every mesh it
references, plus the repository LICENSE. Files are fetched from raw.githubusercontent.com at that commit and every file
is checked against assets.lock.json (robots.assets.verify). Nothing is edited.

Usage: python -m robots.models.h1_2.install [--dest DIR]
Default destination: ~/.cache/gar/robots/h1_2 (robots.assets resolver), or $GAR_ROBOT_ASSETS/h1_2.
"""

import argparse
import os
import re
import sys
import urllib.request
from pathlib import Path

COMMIT = "5994d4faef0a9cadd3287f8de0199a67eeb2a259"
RAW = f"https://raw.githubusercontent.com/unitreerobotics/unitree_ros/{COMMIT}/"
URDF = "robots/h1_2_description/h1_2.urdf"
HERE = Path(__file__).resolve().parent


def files(urdf_text):
    meshes = sorted(set(re.findall(r'filename="(meshes/[^"]+)"', urdf_text)))
    return ["LICENSE", URDF] + ["robots/h1_2_description/" + m for m in meshes]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=None)
    a = ap.parse_args()
    if a.dest:
        dest = Path(a.dest)
    elif os.environ.get("GAR_ROBOT_ASSETS"):
        dest = Path(os.environ["GAR_ROBOT_ASSETS"]).expanduser() / "h1_2"
    else:
        dest = Path.home() / ".cache/gar/robots/h1_2"
    urdf = urllib.request.urlopen(RAW + URDF, timeout=120).read()
    for rel in files(urdf.decode()):
        out = dest / rel
        if out.is_file():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(urdf if rel == URDF else urllib.request.urlopen(RAW + rel, timeout=120).read())
    sys.path.insert(0, str(HERE.parents[2]))
    from robots.assets import verify
    verify("h1_2", dest)
    print(f"h1_2 assets installed and verified in {dest}")


if __name__ == "__main__":
    main()
