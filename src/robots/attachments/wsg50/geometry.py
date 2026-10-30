"""Existing authored WSG geometry, mechanically extracted without retuning."""

import hashlib
import xml.etree.ElementTree as ET
import numpy as np

TCP_Z = 0.107 + 0.008 + 0.0725 + 0.0245 + 0.045


def generated_urdf(model="fr3"):
    from robots.assets import resolve

    path = resolve(model)
    asset = path.parent.parent
    root = ET.parse(path).getroot()
    if model == "panda":
        # Remove the native hand subtree, retaining the manufacturer's arm/flange.
        removed = {
            "panda_hand",
            "panda_leftfinger",
            "panda_rightfinger",
            "panda_hand_tcp",
        }
        for node in list(root):
            if (
                node.tag == "link"
                and node.get("name") in removed
                or node.tag == "joint"
                and node.find("child").get("link") in removed
            ):
                root.remove(node)
    for mesh in root.findall(".//mesh"):
        name = mesh.get("filename")
        if name.startswith("package://"):
            name = name.split("/", 3)[-1]
        mesh.set("filename", str(asset / name))

    def box_link(name, mass, size, origin=(0, 0, 0), rgb=".12 .14 .16 1"):
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, origin)))
        ET.SubElement(inertial, "mass", value=str(mass))
        x, y, z = size
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(mass * (y * y + z * z) / 12),
            iyy=str(mass * (x * x + z * z) / 12),
            izz=str(mass * (x * x + y * y) / 12),
            ixy="0",
            ixz="0",
            iyz="0",
        )
        for role in ("visual", "collision"):
            shape = ET.SubElement(link, role)
            ET.SubElement(shape, "origin", xyz=" ".join(map(str, origin)))
            ET.SubElement(
                ET.SubElement(shape, "geometry"), "box", size=" ".join(map(str, size))
            )
            if role == "visual":
                ET.SubElement(
                    ET.SubElement(shape, "material", name=name + "_finish"),
                    "color",
                    rgba=rgb,
                )
        return link

    def compound_link(name, parts):
        link = ET.SubElement(root, "link", name=name)
        mass = sum(p[0] for p in parts)
        com = sum(p[0] * np.asarray(p[2]) for p in parts) / mass
        inertia = np.zeros((3, 3))
        for m, size, origin, rgb, label in parts:
            x, y, z = size
            r = np.asarray(origin) - com
            inertia += np.diag(
                [
                    m * (y * y + z * z) / 12,
                    m * (x * x + z * z) / 12,
                    m * (x * x + y * y) / 12,
                ]
            ) + m * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
            for role in ("visual", "collision"):
                shape = ET.SubElement(link, role, name=label)
                ET.SubElement(shape, "origin", xyz=" ".join(map(str, origin)))
                ET.SubElement(
                    ET.SubElement(shape, "geometry"),
                    "box",
                    size=" ".join(map(str, size)),
                )
                if role == "visual":
                    ET.SubElement(
                        ET.SubElement(shape, "material", name=label + "_finish"),
                        "color",
                        rgba=rgb,
                    )
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, com)))
        ET.SubElement(inertial, "mass", value=str(mass))
        ET.SubElement(
            inertial,
            "inertia",
            ixx=str(inertia[0, 0]),
            iyy=str(inertia[1, 1]),
            izz=str(inertia[2, 2]),
            ixy=str(inertia[0, 1]),
            ixz=str(inertia[0, 2]),
            iyz=str(inertia[1, 2]),
        )
        return link

    box_link("wsg_adapter", 0.12, (0.080, 0.050, 0.008), (0, 0, 0.004), ".45 .47 .49 1")
    joint = ET.SubElement(root, "joint", name="wsg_adapter_fixed", type="fixed")
    ET.SubElement(joint, "parent", link=model + "_link8")
    ET.SubElement(joint, "child", link="wsg_adapter")
    box_link("wsg_base", 1.11, (0.146, 0.050, 0.0725), (0, 0, 0.03625))
    joint = ET.SubElement(root, "joint", name="wsg_mount_fixed", type="fixed")
    ET.SubElement(joint, "parent", link="wsg_adapter")
    ET.SubElement(joint, "child", link="wsg_base")
    ET.SubElement(joint, "origin", xyz="0 0 .008")
    for i, sign in ((1, 1), (2, -1)):
        name = f"wsg_finger{i}"
        compound_link(
            name,
            [
                (
                    0.042,
                    (0.010, 0.024, 0.060),
                    (sign * 0.006, 0, 0.030),
                    ".45 .47 .49 1",
                    name + "_metal",
                ),
                (
                    0.045,
                    (0.021, 0.030, 0.0245),
                    (sign * 0.0105, 0, -0.01225),
                    ".08 .09 .10 1",
                    name + "_carriage",
                ),
                (
                    0.0012,
                    (0.001, 0.024, 0.045),
                    (sign * 0.0005, 0, 0.0375),
                    ".035 .04 .038 1",
                    name + "_pad",
                ),
            ],
        )
        joint = ET.SubElement(
            root, "joint", name=f"wsg_finger_joint{i}", type="prismatic"
        )
        ET.SubElement(joint, "parent", link="wsg_base")
        ET.SubElement(joint, "child", link=name)
        ET.SubElement(joint, "origin", xyz="0 0 .097")
        ET.SubElement(joint, "axis", xyz=f"{sign} 0 0")
        ET.SubElement(
            joint, "limit", lower="0", upper=".055", effort="40", velocity=".420"
        )
        ET.SubElement(joint, "dynamics", damping=".05", friction="0")
    payload = ET.tostring(root)
    return payload, dict(
        path=str(path),
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        attachment="WSG50-110146x50x97mm envelope,1.2kg including two estimated45g carriages; authored120g adapter and43.2g60mm padded fingers each; compound box inertia estimates",
        tcp_z_from_link7_m=TCP_Z,
    )
