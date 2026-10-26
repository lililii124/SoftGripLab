"""Shared Panda tool assets, independent of the material solver."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from .adhesive_robot import _tool_urdf


def knife_robot_urdf(
    asset_folder, output, vertices, faces, *, world_quaternion=(0.5, 0.5, 0.5, 0.5)
):
    """Mount a supplied physical blade mesh on the Panda."""
    import pybullet as p

    output = Path(output)
    mesh = output / "knife.obj"
    mesh.write_text(
        "".join("v %.12g %.12g %.12g\n" % tuple(v) for v in vertices)
        + "".join(
            "f %d %d %d\n" % tuple(f + 1) for f in np.asarray(faces).reshape(-1, 3)
        ),
        encoding="ascii",
    )
    tree = ET.parse(_tool_urdf(asset_folder, output, 0.018))
    target = tree.getroot().find("link[@name='panda_grasptarget']")
    for visual in list(target.findall("visual")):
        target.remove(visual)
    # Map the supplied blade axes into the requested world orientation.
    _, mount_q = p.multiplyTransforms(
        [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], world_quaternion
    )
    visual = ET.SubElement(target, "visual")
    ET.SubElement(
        visual,
        "origin",
        xyz="0 0 .03",
        rpy=" ".join(map(str, p.getEulerFromQuaternion(mount_q))),
    )
    ET.SubElement(
        ET.SubElement(visual, "geometry"), "mesh", filename=str(mesh.resolve())
    )
    material = ET.SubElement(visual, "material", name="knife_steel")
    ET.SubElement(material, "color", rgba=".64 .66 .64 1")
    # Fixed mounting stem connects the hand to the knife spine. Visual only;
    # the contact geometry remains exactly the upstream blade mesh.
    stem = ET.SubElement(target, "visual")
    ET.SubElement(stem, "origin", xyz="0 0 -.015", rpy="0 0 0")
    ET.SubElement(
        ET.SubElement(stem, "geometry"), "cylinder", radius=".006", length=".052"
    )
    ET.SubElement(
        ET.SubElement(stem, "material", name="knife_mount"),
        "color",
        rgba=".44 .46 .44 1",
    )
    target.find("inertial/origin").set("xyz", "0 0 .03")
    path = output / "panda_cutting.urdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path, mount_q
