"""Dense 3D truth/identified replay with the existing depth-buffer renderer."""

from pathlib import Path
import hashlib
import json
import numpy as np
from PIL import Image

from .mesh_render import _box, _camera, _mesh
from .force_inset import force_inset


def render_shape_identification(output):
    import pybullet as p
    import pybullet_data
    output = Path(output)
    source = output/"dense_validation.npz"
    if not source.exists():
        source = output/"trajectory.npz"
    with np.load(source, allow_pickle=False) as data:
        positions, force = data["position_m"], data["force_n"]
        if positions.ndim == 3:
            positions, force = positions[None], force[None]
        triangles, centers = data["surface_triangles"], data["sphere_center_m"]
        times, radius = data["time_s"], float(data["sphere_radius_m"])
    panels = len(positions)
    frames = []
    client = p.connect(p.DIRECT)
    try:
        for frame, now in enumerate(times):
            p.resetSimulation(physicsClientId=client)
            base = _box(p, client, [.040, .022, .002], color=(.72,.72,.68,1))
            p.resetBasePositionAndOrientation(base, [0,0,-.0118], [0,0,0,1], physicsClientId=client)
            for sign in (-1, 1):
                block = _box(p, client, [0.003, 0.014, 0.012], color=(0.60, 0.61, 0.58, 1))
                p.resetBasePositionAndOrientation(block, [sign*0.032, 0, 0.0022], [0,0,0,1], physicsClientId=client)
            sphere_visual = p.createVisualShape(p.GEOM_MESH,
                                                fileName=str(Path(pybullet_data.getDataPath())/"sphere_smooth.obj"),
                                                meshScale=[radius]*3,
                                                rgbaColor=[0.30, 0.32, 0.31, 1], physicsClientId=client)
            sphere = p.createMultiBody(0, -1, sphere_visual, centers[frame], physicsClientId=client)
            stem_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=.0018, length=.018,
                                             rgbaColor=[.38,.39,.37,1], physicsClientId=client)
            p.createMultiBody(0, -1, stem_visual, centers[frame]+[0,0,radius+.009], physicsClientId=client)
            image = Image.new("RGB", (520*panels, 460), (245,244,240))
            for panel in range(panels):
                body = _mesh(p, client, positions[panel, frame], triangles, color=(0.61,0.54,0.43,1))
                picture = _camera(p, client, [0,0,0.004], .21, 520, 460, yaw=27, pitch=-27)
                image.paste(picture, (520*panel,0))
                p.removeBody(body, physicsClientId=client)
                force_inset(image, times, force[panel], now, (520*panel+25,360,520*panel+275,440),
                            limits=(float(force.min()), float(force.max())))
            frames.append(image)
    finally:
        p.disconnect(physicsClientId=client)
    frames[0].save(output/"shape_identification.gif", save_all=True, append_images=frames[1:],
                   duration=90, loop=0)
    frames[len(frames)//2].save(output/"shape_identification.png")
    (output/"render.json").write_text(json.dumps({
        "trajectory": source.name, "trajectory_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "geometry": "saved dense FEM equilibrium nodes and boundary triangles",
        "deformation_scale": 1.0, "text_overlays": False, "frames": len(frames),
        "force_component": "recorded tool axial reaction", "renderer": "PyBullet TinyRenderer",
        "time_interpretation": "quasistatic loading schedule, not transient dynamics"
    }, indent=2)+"\n", encoding="utf-8")
