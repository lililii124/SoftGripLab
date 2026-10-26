"""Offline depth-buffered rendering of saved solver meshes with Bullet's camera."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from .animations import font, save_frames
from .force_inset import force_inset

BG = "#f5f4f0"
INK = "#292a28"
GRAY = "#666760"


def _mesh(p, cid, points, triangles, color=(0.57, 0.40, 0.30, 1)):
    vertices = points[triangles]
    face_normal = np.cross(
        vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0]
    )
    normals = np.zeros_like(points)
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normal)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-14)
    visual = p.createVisualShape(
        p.GEOM_MESH,
        vertices=points.tolist(),
        indices=triangles.ravel().tolist(),
        normals=normals.tolist(),
        rgbaColor=color,
        physicsClientId=cid,
    )
    return p.createMultiBody(0, -1, visual, physicsClientId=cid)


def _box(p, cid, half, color=(0.67, 0.68, 0.64, 1)):
    visual = p.createVisualShape(
        p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid
    )
    return p.createMultiBody(0, -1, visual, physicsClientId=cid)


def _camera(p, cid, focus, distance, width, height, yaw=42, pitch=-18):
    view = p.computeViewMatrixFromYawPitchRoll(focus, distance, yaw, pitch, 0, 2)
    projection = p.computeProjectionMatrixFOV(32, width / height, 0.001, 2)
    result = p.getCameraImage(
        width * 2,
        height * 2,
        view,
        projection,
        shadow=1,
        lightDirection=[-2, -3, 5],
        lightAmbientCoeff=0.45,
        lightDiffuseCoeff=0.65,
        lightSpecularCoeff=0.05,
        renderer=p.ER_TINY_RENDERER,
        physicsClientId=cid,
    )
    rgb = np.asarray(result[2])[:, :, :3].copy()
    # Replace only clear-depth background pixels, never specimen/tool pixels.
    rgb[np.asarray(result[3]) > 0.99999] = [245, 244, 240]
    return Image.fromarray(rgb.astype(np.uint8)).resize(
        (width, height), Image.Resampling.LANCZOS
    )


def render_fem(source):
    import pybullet as p

    source = Path(source)
    report = json.loads((source / "summary.json").read_text())
    with np.load(source / "trajectory.npz") as d:
        points = d["position_m"]
        triangles = d["surface_triangles"]
        times = d["time_s"]
        tool_positions = d["tool_position_m"]
        half = d["tool_half_extent_m"]
    reference = points[0]
    middle = np.unique(reference[:, 1])[len(np.unique(reference[:, 1])) // 2]
    ids = np.where(np.isclose(reference[:, 1], middle))[0]
    lines = []
    for coordinate in (0, 2):
        for v in np.unique(reference[ids, coordinate]):
            selected = ids[np.isclose(reference[ids, coordinate], v)]
            lines.append(
                selected[np.argsort(reference[selected, 2 if coordinate == 0 else 0])]
            )
    cid = p.connect(p.DIRECT)
    frames = []
    try:
        for i, t in enumerate(times):
            p.resetSimulation(physicsClientId=cid)
            tools = [_box(p, cid, h) for h in half]
            body = _mesh(p, cid, points[i], triangles)
            for tool, pos in zip(tools, tool_positions[i]):
                p.resetBasePositionAndOrientation(
                    tool, pos, [0, 0, 0, 1], physicsClientId=cid
                )
            grasp = report["scene"] == "grasp"
            camera = _camera(
                p,
                cid,
                [0, 0, 0.035 if grasp else 0.023],
                0.26 if grasp else 0.23,
                650,
                560,
                yaw=38,
                pitch=-17,
            )
            image = Image.new("RGB", (960, 560), BG)
            image.paste(camera, (0, 0))
            draw = ImageDraw.Draw(image)

            def project(point):
                return (805 + point[0] * 3300, 450 - point[2] * 3300)

            for line in lines:
                draw.line(
                    [project(v) for v in points[i, line]], fill="#947768", width=1
                )
            for node in points[i, ids]:
                x, z = project(node)
                draw.ellipse((x - 2, z - 2, x + 2, z + 2), fill="#765346")
            for pos, h in zip(tool_positions[i], half):
                x1, y1 = project(pos + [-h[0], 0, h[2]])
                x2, y2 = project(pos + [h[0], 0, -h[2]])
                draw.rectangle(
                    (x1, y1, x2, y2), fill="#b0b0a6", outline="#83847b", width=1
                )
            frames.append(image)
            p.removeBody(body, physicsClientId=cid)
    finally:
        p.disconnect(physicsClientId=cid)
    save_frames(frames, source / f"{report['scene']}.gif", fps=20)
    (source / "render.json").write_text(
        json.dumps(
            {
                "trajectory_sha256": hashlib.sha256(
                    (source / "trajectory.npz").read_bytes()
                ).hexdigest(),
                "geometry": "saved solver nodes and boundary triangles; no smoothing or shape interpolation",
                "renderer": "PyBullet TinyRenderer, depth buffer",
                "deformation_scale": 1.0,
                "frames": len(frames),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render_adhesion_pair(source, comparison, output):
    import pybullet as p
    import pybullet_data

    folders = [Path(comparison), Path(source)]
    data = []
    for folder in folders:
        with np.load(folder / "trajectory.npz") as d:
            data.append({k: d[k].copy() for k in d.files})
    if not np.allclose(data[0]["time_s"], data[1]["time_s"]):
        raise ValueError("adhesion pair requires equal frame times")
    traces = [
        np.genfromtxt(folder / "observables.csv", delimiter=",", names=True)
        for folder in folders
    ]
    limits = (
        min(0.0, *(trace["sphere_reaction_n"].min() for trace in traces)),
        max(0.0, *(trace["sphere_reaction_n"].max() for trace in traces)),
    )
    cid = p.connect(p.DIRECT)
    frames = []
    try:
        radius = float(data[0]["sphere_radius_m"])
        for i, t in enumerate(data[0]["time_s"]):
            # Bullet retains visual shapes after removeBody. Reset the offline
            # renderer between frames so dense meshes do not accumulate in RAM.
            p.resetSimulation(physicsClientId=cid)
            base = _box(p, cid, [0.038, 0.038, 0.003])
            p.resetBasePositionAndOrientation(
                base, [0.05, 0.05, 0.012], [0, 0, 0, 1], physicsClientId=cid
            )
            visual = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(Path(pybullet_data.getDataPath()) / "sphere_smooth.obj"),
                meshScale=[radius] * 3,
                rgbaColor=[0.65, 0.66, 0.63, 1],
                physicsClientId=cid,
            )
            sphere = p.createMultiBody(0, -1, visual, physicsClientId=cid)
            frame = Image.new("RGB", (960, 560), BG)
            draw = ImageDraw.Draw(frame)
            for panel, case in enumerate(data):
                body = _mesh(p, cid, case["position_m"][i], case["surface_triangles"])
                p.resetBasePositionAndOrientation(
                    sphere,
                    case["sphere_center_m"][i],
                    [0, 0, 0, 1],
                    physicsClientId=cid,
                )
                picture = _camera(
                    p, cid, [0.05, 0.05, 0.040], 0.22, 480, 560, yaw=35, pitch=-10
                )
                frame.paste(picture, (480 * panel, 0))
                force_inset(
                    frame,
                    traces[panel]["time_s"],
                    traces[panel]["sphere_reaction_n"],
                    t,
                    (20 + 480 * panel, 425, 260 + 480 * panel, 545),
                    limits=limits,
                )
                p.removeBody(body, physicsClientId=cid)
            frames.append(frame)
    finally:
        p.disconnect(physicsClientId=cid)
    save_frames(frames, output, fps=20)
    Path(output).with_suffix(".json").write_text(
        json.dumps(
            {
                "source_trajectory_sha256": [
                    hashlib.sha256((folder / "trajectory.npz").read_bytes()).hexdigest()
                    for folder in folders
                ],
                "force_trace_sha256": [
                    hashlib.sha256(
                        (folder / "observables.csv").read_bytes()
                    ).hexdigest()
                    for folder in folders
                ],
                "force_component": "sphere_reaction_n",
                "text_overlays": False,
                "deformation_scale": 1.0,
                "frames": len(frames),
                "renderer": "PyBullet TinyRenderer, depth buffer",
                "geometry": "saved outer material sampling layers; no shape interpolation",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
