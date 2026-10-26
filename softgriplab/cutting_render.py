"""Replay PeriPy material states and the measured Panda knife pose."""
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .animations import save_frames
from .mesh_render import BG, _box, _mesh, _camera
from .particle_surface import particle_surface
from .force_inset import force_inset
from .provenance import text_sha256


def render_cutting(source, fps=20):
    import pybullet as p
    import pybullet_data
    from .backends.robot_tools import knife_robot_urdf

    source = Path(source)
    with np.load(source / "trajectory.npz") as f:
        d = {key: f[key].copy() for key in f.files}
    offset, spacing = d["world_offset_m"], float(d["spacing_m"])
    trace = np.genfromtxt(source / "observables.csv", delimiter=",", names=True)
    cid = p.connect(p.DIRECT)
    frames, maximum_pose_error = [], 0.0
    try:
        urdf, mount = knife_robot_urdf(
            Path(pybullet_data.getDataPath()) / "franka_panda",
            source,
            d["knife_vertices"],
            d["knife_faces"],
            world_quaternion=(0, 0, 0, 1),
        )

        def setup_scene():
            p.resetSimulation(physicsClientId=cid)
            robot = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
            for half, pos, color in [
                ([0.85, 0.65, 0.012], [0.25, 0, -0.012], [0.88, 0.87, 0.83, 1]),
                ([0.18, 0.16, 0.05], [0.5, 0, 0.05], [0.65, 0.62, 0.55, 1]),
            ]:
                body = _box(p, cid, half, color)
                p.resetBasePositionAndOrientation(
                    body, pos, [0, 0, 0, 1], physicsClientId=cid
                )
            return robot

        for frame_id, t in enumerate(d["time_s"]):
            robot = setup_scene()
            for j, angle in enumerate(d["robot_joint_position"][frame_id]):
                p.resetJointState(robot, j, angle, physicsClientId=cid)
            link = p.getLinkState(
                robot, 11, computeForwardKinematics=True, physicsClientId=cid
            )
            center, quat = p.multiplyTransforms(link[4], link[5], [0, 0, 0.03], mount)
            expected = d["knife_pose_local"][frame_id]
            maximum_pose_error = max(
                maximum_pose_error,
                np.linalg.norm(np.asarray(center) - offset - expected[:3]),
            )
            if abs(np.dot(quat, expected[3:])) < 1 - 1e-6:
                raise ValueError("knife rotation differs from saved contact geometry")
            nodes, labels = d["position_m"][frame_id], d["component_labels"][frame_id]
            bodies = []
            for component in np.unique(labels):
                vertices, faces = particle_surface(nodes[labels == component], spacing)
                if len(faces):
                    bodies.append(
                        _mesh(p, cid, vertices + offset, faces, [0.79, 0.72, 0.53, 1])
                    )
                else:
                    # A sparse fragment may never reach the density isovalue.
                    # Keep it visible using its actual contact-radius samples.
                    visual = p.createVisualShape(
                        p.GEOM_SPHERE,
                        radius=0.45 * spacing,
                        rgbaColor=[0.79, 0.72, 0.53, 1],
                        physicsClientId=cid,
                    )
                    for point in nodes[labels == component]:
                        bodies.append(
                            p.createMultiBody(
                                0,
                                -1,
                                visual,
                                basePosition=point + offset,
                                physicsClientId=cid,
                            )
                        )
            image = Image.new("RGB", (1120, 560), BG)
            image.paste(
                _camera(p, cid, [0.31, 0, 0.28], 1.65, 460, 560, yaw=43, pitch=-22),
                (0, 0),
            )
            image.paste(
                _camera(p, cid, [0.50, 0, 0.12], 0.42, 660, 560, yaw=145, pitch=-28),
                (460, 0),
            )
            force_inset(
                image, trace["time_s"], trace["reaction_z_n"], t, (20, 425, 280, 545)
            )
            frames.append(image)
            for body in bodies:
                p.removeBody(body, physicsClientId=cid)
            if frame_id % 20 == 0:
                print(f"Render cutting {frame_id}/{len(d['time_s'])}", flush=True)
    finally:
        p.disconnect(physicsClientId=cid)
    if maximum_pose_error > 1e-6:
        raise ValueError(f"knife position mismatch {maximum_pose_error:g} m")
    save_frames(frames, source / "banana_cutting.gif", fps=fps)
    report = dict(
        trajectory_sha256=hashlib.sha256(
            (source / "trajectory.npz").read_bytes()
        ).hexdigest(),
        force_trace_sha256=text_sha256(source / "observables.csv"),
        force_component="reaction_z_n",
        force_trace_hash_normalization="CRLF to LF",
        text_overlays=False,
        frames=len(frames),
        deformation_scale=1,
        max_replayed_tool_center_error_m=float(maximum_pose_error),
        geometry="Marching cubes of particle density, separately per solved connected component",
        voxel_spacing_m=spacing / 3,
        gaussian_sigma_m=spacing * 0.75,
        density_isovalue=0.4,
        geometry_uses_knife_path=False,
        sparse_fragments="contact-radius spheres; no components discarded",
        renderer="PyBullet TinyRenderer",
    )
    (source / "render.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
