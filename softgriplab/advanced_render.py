"""Render coupled demonstrations from archived physical states."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .animations import font, save_frames
from .mesh_render import BG, GRAY, INK, _box, _camera, _mesh
from .force_inset import force_inset


def render_adhesive_robot(source, *, fps=15):
    """Replay measured joints and material coordinates without shape animation."""
    import pybullet as p
    import pybullet_data

    from .backends.adhesive_robot import _tool_urdf

    source = Path(source)
    with np.load(source / "trajectory.npz") as archive:
        data = {key: archive[key].copy() for key in archive.files}
    trace = np.genfromtxt(source / "observables.csv", delimiter=",", names=True)
    offset = data["world_offset_m"]
    cid = p.connect(p.DIRECT)
    frames = []
    max_pose_error = 0.0
    try:
        urdf = _tool_urdf(
            Path(pybullet_data.getDataPath()) / "franka_panda",
            source,
            float(data["sphere_radius_m"]),
        )

        def setup_scene():
            p.resetSimulation(physicsClientId=cid)
            robot = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
            floor = _box(p, cid, [0.85, 0.65, 0.012], [0.88, 0.87, 0.83, 1])
            p.resetBasePositionAndOrientation(
                floor, [0.25, 0, -0.012], [0, 0, 0, 1], physicsClientId=cid
            )
            table = _box(p, cid, [0.18, 0.16, 0.05], [0.68, 0.66, 0.61, 1])
            p.resetBasePositionAndOrientation(
                table, [0.5, -0.04, 0.05], [0, 0, 0, 1], physicsClientId=cid
            )
            marker = _box(p, cid, [0.021, 0.021, 0.00015], [0.41, 0.43, 0.39, 1])
            p.resetBasePositionAndOrientation(
                marker,
                offset + [0.06, 0.13, 0.03015],
                [0, 0, 0, 1],
                physicsClientId=cid,
            )
            return robot

        for frame_id, time_s in enumerate(data["time_s"]):
            robot = setup_scene()
            for joint_id, angle in enumerate(data["robot_joint_position"][frame_id]):
                p.resetJointState(robot, joint_id, angle, physicsClientId=cid)
            link = p.getLinkState(
                robot, 11, computeForwardKinematics=True, physicsClientId=cid
            )
            center, _ = p.multiplyTransforms(
                link[4], link[5], [0, 0, 0.018], [0, 0, 0, 1]
            )
            expected = data["sphere_center_m"][frame_id] + offset
            max_pose_error = max(
                max_pose_error, float(np.linalg.norm(np.array(center) - expected))
            )
            nodes = data["position_m"][frame_id] + offset
            body = _mesh(p, cid, nodes, data["surface_triangles"])
            image = Image.new("RGB", (1120, 560), BG)
            image.paste(
                _camera(p, cid, [0.31, -0.02, 0.29], 1.65, 600, 560, yaw=47, pitch=-21),
                (0, 0),
            )
            focus = (nodes.mean(axis=0) + expected) / 2 - [0, 0, 0.004]
            image.paste(
                _camera(p, cid, focus, 0.26, 520, 560, yaw=-65, pitch=-12), (600, 0)
            )
            force_inset(
                image,
                trace["time_s"],
                -trace["reaction_z_n"],
                time_s,
                (20, 425, 280, 545),
            )
            frames.append(image)
            p.removeBody(body, physicsClientId=cid)
    finally:
        p.disconnect(physicsClientId=cid)
    if max_pose_error > 1e-6:
        raise ValueError(
            f"Replayed tool pose differs from solver contact geometry: {max_pose_error:g} m"
        )
    save_frames(frames, source / "adhesive_robot.gif", fps=fps)
    (source / "render.json").write_text(
        json.dumps(
            {
                "trajectory_sha256": hashlib.sha256(
                    (source / "trajectory.npz").read_bytes()
                ).hexdigest(),
                "deformation_scale": 1.0,
                "geometry": "saved material surface nodes and measured Panda joints",
                "force_trace_sha256": hashlib.sha256(
                    (source / "observables.csv").read_bytes()
                ).hexdigest(),
                "force_component": "-reaction_z_n",
                "text_overlays": False,
                "max_replayed_tool_center_error_m": max_pose_error,
                "frames": len(frames),
                "renderer": "PyBullet TinyRenderer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def render_banana_cutting(source, *, fps=15):
    """Replay the FEM surface, cut faces and measured knife-carrying robot."""
    import pybullet as p
    import pybullet_data
    from .backends.banana_cutting import knife_robot_urdf

    source = Path(source)
    report = json.loads((source / "summary.json").read_text())
    with np.load(source / "trajectory.npz") as archive:
        data = {k: archive[k].copy() for k in archive.files}
    trace = np.genfromtxt(source / "observables.csv", delimiter=",", names=True)
    rotation, offset = data["world_rotation"], data["world_offset_m"]
    cid = p.connect(p.DIRECT)
    frames = []
    pose_error = 0.0
    try:
        urdf, mount_q = knife_robot_urdf(
            Path(pybullet_data.getDataPath()) / "franka_panda",
            source,
            data["knife_vertices"],
            data["knife_faces"],
        )
        robot = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
        floor = _box(p, cid, [0.85, 0.65, 0.012], [0.88, 0.87, 0.83, 1])
        p.resetBasePositionAndOrientation(
            floor, [0.25, 0, -0.012], [0, 0, 0, 1], physicsClientId=cid
        )
        table = _box(p, cid, [0.18, 0.18, 0.05], [0.64, 0.61, 0.54, 1])
        p.resetBasePositionAndOrientation(
            table, [0.5, 0, 0.05], [0, 0, 0, 1], physicsClientId=cid
        )
        # The visible fixture identifies the prescribed clamped end band.
        fixed = data["reference_position_m"][data["clamped_nodes"]]
        center = fixed.mean(axis=0) @ rotation.T + offset
        clamp = _box(p, cid, [0.011, 0.0048, 0.012], [0.39, 0.41, 0.39, 1])
        p.resetBasePositionAndOrientation(
            clamp, center, [0, 0, 0, 1], physicsClientId=cid
        )
        for frame_id, time_s in enumerate(data["time_s"]):
            for joint_id, angle in enumerate(data["robot_joint_position"][frame_id]):
                p.resetJointState(robot, joint_id, angle, physicsClientId=cid)
            link = p.getLinkState(
                robot, 11, computeForwardKinematics=True, physicsClientId=cid
            )
            center, _ = p.multiplyTransforms(link[4], link[5], [0, 0, 0.03], mount_q)
            expected = data["knife_pose_local"][frame_id, :3] @ rotation.T + offset
            pose_error = max(
                pose_error, float(np.linalg.norm(np.asarray(center) - expected))
            )
            nodes = data["position_m"][frame_id] @ rotation.T + offset
            body = _mesh(
                p, cid, nodes, data["surface_triangles"], color=[0.80, 0.73, 0.54, 1]
            )
            row = trace[np.argmin(abs(trace["time_s"] - time_s))]
            frame = Image.new("RGB", (1120, 640), BG)
            frame.paste(
                _camera(p, cid, [0.31, 0, 0.28], 1.65, 500, 475, yaw=43, pitch=-22),
                (0, 85),
            )
            frame.paste(
                _camera(p, cid, [0.50, 0, 0.125], 0.49, 620, 475, yaw=135, pitch=-28),
                (500, 85),
            )
            draw = ImageDraw.Draw(frame)
            draw.text(
                (24, 14), "Panda / three sequential cuts", font=font(26), fill=INK
            )
            cut = min(int(time_s / 1.4), 2) + 1
            phase_t = time_s - (cut - 1) * 1.4
            phase = (
                "Align"
                if phase_t < 0.2
                else "Cut"
                if phase_t < 0.8
                else "Retract"
                if phase_t < 1.1
                else "Move"
            )
            if time_s >= 3.9:
                phase = "Complete"
            draw.text(
                (25, 52),
                f"{time_s:.2f} s    Cut {cut} / {phase}",
                font=font(18),
                fill=GRAY,
            )
            draw.text(
                (522, 57),
                "Knife contact and material separation",
                font=font(17),
                fill=INK,
            )
            draw.text(
                (25, 568),
                f"Vertical knife reaction {row['reaction_y_n']:+.2f} N",
                font=font(18),
                fill=INK,
            )
            complete = sum(row[f"broken_cut_{i}"] >= 1 for i in (1, 2, 3))
            pieces = (
                int(row["connected_components"])
                if "connected_components" in trace.dtype.names
                else complete + 1
            )
            draw.text(
                (522, 568),
                f"{complete} / 3 cuts complete    |    {pieces} connected material pieces",
                font=font(18),
                fill=INK,
            )
            draw.text(
                (25, 604),
                f"{report['tetrahedra']} tetrahedra / clamped end / uncalibrated fruit surrogate",
                font=font(14),
                fill=GRAY,
            )
            draw.text(
                (660, 604),
                "Saved solver mesh / deformation scale 1:1",
                font=font(14),
                fill=GRAY,
            )
            frames.append(frame)
            p.removeBody(body, physicsClientId=cid)
    finally:
        p.disconnect(physicsClientId=cid)
    if pose_error > 1e-6:
        raise ValueError(f"Knife replay center mismatch: {pose_error:g} m")
    save_frames(frames, source / "banana_cutting.gif", fps=fps)
    (source / "render.json").write_text(
        json.dumps(
            {
                "trajectory_sha256": hashlib.sha256(
                    (source / "trajectory.npz").read_bytes()
                ).hexdigest(),
                "frames": len(frames),
                "deformation_scale": 1.0,
                "max_replayed_tool_center_error_m": pose_error,
                "geometry": "saved FEM and interpolated virtual boundary nodes; measured robot joints",
                "renderer": "PyBullet TinyRenderer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
