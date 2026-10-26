"""Render coupled demonstrations from archived physical states."""

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .animations import font, save_frames
from .mesh_render import BG, GRAY, INK, _box, _camera, _mesh


def render_adhesive_robot(source, *, fps=15):
    """Replay measured joints and material coordinates without shape animation."""
    import pybullet as p
    import pybullet_data

    from .backends.adhesive_robot import _tool_urdf

    source = Path(source)
    report = json.loads((source / "summary.json").read_text())
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
            marker, offset + [0.06, 0.13, 0.03015], [0, 0, 0, 1], physicsClientId=cid
        )
        for frame_id, time_s in enumerate(data["time_s"]):
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
            row = trace[np.argmin(np.abs(trace["time_s"] - time_s))]
            image = Image.new("RGB", (1120, 640), BG)
            image.paste(
                _camera(p, cid, [0.31, -0.02, 0.29], 1.65, 600, 475, yaw=47, pitch=-21),
                (0, 85),
            )
            focus = (nodes.mean(axis=0) + expected) / 2 - [0, 0, 0.004]
            image.paste(
                _camera(p, cid, focus, 0.26, 520, 475, yaw=-65, pitch=-12), (600, 85)
            )
            draw = ImageDraw.Draw(image)
            phase_id = np.searchsorted(
                [0.4, 1.2, 1.7, 2.7, 4.0, 4.8, 5.05, 5.3, 5.8], time_s, side="right"
            )
            phase = [
                "Approach",
                "Make contact",
                "Adhesion on",
                "Lift",
                "Carry",
                "Lower",
                "Set down",
                "Release",
                "Retreat",
                "Complete",
            ][phase_id]
            draw.text(
                (24, 14),
                "Adhesive pick and place / deformable payload",
                font=font(26),
                fill=INK,
            )
            draw.text((25, 52), f"{time_s:.2f} s    {phase}", font=font(18), fill=GRAY)
            draw.text(
                (622, 57),
                "Contact detail / actual solver mesh",
                font=font(17),
                fill=INK,
            )
            draw.text(
                (25, 568),
                f"Payload {report['mass_kg']*1000:.1f} g   |   weight {report['weight_n']:.3f} N",
                font=font(18),
                fill=INK,
            )
            draw.text(
                (622, 568),
                f"Upward contact force {-row['reaction_z_n']:+.3f} N",
                font=font(18),
                fill=INK,
            )
            draw.text(
                (25, 604),
                "Panda joint motors + Neo-Hookean MPM / deformation scale 1:1",
                font=font(14),
                fill=GRAY,
            )
            draw.text(
                (622, 604),
                f"Adhesion {'ON' if row['adhesion_on'] else 'OFF'}   |   {int(row['adhesive_points'])} active surface samples",
                font=font(14),
                fill=GRAY,
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
                "max_replayed_tool_center_error_m": max_pose_error,
                "frames": len(frames),
                "renderer": "PyBullet TinyRenderer",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
