"""Panda pick-and-place carried by bilateral finger contact and Coulomb friction."""
import json
from pathlib import Path
import numpy as np


def _target(time_s):
    times = np.array([0, 0.5, 1.8, 2.8, 4.0, 5.6, 6.9, 7.6, 8.7, 9.0])
    points = np.array(
        [
            [0.5, -0.18, 0.32],
            [0.5, -0.18, 0.32],
            [0.5, -0.18, 0.13],
            [0.5, -0.18, 0.13],
            [0.5, -0.18, 0.36],
            [0.5, 0.18, 0.36],
            [0.5, 0.18, 0.13],
            [0.5, 0.18, 0.13],
            [0.5, 0.18, 0.32],
            [0.5, 0.18, 0.32],
        ]
    )
    i = int(
        np.clip(np.searchsorted(times, time_s, side="right") - 1, 0, len(times) - 2)
    )
    s = np.clip((time_s - times[i]) / (times[i + 1] - times[i]), 0, 1)
    blend = 10 * s**3 - 15 * s**4 + 6 * s**5
    phases = [
        "Approach",
        "Lower",
        "Close fingers",
        "Lift",
        "Transfer",
        "Lower to table",
        "Open fingers",
        "Retreat",
        "Complete",
    ]
    return points[i] + blend * (points[i + 1] - points[i]), phases[i]


def run_robot(
    output, *, friction=0.8, finger_force_n=4.0, render=True, fps=15, timestep_s=1 / 240
):
    if type(fps) is not int or not 1 <= fps <= 30 or not 0 < timestep_s <= 1 / 120:
        raise ValueError("fps must be 1..30; timestep must be in (0, 1/120]")
    if (
        not np.isfinite(friction)
        or friction < 0
        or not np.isfinite(finger_force_n)
        or finger_force_n <= 0
    ):
        raise ValueError(
            "friction must be finite/nonnegative and finger force positive"
        )
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as error:
        raise ImportError(
            "robot demo requires python -m pip install '.[robot]'"
        ) from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.DIRECT)
    video, records, snapshots, events = [], [], [], []
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(timestep_s, physicsClientId=cid)
        p.setPhysicsEngineParameter(
            numSolverIterations=150,
            deterministicOverlappingPairs=1,
            physicsClientId=cid,
        )
        floor = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeVisualShape(
            floor,
            -1,
            textureUniqueId=-1,
            rgbaColor=[0.91, 0.90, 0.88, 1],
            physicsClientId=cid,
        )

        def box(half, pos, mass, color):
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half, physicsClientId=cid
            )
            visual = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid
            )
            return p.createMultiBody(mass, collision, visual, pos, physicsClientId=cid)

        box([0.25, 0.38, 0.05], [0.5, 0, 0.05], 0, [0.64, 0.62, 0.58, 1])
        marker = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.042, 0.042, 0.0003],
            rgbaColor=[0.40, 0.42, 0.38, 1],
            physicsClientId=cid,
        )
        p.createMultiBody(0, -1, marker, [0.5, 0.18, 0.1004], physicsClientId=cid)
        payload = box([0.025] * 3, [0.5, -0.18, 0.125], 0.05, [0.55, 0.32, 0.24, 1])
        p.changeDynamics(
            payload,
            -1,
            lateralFriction=friction,
            spinningFriction=0,
            rollingFriction=0,
            restitution=0,
            physicsClientId=cid,
        )
        robot = p.loadURDF(
            "franka_panda/panda.urdf", useFixedBase=True, physicsClientId=cid
        )
        orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        for i, angle in enumerate([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]):
            p.resetJointState(robot, i, angle, physicsClientId=cid)
        ik = p.calculateInverseKinematics(
            robot,
            11,
            _target(0)[0],
            orientation,
            maxNumIterations=200,
            residualThreshold=1e-6,
            physicsClientId=cid,
        )
        for i in range(7):
            p.resetJointState(robot, i, ik[i], physicsClientId=cid)
        for i in (9, 10):
            p.resetJointState(robot, i, 0.04, physicsClientId=cid)
            p.changeDynamics(
                robot,
                i,
                lateralFriction=1.0,
                spinningFriction=0,
                rollingFriction=0,
                contactStiffness=30000,
                contactDamping=100,
                physicsClientId=cid,
            )
        # No attachment joint, no disabled payload collisions, no state resets in the loop.
        view = p.computeViewMatrixFromYawPitchRoll([0.32, 0, 0.30], 1.30, 48, -24, 0, 2)
        projection = p.computeProjectionMatrixFOV(48, 560 / 430, 0.03, 3)
        stride = max(1, round(1 / (fps * timestep_s)))
        max_height, airborne_steps, bilateral_steps = 0.125, 0, 0
        penetration, first_contact = 0.0, False
        for step in range(round(9 / timestep_s) + 1):
            t = step * timestep_s
            target, phase = _target(t)
            ik = p.calculateInverseKinematics(
                robot,
                11,
                target,
                orientation,
                maxNumIterations=80,
                residualThreshold=1e-5,
                physicsClientId=cid,
            )
            p.setJointMotorControlArray(
                robot,
                list(range(7)),
                p.POSITION_CONTROL,
                targetPositions=ik[:7],
                forces=[120.0] * 7,
                positionGains=[0.18] * 7,
                velocityGains=[1.0] * 7,
                physicsClientId=cid,
            )
            opening = 0.04 if t < 1.85 or t >= 7.1 else 0.0
            for joint in (9, 10):
                p.setJointMotorControl2(
                    robot,
                    joint,
                    p.POSITION_CONTROL,
                    opening,
                    force=finger_force_n,
                    maxVelocity=0.06,
                    physicsClientId=cid,
                )
            p.stepSimulation(physicsClientId=cid)
            payload_pos, payload_orn = p.getBasePositionAndOrientation(
                payload, physicsClientId=cid
            )
            contacts = [
                p.getContactPoints(robot, payload, linkIndexA=i, physicsClientId=cid)
                for i in (9, 10)
            ]
            forces = [sum(c[9] for c in side) for side in contacts]
            gaps = [min((c[8] for c in side), default=np.nan) for side in contacts]
            bilateral = all(f > 0.01 for f in forces)
            if bilateral and not first_contact:
                events.append(
                    {
                        "event": "bilateral_contact",
                        "time_s": t + timestep_s,
                        "normal_force_n": forces,
                        "signed_distance_m": gaps,
                    }
                )
                first_contact = True
            for side in contacts:
                penetration = max(penetration, max((-c[8] for c in side), default=0.0))
            if 3.3 < t < 6.2 and payload_pos[2] > 0.18:
                airborne_steps += 1
                bilateral_steps += int(bilateral)
            max_height = max(max_height, payload_pos[2])
            actual_tcp = p.getLinkState(robot, 11, physicsClientId=cid)[4]
            tracking_error = float(np.linalg.norm(np.asarray(actual_tcp) - target))
            records.append(
                [t + timestep_s, *payload_pos, *forces, *gaps, tracking_error]
            )
            if step % stride == 0:
                joints = [
                    p.getJointState(robot, i, physicsClientId=cid)[0] for i in range(12)
                ]
                snapshots.append([t + timestep_s, *joints, *payload_pos, *payload_orn])
                if render:
                    camera = p.getCameraImage(
                        560,
                        430,
                        view,
                        projection,
                        shadow=1,
                        lightDirection=[-2, -1, 4],
                        renderer=p.ER_TINY_RENDERER,
                        physicsClientId=cid,
                    )
                    focus = (np.asarray(payload_pos) + np.asarray(actual_tcp)) / 2 + [
                        0,
                        0,
                        0.015,
                    ]
                    detail_view = p.computeViewMatrixFromYawPitchRoll(
                        focus, 0.24, 85, -22, 0, 2
                    )
                    detail_projection = p.computeProjectionMatrixFOV(
                        42, 400 / 430, 0.005, 2
                    )
                    detail = p.getCameraImage(
                        400,
                        430,
                        detail_view,
                        detail_projection,
                        shadow=1,
                        lightDirection=[-2, -1, 4],
                        renderer=p.ER_TINY_RENDERER,
                        physicsClientId=cid,
                    )
                    from ..rendering import robot_frame

                    video.append(
                        robot_frame(
                            np.asarray(camera[2])[:, :, :3],
                            np.asarray(detail[2])[:, :, :3],
                            t + timestep_s,
                            phase,
                            forces,
                            gaps,
                        )
                    )
        final_pos = np.asarray(
            p.getBasePositionAndOrientation(payload, physicsClientId=cid)[0]
        )
        fraction = bilateral_steps / max(airborne_steps, 1)
        success = bool(
            max_height > 0.30
            and np.linalg.norm(final_pos[:2] - [0.5, 0.18]) < 0.025
            and abs(final_pos[2] - 0.125) < 0.008
            and fraction > 0.95
        )
        report = {
            "backend": "PyBullet 3.2.7",
            "model": "pybullet_data/franka_panda/panda.urdf",
            "payload": "rigid 50 mm cube, mass 0.05 kg",
            "friction": friction,
            "finger_motor_force_n": finger_force_n,
            "payload_attachment_constraints": 0,
            "events": events,
            "max_payload_height_m": max_height,
            "final_payload_position_m": final_pos.tolist(),
            "pick_place_success": success,
            "airborne_bilateral_contact_fraction": fraction,
            "airborne_steps": airborne_steps,
            "max_contact_penetration_m": penetration,
            "timestep_s": timestep_s,
            "render_fps": 1 / (stride * timestep_s),
            "recorded_frames": len(snapshots),
            "limitations": [
                "Rigid payload; deformation is covered by the separate tetrahedral FEM scene.",
                "Scripted Cartesian path, numerical IK, joint motors and Coulomb contact.",
                "URDF collision meshes and visible meshes are distinct; solver permits small penetration.",
            ],
        }
        np.savetxt(
            output / "trajectory.csv",
            records,
            delimiter=",",
            comments="",
            header="time_s,payload_x_m,payload_y_m,payload_z_m,left_normal_n,right_normal_n,left_gap_m,right_gap_m,tcp_tracking_error_m",
        )
        np.savez_compressed(output / "states.npz", states=np.asarray(snapshots))
        (output / "summary.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        if render:
            from ..animations import save_frames

            save_frames(
                video, output / "robot.gif", fps=round(1 / (stride * timestep_s))
            )
        return report
    finally:
        p.disconnect(physicsClientId=cid)
