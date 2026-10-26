"""Motor-driven Panda pick-and-place with a contact-gated suction tool in PyBullet.

The payload is rigid. Soft-material MPM and JKR examples remain separate models.
No payload or robot state is reset after the initial configuration.
"""

from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from ..suction import SuctionSpec


def _target(time_s):
    times = np.array([0, .6, 1.8, 2.5, 3.8, 5.5, 6.8, 7.4, 8.5, 9.0])
    points = np.array([[.5, -.18, .34], [.5, -.18, .34], [.5, -.18, .1705],
                       [.5, -.18, .1705], [.5, -.18, .40], [.5, .18, .40],
                       [.5, .18, .171], [.5, .18, .171], [.5, .18, .34], [.5, .18, .34]])
    i = int(np.clip(np.searchsorted(times, time_s, side="right") - 1, 0, len(times) - 2))
    s = np.clip((time_s - times[i]) / (times[i + 1] - times[i]), 0, 1)
    blend = 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5
    phases = ["Approach", "Lower to seal", "Acquire seal", "Lift", "Transfer",
              "Lower to target", "Release", "Retreat", "Complete"]
    return points[i] + blend * (points[i + 1] - points[i]), phases[i]


def run_robot(output, *, pressure_pa=30000., render=True, fps=15, timestep_s=1 / 240):
    if type(fps) is not int or not 1 <= fps <= 30 or not 0 < timestep_s <= 1 / 120:
        raise ValueError("fps must be 1..30; timestep must be in (0, 1/120]")
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as error:
        raise ImportError("robot demo requires python -m pip install '.[robot]'") from error
    spec = SuctionSpec(pressure_pa=pressure_pa)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cid = p.connect(p.DIRECT)
    video = []
    events, records = [], []
    suction_id = None
    locked_out = False
    max_reaction = 0.
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(timestep_s, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100, deterministicOverlappingPairs=1, physicsClientId=cid)
        floor = p.loadURDF("plane.urdf", physicsClientId=cid)
        p.changeVisualShape(floor, -1, textureUniqueId=-1,
                             rgbaColor=[.82, .87, .91, 1], physicsClientId=cid)

        def box(half, pos, mass, color):
            collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half, physicsClientId=cid)
            visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half, rgbaColor=color, physicsClientId=cid)
            return p.createMultiBody(mass, collision, visual, pos, physicsClientId=cid)

        box([.25, .38, .05], [.5, 0, .05], 0, [.20, .25, .30, 1])
        # Non-colliding target patch on top of the workbench.
        marker = p.createVisualShape(p.GEOM_BOX, halfExtents=[.045, .045, .0003],
                                     rgbaColor=[.23, .65, .52, 1], physicsClientId=cid)
        p.createMultiBody(0, -1, marker, [.5, .18, .1004], physicsClientId=cid)
        payload = box([.025] * 3, [.5, -.18, .125], .05, [.95, .56, .18, 1])
        robot = p.loadURDF("franka_panda/panda.urdf", useFixedBase=True, physicsClientId=cid)
        orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        initial = [.0, -.4, .0, -2.2, .0, 1.8, .8]
        for i, angle in enumerate(initial):
            p.resetJointState(robot, i, angle, physicsClientId=cid)
        # Initial IK configuration is set once; subsequent motion uses joint motors.
        ik = p.calculateInverseKinematics(robot, 11, _target(0)[0], orientation,
                                          maxNumIterations=200, residualThreshold=1e-6, physicsClientId=cid)
        for i in range(7):
            p.resetJointState(robot, i, ik[i], physicsClientId=cid)
        for i in (9, 10):
            p.resetJointState(robot, i, .04, physicsClientId=cid)
            p.setJointMotorControl2(robot, i, p.POSITION_CONTROL, .04, force=20, physicsClientId=cid)
        tcp = p.getLinkState(robot, 11, computeForwardKinematics=True, physicsClientId=cid)
        cup_pos, cup_orn = p.multiplyTransforms(tcp[4], tcp[5], [0, 0, .015], [0, 0, 0, 1])
        cup_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=spec.radius_m, height=.012, physicsClientId=cid)
        cup_visual = p.createVisualShape(p.GEOM_CYLINDER, radius=spec.radius_m, length=.012,
                                         rgbaColor=[.10, .72, .77, 1], physicsClientId=cid)
        cup = p.createMultiBody(.02, cup_shape, cup_visual, cup_pos, cup_orn, physicsClientId=cid)
        mounting = p.createConstraint(robot, 11, cup, -1, p.JOINT_FIXED, [0, 0, 0],
                                      [0, 0, .015], [0, 0, 0], physicsClientId=cid)
        p.changeConstraint(mounting, maxForce=500, physicsClientId=cid)
        # The demonstration uses only the suction cup as the payload contact tool.
        for link in range(-1, p.getNumJoints(robot, physicsClientId=cid)):
            p.setCollisionFilterPair(robot, cup, link, -1, 0, physicsClientId=cid)
            p.setCollisionFilterPair(robot, payload, link, -1, 0, physicsClientId=cid)
        view = p.computeViewMatrixFromYawPitchRoll([.30, 0, .35], 1.45, 47, -25, 0, 2)
        projection = p.computeProjectionMatrixFOV(48, 800 / 500, .05, 3)
        stride = max(1, round(1 / (fps * timestep_s)))
        start_height = .125
        max_height = start_height
        for step in range(round(9 / timestep_s) + 1):
            t = step * timestep_s
            target, phase = _target(t)
            ik = p.calculateInverseKinematics(robot, 11, target, orientation,
                                              maxNumIterations=80, residualThreshold=1e-5, physicsClientId=cid)
            p.setJointMotorControlArray(robot, list(range(7)), p.POSITION_CONTROL,
                                        targetPositions=ik[:7], forces=[120.] * 7,
                                        positionGains=[.18] * 7, velocityGains=[1.] * 7,
                                        physicsClientId=cid)
            p.stepSimulation(physicsClientId=cid)
            cup_pos, cup_orn = p.getBasePositionAndOrientation(cup, physicsClientId=cid)
            payload_pos, payload_orn = p.getBasePositionAndOrientation(payload, physicsClientId=cid)
            tip, _ = p.multiplyTransforms(cup_pos, cup_orn, [0, 0, .006], [0, 0, 0, 1])
            inv_pos, inv_orn = p.invertTransform(payload_pos, payload_orn)
            local_tip, _ = p.multiplyTransforms(inv_pos, inv_orn, tip, [0, 0, 0, 1])
            gap_local = np.asarray(local_tip) - [0, 0, .025]
            cup_axis = np.asarray(p.getMatrixFromQuaternion(cup_orn)).reshape(3, 3)[:, 2]
            top_normal = np.asarray(p.getMatrixFromQuaternion(payload_orn)).reshape(3, 3)[:, 2]
            alignment = float(np.clip(-np.dot(cup_axis, top_normal), -1, 1))
            touching = bool(p.getContactPoints(cup, payload, physicsClientId=cid))
            if (suction_id is None and not locked_out and 1.9 <= t < 2.5
                    and spec.permits_seal(gap_local, alignment, .025, touching=touching)):
                # Preserve the actual relative transform at contact; do not snap the payload.
                inverse = p.invertTransform(cup_pos, cup_orn)
                relative_pos, relative_orn = p.multiplyTransforms(*inverse, payload_pos, payload_orn)
                suction_id = p.createConstraint(cup, -1, payload, -1, p.JOINT_FIXED, [0, 0, 0],
                                                relative_pos, [0, 0, 0],
                                                parentFrameOrientation=relative_orn,
                                                childFrameOrientation=[0, 0, 0, 1], physicsClientId=cid)
                # A stiff trial joint supplies a demand reaction; overload opens it on the next step.
                p.changeConstraint(suction_id, maxForce=100, physicsClientId=cid)
                events.append({"time_s": t, "event": "seal_acquired", "gap_m": float(gap_local[2]),
                               "alignment": alignment, "contact_confirmed": touching})
            reaction = np.zeros(6)
            if suction_id is not None:
                reaction = np.asarray(p.getConstraintState(suction_id, physicsClientId=cid))
                max_reaction = max(max_reaction, float(np.linalg.norm(reaction[:3])))
                if t >= 7.05 or spec.overloaded(reaction):
                    event = "commanded_release" if t >= 7.05 else "capacity_exceeded"
                    p.removeConstraint(suction_id, physicsClientId=cid)
                    suction_id, locked_out = None, True
                    events.append({"time_s": t, "event": event, "force_n": float(np.linalg.norm(reaction[:3])),
                                   "moment_nm": float(np.linalg.norm(reaction[3:]))})
            max_height = max(max_height, payload_pos[2])
            if step % stride == 0:
                actual_tcp = p.getLinkState(robot, 11, physicsClientId=cid)[4]
                tracking_error = float(np.linalg.norm(np.asarray(actual_tcp) - target))
                records.append([t, *payload_pos, float(suction_id is not None),
                                float(np.linalg.norm(reaction[:3])), tracking_error])
                if render:
                    camera = p.getCameraImage(800, 500, view, projection, shadow=1,
                                              lightDirection=[-1, -1, 3], renderer=p.ER_TINY_RENDERER,
                                              physicsClientId=cid)
                    detail_view = p.computeViewMatrixFromYawPitchRoll(
                        np.asarray(payload_pos) + [0, 0, .025], .28, 55, -22, 0, 2)
                    detail_projection = p.computeProjectionMatrixFOV(50, 240 / 160, .005, 2)
                    detail = p.getCameraImage(240, 160, detail_view, detail_projection, shadow=1,
                                              lightDirection=[-1, -1, 3], renderer=p.ER_TINY_RENDERER,
                                              physicsClientId=cid)
                    from ..animations import robot_frame
                    video.append(robot_frame(np.asarray(camera[2])[:, :, :3], t, phase,
                                             suction_id is not None, spec.force_capacity_n,
                                             float(np.linalg.norm(reaction[:3])), pressure_pa,
                                             detail_rgb=np.asarray(detail[2])[:, :, :3]))
        final_pos = np.asarray(p.getBasePositionAndOrientation(payload, physicsClientId=cid)[0])
        success = bool(max_height > .25 and np.linalg.norm(final_pos[:2] - [.5, .18]) < .035
                       and abs(final_pos[2] - .125) < .015
                       and any(event["event"] == "commanded_release" for event in events))
        report = {"backend": "PyBullet", "model": "pybullet_data/franka_panda/panda.urdf",
                  "payload": "rigid 50 mm cube, mass 0.05 kg", "suction": asdict(spec),
                  "capacity_n": spec.force_capacity_n, "moment_capacity_nm": spec.moment_capacity_nm,
                  "events": events, "max_payload_height_m": max_height,
                  "final_payload_position_m": final_pos.tolist(), "pick_place_success": success,
                  "max_trial_reaction_n": max_reaction, "timestep_s": timestep_s,
                  "render_fps": 1 / (stride * timestep_s), "recorded_frames": len(records),
                  "limitations": ["Rigid payload; no deformable-cup sealing or fluid simulation.",
                                  "Conservative isotropic force and moment capacity; no identified leakage model.",
                                  "Trial fixed constraint may exceed capacity for one integration step before breakage.",
                                  "Scripted Cartesian waypoints, numerical IK and joint position motors; no learned policy.",
                                  "This robot scene is not co-simulated with the separate MPM or JKR demos."]}
        np.savetxt(output / "trajectory.csv", records, delimiter=",", comments="",
                   header="time_s,payload_x_m,payload_y_m,payload_z_m,sealed,trial_force_n,tcp_tracking_error_m")
        (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        if render:
            from ..animations import save_frames
            save_frames(video, output / "robot.gif", fps=round(1 / (stride * timestep_s)))
        return report
    finally:
        p.disconnect(physicsClientId=cid)
