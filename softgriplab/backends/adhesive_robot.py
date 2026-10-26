"""Partitioned Panda / deformable MPM coupling through actual adhesive tractions.

Robot motion is integrated with joint motors. Its measured tool pose drives MPM
contact; equal and opposite interval-averaged reaction is returned to the arm.
The payload has gravity and no anchors, fixed base or attachment constraint.
"""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .continuum_adhesion import _surface_triangles


def _tool_urdf(asset_folder, output, radius):
    """Replace the parallel fingers with a spherical adhesive head on the Panda."""
    tree = ET.parse(asset_folder / "panda.urdf")
    root = tree.getroot()
    for mesh in root.findall(".//mesh"):
        mesh.set(
            "filename",
            str(asset_folder / mesh.get("filename").replace("package://", "")),
        )
    for name in ("panda_leftfinger", "panda_rightfinger"):
        link = root.find(f"link[@name='{name}']")
        for tag in ("visual", "collision"):
            for node in list(link.findall(tag)):
                link.remove(node)
    target = root.find("link[@name='panda_grasptarget']")
    target.find("inertial/mass").set("value", ".03")
    target.find("inertial/origin").set("xyz", "0 0 .018")
    inertia = target.find("inertial/inertia")
    for field in ("ixx", "iyy", "izz"):
        inertia.set(field, "0.00001")
    for kind in ("sphere", "cylinder"):
        visual = ET.SubElement(target, "visual")
        ET.SubElement(
            visual,
            "origin",
            xyz="0 0 .018" if kind == "sphere" else "0 0 -.012",
            rpy="0 0 0",
        )
        geometry = ET.SubElement(visual, "geometry")
        if kind == "sphere":
            ET.SubElement(
                geometry,
                "mesh",
                filename=str(asset_folder.parent / "sphere_smooth.obj"),
                scale=" ".join([str(radius)] * 3),
            )
        else:
            ET.SubElement(geometry, "cylinder", radius=".006", length=".060")
        material = ET.SubElement(visual, "material", name="adhesive_head_" + kind)
        ET.SubElement(
            material,
            "color",
            rgba=".43 .46 .40 1" if kind == "sphere" else ".65 .65 .62 1",
        )
    path = output / "panda_adhesive.urdf"
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return path


def _target(time_s, release_mode="switch", release_duration_s=0.12):
    if release_mode == "inertial" and time_s >= 4.0:
        u = np.clip((time_s - 4.4) / release_duration_s, 0.0, 1.0)
        # Constant upward acceleration for half the stroke, then braking.
        # The measured robot motion, rather than this command, drives contact.
        s = 2 * u**2 if u <= 0.5 else 1 - 2 * (1 - u) ** 2
        return np.array([0.06, 0.13, 0.127 + 0.04 * s])
    times = np.array([0.0, 0.4, 1.2, 1.7, 2.7, 4.0, 4.8, 5.3, 5.8, 6.0])
    centers = np.array(
        [
            [0.06, 0.05, 0.117],
            [0.06, 0.05, 0.117],
            [0.06, 0.05, 0.076],
            [0.06, 0.05, 0.076],
            [0.06, 0.05, 0.127],
            [0.06, 0.13, 0.127],
            [0.06, 0.13, 0.076],
            [0.06, 0.13, 0.076],
            [0.06, 0.13, 0.117],
            [0.06, 0.13, 0.117],
        ]
    )
    i = int(
        np.clip(np.searchsorted(times, time_s, side="right") - 1, 0, len(times) - 2)
    )
    u = np.clip((time_s - times[i]) / (times[i + 1] - times[i]), 0, 1)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    return centers[i] + s * (centers[i + 1] - centers[i])


def run_adhesive_robot(
    output,
    *,
    work_j_m2=4.0,
    grid=40,
    render=False,
    fps=15,
    coupling_dt=1 / 120,
    dt_factor=0.5,
    young_pa=12000.0,
    release_mode="inertial",
    release_duration_s=0.04,
    release_position_gain=0.6,
):
    if (
        release_mode not in ("switch", "inertial")
        or not 0.04 <= release_duration_s <= 1.4
        or not 0 < release_position_gain <= 1
    ):
        raise ValueError(
            "release mode must be switch/inertial and duration between .04 and 1.4 s"
        )
    if (
        work_j_m2 < 0
        or not np.isfinite(work_j_m2)
        or grid < 24
        or not 0 < dt_factor <= 1
    ):
        raise ValueError(
            "require nonnegative finite adhesion, grid >=24 and dt factor in (0,1]"
        )
    if not 0 < coupling_dt <= 1 / 120 or not 1 <= fps <= 30 or young_pa <= 0:
        raise ValueError("invalid coupling time step, fps or elastic modulus")
    import jax
    import jax.numpy as jnp
    import mpm_jax
    from mpm_jax import Config, PlaneWall, block, neo_hookean, step, validate_state
    from mpm_jax.adhesion import DugdaleLaw, sphere_interaction, penalty_step_number
    import pybullet as p
    import pybullet_data

    jax.config.update("jax_enable_x64", True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    domain, density, nu, radius = 0.18, 1000.0, 0.3, 0.018
    lower, upper = np.array([0.045, 0.035, 0.03]), np.array([0.075, 0.065, 0.06])
    counts = np.ceil((upper - lower) / (domain / grid) * 3).astype(int)
    spacing = (upper - lower) / counts
    area = float(spacing[0] * spacing[1])
    law = DugdaleLaw(
        work_of_adhesion=work_j_m2, interaction_range=0.0008, penalty_stiffness=4.8e7
    )
    material = neo_hookean(young_pa, nu)
    wave = np.sqrt((material.bulk + 4 * material.mu / 3) / density)
    limit = (
        min(
            0.12 * domain / grid / wave,
            0.25 * np.sqrt(density * np.prod(spacing) / (law.penalty_stiffness * area)),
        )
        * dt_factor
    )
    substeps = int(np.ceil(coupling_dt / limit))
    dt = coupling_dt / substeps
    cfg = Config(
        dim=3,
        n_grid=grid,
        domain_size=domain,
        dt=dt,
        material="neo-hookean",
        gravity=(0.0, 0.0, -9.81),
        boundary_cells=0,
        boundaries=(PlaneWall(2, 0.03, "lower", 0.5),),
    )
    state = block(
        cfg, lower, upper, particles_per_cell=3, density=density, dtype=jnp.float64
    )
    reference = np.asarray(state.x)
    top = jnp.asarray(np.flatnonzero(reference[:, 2] == reference[:, 2].max()))
    total_mass = float(jnp.sum(state.mass))
    offset = np.array([0.42, -0.13, 0.07])

    @jax.jit
    def advance(current, start_center, end_center, activation):
        def body(carry, index):
            s, impulse, penetration = carry
            center = (
                start_center + (end_center - start_center) * (index + 0.5) / substeps
            )
            contact = sphere_interaction(s.x[top], area, center, radius, law)
            # The ideal actuator changes only attraction; repulsive contact stays on.
            factor = jnp.where(contact.gap < 0, 1.0, activation)
            surface_force = contact.forces * factor[:, None]
            force = jnp.zeros_like(s.x).at[top].add(surface_force)
            reaction = -jnp.sum(surface_force, axis=0)
            return (
                step(s, material, cfg, force),
                impulse + reaction * dt,
                jnp.maximum(penetration, jnp.maximum(-contact.gap.min(), 0.0)),
            ), None

        final, _ = jax.lax.scan(
            body, (current, jnp.zeros(3), jnp.asarray(0.0)), jnp.arange(substeps)
        )
        return final[0], final[1] / coupling_dt, final[2]

    cid = p.connect(p.DIRECT)
    positions, times, centers, joints, rows = [], [], [], [], []
    reaction = np.zeros(3)
    max_penetration = 0.0
    source = Path(mpm_jax.__file__).parent
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(coupling_dt, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=cid)
        urdf = _tool_urdf(
            Path(pybullet_data.getDataPath()) / "franka_panda", output, radius
        )
        robot = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
        orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        for i, q in enumerate([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]):
            p.resetJointState(robot, i, q, physicsClientId=cid)
        target = _target(0, release_mode, release_duration_s) + offset + [0, 0, 0.018]
        ik = p.calculateInverseKinematics(
            robot,
            11,
            target,
            orientation,
            maxNumIterations=300,
            residualThreshold=1e-7,
            physicsClientId=cid,
        )
        for i in range(7):
            p.resetJointState(robot, i, ik[i], physicsClientId=cid)
        for i in (9, 10):
            p.resetJointState(robot, i, 0, physicsClientId=cid)
            p.setJointMotorControl2(
                robot, i, p.POSITION_CONTROL, 0, force=10, physicsClientId=cid
            )

        def measured_center():
            link = p.getLinkState(
                robot, 11, computeForwardKinematics=True, physicsClientId=cid
            )
            world, _ = p.multiplyTransforms(
                link[4], link[5], [0, 0, 0.018], [0, 0, 0, 1]
            )
            return np.asarray(world) - offset

        current = measured_center()
        head_velocity = np.zeros(3)
        head_acceleration = np.zeros(3)
        stride = max(1, round(1 / (fps * coupling_dt)))
        steps = round(6 / coupling_dt)
        established_contact = False
        first_compressive_contact = None
        for tick in range(steps + 1):
            t = tick * coupling_dt
            activation = float(
                work_j_m2 > 0
                and established_contact
                and t >= 1.25
                and (release_mode == "inertial" or t < 5.05)
            )
            if tick:
                target = (
                    _target(t, release_mode, release_duration_s)
                    + offset
                    + [0, 0, 0.018]
                )
                ik = p.calculateInverseKinematics(
                    robot,
                    11,
                    target,
                    orientation,
                    maxNumIterations=80,
                    residualThreshold=1e-6,
                    physicsClientId=cid,
                )
                p.setJointMotorControlArray(
                    robot,
                    list(range(7)),
                    p.POSITION_CONTROL,
                    targetPositions=ik[:7],
                    forces=[120.0] * 7,
                    positionGains=[
                        1
                        - (
                            1
                            - (
                                release_position_gain
                                if release_mode == "inertial" and t >= 4.4
                                else 0.18
                            )
                        )
                        ** (coupling_dt * 120)
                    ]
                    * 7,
                    velocityGains=[1.0] * 7,
                    physicsClientId=cid,
                )
                # First-order partitioned coupling: previous interval reaction loads the next arm step.
                p.applyExternalForce(
                    robot,
                    11,
                    reaction,
                    current + offset,
                    p.WORLD_FRAME,
                    physicsClientId=cid,
                )
                p.stepSimulation(physicsClientId=cid)
                after = measured_center()
                next_velocity = (after - current) / coupling_dt
                head_acceleration = (next_velocity - head_velocity) / coupling_dt
                head_velocity = next_velocity
                state, reaction_j, penetration = advance(
                    state, jnp.asarray(current), jnp.asarray(after), activation
                )
                reaction = np.asarray(reaction_j)
                max_penetration = max(max_penetration, float(penetration))
                current = after
                try:
                    validate_state(state, cfg)
                except (FloatingPointError, ValueError):
                    np.savez_compressed(
                        output / "failure_state.npz",
                        time_s=t,
                        position_m=np.asarray(state.x),
                        velocity_m_s=np.asarray(state.v),
                        deformation_gradient=np.asarray(state.F),
                        sphere_center_m=current,
                        mass_kg=np.asarray(state.mass),
                    )
                    raise
            x = np.asarray(state.x)
            com = np.average(x, axis=0, weights=np.asarray(state.mass))
            velocity = np.average(
                np.asarray(state.v), axis=0, weights=np.asarray(state.mass)
            )
            load = sphere_interaction(
                state.x[top], area, jnp.asarray(current), radius, law
            )
            if float(load.compressed_points) > 0:
                established_contact = True
                if first_compressive_contact is None:
                    first_compressive_contact = t
            rows.append(
                [
                    t,
                    *com,
                    *velocity,
                    *reaction,
                    activation,
                    activation * float(load.adhesive_points),
                    float(load.gap.min()),
                    float(np.linalg.det(np.asarray(state.F)).min()),
                    float(
                        np.linalg.norm(
                            current - _target(t, release_mode, release_duration_s)
                        )
                    ),
                    float(x[:, 2].min()),
                    float(x[:, 2].max()),
                    float(load.compressed_points),
                    *current,
                    *head_velocity,
                    *head_acceleration,
                ]
            )
            if tick % stride == 0 or (release_mode == "inertial" and 4.35 <= t <= 4.8):
                positions.append(x.copy())
                times.append(t)
                centers.append(current.copy())
                joints.append(
                    [
                        p.getJointState(robot, i, physicsClientId=cid)[0]
                        for i in range(12)
                    ]
                )
            if tick % 120 == 0:
                print(
                    f"Adhesive robot W={work_j_m2:g}: {t:.1f} s, COM z={com[2]*1000:.1f} mm, Fz={-reaction[2]:+.3f} N",
                    flush=True,
                )
        rows = np.asarray(rows)
        airborne = (rows[:, 0] > 2.8) & (rows[:, 0] < 3.8)
        max_height = float(rows[:, 3].max())
        final_com = rows[-1, 1:4]
        airborne_fraction = float(
            np.mean(rows[airborne, 15] > 0.03 + 2 * domain / grid)
        )
        active_fraction = float(np.mean(rows[airborne, 11] > 0))
        first, last = np.flatnonzero(airborne)[[0, -1]]
        impulse = (
            np.sum(
                -rows[first + 1 : last + 1, 7:10] - [0, 0, total_mass * 9.81], axis=0
            )
            * coupling_dt
        )
        momentum_change = total_mass * (rows[last, 4:7] - rows[first, 4:7])
        momentum_residual = float(np.linalg.norm(momentum_change - impulse))
        if airborne_fraction < 0.95:
            momentum_residual = (
                None  # Floor impulse is not included in this airborne check.
            )
        success = bool(
            max_height > 0.08
            and np.linalg.norm(final_com[:2] - [0.06, 0.13]) < 0.015
            and final_com[2] < 0.055
            and airborne_fraction > 0.95
            and active_fraction > 0.95
            and rows[-1, 11] == 0
            and np.linalg.norm(rows[-1, 7:10]) < 1e-6
        )
        np.savez_compressed(
            output / "trajectory.npz",
            time_s=times,
            position_m=positions,
            surface_triangles=_surface_triangles(reference),
            sphere_center_m=centers,
            sphere_radius_m=radius,
            robot_joint_position=joints,
            world_offset_m=offset,
        )
        np.savetxt(
            output / "observables.csv",
            rows,
            delimiter=",",
            comments="",
            header="time_s,com_x_m,com_y_m,com_z_m,com_vx_m_s,com_vy_m_s,com_vz_m_s,reaction_x_n,reaction_y_n,reaction_z_n,adhesion_on,adhesive_points,min_gap_m,min_det_f,tool_tracking_error_m,bottom_z_m,top_z_m,compressed_points,head_x_m,head_y_m,head_z_m,head_vx_m_s,head_vy_m_s,head_vz_m_s,head_ax_m_s2,head_ay_m_s2,head_az_m_s2",
        )
        summary = {
            "backend": "partitioned PyBullet Panda + JAX-MPM",
            "jax_version": jax.__version__,
            "numpy_version": np.__version__,
            "release_mode": release_mode,
            "release_duration_s": release_duration_s,
            "release_position_gain": release_position_gain,
            "release_stroke_m": 0.04,
            "release_detail_frames": release_mode == "inertial",
            "release_start_s": 4.4 if release_mode == "inertial" else 5.05,
            "commanded_upward_acceleration_m_s2": 4 * 0.04 / release_duration_s**2
            if release_mode == "inertial"
            else None,
            "max_measured_upward_acceleration_m_s2": float(
                rows[rows[:, 0] >= 4.4, 26].max()
            ),
            "detached_at_end": bool(
                rows[-1, 11] == 0
                and rows[-1, 17] == 0
                and rows[-1, 12] > law.interaction_range
            ),
            "adhesion_switched_off_for_release": release_mode == "switch",
            "work_j_m2": work_j_m2,
            "young_pa": young_pa,
            "poisson": nu,
            "grid": grid,
            "grid_domain_m": domain,
            "particles": len(reference),
            "mass_kg": total_mass,
            "weight_n": total_mass * 9.81,
            "coupling_dt_s": coupling_dt,
            "mpm_dt_s": dt,
            "mpm_substeps_per_robot_step": substeps,
            "interaction_range_m": law.interaction_range,
            "max_penetration_m": max_penetration,
            "max_penalty_step_number": float(
                penalty_step_number(state.mass[top], area, dt, law).max()
            ),
            "payload_anchors": 0,
            "payload_attachment_constraints": 0,
            "reaction_feedback": True,
            "first_compressive_contact_s": first_compressive_contact,
            "max_com_height_m": max_height,
            "final_com_m": final_com.tolist(),
            "pick_place_success": success,
            "mean_airborne_support_n": float(-rows[airborne, 9].mean()),
            "airborne_fraction": airborne_fraction,
            "airborne_active_contact_fraction": active_fraction,
            "airborne_momentum_balance_residual_n_s": momentum_residual,
            "final_contact_force_n": float(np.linalg.norm(rows[-1, 7:10])),
            "max_tracking_error_m": float(rows[:, 14].max()),
            "min_det_f": float(rows[:, 13].min()),
            "loaded_source_sha256": {
                name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                for name in ("core.py", "adhesion.py", "boundaries.py")
            },
            "limitations": [
                "Finite-range dry adhesion; no vacuum, electrostatic field or leakage model.",
                "Spherical head acts on the original top material sampling layer only.",
                "One coupling interval reaction lag; grid-based floor contact has spatial error.",
                "Robot meshes outside the adhesive head do not collide with MPM particles.",
                "Elastic uncalibrated material; no fracture or damage in this scene.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        p.disconnect(physicsClientId=cid)
    if render:
        from ..advanced_render import render_adhesive_robot

        render_adhesive_robot(output)
    return summary
