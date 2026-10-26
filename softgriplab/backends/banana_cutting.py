"""Optional research-only DiSECt adapter for sequential Panda knife cuts.

DiSECt is installed separately under its NVIDIA research license. This adapter
defines a specimen, planned cohesive interfaces and a partitioned robot driver;
the upstream FEM/contact/damage kernels advance the material state continuously.
"""

import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import subprocess

import numpy as np

from ..cut_quadrature import apply_cut_volume_weights
from ..meshes import tetrahedral_block, write_mesh
from .robot_tools import knife_robot_urdf

DISECT_COMMIT = "ff90165070335bb8033000373d2d20c6aac13edd"
CUT_PLANES = np.array([0.05, 1 / 300, -13 / 300])
WORLD_ROTATION = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
WORLD_OFFSET = np.array([0.50, 0.0, 0.10])


def banana_mesh():
    """A homogeneous peeled-fruit surrogate; parameters are not banana fits."""
    points, tets, faces = tetrahedral_block((0.16, 2.0, 2.0), (24, 4, 4))
    s = points[:, 0] / 0.08
    u, v = points[:, 1].copy(), points[:, 2].copy()
    radius = 0.014 * (1 - 0.6 * s**4)
    points[:, 1] = 0.015 + radius * u * np.sqrt(1 - v * v / 2)
    points[:, 2] = 0.035 * (1 - s * s) + radius * v * np.sqrt(1 - u * u / 2)
    points[:, 2] -= (points[:, 2].min() + points[:, 2].max()) / 2
    return points, tets, faces


def knife_target(t):
    high, low = 0.065, 0.0202
    times, points = [0.0], [[CUT_PLANES[0], high, 0.0]]
    for i, x in enumerate(CUT_PLANES):
        start = 1.4 * i
        times.extend([start + 0.2, start + 0.6, start + 0.8, start + 1.1])
        points.extend([[x, high, 0.0], [x, low, 0.0], [x, low, 0.0], [x, high, 0.0]])
        if i < 2:
            times.append(start + 1.4)
            points.append([CUT_PLANES[i + 1], high, 0.0])
    times.append(4.2)
    points.append(points[-1])
    i = int(np.clip(np.searchsorted(times, t, side="right") - 1, 0, len(times) - 2))
    u = np.clip((t - times[i]) / (times[i + 1] - times[i]), 0.0, 1.0)
    s = 10 * u**3 - 15 * u**4 + 6 * u**5
    return np.asarray(points[i]) + s * (np.asarray(points[i + 1]) - points[i])


def run_banana_cutting(
    output,
    *,
    source=None,
    dt=2e-5,
    coupling_dt=1 / 120,
    duration_s=4.2,
    softening_rate=2.5e7,
    fps=15,
    render=False,
):
    if (
        not 0 < dt <= 2e-5
        or not 0 < coupling_dt <= 1 / 120
        or not 0 < duration_s <= 4.2
    ):
        raise ValueError(
            "require dt <= 2e-5, coupling dt <= 1/120 and duration <=4.2 s"
        )
    if not np.isfinite(softening_rate) or softening_rate < 0 or not 1 <= fps <= 30:
        raise ValueError("invalid softening rate or frame rate")
    source = source or os.environ.get("SOFTGRIPLAB_DISECT")
    if not source:
        raise ImportError(
            "Set SOFTGRIPLAB_DISECT to a separately installed DiSECt research backend; see docs/CUTTING.md"
        )
    source = Path(source).resolve()
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=source, text=True
    ).strip()
    if revision != DISECT_COMMIT:
        raise ValueError(
            f"DiSECt revision must be {DISECT_COMMIT}; run tools/setup_disect.py"
        )
    source_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in (
            "dflex/sim.py",
            "dflex/model.py",
            "dflex/adjoint.py",
            "cutting/cutting_sim.py",
            "meshing/meshing.cpp",
        )
    }
    sys.path[:0] = [str(source), str(source / "meshing" / "build")]
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    import torch
    import dflex as df
    import pybullet as p
    import pybullet_data
    from cutting.cutting_sim import CuttingSim
    from cutting.settings import default_settings

    torch.set_num_threads(1)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    points, tets, faces = banana_mesh()
    _, vtk = write_mesh(output, points, tets, faces)
    settings = copy.deepcopy(default_settings)
    settings.generator = "meshio"
    settings.generators.meshio.filename = str(vtk)
    settings.geometry.position = [0, 0, 0]
    settings.young = 50000.0
    settings.poisson = 0.3
    settings.density = 1000.0
    settings.damping = 0.2
    substeps = int(np.ceil(coupling_dt / dt))
    dt = coupling_dt / substeps
    settings.sim_dt = dt
    settings.sim_duration = duration_s
    settings.sim_substeps = substeps
    settings.static_vertices.include = []
    settings.static_vertices.exclude = []
    settings.ground_ke = 10000.0
    settings.ground_kd = 0.003
    settings.ground_kf = 0.1
    settings.ground_mu = 0.6
    sim = CuttingSim(
        settings,
        adapter="cpu",
        requires_grad=False,
        experiment_name="softgriplab_sequential_cuts",
    )
    sim.setup_free_floating_knife()
    original_particles = len(sim.builder.particle_q)
    original_mass = float(sum(sim.builder.particle_mass))
    surfaces = []
    for x in CUT_PLANES:
        a, b, c, d = (
            [x, -0.01, -0.10],
            [x, 0.06, -0.10],
            [x, 0.06, 0.10],
            [x, -0.01, 0.10],
        )
        surfaces.extend([[a, b, c], [a, c, d]])
    sim.builder.prepare_cut(
        sim.builder.tet_indices,
        np.array(surfaces),
        use_cpp=True,
        cut_spring_ke=500.0,
        surface_cut_spring_ke=500.0,
        cut_spring_kd=0.003,
        surface_cut_spring_kd=0.003,
        # Upstream update is per step; scale it to an opening-dependent rate law.
        cut_spring_softness=softening_rate * dt,
        surface_cut_spring_softness=softening_rate * dt,
        contact_ke=1000.0,
        surface_contact_ke=1000.0,
        contact_kd=0.003,
        surface_contact_kd=0.003,
        contact_mu=0.5,
        surface_contact_mu=0.5,
    )
    raw_mass = float(sum(sim.builder.particle_mass))
    fractions = apply_cut_volume_weights(
        sim.builder, original_particles, CUT_PLANES, settings.density
    )
    physical_masses = np.asarray(sim.builder.particle_mass).copy()
    if not np.isclose(physical_masses.sum(), original_mass, rtol=1e-10):
        raise RuntimeError("cut topology quadrature does not conserve reference mass")
    reference = np.asarray(sim.builder.particle_q)
    clamp = reference[:, 0] < -0.072
    for i in np.flatnonzero(clamp):
        sim.builder.particle_mass[i] = 0.0
    sim.create_model_()
    sim.state = sim.model.state()
    indices = np.asarray(sim.builder.tet_indices)
    inverse = np.linalg.inv(
        np.transpose(
            reference[indices[:, 1:]] - reference[indices[:, 0, None]], (0, 2, 1)
        )
    )
    edges = np.asarray(sim.builder.cut_edge_indices)
    edge_coordinates = np.asarray(sim.builder.cut_edge_coords)
    springs = np.asarray(sim.builder.cut_spring_indices)
    virtual_ref = (
        reference[edges[:, 0]] * (1 - edge_coordinates[:, None])
        + reference[edges[:, 1]] * edge_coordinates[:, None]
    )
    spring_planes = np.argmin(
        abs(virtual_ref[springs[:, 0], 0, None] - CUT_PLANES[None, :]), axis=1
    )
    surface = np.vstack(
        (np.asarray(sim.builder.tri_indices), np.asarray(sim.builder.cut_tri_indices))
    )
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components

    hard_graph = coo_matrix(
        (
            np.ones(len(indices) * 3),
            (np.repeat(indices[:, 0], 3), indices[:, 1:].ravel()),
        ),
        shape=(len(reference), len(reference)),
    )
    base_components, labels = connected_components(hard_graph, directed=False)
    if not np.array_equal(labels[edges[:, 0]], labels[edges[:, 1]]):
        raise RuntimeError("virtual interpolation edge crosses a material component")
    spring_a, spring_b = (
        labels[edges[springs[:, 0], 0]],
        labels[edges[springs[:, 1], 0]],
    )

    class MeasuredMotion:
        def update_state(self, state, time_s, step_dt):
            fraction = np.clip((time_s - self.time0) / coupling_dt, 0.0, 1.0)
            q = self.start + fraction * (self.end - self.start)
            q[3:] /= torch.linalg.vector_norm(q[3:])
            state.joint_q[:] = q
            state.joint_qd[:] = self.velocity

    motion = MeasuredMotion()
    sim.motion = motion
    cid = p.connect(p.DIRECT)
    snapshots, robot_states, times, tool_centers, stiffness, rows = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    component_history = []
    force = np.zeros(3)
    minimum_j = 1.0
    max_tool_error = 0.0
    max_penetration = 0.0
    broken_previous = np.zeros(len(springs), bool)
    first_complete = [None] * 3
    try:
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(coupling_dt, physicsClientId=cid)
        p.setPhysicsEngineParameter(numSolverIterations=100, physicsClientId=cid)
        urdf, mount_q = knife_robot_urdf(
            Path(pybullet_data.getDataPath()) / "franka_panda",
            output,
            sim.knife_vertices,
            sim.knife_faces,
        )
        robot = p.loadURDF(str(urdf), useFixedBase=True, physicsClientId=cid)
        orientation = p.getQuaternionFromEuler([np.pi, 0, 0])
        for i, q in enumerate([0.0, -0.4, 0.0, -2.2, 0.0, 1.8, 0.8]):
            p.resetJointState(robot, i, q, physicsClientId=cid)
        initial = WORLD_ROTATION @ knife_target(0) + WORLD_OFFSET + [0, 0, 0.03]
        ik = p.calculateInverseKinematics(
            robot,
            11,
            initial,
            orientation,
            maxNumIterations=300,
            residualThreshold=1e-7,
            physicsClientId=cid,
        )
        for i in range(7):
            p.resetJointState(robot, i, ik[i], physicsClientId=cid)
        for i in (9, 10):
            p.resetJointState(robot, i, 0.0, physicsClientId=cid)

        def measured_pose():
            link = p.getLinkState(
                robot, 11, computeForwardKinematics=True, physicsClientId=cid
            )
            center, world_q = p.multiplyTransforms(
                link[4], link[5], [0, 0, 0.03], mount_q
            )
            _, local_q = p.multiplyTransforms(
                [0, 0, 0], [-0.5, -0.5, -0.5, 0.5], [0, 0, 0], world_q
            )
            return np.r_[(np.asarray(center) - WORLD_OFFSET) @ WORLD_ROTATION, local_q]

        current = measured_pose()
        sim.state.joint_q[:] = torch.tensor(current, dtype=torch.float32)
        stride = max(1, round(1 / (fps * coupling_dt)))
        with torch.no_grad():
            for tick in range(round(duration_s / coupling_dt) + 1):
                t = tick * coupling_dt
                if tick:
                    target = (
                        WORLD_ROTATION @ knife_target(t) + WORLD_OFFSET + [0, 0, 0.03]
                    )
                    ik = p.calculateInverseKinematics(
                        robot,
                        11,
                        target,
                        orientation,
                        maxNumIterations=100,
                        residualThreshold=1e-6,
                        physicsClientId=cid,
                    )
                    p.setJointMotorControlArray(
                        robot,
                        list(range(7)),
                        p.POSITION_CONTROL,
                        targetPositions=ik[:7],
                        forces=[120.0] * 7,
                        positionGains=[1 - 0.82 ** (coupling_dt * 120)] * 7,
                        velocityGains=[1.0] * 7,
                        physicsClientId=cid,
                    )
                    p.applyExternalForce(
                        robot,
                        11,
                        WORLD_ROTATION @ force,
                        WORLD_ROTATION @ current[:3] + WORLD_OFFSET,
                        p.WORLD_FRAME,
                        physicsClientId=cid,
                    )
                    p.stepSimulation(physicsClientId=cid)
                    after = measured_pose()
                    if np.dot(after[3:], current[3:]) < 0:
                        after[3:] *= -1
                    delta_q = df.quat_multiply(after[3:], df.quat_inverse(current[3:]))
                    angle = 2 * np.arctan2(np.linalg.norm(delta_q[:3]), delta_q[3])
                    omega = (
                        np.asarray(delta_q[:3])
                        * angle
                        / max(np.linalg.norm(delta_q[:3]), 1e-14)
                        / coupling_dt
                    )
                    motion.start = torch.tensor(current, dtype=torch.float32)
                    motion.end = torch.tensor(after, dtype=torch.float32)
                    motion.velocity = torch.tensor(
                        np.r_[(after[:3] - current[:3]) / coupling_dt, omega],
                        dtype=torch.float32,
                    )
                    motion.time0 = sim.sim_time
                    accumulated = np.zeros(3)
                    for _ in range(substeps):
                        sim.simulation_step()
                        # Upstream knife_f stores forces ON material, despite its name.
                        accumulated -= sim.state.knife_f.sum(dim=0).numpy()
                    force = accumulated / substeps
                    current = after
                x = sim.state.particle_q.numpy()
                ke = sim.state.cut_spring_ke.numpy()
                if not np.isfinite(x).all() or not np.isfinite(ke).all():
                    raise RuntimeError(f"nonfinite cutting state at {t:g} s")
                F = (
                    np.transpose(x[indices[:, 1:]] - x[indices[:, 0, None]], (0, 2, 1))
                    @ inverse
                )
                min_j = float(np.linalg.det(F).min())
                minimum_j = min(minimum_j, min_j)
                if min_j <= 0:
                    raise RuntimeError(
                        f"inverted cutting element at {t:g} s: {min_j:g}"
                    )
                broken = ke < 1e-5
                if np.any(broken_previous & ~broken):
                    raise RuntimeError("a broken cutting spring healed")
                broken_previous = broken.copy()
                graph = np.eye(base_components, dtype=bool)
                graph[spring_a[~broken], spring_b[~broken]] = True
                pieces = int(
                    connected_components(
                        csr_matrix(graph), directed=False, return_labels=False
                    )
                )
                fractions_broken = [
                    float(broken[spring_planes == i].mean()) for i in range(3)
                ]
                for i, value in enumerate(fractions_broken):
                    if value == 1 and first_complete[i] is None:
                        first_complete[i] = t
                virtual = (
                    x[edges[:, 0]] * (1 - edge_coordinates[:, None])
                    + x[edges[:, 1]] * edge_coordinates[:, None]
                )
                visible = np.vstack((x, virtual))
                error = float(np.linalg.norm(current[:3] - knife_target(t)))
                max_tool_error = max(max_tool_error, error)
                max_penetration = max(
                    max_penetration, max(0.0, -float(visible[:, 1].min()))
                )
                rows.append(
                    [t, *current[:3], *force, *fractions_broken, min_j, error, pieces]
                )
                if tick % stride == 0 or tick == round(duration_s / coupling_dt):
                    snapshots.append(visible.copy())
                    stiffness.append(ke.copy())
                    times.append(t)
                    robot_states.append(
                        [
                            p.getJointState(robot, j, physicsClientId=cid)[0]
                            for j in range(12)
                        ]
                    )
                    tool_centers.append(current.copy())
                    component_history.append(pieces)
                if tick % 60 == 0:
                    print(
                        f"Cutting {t:.2f} s / broken interfaces {fractions_broken} / min J {min_j:.3f}",
                        flush=True,
                    )
        np.savez_compressed(
            output / "trajectory.npz",
            time_s=times,
            position_m=snapshots,
            surface_triangles=surface,
            robot_joint_position=robot_states,
            knife_pose_local=tool_centers,
            knife_vertices=sim.knife_vertices,
            knife_faces=np.asarray(sim.knife_faces).reshape(-1, 3),
            cut_stiffness_n_m=stiffness,
            cut_spring_plane=spring_planes,
            world_rotation=WORLD_ROTATION,
            world_offset_m=WORLD_OFFSET,
            reference_position_m=np.vstack((reference, virtual_ref)),
            tet_indices=indices,
            physical_mass_kg=physical_masses,
            clamped_nodes=np.flatnonzero(clamp),
            cut_volume_fraction=fractions,
            connected_components=component_history,
            cut_spring_indices=springs,
            cut_edge_indices=edges,
            cut_edge_coordinates=edge_coordinates,
            hard_particle_count=len(reference),
            tet_component_labels=labels,
        )
        np.savetxt(
            output / "observables.csv",
            rows,
            delimiter=",",
            comments="",
            header="time_s,knife_x_m,knife_y_m,knife_z_m,reaction_x_n,reaction_y_n,reaction_z_n,broken_cut_1,broken_cut_2,broken_cut_3,min_det_f,tracking_error_m,connected_components",
        )
        summary = {
            "backend": "DiSECt CPU FEM with PyBullet Panda",
            "disect_commit": DISECT_COMMIT,
            "backend_source_sha256": source_hashes,
            "young_pa": settings.young,
            "poisson": settings.poisson,
            "density_kg_m3": settings.density,
            "reference_mass_kg": original_mass,
            "raw_virtual_node_mass_kg": raw_mass,
            "integrated_reference_mass_kg": float(physical_masses.sum()),
            "clamped_nodes": int(clamp.sum()),
            "tetrahedra": len(indices),
            "cutting_springs": len(springs),
            "min_cut_volume_fraction": float(fractions.min()),
            "dt_s": dt,
            "coupling_dt_s": coupling_dt,
            "duration_s": duration_s,
            "softening_rate_n_m2_s": softening_rate,
            "ground_penalty_n_m": settings.ground_ke,
            "commanded_lowest_blade_tip_m": 0.0202 - 0.02004,
            "completed_cut_times_s": first_complete,
            "final_broken_fractions": fractions_broken,
            "min_det_f": minimum_j,
            "max_support_node_penetration_m": max_penetration,
            "max_saved_surface_penetration_m": max(
                0.0, -float(np.asarray(snapshots)[:, np.unique(surface), 1].min())
            ),
            "max_tracking_error_m": max_tool_error,
            "success": all(t is not None for t in first_complete) and pieces == 4,
            "reaction_force_feedback": True,
            "initial_connected_components": component_history[0],
            "final_connected_components": pieces,
            "limitations": [
                "Homogeneous uncalibrated peeled-banana geometry; no peel, fibres or viscoelasticity.",
                "Three planned cohesive planes; interfaces remain connected until knife-contact damage breaks their springs.",
                "Opening-dependent softening, scaled by dt; not an experimentally fitted fracture-energy law.",
                "End band is clamped; separated pieces retain state and remain free.",
                "Upstream paired-interface contact; no general collision between arbitrarily rotated fragments.",
                "Force-only partitioned robot feedback; no knife torque feedback or arm/material collision.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        p.disconnect(physicsClientId=cid)
    if render:
        from ..advanced_render import render_banana_cutting

        render_banana_cutting(output)
    return summary
