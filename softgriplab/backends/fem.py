"""Volumetric Neo-Hookean specimens solved with Bullet's deformable FEM.

Only rigid tool boundary conditions are prescribed. Specimen nodes, velocities
and strains always come from the solver, without shape interpolation or anchors.
"""
import json
from pathlib import Path
import numpy as np

from ..meshes import tetrahedral_block, write_mesh


def _smooth(t, start, end):
    s = np.clip((t - start) / (end - start), 0.0, 1.0)
    return s * s * (3 - 2 * s)


def tool_geometry(scene, t):
    if scene == "compression":
        bottom = 0.045 - 0.014 * _smooth(t, 0.4, 1.3) + 0.014 * _smooth(t, 1.8, 2.7)
        return np.array([[0.0, 0.0, -0.005], [0.0, 0.0, bottom + 0.005]]), np.array(
            [[0.035, 0.035, 0.005]] * 2
        )
    closure = 0.008 * _smooth(t, 0.4, 1.2) - 0.008 * _smooth(t, 2.5, 2.8)
    lift = 0.035 * _smooth(t, 1.4, 2.1)
    return np.array(
        [
            [0, 0, -0.005],
            [-0.034 + closure, 0, 0.03 + lift],
            [0.034 - closure, 0, 0.03 + lift],
        ]
    ), np.array([[0.045, 0.045, 0.005], [0.006, 0.022, 0.020], [0.006, 0.022, 0.020]])


def run_fem(
    output,
    *,
    scene="compression",
    cells=12,
    young_pa=15000.0,
    friction=0.7,
    timestep_s=0.0005,
    frames=91,
    render=True,
):
    if scene not in ("compression", "grasp"):
        raise ValueError("scene must be compression or grasp")
    if type(cells) is not int or cells < 3 or type(frames) is not int or frames < 10:
        raise ValueError("cells >= 3 and frames >= 10 are required")
    if (
        not (0 < timestep_s <= 0.001)
        or not np.isfinite(young_pa)
        or young_pa <= 0
        or not np.isfinite(friction)
        or friction < 0
    ):
        raise ValueError(
            "require dt in (0,.001], positive modulus and nonnegative friction"
        )
    try:
        import pybullet as p
    except ImportError as error:
        raise ImportError("FEM requires python -m pip install '.[robot]'") from error
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    x, tets, faces = tetrahedral_block(cells=(cells, cells, max(3, cells - 1)))
    obj, vtk = write_mesh(output / "mesh", x, tets, faces)
    duration, poisson, mass = 3.0, 0.35, 0.1
    steps = int(np.ceil(duration / timestep_s))
    dt = duration / steps
    frame_steps = set(np.linspace(0, steps, frames).round().astype(int))
    cid = p.connect(p.DIRECT)
    try:
        p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD, physicsClientId=cid)
        p.setGravity(0, 0, -9.81, physicsClientId=cid)
        p.setTimeStep(dt, physicsClientId=cid)
        p.setPhysicsEngineParameter(
            sparseSdfVoxelSize=0.001, numSolverIterations=100, physicsClientId=cid
        )
        initial_tools, half = tool_geometry(scene, 0)
        tools = []
        for pos, shape in zip(initial_tools, half):
            collision = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=shape, physicsClientId=cid
            )
            tools.append(p.createMultiBody(0, collision, -1, pos, physicsClientId=cid))
        soft = p.loadSoftBody(
            str(obj),
            simFileName=str(vtk),
            basePosition=[0, 0, 0.0205],
            mass=mass,
            useNeoHookean=1,
            NeoHookeanMu=young_pa / (2 * (1 + poisson)),
            NeoHookeanLambda=young_pa * poisson / ((1 + poisson) * (1 - 2 * poisson)),
            NeoHookeanDamping=0.03,
            collisionMargin=0.0003,
            useSelfCollision=0,
            frictionCoeff=friction,
            repulsionStiffness=1000,
            physicsClientId=cid,
        )
        ref = np.asarray(
            p.getMeshData(soft, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=cid)[
                1
            ]
        )
        if ref.shape != x.shape or not np.allclose(ref, x + [0, 0, 0.0205], atol=1e-10):
            raise RuntimeError(
                "Bullet changed node ordering; cannot associate tetrahedra safely"
            )
        inverse = np.linalg.inv(
            np.transpose(ref[tets[:, 1:]] - ref[tets[:, 0, None]], (0, 2, 1))
        )
        initial_volume = np.linalg.det(np.linalg.inv(inverse)) / 6
        positions, times, tool_positions, strains, rows = [], [], [], [], []
        all_step_min_j = 1.0
        for step in range(steps + 1):
            t = step * dt
            tool_pos, _ = tool_geometry(scene, t)
            if step:
                # Kinematic fixtures are displacement boundary conditions, not specimen animation.
                previous_tools, _ = tool_geometry(scene, t - dt)
                for body, pos, previous in zip(tools, tool_pos, previous_tools):
                    p.resetBasePositionAndOrientation(
                        body, previous, [0, 0, 0, 1], physicsClientId=cid
                    )
                    p.resetBaseVelocity(
                        body, linearVelocity=(pos - previous) / dt, physicsClientId=cid
                    )
                p.stepSimulation(physicsClientId=cid)
            vertices = np.asarray(
                p.getMeshData(
                    soft, flags=p.MESH_DATA_SIMULATION_MESH, physicsClientId=cid
                )[1]
            )
            F = (
                np.transpose(
                    vertices[tets[:, 1:]] - vertices[tets[:, 0, None]], (0, 2, 1)
                )
                @ inverse
            )
            jacobian = np.linalg.det(F)
            min_j = float(jacobian.min())
            if not np.all(np.isfinite(vertices)) or min_j <= 0:
                raise RuntimeError(
                    f"invalid tetrahedral state at {t:.6f} s: min J={min_j}"
                )
            all_step_min_j = min(all_step_min_j, min_j)
            if step in frame_steps:
                green = 0.5 * (F.transpose(0, 2, 1) @ F - np.eye(3))
                strain = np.linalg.norm(green, axis=(1, 2))
                center = np.average(
                    vertices[tets].mean(axis=1), axis=0, weights=initial_volume
                )
                contacts = [
                    len(p.getContactPoints(soft, tool, physicsClientId=cid))
                    for tool in tools
                ]
                positions.append(vertices.copy())
                times.append(t)
                actual_tools = np.asarray(
                    [
                        p.getBasePositionAndOrientation(body, physicsClientId=cid)[0]
                        for body in tools
                    ]
                )
                if not np.allclose(actual_tools, tool_pos, atol=1e-9):
                    raise RuntimeError(
                        "tool motion does not match prescribed boundary conditions"
                    )
                tool_positions.append(actual_tools)
                strains.append(strain)
                rows.append(
                    [
                        t,
                        *np.ptp(vertices, axis=0),
                        center[2],
                        min_j,
                        float(strain.max()),
                        *contacts,
                    ]
                )
                if len(times) % 20 == 0:
                    print(
                        f"FEM {scene}: {len(times)}/{frames} frames, min J={min_j:.3f}",
                        flush=True,
                    )
        positions = np.asarray(positions)
        np.savez_compressed(
            output / "trajectory.npz",
            time_s=times,
            position_m=positions,
            reference_m=ref,
            tetrahedra=tets,
            surface_triangles=faces,
            tool_position_m=tool_positions,
            tool_half_extent_m=half,
            green_strain_norm=strains,
        )
        np.savetxt(
            output / "observables.csv",
            rows,
            delimiter=",",
            comments="",
            header="time_s,width_x_m,width_y_m,height_m,center_z_m,min_det_f,max_green_strain_norm,"
            + ",".join(f"tool_{i}_contacts" for i in range(len(tools))),
        )
        summary = {
            "backend": "PyBullet 3.2.7 deformable FEM",
            "scene": scene,
            "material": "compressible Neo-Hookean",
            "young_pa": young_pa,
            "poisson": poisson,
            "density_kg_m3": 1000.0,
            "mass_kg": mass,
            "nodes": len(ref),
            "tetrahedra": len(tets),
            "friction": friction,
            "timestep_s": dt,
            "steps": steps,
            "frames": len(times),
            "min_det_f_all_steps": all_step_min_j,
            "collision_margin_m": 0.0003,
            "initial_height_m": float(np.ptp(ref[:, 2])),
            "minimum_height_m": float(np.ptp(positions[:, :, 2], axis=1).min()),
            "maximum_width_m": float(np.ptp(positions[:, :, 0], axis=1).max()),
            "maximum_center_height_m": float(np.asarray(rows)[:, 4].max()),
            "final_height_m": float(np.ptp(positions[-1, :, 2])),
            "specimen_anchors": 0,
            "render_deformation_scale": 1.0,
            "limitations": [
                "Prescribed rigid fixture displacement; no fixture actuator dynamics.",
                "Coarse linear tetrahedra and 0.3 mm contact margin; refinement is reported separately.",
                "Numerical damping 0.03; material values are illustrative, not identified rubber.",
            ],
        }
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    finally:
        p.disconnect(physicsClientId=cid)
    if render:
        from ..mesh_render import render_fem

        render_fem(output)
    return summary
