"""Advected-surface adhesion coupled to the existing 3D JAX-MPM solver.

Uses upstream DugdaleLaw and sphere_interaction, rather than imposing a drawn
neck or a prescribed Hertz/JKR footprint. The geometry is read back from MPM.
"""
import hashlib
import json
from pathlib import Path
import numpy as np


def _surface_triangles(reference):
    values = [np.unique(reference[:, i]) for i in range(3)]
    ids = np.empty(tuple(len(v) for v in values), dtype=int)
    indices = tuple(np.searchsorted(v, reference[:, i]) for i, v in enumerate(values))
    ids[indices] = np.arange(len(reference))
    triangles = []
    for axis in range(3):
        for side in (0, -1):
            grid = np.take(ids, side, axis=axis)
            a, b, c, d = grid[:-1, :-1], grid[1:, :-1], grid[1:, 1:], grid[:-1, 1:]
            for vertices in ((a, b, c), (a, c, d)):
                triangles.extend(np.stack(vertices, axis=-1).reshape(-1, 3))
    triangles = np.asarray(triangles, dtype=np.int32)
    vertices = reference[triangles]
    normal = np.cross(vertices[:, 1] - vertices[:, 0], vertices[:, 2] - vertices[:, 0])
    inward = (
        np.einsum("ij,ij->i", normal, vertices.mean(axis=1) - reference.mean(axis=0))
        < 0
    )
    triangles[inward] = triangles[inward][:, [0, 2, 1]]
    return triangles


def run_continuum_adhesion(
    output, *, grid=24, frames=101, work_j_m2=14.4, dt_factor=1.0
):
    if type(grid) is not int or grid < 16 or type(frames) is not int or frames < 10:
        raise ValueError("grid >= 16 and frames >= 10 are required")
    if not np.isfinite(work_j_m2) or work_j_m2 < 0 or not 0 < dt_factor <= 1:
        raise ValueError("work must be finite/nonnegative; dt factor must lie in (0,1]")
    import jax
    import jax.numpy as jnp
    import mpm_jax
    from mpm_jax import Config, block, fixed_box, neo_hookean, step, validate_state
    from mpm_jax.adhesion import (
        DugdaleLaw,
        SpherePath,
        sphere_interaction,
        penalty_step_number,
    )

    jax.config.update("jax_enable_x64", True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    domain, duration, E, nu, density = 0.1, 0.13, 8000.0, 0.25, 1000.0
    lower, upper = np.array([0.014, 0.015, 0.014]), np.array([0.086, 0.038, 0.086])
    counts = np.ceil((upper - lower) / (domain / grid) * 2).astype(int)
    spacing = (upper - lower) / counts
    area = float(spacing[0] * spacing[2])
    surface_y = float(upper[1] - 0.5 * spacing[1])
    law = DugdaleLaw(
        work_of_adhesion=work_j_m2, interaction_range=0.0025, penalty_stiffness=4.8e7
    )
    material = neo_hookean(E, nu)
    wave = np.sqrt((material.bulk + 4 * material.mu / 3) / density)
    limit = (
        min(
            0.15 * domain / grid / wave,
            0.3 * np.sqrt(density * np.prod(spacing) / (law.penalty_stiffness * area)),
        )
        * dt_factor
    )
    substeps = int(np.ceil(duration / ((frames - 1) * limit)))
    dt = duration / (frames - 1) / substeps
    config = Config(
        dim=3,
        n_grid=grid,
        domain_size=domain,
        dt=dt,
        gravity=(0.0, 0.0, 0.0),
        boundary_cells=0,
        material="neo-hookean",
        boundaries=(fixed_box((0.0, 0.0, 0.0), (domain, lower[1], domain)),),
    )
    state = block(
        config, lower, upper, particles_per_cell=2, density=density, dtype=jnp.float64
    )
    reference = np.asarray(state.x)
    top = jnp.asarray(np.flatnonzero(reference[:, 1] == reference[:, 1].max()))
    radius = 0.016
    high, low, hold = (
        surface_y + radius + 0.0045,
        surface_y + radius - 0.0035,
        surface_y + radius + 0.0035,
    )
    path = SpherePath(
        radius=radius,
        times=(0.0, 0.008, 0.028, 0.042, 0.072, 0.094, 0.112, duration),
        centers=(
            (0.05, high, 0.05),
            (0.05, high, 0.05),
            (0.05, low, 0.05),
            (0.05, low, 0.05),
            (0.05, hold, 0.05),
            (0.05, hold, 0.05),
            (0.05, high + 0.025, 0.05),
            (0.05, high + 0.025, 0.05),
        ),
    )

    @jax.jit
    def advance(current):
        def body(s, _):
            center, _ = path.motion(s.time)
            contact = sphere_interaction(s.x[top], area, center, radius, law)
            force = jnp.zeros_like(s.x).at[top].add(contact.forces)
            return step(s, material, config, force), None

        return jax.lax.scan(body, current, None, length=substeps)[0]

    positions, times, centers, fields, rows = [], [], [], [], []
    for i in range(frames):
        if i:
            state = advance(state)
        validate_state(state, config)
        xyz, F = np.asarray(state.x), np.asarray(state.F)
        center, _ = path.motion(state.time)
        load = sphere_interaction(state.x[top], area, center, radius, law)
        strain = 0.5 * (F.transpose(0, 2, 1) @ F - np.eye(3))
        norm = np.linalg.norm(strain, axis=(1, 2))
        positions.append(xyz)
        times.append(float(state.time))
        centers.append(np.asarray(center))
        fields.append(norm)
        rows.append(
            [
                float(state.time),
                float(load.reaction[1]),
                float(load.gap.min()),
                float(load.adhesive_points),
                float(load.compressed_points),
                float((state.x[top, 1] - surface_y).max()),
                float(np.linalg.det(F).min()),
            ]
        )
        if i % 25 == 0:
            print(
                f"MPM adhesion W={work_j_m2:g}: {i}/{frames-1}, F={rows[-1][1]:+.3f} N",
                flush=True,
            )
    # Convert y-up solver coordinates to z-up display coordinates; no deformation scaling.
    np.savez_compressed(
        output / "trajectory.npz",
        time_s=times,
        position_m=np.asarray(positions)[:, :, [0, 2, 1]],
        sphere_center_m=np.asarray(centers)[:, [0, 2, 1]],
        sphere_radius_m=radius,
        surface_triangles=_surface_triangles(reference[:, [0, 2, 1]]),
        green_strain_norm=fields,
    )
    np.savetxt(
        output / "observables.csv",
        rows,
        delimiter=",",
        comments="",
        header="time_s,sphere_reaction_n,min_gap_m,adhesive_points,compressed_points,max_surface_u_m,min_det_f",
    )
    rows = np.asarray(rows)
    summary = {
        "backend": "JAX-MPM",
        "upstream_commit": "fd8ed2caa40a50d01992420660f68ae6ce0d084f",
        "model": "Neo-Hookean bulk + upstream finite-range Dugdale surface traction",
        "grid": grid,
        "particles": len(reference),
        "frames": frames,
        "dt_s": dt,
        "young_pa": E,
        "poisson": nu,
        "work_j_m2": work_j_m2,
        "interaction_range_m": law.interaction_range,
        "penalty_stiffness_pa_m": law.penalty_stiffness,
        "radius_m": radius,
        "max_penalty_step_number": float(
            penalty_step_number(state.mass[top], area, dt, law).max()
        ),
        "min_det_f": float(rows[:, 6].min()),
        "max_surface_lift_m": float(rows[:, 5].max()),
        "minimum_reaction_n": float(rows[:, 1].min()),
        "maximum_penetration_m": float(max(0, -rows[:, 2].min())),
        "tensile_samples": int(np.sum(rows[:, 1] < -0.001)),
        "detached_at_end": bool(rows[-1, 3] == 0 and rows[-1, 4] == 0),
        "loaded_source_sha256": {
            name: hashlib.sha256(
                (Path(mpm_jax.__file__).parent / name).read_bytes()
            ).hexdigest()
            for name in ("core.py", "adhesion.py", "materials/elastic.py")
        },
        "limitations": [
            "Dry normal adhesion, not a vacuum cup or an identified sealing model.",
            "Top material sampling layer is the contact surface; coarse particle resolution.",
            "Fixed base, prescribed sphere path and no tangential surface traction.",
            "Illustrative material/interface parameters; not experimental validation.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary
