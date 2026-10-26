"""Three-dimensional grasp scene using JAX-MPM's existing solver and contacts.

This separate Neo-Hookean integration example is NOT the SLS material model.
The frame acceleration pulse is distributed inertial loading, not local impact.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np


def run_grasp(output, *, grid=24, frames=40, friction=0.6, shock_peak=20.0,
              dt_factor=1.0, young_pa=35000, gif=False):
    if type(grid) is not int or grid < 16 or type(frames) is not int or frames < 4:
        raise ValueError("grid must be an integer >=16 and frames an integer >=4")
    if not np.isfinite(friction) or friction < 0 or not np.isfinite(shock_peak):
        raise ValueError("friction must be finite/nonnegative and shock peak finite")
    if not np.isfinite(young_pa) or young_pa <= 0 or not 0 < dt_factor <= 1:
        raise ValueError("Young's modulus must be positive; dt-factor must lie in (0,1]")
    try:
        import jax
        import jax.numpy as jnp
        import mpm_jax
        from mpm_jax import Config, PlaneWall, block, neo_hookean, step, validate_state
        from mpm_jax.export import export_series
    except ImportError as error:
        raise ImportError("mpm-grasp requires Python >=3.12 and the optional jax-mpm dependency; "
                          "install with python -m pip install '.[mpm]' in a separate environment") from error

    jax.config.update("jax_enable_x64", True)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    density, poisson, domain = 1000.0, 0.3, 0.12
    close_s, end_s, closure_m = 0.03, 0.12, 0.003
    left, right = 0.045, 0.075
    wave_speed = np.sqrt(young_pa * (1 - poisson) / ((1 + poisson) * (1 - 2 * poisson) * density))
    substeps = int(np.ceil((end_s / frames) / (0.15 * domain / grid / wave_speed * dt_factor)))
    dt = end_s / frames / substeps
    speed = closure_m / (2 * close_s)
    moving = (PlaneWall(0, left, "lower", friction, (speed, 0., 0.)),
              PlaneWall(0, right, "upper", friction, (-speed, 0., 0.)))
    held = (PlaneWall(0, left + closure_m / 2, "lower", friction),
            PlaneWall(0, right - closure_m / 2, "upper", friction))
    config = Config(n_grid=grid, dim=3, domain_size=domain, dt=dt,
                    gravity=(0., 0., 0.), boundary_cells=0,
                    material="neo-hookean", boundaries=moving)
    hold_config = replace(config, boundaries=held)
    material = neo_hookean(young=young_pa, poisson=poisson)
    state = block(config, (left, 0.045, 0.045), (right, 0.075, 0.075),
                  particles_per_cell=2, density=density, dtype=jnp.float64)

    @jax.jit
    def advance_frame(state):
        def body(state, _):
            t = state.time
            # Smoothly transfer gravity to the grasp after squeeze, in the gripper frame.
            support = jnp.clip((t - close_s) / 0.015, 0., 1.)
            phase = (t - 0.075) / 0.015
            shock = jnp.where((phase >= 0) & (phase <= 1), shock_peak * jnp.sin(jnp.pi * phase), 0.)
            force = state.mass[:, None] * jnp.array([0., -9.81 * support - shock, 0.])[None, :]
            new = jax.lax.cond(t < close_s,
                               lambda s: step(s, material, config, force),
                               lambda s: step(s, material, hold_config, force), state)
            return new, None
        return jax.lax.scan(body, state, None, length=substeps)[0]

    start = perf_counter()
    warm = advance_frame(state)
    warm.x.block_until_ready()
    compile_seconds = perf_counter() - start
    positions, fields, times, rows = [], [], [], []
    reference = np.asarray(state.x)
    reference_com = np.average(reference, axis=0, weights=np.asarray(state.mass))
    start = perf_counter()
    for i in range(frames + 1):
        if i:
            state = advance_frame(state)
        validate_state(state, config)
        x, velocity, F = np.asarray(state.x), np.asarray(state.v), np.asarray(state.F)
        J = np.linalg.det(F)
        time_s = float(state.time)
        com = np.average(x, axis=0, weights=np.asarray(state.mass))
        strain = 0.5 * (np.swapaxes(F, -1, -2) @ F - np.eye(3))
        strain_norm = np.sqrt(np.sum(strain ** 2, axis=(1, 2)))
        positions.append(x)
        fields.append(strain_norm)
        times.append(time_s)
        rows.append((time_s, com[1] - reference_com[1], float(np.min(J)),
                     float(np.max(strain_norm)), float(.5 * np.sum(np.asarray(state.mass) * np.sum(velocity ** 2, axis=1)))))
        if i % max(1, frames // 4) == 0:
            print(f"MPM frame {i}/{frames}, t={time_s:.4f} s, COM dy={rows[-1][1]:.6f} m", flush=True)
    solver_seconds = perf_counter() - start
    positions, fields, times = np.asarray(positions), np.asarray(fields), np.asarray(times)
    np.savez_compressed(output / "trajectory.npz", time_s=times, position_m=positions,
                        green_strain_norm=fields)
    np.savetxt(output / "observables.csv", rows, delimiter=",", comments="",
               header="time_s,com_vertical_displacement_m,min_det_f,max_green_strain_norm,kinetic_energy_j")
    export_series(positions, {"green_strain_norm": fields}, times, output, dim=3, displacement=True)
    source = Path(mpm_jax.__file__).parent
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
              for name in ("core.py", "boundaries.py", "initialization.py", "materials/elastic.py")}
    summary = {
        "backend": "jax-mpm", "material": "Neo-Hookean; independent of SLS identification demo",
        "upstream_reference_commit": "fd8ed2caa40a50d01992420660f68ae6ce0d084f",
        "loaded_source_sha256": hashes,
        "grid": grid, "particles": len(reference), "frames": frames, "steps": frames * substeps,
        "dt_s": dt, "young_pa": young_pa, "poisson": poisson, "density_kg_m3": density,
        "friction": friction, "total_closure_m": closure_m, "duration_s": end_s,
        "shock": {"start_s": .075, "duration_s": .015, "peak_m_s2": shock_peak},
        "min_det_f": min(row[2] for row in rows), "final_com_drop_m": -rows[-1][1],
        "compile_seconds": compile_seconds, "solver_seconds": solver_seconds,
        "jax_version": jax.__version__, "device": str(jax.devices()[0]),
        "limitations": ["Infinite planar jaws with upstream grid-based Coulomb boundaries.",
                        "Contact positions have grid-scale error; convergence is not established.",
                        "No actuator dynamics, finite fingers, damage, or local impactor.",
                        "The material is elastic, not calibrated rate-dependent rubber.",
                        "Frame-sampled determinant minima do not replace upstream per-step state checks."],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _plot(output, times, positions, fields, rows, left, right, closure_m, close_s, gif)
    return summary


def _plot(output, times, positions, fields, rows, left, right, closure, close_s, gif):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    rows = np.asarray(rows)
    axes[0].plot(times, -rows[:, 1] * 1000)
    axes[0].set(xlabel="Time (s)", ylabel="COM drop in jaw frame (mm)", title="Squeeze, gravity transfer, base shock")
    axes[0].axvspan(.075, .09, alpha=.15, color="orange")
    axes[1].plot(times, rows[:, 3])
    axes[1].set(xlabel="Time (s)", ylabel="Max Green strain norm", title="Deformation diagnostic (not damage)")
    fig.savefig(output / "mpm_observables.png", dpi=150)
    plt.close(fig)
    if not gif:
        return
    # Show a true particle section of the 3D trajectory, not a separate 2D simulation.
    mask = np.abs(positions[0, :, 2] - .06) < .0026
    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.set(xlim=(.035, .085), ylim=(.02, .085), aspect="equal",
           xlabel="x (m)", ylabel="y (m)")
    cloud = ax.scatter(positions[0, mask, 0], positions[0, mask, 1], c=fields[0, mask],
                       s=12, cmap="viridis", vmin=0, vmax=max(float(fields.max()), .01))
    jaw_l, jaw_r = ax.axvline(left, color="#666666", lw=5), ax.axvline(right, color="#666666", lw=5)
    fig.colorbar(cloud, ax=ax, label="Green strain norm")

    def update(i):
        shift = closure / 2 * min(times[i] / close_s, 1)
        jaw_l.set_xdata([left + shift] * 2)
        jaw_r.set_xdata([right - shift] * 2)
        cloud.set_offsets(positions[i, mask, :2])
        cloud.set_array(fields[i, mask])
        ax.set_title(f"JAX-MPM · 3D central section\nt = {times[i]:.3f} s")
        return cloud, jaw_l, jaw_r

    animation = FuncAnimation(fig, update, frames=len(times), interval=60)
    animation.save(output / "grasp.gif", writer=PillowWriter(fps=15))
    plt.close(fig)
