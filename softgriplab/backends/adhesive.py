"""Displacement-controlled dry adhesion, delegated to JAX-MPM's JKR law."""

import json
from pathlib import Path

import numpy as np

from ..materials import positive


def contact_cycle(*, frames=151, radius_m=0.01, modulus_pa=100000., work_j_m2=.08):
    if type(frames) is not int or frames < 12:
        raise ValueError("contact cycle requires at least 12 frames")
    positive(radius_m=radius_m, modulus_pa=modulus_pa)
    if not np.isfinite(work_j_m2) or work_j_m2 < 0:
        raise ValueError("work of adhesion must be finite and nonnegative")
    try:
        import jax
        import jax.numpy as jnp
        from mpm_jax.contact_laws import hertz_contact, jkr_detachment, jkr_history
    except ImportError as error:
        raise ImportError("adhesion requires the optional JAX-MPM environment") from error
    jax.config.update("jax_enable_x64", True)
    t = np.linspace(0, 5, frames)
    # Smoothly approach, hold briefly, then retract past the displacement instability.
    critical_a, critical_delta = jkr_detachment(radius_m, modulus_pa, max(work_j_m2, 1e-12))
    gap = max(4 * abs(float(critical_delta)), 1.2e-4)
    tensile_hold = .5 * float(critical_delta) if work_j_m2 else -1.5e-5
    indentation = np.interp(t, [0, .3, 1.3, 1.7, 2.7, 3.4, 4.4, 5.],
                            [-gap, -gap, 1.5e-4, 1.5e-4, tensile_hold, tensile_hold, -gap, -gap])
    adhesive = jkr_history(jnp.asarray(indentation), radius_m, modulus_pa, work_j_m2)
    hertz = hertz_contact(jnp.asarray(indentation), radius_m, modulus_pa)
    return {
        "time_s": t, "indentation_m": indentation,
        "jkr_radius_m": np.asarray(adhesive.radius), "jkr_force_n": np.asarray(adhesive.force),
        "jkr_active": np.asarray(adhesive.active),
        "hertz_radius_m": np.asarray(hertz.radius), "hertz_force_n": np.asarray(hertz.force),
        "hertz_active": np.asarray(hertz.active),
        "settings": {"radius_m": radius_m, "reduced_modulus_pa": modulus_pa, "work_j_m2": work_j_m2,
                     "displacement_detachment_m": float(critical_delta) if work_j_m2 else 0.,
                     "critical_contact_radius_m": float(critical_a) if work_j_m2 else 0.,
                     "load_control_pull_off_n": -1.5 * np.pi * work_j_m2 * radius_m,
                     "backend": "mpm_jax.contact_laws.jkr_history; displacement-controlled stable branch"},
    }


def run_adhesion(output, *, frames=151):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    data = contact_cycle(frames=frames)
    keys = [key for key in data if key != "settings"]
    np.savetxt(output / "contact.csv", np.column_stack([data[key] for key in keys]),
               delimiter=",", comments="", header=",".join(keys))
    attached_in_tension = (data["indentation_m"] < 0) & data["jkr_active"]
    report = {**data["settings"], "frames": frames,
              "tensile_attachment_samples": int(np.count_nonzero(attached_in_tension)),
              "minimum_force_n": float(data["jkr_force_n"].min()),
              "detached_at_end": not bool(data["jkr_active"][-1]),
              "geometry_display": "schematic local contact section; not a resolved displacement field",
              "mechanism": "dry surface adhesion; not vacuum suction"}
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    from ..animations import render_adhesion
    render_adhesion(data, output / "adhesion.gif")
    return report
