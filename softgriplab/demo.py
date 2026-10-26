"""Reproducible identification-to-grasp reference benchmark."""

from dataclasses import asdict
import json
from pathlib import Path
import platform

import numpy as np
import scipy

from .grasp import GraspSpec, assess_grasp, half_sine_acceleration, plan_fixed_closure
from .identification import CompressionTrace, fit_material
from .materials import Geometry, StandardLinearSolid, ramp_hold


def analytic_ramp_force(material, time_s, amplitude_m, ramp_s, geometry):
    """Independent closed-form ramp/hold oracle for synthetic verification."""
    t = np.asarray(time_s)
    rate = amplitude_m / geometry.thickness_m / ramp_s
    loaded = np.minimum(t, ramp_s)
    elastic = material.equilibrium_pa * rate * loaded
    branch = (material.branch_pa * rate * material.relaxation_s *
              (-np.expm1(-loaded / material.relaxation_s)) *
              np.exp(-np.maximum(t - ramp_s, 0) / material.relaxation_s))
    return geometry.area_m2 * (elastic + branch)


def run_demo(output, seed=7):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    geometry = Geometry()
    truth = StandardLinearSolid(35000, 65000, 0.6)
    noise = 0.005
    training = []
    protocols = [(0.0008, 0.12), (0.0012, 0.8)]
    for i, (amplitude, ramp) in enumerate(protocols):
        t = np.linspace(0, 4, 401)
        u = ramp_hold(t, amplitude, ramp)
        clean = analytic_ramp_force(truth, t, amplitude, ramp, geometry)
        trace = CompressionTrace(t, u, clean + rng.normal(0, noise, t.size), geometry, noise)
        trace.to_csv(output / f"training_{i + 1}.csv")
        training.append(trace)
    fit = fit_material(training)

    tv = np.linspace(0, 3, 601)
    uv = ramp_hold(tv, 0.0005, 0.035)
    fv = analytic_ramp_force(truth, tv, 0.0005, 0.035, geometry)
    predicted = fit.material.force(tv, uv, geometry)
    CompressionTrace(tv, uv, fv, geometry).to_csv(output / "held_out.csv")

    # Deliberate constitutive mismatch, reported rather than used for fitting.
    mismatch = (analytic_ramp_force(StandardLinearSolid(17500, 50000, 0.25),
                                   tv, 0.0005, 0.035, geometry)
                + analytic_ramp_force(StandardLinearSolid(17500, 15000, 1.5),
                                      tv, 0.0005, 0.035, geometry))
    spec = GraspSpec()
    t = np.linspace(0, 4, 801)
    acceleration = half_sine_acceleration(t, 2.5, 0.2, 3.0)
    plan = plan_fixed_closure(fit.material, t, acceleration, spec, geometry)
    fixed_u = ramp_hold(t, 0.04 * geometry.thickness_m, 0.2)
    force_control = 0.8 * np.clip(t / 0.2, 0, 1)
    planned_u = ramp_hold(t, plan["closure_m"], 0.2)
    # Evaluate all policies with truth, not the fitted material used by the planner.
    policies = {
        "fixed_closure": (fixed_u, truth.force(t, fixed_u, geometry)),
        "ideal_force_control": (truth.compression(t, force_control, geometry), force_control),
        "identified_closure": (planned_u, truth.force(t, planned_u, geometry)),
    }
    results = {}
    for name, (u, normal) in policies.items():
        result = assess_grasp(t, normal, u, acceleration, spec, geometry, 0.3)
        results[name] = result
        np.savetxt(output / f"grasp_{name}.csv",
                   np.column_stack((t, u, normal, acceleration, result["capacity_n"],
                                    result["demand_n"], result["margin_n"], result["strain"])),
                   delimiter=",", comments="",
                   header="time_s,compression_m,normal_force_per_finger_n,acceleration_m_s2,capacity_n,demand_n,margin_n,strain")

    # Envelope over pulse peak and permitted strain. All samples use the same ramp/hold.
    peaks = np.linspace(0, 12, 49)
    envelope = []
    for peak in peaks:
        a = half_sine_acceleration(t, 2.5, 0.2, peak)
        item = plan_fixed_closure(fit.material, t, a, spec, geometry)
        envelope.append((peak, item["required_strain"], item["feasible"]))
    np.savetxt(output / "shock_envelope.csv", envelope, delimiter=",", comments="",
               header="peak_acceleration_m_s2,required_strain,feasible")

    report = {
        "version": "0.1.0", "seed": seed,
        "data_source": "synthetic analytic uniaxial ramp-hold; not physical experiments",
        "geometry": asdict(geometry), "truth": asdict(truth), "fit": fit.to_dict(),
        "training_protocols": [{"amplitude_m": a, "ramp_s": r} for a, r in protocols],
        "noise_std_n": noise,
        "held_out": {"amplitude_m": 0.0005, "ramp_s": 0.035,
                     "relative_l2_error": float(np.linalg.norm(predicted - fv) / np.linalg.norm(fv))},
        "two_branch_mismatch_relative_l2_error": float(np.linalg.norm(predicted - mismatch) /
                                                       np.linalg.norm(mismatch)),
        "parameter_relative_errors": {key: abs(asdict(fit.material)[key] / value - 1)
                                      for key, value in asdict(truth).items()},
        "grasp_spec": asdict(spec), "plan": plan,
        "policies": {name: result["summary"] for name, result in results.items()},
        "pulse": {"start_s": 2.5, "duration_s": 0.2, "peak_m_s2": 3.0,
                  "equivalent_inertial_impulse_ns": float(spec.mass_kg * 2 * 3 * 0.2 / np.pi)},
        "limitations": ["Uniform small-strain model; no spatial contact pressure or damage.",
                        "Friction is supplied, not identified from normal compression.",
                        "Sticking feasibility only; no slip trajectory or robot dynamics.",
                        "Acceleration pulse is prescribed base excitation, not a resolved impact.",
                        "Planner knows the pulse envelope; force-control baseline has ideal bandwidth.",
                        "Faster held-out synthetic loading does not establish real impact validity."],
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": scipy.__version__},
    }
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _plot(output, training, fit, tv, fv, predicted, mismatch, t, results, spec, envelope)
    return report


def _plot(output, training, fit, tv, fv, predicted, mismatch, t, results, spec, envelope):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#287c8e", "#ce7146", "#6474b5"]
    with plt.rc_context({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 140}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
        for i, tr in enumerate(training):
            axes[0, 0].plot(tr.time_s, tr.force_n, ".", ms=2, alpha=.4, color=colors[i])
            axes[0, 0].plot(tr.time_s, fit.material.force(tr.time_s, tr.compression_m, tr.geometry),
                            color=colors[i], label=f"Ramp {i + 1}: fitted")
        axes[0, 0].set(title="Identify from two loading rates", xlabel="Time (s)", ylabel="Normal force (N)")
        axes[0, 0].legend()
        axes[0, 1].plot(tv, fv, color=colors[0], label="Held-out analytic truth")
        axes[0, 1].plot(tv, predicted, "--", color=colors[1], label="Fitted model prediction")
        axes[0, 1].plot(tv, mismatch, ":", color=colors[2], label="Two-branch mismatch probe")
        axes[0, 1].set(title="Predict a faster, unseen ramp", xlabel="Time (s)", ylabel="Normal force (N)")
        axes[0, 1].legend(fontsize=8)
        envelope = np.asarray(envelope)
        axes[0, 2].plot(envelope[:, 0], envelope[:, 1] * 100, color=colors[0])
        axes[0, 2].axhline(spec.strain_limit * 100, color=colors[1], ls="--", label="Strain limit")
        axes[0, 2].set(title="Required closure vs. shock envelope", xlabel="Peak acceleration (m/s²)", ylabel="Required strain (%)")
        axes[0, 2].legend()
        for (name, result), color in zip(results.items(), colors):
            label = name.replace("_", " ")
            axes[1, 0].plot(t, result["capacity_n"], color=color, label=label)
            axes[1, 1].plot(t, result["margin_n"], color=color)
            axes[1, 2].plot(t, result["strain"] * 100, color=color)
        axes[1, 0].plot(t, result["demand_n"], "k--", label="Support demand")
        axes[1, 0].set(title="Two-finger friction capacity", xlabel="Time (s)", ylabel="Tangential force (N)")
        axes[1, 0].legend(fontsize=8)
        axes[1, 1].axhline(0, color="black", ls="--")
        axes[1, 1].set(title="Negative margin: sticking infeasible", xlabel="Time (s)", ylabel="Capacity − demand (N)")
        axes[1, 2].axhline(spec.strain_limit * 100, color="black", ls="--")
        axes[1, 2].set(title="Compression and creep", xlabel="Time (s)", ylabel="Total compression strain (%)")
        for ax in axes[1]:
            ax.set_xlim(.3, 4)
            ax.axvspan(2.5, 2.7, color="#999999", alpha=.12)
        fig.suptitle("SoftGripLab · identify → plan → challenge\nSynthetic uniaxial reference benchmark", fontsize=16)
        for ext in ("png", "svg"):
            fig.savefig(output / f"benchmark.{ext}")
        plt.close(fig)
