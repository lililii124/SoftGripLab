"""Collect reproducible demo outputs into the small committed evidence gallery.

Run from repository root after the commands in docs/VALIDATION.md.
Full trajectories remain under outputs/ and are not committed.
"""

import json
from pathlib import Path
import shutil

import numpy as np
from scipy.integrate import quad


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/assets"
    results = root / "docs/results"
    assets.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    demo = root / "outputs/demo"
    for name in ("benchmark.png", "benchmark.svg"):
        shutil.copyfile(demo / name, assets / name)
    for name in ("report.json", "training_1.csv", "training_2.csv", "held_out.csv", "shock_envelope.csv"):
        shutil.copyfile(demo / name, results / name)
    for path in demo.glob("grasp_*.csv"):
        shutil.copyfile(path, results / path.name)

    def load_case(name):
        directory = root / "outputs" / name
        summary = json.loads((directory / "summary.json").read_text())
        rows = np.genfromtxt(directory / "observables.csv", delimiter=",", names=True)
        shutil.copyfile(directory / "summary.json", results / f"{name}_summary.json")
        shutil.copyfile(directory / "observables.csv", results / f"{name}_observables.csv")
        return summary, rows

    names = ("mpm", "mpm_frictionless", "mpm_frictionless_halfdt", "mpm_grid32")
    cases = {name: load_case(name) for name in names}
    shutil.copyfile(root / "outputs/mpm/grasp.gif", assets / "grasp.gif")

    def integrand(t):
        gravity = 9.81 * np.clip((t - .03) / .015, 0, 1)
        shock = 20 * np.sin(np.pi * (t - .075) / .015) if .075 < t < .09 else 0
        return (.12 - t) * (gravity + shock)

    exact = quad(integrand, 0, .12, points=[.03, .045, .075, .09], epsabs=1e-12)[0]
    check = {"free_motion_exact_drop_m": exact,
             "free_motion_time_step_errors_m": {
                 name: abs(cases[name][0]["final_com_drop_m"] - exact)
                 for name in ("mpm_frictionless", "mpm_frictionless_halfdt")},
             "frictional_peak_drop_m": {
                 name: float(np.max(-cases[name][1]["com_vertical_displacement_m"]))
                 for name in ("mpm", "mpm_grid32")},
             "spatial_convergence_established": False}
    errors = list(check["free_motion_time_step_errors_m"].values())
    if not errors[1] < errors[0] < 1e-4:
        raise RuntimeError("free-motion validation failed or refinement did not reduce error")
    if any(summary["min_det_f"] <= 0 for summary, _ in cases.values()):
        raise RuntimeError("invalid continuum deformation")
    (results / "mpm_verification.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    labels = {"mpm": "Friction 0.6 · grid 24", "mpm_grid32": "Friction 0.6 · grid 32",
              "mpm_frictionless": "Frictionless · grid 24"}
    for name in labels:
        _, rows = cases[name]
        axes[0].plot(rows["time_s"], -1000 * rows["com_vertical_displacement_m"], label=labels[name])
        if name != "mpm_frictionless":
            axes[1].plot(rows["time_s"], -1000 * rows["com_vertical_displacement_m"], label=labels[name])
    for ax in axes:
        ax.axvspan(.075, .09, color="orange", alpha=.15)
        ax.set(xlabel="Time (s)", ylabel="COM drop in gripper frame (mm)")
        ax.legend(fontsize=8)
    axes[0].set_title("Contact prevents free fall in this scene")
    axes[1].set_title("Grid sensitivity remains; not a convergence claim")
    fig.savefig(assets / "mpm_comparison.png", dpi=160)
    plt.close(fig)
    print(json.dumps(check, indent=2))


if __name__ == "__main__":
    main()
