"""Validate and archive dynamic cutting; never replace the gallery with a failed run."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.integrate import trapezoid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(folder):
    trace = np.genfromtxt(folder / "observables.csv", delimiter=",", names=True)
    with np.load(folder / "trajectory.npz") as data:
        x = data["position_m"]
        labels = data["component_labels"]
        n = x.shape[1]
        major = [int((np.bincount(row) > n * 0.03).sum()) for row in labels]
        sizes = np.sort(np.bincount(labels[-1]))[::-1]
        assert np.isfinite(x).all() and np.all(data["particle_volume_m3"] > 0)
        assert major[0] == 1
        return dict(
            material_points=n,
            initial_major_components=major[0],
            final_major_components=major[-1],
            peak_axial_contact_force_n=float(abs(trace["reaction_z_n"]).max()),
            absolute_axial_contact_impulse_n_s=float(
                trapezoid(abs(trace["reaction_z_n"]), trace["time_s"])
            ),
            final_major_sizes=sizes[sizes > n * 0.03].tolist(),
            debris_mass_fraction=float(sizes[sizes <= n * 0.03].sum() / n),
            component_transition_times_s={
                str(k): float(data["time_s"][major.index(k)])
                for k in sorted(set(major))
            },
            trajectory_sha256=digest(folder / "trajectory.npz"),
        )


def main():
    sources = {
        name: ROOT / "outputs" / folder
        for name, folder in [
            ("nominal", "pd_nominal"),
            ("shifted", "pd_shifted"),
            ("intact", "pd_intact"),
        ]
    }
    reports = {
        name: json.loads((folder / "summary.json").read_text())
        for name, folder in sources.items()
    }
    nominal = reports["nominal"]
    assert nominal["success"] and nominal["final_major_components"] == 4
    assert (
        nominal["initial_broken_bonds"]
        == nominal["preset_interfaces"]
        == nominal["clamped_nodes"]
        == 0
    )
    assert (
        reports["shifted"]["success"]
        and reports["shifted"]["final_major_components"] == 2
    )
    assert reports["intact"]["final_connected_components"] == 1
    assert not reports["intact"]["geometric_knife_cutting"]
    with np.load(sources["nominal"] / "trajectory.npz") as a, np.load(
        sources["shifted"] / "trajectory.npz"
    ) as b:
        np.testing.assert_array_equal(a["initial_neighbours"], b["initial_neighbours"])
        np.testing.assert_array_equal(
            a["reference_position_m"], b["reference_position_m"]
        )
        assert abs(a["knife_pose_local"][0, 0] - b["knife_pose_local"][0, 0]) > 0.009
    render = json.loads((sources["nominal"] / "render.json").read_text())
    assert render["trajectory_sha256"] == digest(sources["nominal"] / "trajectory.npz")
    assert render["force_trace_sha256"] == digest(
        sources["nominal"] / "observables.csv"
    )
    assert render["max_replayed_tool_center_error_m"] < 1e-6
    comparison = {
        "fine": metrics(sources["nominal"]),
        "coarse": metrics(ROOT / "outputs/pd_ductile"),
        "fine_dt_s": nominal["dt_s"],
        "coarse_dt_s": (1 / 120) / 84,
        "note": "Temporal sensitivity at fixed material discretization; not spatial convergence.",
    }
    assert (
        comparison["fine"]["final_major_components"]
        == comparison["coarse"]["final_major_components"]
        == 4
    )
    for metric in ["fine", "coarse"]:
        assert comparison[metric]["debris_mass_fraction"] < 0.01
    destination = ROOT / "docs/results/dynamic_cutting"
    destination.mkdir(parents=True, exist_ok=True)
    for name, folder in sources.items():
        dest = destination / name
        dest.mkdir(exist_ok=True)
        for filename in [
            "summary.json",
            "observables.csv",
            "trajectory.npz",
            "render.json",
        ]:
            if (folder / filename).exists():
                shutil.copyfile(folder / filename, dest / filename)
    coarse = destination / "coarse_dt"
    coarse.mkdir(exist_ok=True)
    for filename in ["trajectory.npz", "observables.csv"]:
        shutil.copyfile(ROOT / "outputs/pd_ductile" / filename, coarse / filename)
    (destination / "temporal_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    for suffix in [".gif", ".png"]:
        shutil.copyfile(
            sources["nominal"] / ("banana_cutting" + suffix),
            ROOT / "docs/assets" / ("06_banana_cutting" + suffix),
        )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
