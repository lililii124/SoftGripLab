"""Archive a matched fast/slow withdrawal and a refined coupling step."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path, text=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if text else data).hexdigest()


def separation(trace, reach):
    ids = np.flatnonzero(
        (trace["time_s"] > 4.4)
        & (trace["adhesive_points"] == 0)
        & (trace["compressed_points"] == 0)
        & (trace["min_gap_m"] > reach)
    )
    return int(ids[0]) if len(ids) else None


def main():
    sources = {
        name: ROOT / "outputs" / ("inertial_w4_" + name)
        for name in ["fast", "slow", "fine"]
    }
    reports = {
        name: json.loads((folder / "summary.json").read_text())
        for name, folder in sources.items()
    }
    traces = {
        name: np.genfromtxt(folder / "observables.csv", delimiter=",", names=True)
        for name, folder in sources.items()
    }
    for name in sources:
        report, trace = reports[name], traces[name]
        assert report["release_mode"] == "inertial" and report["work_j_m2"] == 4
        assert report["grid"] == 40 and report["min_det_f"] > 0
        assert (
            report["payload_anchors"] == report["payload_attachment_constraints"] == 0
        )
        assert np.all(trace["adhesion_on"][trace["time_s"] >= 1.25] == 1)
        carry = (trace["time_s"] >= 2.8) & (trace["time_s"] <= 4.4)
        assert np.all(trace["bottom_z_m"][carry] > 0.05)
        assert np.all(trace["adhesive_points"][carry] > 0)
    prefix = traces["fast"]["time_s"] <= 4.4
    for key in traces["fast"].dtype.names:
        np.testing.assert_allclose(
            traces["fast"][key][prefix], traces["slow"][key][prefix], rtol=0, atol=1e-12
        )
    assert (
        not reports["slow"]["detached_at_end"]
        and separation(traces["slow"], 0.0008) is None
    )
    comparison = {}
    for name in ["fast", "fine"]:
        report, trace = reports[name], traces[name]
        assert report["pick_place_success"] and report["detached_at_end"]
        index = separation(trace, report["interaction_range_m"])
        assert index is not None and trace["head_az_m_s2"][index] > 0
        assert np.max(abs(trace["reaction_z_n"][index + 1 :])) < 1e-10
        comparison[name] = {
            "first_separated_sample_s": float(trace["time_s"][index]),
            "head_acceleration_at_separation_m_s2": float(trace["head_az_m_s2"][index]),
            "coupling_dt_s": report["coupling_dt_s"],
            "max_measured_upward_acceleration_m_s2": report[
                "max_measured_upward_acceleration_m_s2"
            ],
            "final_com_m": report["final_com_m"],
        }
    assert reports["fine"]["coupling_dt_s"] == reports["fast"]["coupling_dt_s"] / 2
    assert (
        abs(
            comparison["fine"]["first_separated_sample_s"]
            - comparison["fast"]["first_separated_sample_s"]
        )
        <= reports["fast"]["coupling_dt_s"] + 1e-12
    )
    destination = ROOT / "docs/results/inertial_release"
    destination.mkdir(parents=True, exist_ok=True)
    for name, folder in sources.items():
        dest = destination / name
        dest.mkdir(exist_ok=True)
        for filename in ["summary.json", "observables.csv"]:
            shutil.copyfile(folder / filename, dest / filename)
        with np.load(folder / "trajectory.npz") as archive:
            data = {key: archive[key].copy() for key in archive.files}
        ids = np.unique(data["surface_triangles"])
        inverse = np.full(data["position_m"].shape[1], -1, dtype=int)
        inverse[ids] = np.arange(len(ids))
        data["position_m"] = data["position_m"][:, ids]
        data["surface_triangles"] = inverse[data["surface_triangles"]]
        data["source_trajectory_sha256"] = digest(folder / "trajectory.npz")
        np.savez_compressed(dest / "trajectory.npz", **data)
    render = json.loads((sources["fast"] / "render.json").read_text())
    assert render["trajectory_sha256"] == digest(sources["fast"] / "trajectory.npz")
    assert render["force_trace_sha256"] in [
        digest(sources["fast"] / "observables.csv"),
        digest(sources["fast"] / "observables.csv", text=True),
    ]
    render["force_trace_sha256"] = digest(
        sources["fast"] / "observables.csv", text=True
    )
    render["force_trace_hash_normalization"] = "CRLF to LF"
    render["surface_replay_sha256"] = digest(destination / "fast/trajectory.npz")
    render["release_slow_motion_factor"] = 8
    (destination / "fast/render.json").write_text(
        json.dumps(render, indent=2) + "\n", encoding="utf-8"
    )
    comparison["slow"] = {
        "detached": False,
        "max_measured_upward_acceleration_m_s2": reports["slow"][
            "max_measured_upward_acceleration_m_s2"
        ],
    }
    comparison["identical_fast_slow_prefix_through_s"] = 4.4
    comparison["adhesion_parameters_changed_during_release"] = False
    comparison[
        "note"
    ] = "Qualitative release robustness under a halved robot coupling step; not spatial convergence or experimental validation."
    (destination / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    for suffix in [".gif", ".png"]:
        shutil.copyfile(
            sources["fast"] / ("adhesive_robot" + suffix),
            ROOT / "docs/assets" / ("05_adhesive_robot" + suffix),
        )
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
