"""Publish solver-backed animations only after checking their physical evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/assets"
    evidence = root / "docs/results/animations"
    assets.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    cases = [
        "fem_compression",
        "fem_grasp",
        "continuum_adhesion",
        "continuum_nonadhesive",
        "robot",
        "robot_frictionless",
        "fem_grasp_zero",
        "fem_compression_coarse",
        "fem_compression_halfdt",
    ]
    reports = {
        name: json.loads((root / "outputs" / name / "summary.json").read_text())
        for name in cases
    }
    robot = reports["robot"]
    if (
        not robot["pick_place_success"]
        or reports["robot_frictionless"]["pick_place_success"]
        or robot["payload_attachment_constraints"] != 0
        or robot["airborne_bilateral_contact_fraction"] < 0.99
        or robot["max_contact_penetration_m"] > 0.0008
    ):
        raise RuntimeError("robot bilateral-contact verification failed")
    if (
        reports["continuum_adhesion"]["tensile_samples"] <= 0
        or not reports["continuum_adhesion"]["detached_at_end"]
        or reports["continuum_nonadhesive"]["tensile_samples"] != 0
    ):
        raise RuntimeError("adhesion causal control or detachment failed")
    if (
        reports["fem_grasp"]["maximum_center_height_m"] < 0.04
        or reports["fem_grasp_zero"]["maximum_center_height_m"] > 0.03
    ):
        raise RuntimeError("soft grasp friction control failed")
    for name in ("fem_compression", "fem_grasp"):
        folder = root / "outputs" / name
        render = json.loads((folder / "render.json").read_text())
        if (
            reports[name]["min_det_f_all_steps"] <= 0
            or reports[name]["specimen_anchors"] != 0
            or render["trajectory_sha256"] != digest(folder / "trajectory.npz")
        ):
            raise RuntimeError(f"invalid or stale FEM render: {name}")
    rendered = json.loads((root / "outputs/gallery/02_adhesion.json").read_text())
    if rendered["source_trajectory_sha256"] != [
        digest(root / "outputs" / name / "trajectory.npz")
        for name in ("continuum_nonadhesive", "continuum_adhesion")
    ]:
        raise RuntimeError("adhesion render does not match the current solver states")
    sources = {
        "01_compression": "outputs/fem_compression/compression.gif",
        "02_adhesion": "outputs/gallery/02_adhesion.gif",
        "03_grasp": "outputs/fem_grasp/grasp.gif",
        "04_robot": "outputs/robot/robot.gif",
    }
    manifest = {}
    for name, relative in sources.items():
        source = root / relative
        with Image.open(source) as image:
            if image.n_frames < 70:
                raise RuntimeError(f"{source} has too few frames")
            manifest[name] = {
                "frames": image.n_frames,
                "pixels": list(image.size),
                "sha256": digest(source),
                "bytes": source.stat().st_size,
            }
        shutil.copyfile(source, assets / f"{name}.gif")
        shutil.copyfile(source.with_suffix(".png"), assets / f"{name}.png")
    for name in cases:
        folder = root / "outputs" / name
        for file in (
            "summary.json",
            "observables.csv",
            "trajectory.csv",
            "render.json",
            "states.npz",
        ):
            if (folder / file).exists():
                shutil.copyfile(folder / file, evidence / f"{name}_{file}")
    # Compact replay preserves every rendered surface node at every saved time,
    # in float64. Interior MPM particles are omitted, not displaced or smoothed.
    for name in (
        "fem_compression",
        "fem_grasp",
        "continuum_adhesion",
        "continuum_nonadhesive",
    ):
        folder = root / "outputs" / name
        with np.load(folder / "trajectory.npz") as data:
            ids = np.unique(data["surface_triangles"])
            inverse = np.full(data["position_m"].shape[1], -1, dtype=int)
            inverse[ids] = np.arange(len(ids))
            replay = {
                "time_s": data["time_s"],
                "position_m": data["position_m"][:, ids],
                "surface_triangles": inverse[data["surface_triangles"]],
                "source_trajectory_sha256": digest(folder / "trajectory.npz"),
            }
            for key in (
                "tool_position_m",
                "tool_half_extent_m",
                "sphere_center_m",
                "sphere_radius_m",
            ):
                if key in data:
                    replay[key] = data[key]
            target = evidence / f"{name}_surface.npz"
            np.savez_compressed(target, **replay)
            manifest[name + "_surface"] = {
                "sha256": digest(target),
                "bytes": target.stat().st_size,
                "surface_nodes": len(ids),
                "frames": len(replay["time_s"]),
            }
    fine = reports["fem_compression"]
    comparisons = {}
    for control in ("fem_compression_coarse", "fem_compression_halfdt"):
        comparisons[control] = {
            key + "_difference_m": abs(fine[key] - reports[control][key])
            for key in ("minimum_height_m", "maximum_width_m", "final_height_m")
        }
    (evidence / "refinement.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    (evidence / "gallery.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
