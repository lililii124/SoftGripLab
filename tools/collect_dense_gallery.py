"""Archive the dense solver runs and their minimal-caption renders."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    destination = ROOT/"docs/results/animations"
    mapping = dict(fem_compression="dense_compression", fem_grasp="dense_grasp",
        fem_grasp_zero="dense_grasp_zero", continuum_adhesion="dense_adhesion_14.4",
        continuum_nonadhesive="dense_adhesion_0", robot="robot_minimal")
    reports = {key:json.loads((ROOT/"outputs"/value/"summary.json").read_text()) for key, value in mapping.items()}
    assert reports["fem_compression"]["tetrahedra"] == reports["fem_grasp"]["tetrahedra"] == 9504
    assert reports["fem_compression"]["min_det_f_all_steps"] > 0
    assert reports["fem_grasp"]["min_det_f_all_steps"] > 0
    assert reports["fem_grasp"]["maximum_center_height_m"] > .04
    assert reports["fem_grasp_zero"]["maximum_center_height_m"] < .03
    assert reports["continuum_adhesion"]["grid"] == 36
    assert reports["continuum_adhesion"]["detached_at_end"]
    assert reports["continuum_adhesion"]["tensile_samples"] > 0
    assert reports["continuum_nonadhesive"]["tensile_samples"] == 0
    assert reports["robot"]["pick_place_success"]
    assert reports["robot"]["airborne_bilateral_contact_fraction"] > .99
    # Preserve the earlier, independently performed mesh/time-step comparison.
    legacy = destination/"legacy_refinement"
    legacy.mkdir(exist_ok=True)
    for file in ["refinement.json", "fem_compression_summary.json", "fem_compression_observables.csv"]:
        if not (legacy/file).exists():
            shutil.copyfile(destination/file, legacy/file)
    for name, folder_name in mapping.items():
        folder = ROOT/"outputs"/folder_name
        for file in ["summary.json", "observables.csv", "render.json", "trajectory.csv", "states.npz"]:
            if (folder/file).exists():
                shutil.copyfile(folder/file, destination/f"{name}_{file}")
        if name in ("fem_grasp_zero", "robot"):
            continue
        if name.startswith("fem_"):
            render = json.loads((folder/"render.json").read_text())
            assert render["trajectory_sha256"] == digest(folder/"trajectory.npz")
        with np.load(folder/"trajectory.npz") as data:
            ids = np.unique(data["surface_triangles"])
            inverse = np.full(data["position_m"].shape[1], -1, dtype=int)
            inverse[ids] = np.arange(len(ids))
            replay = dict(time_s=data["time_s"], position_m=data["position_m"][:, ids],
                surface_triangles=inverse[data["surface_triangles"]],
                source_trajectory_sha256=digest(folder/"trajectory.npz"))
            for key in ("tool_position_m", "tool_half_extent_m", "sphere_center_m", "sphere_radius_m"):
                if key in data:
                    replay[key] = data[key]
            np.savez_compressed(destination/f"{name}_surface.npz", **replay)
    rendered = json.loads((ROOT/"outputs/gallery/dense_adhesion.json").read_text())
    assert rendered["source_trajectory_sha256"] == [digest(ROOT/"outputs"/mapping[key]/"trajectory.npz")
                                                   for key in ("continuum_nonadhesive", "continuum_adhesion")]
    sources = {"01_compression":"dense_compression/compression.gif", "02_adhesion":"gallery/dense_adhesion.gif",
               "03_grasp":"dense_grasp/grasp.gif", "04_robot":"robot_minimal/robot.gif"}
    manifest = {}
    for key, path in sources.items():
        source = ROOT/"outputs"/path
        with Image.open(source) as gif:
            assert gif.n_frames >= 70
            manifest[key] = dict(frames=gif.n_frames, pixels=list(gif.size), sha256=digest(source), bytes=source.stat().st_size)
        for extension in (".gif", ".png"):
            shutil.copyfile(source.with_suffix(extension), ROOT/"docs/assets"/(key+extension))
    (destination/"gallery.json").write_text(json.dumps(manifest, indent=2)+"\n")
    coarse = json.loads((destination/"fem_compression_halfdt_summary.json").read_text())
    dense = reports["fem_compression"]
    comparison = dict(coarse_tetrahedra=coarse["tetrahedra"], dense_tetrahedra=dense["tetrahedra"],
                      coarse_dt_s=coarse["timestep_s"], dense_dt_s=dense["timestep_s"],
                      note="Spatial sensitivity at equal dt; no claim of spatial convergence.",
                      previous_temporal_probe="legacy_refinement/refinement.json")
    comparison["absolute_differences_m"] = {key:abs(dense[key]-coarse[key]) for key in
                  ("minimum_height_m", "maximum_width_m", "final_height_m")}
    (destination/"refinement.json").write_text(json.dumps(comparison, indent=2)+"\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
