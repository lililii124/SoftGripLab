"""Archive solved cutting data and its rendered frames after validation."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np


def collect(nominal, control, fine, destination):
    reports = [
        json.loads((folder / "summary.json").read_text())
        for folder in (nominal, control, fine)
    ]
    a, b, c = reports
    if not (
        a["success"] and a["final_connected_components"] == 4 and a["min_det_f"] > 0
    ):
        raise ValueError("Nominal run did not complete three stable cuts")
    if not np.isclose(
        a["reference_mass_kg"], a["integrated_reference_mass_kg"], rtol=1e-10
    ):
        raise ValueError("Cut topology changed the reference mass")
    if (
        b["softening_rate_n_m2_s"] != 0
        or b["final_broken_fractions"] != [0, 0, 0]
        or b["duration_s"] < 0.5
    ):
        raise ValueError("No-damage contact prefix did not remain intact")
    delta = abs(c["completed_cut_times_s"][0] - a["completed_cut_times_s"][0])
    if c["dt_s"] > 0.51 * a["dt_s"] or delta > a["coupling_dt_s"] + 1e-9:
        raise ValueError(
            "Half-step first-cut check differs by more than one robot step"
        )
    render = json.loads((nominal / "render.json").read_text())
    digest = hashlib.sha256((nominal / "trajectory.npz").read_bytes()).hexdigest()
    if (
        render["trajectory_sha256"] != digest
        or render["max_replayed_tool_center_error_m"] > 1e-6
    ):
        raise ValueError("Rendered mesh/robot pose does not match saved simulation")
    for name, source in (
        ("nominal", nominal),
        ("no_damage_prefix", control),
        ("half_step_first_cut", fine),
    ):
        target = destination / name
        target.mkdir(parents=True, exist_ok=True)
        files = ["summary.json", "observables.csv"]
        if name == "nominal":
            files += ["trajectory.npz", "render.json"]
        for filename in files:
            shutil.copy2(source / filename, target / filename)
    assets = Path("docs/assets")
    assets.mkdir(parents=True, exist_ok=True)
    for suffix in ("gif", "png"):
        shutil.copy2(
            nominal / f"banana_cutting.{suffix}", assets / f"06_banana_cutting.{suffix}"
        )
    validation = {
        "nominal_duration_s": a["duration_s"],
        "no_damage_control_duration_s": b["duration_s"],
        "half_step_check_duration_s": c["duration_s"],
        "first_cut_completion_difference_s": delta,
        "nominal_trajectory_sha256": digest,
        "scope": "Complete nominal three-cut run; controls cover the stated prefixes only. No spatial convergence or experimental calibration claim.",
    }
    (destination / "validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nominal", type=Path, default=Path("outputs/banana_robot"))
    parser.add_argument("--control", type=Path, default=Path("outputs/banana_control"))
    parser.add_argument("--fine", type=Path, default=Path("outputs/banana_fine"))
    parser.add_argument(
        "--destination", type=Path, default=Path("docs/results/cutting")
    )
    args = parser.parse_args()
    collect(args.nominal, args.control, args.fine, args.destination)
