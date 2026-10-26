"""Gate and collect the adhesive robot animation and exact surface replay."""
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def collect(root=Path(".")):
    source = root / "outputs/adhesive_robot_6.0"
    control = root / "outputs/adhesive_robot_0.0"
    fine = root / "outputs/adhesive_robot_fine_coupling"
    nominal, zero, refined = [
        json.loads((p / "summary.json").read_text()) for p in (source, control, fine)
    ]
    if (
        not nominal["pick_place_success"]
        or zero["pick_place_success"]
        or not refined["pick_place_success"]
    ):
        raise ValueError("nominal/refined must succeed and zero adhesion must fail")
    for r in (nominal, refined):
        if (
            r["payload_anchors"]
            or r["payload_attachment_constraints"]
            or r["min_det_f"] <= 0
        ):
            raise ValueError("invalid free-payload physics")
        if (
            abs(r["mean_airborne_support_n"] / r["weight_n"] - 1) > 0.01
            or r["airborne_momentum_balance_residual_n_s"] > 1e-6
        ):
            raise ValueError("force/weight or airborne momentum check failed")
        if (
            r["first_compressive_contact_s"] >= 1.25
            or r["final_contact_force_n"] > 1e-6
        ):
            raise ValueError(
                "contact before engagement and complete release are required"
            )
    if (
        np.linalg.norm(np.array(nominal["final_com_m"]) - refined["final_com_m"])
        > 0.0005
    ):
        raise ValueError("coupling refinement moved landing by >0.5 mm")
    rendering = json.loads((source / "render.json").read_text())
    if (
        rendering["trajectory_sha256"]
        != hashlib.sha256((source / "trajectory.npz").read_bytes()).hexdigest()
    ):
        raise ValueError("rendered trajectory does not match solver output")
    if rendering["max_replayed_tool_center_error_m"] > 1e-6:
        raise ValueError("visible and physical tool centers disagree")
    archive = root / "docs/results/adhesive_robot"
    for name, folder in [
        ("nominal", source),
        ("nonadhesive", control),
        ("fine_coupling", fine),
    ]:
        target = archive / name
        target.mkdir(parents=True, exist_ok=True)
        for file in ("observables.csv", "summary.json"):
            shutil.copy2(folder / file, target / file)
        if name != "fine_coupling":
            with np.load(folder / "trajectory.npz") as data:
                saved = {k: data[k].copy() for k in data.files}
            ids = np.unique(saved["surface_triangles"])
            saved["position_m"] = saved["position_m"][:, ids]
            saved["surface_triangles"] = np.searchsorted(
                ids, saved["surface_triangles"]
            ).astype(np.int32)
            np.savez_compressed(target / "trajectory.npz", **saved)
    rendering["surface_replay_sha256"] = hashlib.sha256(
        (archive / "nominal/trajectory.npz").read_bytes()
    ).hexdigest()
    (archive / "nominal/render.json").write_text(
        json.dumps(rendering, indent=2) + "\n", encoding="utf-8"
    )
    for suffix in ("gif", "png"):
        shutil.copy2(
            source / f"adhesive_robot.{suffix}",
            root / f"docs/assets/05_adhesive_robot.{suffix}",
        )
    print("Collected validated adhesive robot animation and controls.")


if __name__ == "__main__":
    collect()
