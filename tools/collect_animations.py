"""Publish only verified animation artifacts into the repository gallery."""

import hashlib
import json
from pathlib import Path
import shutil

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    assets = root / "docs/assets"
    evidence = root / "docs/results/animations"
    assets.mkdir(parents=True, exist_ok=True)
    evidence.mkdir(parents=True, exist_ok=True)
    adhesion = json.loads((root / "outputs/adhesion/summary.json").read_text())
    robot = json.loads((root / "outputs/robot/summary.json").read_text())
    weak = json.loads((root / "outputs/robot_weak/summary.json").read_text())
    if not adhesion["detached_at_end"] or adhesion["tensile_attachment_samples"] <= 0:
        raise RuntimeError("adhesive cycle did not demonstrate attachment and detachment")
    if not robot["pick_place_success"] or weak["pick_place_success"]:
        raise RuntimeError("nominal/weak suction outcomes do not match the documented demonstration")
    sources = {"01_compression": "outputs/gallery/01_compression.gif",
               "02_adhesion": "outputs/adhesion/adhesion.gif",
               "03_grasp": "outputs/gallery/03_grasp.gif",
               "04_robot": "outputs/robot/robot.gif"}
    manifest = {}
    for name, relative in sources.items():
        source = root / relative
        with Image.open(source) as image:
            if image.n_frames < 12:
                raise RuntimeError(f"{source} is not a sufficiently sampled animation")
            manifest[name] = {"frames": image.n_frames, "pixels": list(image.size),
                              "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                              "bytes": source.stat().st_size}
        shutil.copyfile(source, assets / f"{name}.gif")
        shutil.copyfile(source.with_suffix(".png"), assets / f"{name}.png")
    for folder, names in (("adhesion", ["summary.json", "contact.csv"]),
                          ("robot", ["summary.json", "trajectory.csv"]),
                          ("robot_weak", ["summary.json", "trajectory.csv"])):
        for name in names:
            shutil.copyfile(root / "outputs" / folder / name, evidence / f"{folder}_{name}")
    (evidence / "gallery.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
