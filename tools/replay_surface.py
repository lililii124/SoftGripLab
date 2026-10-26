"""Render a committed surface replay file without running either physics solver."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pybullet as p
import pybullet_data
from PIL import ImageDraw
from softgriplab.animations import font, save_frames
from softgriplab.mesh_render import _mesh, _box, _camera, INK


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/replay.gif"))
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()
    if not 1 <= args.fps <= 60:
        parser.error("fps must be between 1 and 60")
    with np.load(args.source) as npz:
        data = {key: npz[key] for key in npz.files}
    cid = p.connect(p.DIRECT)
    frames = []
    try:
        adhesion = "sphere_center_m" in data
        if adhesion:
            base = _box(p, cid, [0.038, 0.038, 0.003])
            p.resetBasePositionAndOrientation(
                base, [0.05, 0.05, 0.012], [0, 0, 0, 1], physicsClientId=cid
            )
            radius = float(data["sphere_radius_m"])
            visual = p.createVisualShape(
                p.GEOM_MESH,
                fileName=str(Path(pybullet_data.getDataPath()) / "sphere_smooth.obj"),
                meshScale=[radius] * 3,
                rgbaColor=[0.65, 0.66, 0.63, 1],
                physicsClientId=cid,
            )
            sphere = p.createMultiBody(0, -1, visual, physicsClientId=cid)
        else:
            fixtures = [_box(p, cid, h) for h in data["tool_half_extent_m"]]
        for i, t in enumerate(data["time_s"]):
            body = _mesh(p, cid, data["position_m"][i], data["surface_triangles"])
            if adhesion:
                p.resetBasePositionAndOrientation(
                    sphere,
                    data["sphere_center_m"][i],
                    [0, 0, 0, 1],
                    physicsClientId=cid,
                )
                frame = _camera(
                    p, cid, [0.05, 0.05, 0.04], 0.20, 640, 480, yaw=35, pitch=-10
                )
            else:
                for fixture, pos in zip(fixtures, data["tool_position_m"][i]):
                    p.resetBasePositionAndOrientation(
                        fixture, pos, [0, 0, 0, 1], physicsClientId=cid
                    )
                frame = _camera(
                    p, cid, [0, 0, 0.035], 0.26, 640, 480, yaw=38, pitch=-17
                )
            ImageDraw.Draw(frame).text(
                (16, 16),
                f"Surface replay / t = {t:.3f} s / scale 1:1",
                font=font(17),
                fill=INK,
            )
            frames.append(frame)
            p.removeBody(body, physicsClientId=cid)
    finally:
        p.disconnect(physicsClientId=cid)
    save_frames(frames, args.output, fps=args.fps)
    print(args.output)


if __name__ == "__main__":
    main()
