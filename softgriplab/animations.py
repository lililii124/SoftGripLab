"""Scene-first renderers driven by saved data; no physics is advanced here.

Legacy MPM particle sections remain available as a supplementary view.
The main gallery uses depth-buffered surfaces in mesh_render.py.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = "#f5f4f0"
PANEL = "#e6e5df"
WHITE = "#292a28"
MUTED = "#666760"
TEAL = "#6b7565"
ORANGE = "#9b785d"
PINK = "#85655a"


@lru_cache(maxsize=12)
def font(size):
    from matplotlib import get_data_path
    return ImageFont.truetype(str(Path(get_data_path()) / "fonts/ttf/DejaVuSans.ttf"), size)


def label(draw, xy, text, size=18, fill=WHITE, **kwargs):
    draw.text(xy, text, font=font(size), fill=fill, **kwargs)


def scene(title, subtitle, number, *, width=960, height=600):
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    label(draw, (26, 17), f"{number:02d}  {title}", 27)
    label(draw, (27, 55), subtitle, 15, MUTED)
    return image, draw


def progress(draw, time_s, end_s, *, y=574, width=960):
    draw.rounded_rectangle((26, y, width - 26, y + 4), radius=2, fill=PANEL)
    draw.rounded_rectangle((26, y, 26 + (width - 52) * np.clip(time_s / end_s, 0, 1), y + 4), radius=2, fill=TEAL)


def save_frames(frames, path, *, fps=15):
    path = Path(path)
    if not frames or fps <= 0:
        raise ValueError("animation needs frames and a positive fps")
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[len(frames) // 2].save(path.with_suffix(".png"))
    palette_frames = [frame.convert("RGB").quantize(colors=128, method=Image.Quantize.MEDIANCUT)
                      for frame in frames]
    palette_frames[0].save(path, save_all=True, append_images=palette_frames[1:],
                           duration=round(1000 / fps), loop=0, optimize=False, disposal=2)


def render_grasp(source, comparison, output):
    sources = [Path(comparison), Path(source)]
    data = [np.load(path / "trajectory.npz") for path in sources]
    summaries = [json.loads((path / "summary.json").read_text()) for path in sources]
    try:
        times = data[0]["time_s"]
        if not np.allclose(times, data[1]["time_s"]):
            raise ValueError("comparison trajectories must have identical frame times")
        frames = []
        vmax = max(float(case["green_strain_norm"].max()) for case in data)
        for i, time_s in enumerate(times):
            image, draw = scene("Grip a soft block under a disturbance", "3D JAX-MPM · central particle section · identical loading", 3)
            for panel, (case, summary) in enumerate(zip(data, summaries)):
                x0 = 25 + panel * 470
                draw.rounded_rectangle((x0, 93, x0 + 445, 521), radius=16, fill=PANEL)
                label(draw, (x0 + 20, 108), f"Friction coefficient {summary['friction']:.1f}", 21)

                def project(x, y):
                    return x0 + 26 + (x - .035) / .05 * 393, 481 - y / .085 * 325

                shift = .0015 * min(float(time_s) / .03, 1)
                for boundary in (.045 + shift, .075 - shift):
                    px, _ = project(boundary, 0)
                    draw.line((px, 145, px, 480), fill="#a7b6c7", width=9)
                positions = case["position_m"][i]
                mask = np.abs(case["position_m"][0, :, 2] - .06) < .0026
                for (x, y, _), strain in zip(positions[mask], case["green_strain_norm"][i][mask]):
                    px, py = project(x, y)
                    fraction = np.clip(strain / max(vmax, 1e-12), 0, 1)
                    color = tuple((np.array([143, 150, 135]) * (1 - fraction)
                                   + np.array([143, 101, 79]) * fraction).astype(int))
                    draw.ellipse((px - 3.5, py - 3.5, px + 3.5, py + 3.5), fill=color)
                drop = (case["position_m"][0, :, 1].mean() - positions[:, 1].mean()) * 1000
                label(draw, (x0 + 20, 490), f"COM drop {drop:+.2f} mm", 18, TEAL)
            state = "Squeeze" if time_s < .03 else "Transfer gravity to jaws"
            if time_s >= .045:
                state = "Hold" if time_s < .075 else ("Base shock pulse" if time_s <= .09 else "Recover")
            label(draw, (27, 539), f"t = {time_s * 1000:.0f} ms    {state}    Color: Green strain norm", 18,
                  ORANGE if .075 <= time_s <= .09 else WHITE)
            progress(draw, time_s, times[-1])
            frames.append(image)
        save_frames(frames, output, fps=10)
    finally:
        for case in data:
            case.close()
