"""Scene-first renderers driven by saved data; no physics is advanced here.

Reduced-model geometry is explicitly a schematic. MPM particles and the robot
camera come directly from their respective simulators.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


BG = "#101c2b"
PANEL = "#1a2a3d"
WHITE = "#ecf4fa"
MUTED = "#afc2d4"
TEAL = "#53d6c7"
ORANGE = "#ffb15c"
PINK = "#f39ab5"


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


def render_compression(source, output):
    source = Path(source)
    traces = [np.genfromtxt(source / f"training_{i}.csv", names=True, delimiter=",") for i in (1, 2)]
    report = json.loads((source / "report.json").read_text())
    h = report["geometry"]["thickness_m"]
    selected = np.linspace(0, len(traces[0]) - 1, 101).astype(int)
    frames = []
    for index in selected:
        time_s = float(traces[0]["time_s"][index])
        image, draw = scene("Press, hold, identify", "Synthetic compression records · uniform-strain illustration", 1)
        for panel, (trace, caption) in enumerate(zip(traces, ("Fast loading · 0.12 s", "Slow loading · 0.80 s"))):
            x0 = 25 + panel * 470
            draw.rounded_rectangle((x0, 93, x0 + 445, 521), radius=16, fill=PANEL)
            label(draw, (x0 + 22, 110), caption, 21)
            d, force = float(trace["compression_m"][index]), float(trace["force_n"][index])
            bottom, height, left, width = 389, 205 * (1 - d / h), x0 + 114, 218
            top = bottom - height
            color = tuple(np.array([49, 135, 155]) + (np.clip(force / 2., 0, 1) * np.array([130, 60, -30])).astype(int))
            draw.rounded_rectangle((left, top, left + width, bottom), radius=8, fill=color)
            for j in range(1, 8):
                draw.line((left + j * width / 8, top + 4, left + j * width / 8, bottom - 4), fill="#7cb2ba")
            for j in range(1, 7):
                yy = top + height * j / 7
                draw.line((left + 4, yy, left + width - 4, yy), fill="#7cb2ba")
            draw.rounded_rectangle((left - 15, top - 17, left + width + 15, top), radius=4, fill="#aab8c7")
            draw.rounded_rectangle((left - 15, bottom, left + width + 15, bottom + 17), radius=4, fill="#687b91")
            mid = left + width / 2
            draw.line((mid, top - 42, mid, top - 21), fill=ORANGE, width=4)
            draw.polygon([(mid, top - 19), (mid - 6, top - 29), (mid + 6, top - 29)], fill=ORANGE)
            label(draw, (x0 + 22, 427), f"Compression {d * 1000:.2f} mm", 18)
            label(draw, (x0 + 22, 455), f"Force {force:.3f} N", 23, TEAL)
            draw.rounded_rectangle((x0 + 235, 459, x0 + 415, 474), radius=5, fill=BG)
            draw.rounded_rectangle((x0 + 235, 459, x0 + 235 + 180 * np.clip(force / 2., 0, 1), 474), radius=5, fill=TEAL)
            label(draw, (x0 + 22, 490), "Holding: force relaxes" if time_s > (.12 if panel == 0 else .8) else "Loading", 15, MUTED)
        label(draw, (27, 539), f"t = {time_s:.2f} s    Same material · different loading rates · fit uses both records", 17)
        progress(draw, time_s, 4)
        frames.append(image)
    save_frames(frames, output, fps=20)


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
                    color = tuple((np.array([68, 210, 194]) * (1 - fraction)
                                   + np.array([255, 151, 83]) * fraction).astype(int))
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


def render_adhesion(data, output):
    settings = data["settings"]
    radius_mm = settings["radius_m"] * 1000
    frames = []
    force_scale = max(float(np.max(np.abs(data[key]))) for key in ("hertz_force_n", "jkr_force_n"))
    for i, time_s in enumerate(data["time_s"]):
        image, draw = scene("Touch, adhere, pull off", "Dry surface adhesion · JAX-MPM Hertz / JKR contact laws", 2)
        for panel, (prefix, title) in enumerate((("hertz", "Hertz · no adhesion"), ("jkr", "JKR · adhesive interface"))):
            x0 = 25 + panel * 470
            draw.rounded_rectangle((x0, 93, x0 + 445, 521), radius=16, fill=PANEL)
            label(draw, (x0 + 19, 109), title, 21)
            delta = data["indentation_m"][i] * 1000
            a = data[f"{prefix}_radius_m"][i] * 1000
            active = bool(data[f"{prefix}_active"][i])
            force = float(data[f"{prefix}_force_n"][i])

            def project(x, z):
                return x0 + 222 + x * 76, 320 - z * 350

            draw.rectangle((x0 + 28, 320, x0 + 417, 393), fill="#436777")
            draw.line((x0 + 28, 320, x0 + 417, 320), fill=TEAL, width=2)
            xs = np.linspace(-2.4, 2.4, 100)
            probe = [project(x, x * x / (2 * radius_mm) - delta) for x in xs]
            draw.polygon([project(-2.4, .36), *probe, project(2.4, .36)], fill="#afbdce")
            if active:
                local = np.linspace(-a, a, 40)
                patch = [project(x, x * x / (2 * radius_mm) - delta) for x in local]
                draw.polygon([project(-a, 0), *patch, project(a, 0)], fill=PINK if force < 0 else TEAL)
                draw.line((*project(-a, 0), *project(a, 0)), fill=PINK if force < 0 else TEAL, width=6)
            label(draw, (x0 + 24, 410), f"Contact radius {a:.3f} mm", 17)
            label(draw, (x0 + 24, 438), f"Force {force * 1000:+.2f} mN", 23, PINK if force < 0 else TEAL)
            status = "Separated" if not active else ("Tensile attachment" if force < 0 else "Compressed contact")
            label(draw, (x0 + 24, 475), status, 20, PINK if force < 0 else WHITE)
            # Signed force gauge, subordinate to the contact scene.
            center = x0 + 340
            draw.line((center, 432, center, 464), fill=MUTED, width=2)
            length = 65 * force / force_scale
            draw.line((center, 448, center + length, 448), fill=PINK if force < 0 else TEAL, width=10)
        label(draw, (27, 536), f"t = {time_s:.2f} s    Contact geometry is schematic; radius and force are computed", 16)
        progress(draw, time_s, data["time_s"][-1])
        frames.append(image)
    save_frames(frames, output, fps=20)


def robot_frame(rgb, time_s, phase, sealed, capacity_n, force_n, pressure_pa, detail_rgb=None):
    frame = Image.new("RGB", (800, 625), BG)
    frame.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 80))
    draw = ImageDraw.Draw(frame)
    label(draw, (21, 12), "04  Robot suction pick-and-place", 26)
    label(draw, (22, 47), "Panda URDF · motor-driven PyBullet simulation · rigid payload", 16, MUTED)
    draw.rounded_rectangle((18, 96, 245, 132), radius=8, fill=PANEL)
    label(draw, (30, 102), f"{time_s:4.1f} s  |  {phase}", 17)
    state = "SEALED" if sealed else "OPEN"
    draw.rounded_rectangle((622, 96, 780, 132), radius=8, fill=PANEL)
    label(draw, (642, 102), state, 18, TEAL if sealed else MUTED)
    if detail_rgb is not None:
        draw.rounded_rectangle((537, 367, 787, 568), radius=7, fill=PANEL)
        frame.paste(Image.fromarray(detail_rgb.astype(np.uint8)), (542, 402))
        label(draw, (551, 375), "Tool / payload close-up", 16)
    label(draw, (20, 589), f"Vacuum {pressure_pa / 1000:.1f} kPa   |   Trial load {force_n:.2f} / {capacity_n:.2f} N", 18)
    return frame
