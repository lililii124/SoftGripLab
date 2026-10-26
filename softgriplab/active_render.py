"""Text-free replay of solved axial states; no imposed deformation or robot pose."""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .force_inset import force_inset


def render_trace(source, output, *, fps=20, frames=80):
    with np.load(source) as data:
        times, tool = data["time_s"], data["tool_m"]
        state, forces = data["state"], data["force_n"]
    if not np.all(state[:, :, 0] == 0):
        # Show the release onset in slow motion rather than shrinking a 20 mm
        # specimen to a few pixels to fit the full subsequent ballistic flight.
        count = int(np.searchsorted(times, min(0.09, times[-1]), side="right"))
        times, tool, state, forces = times[:count], tool[:count], state[:, :count], forces[:, :count]
    length, width = 0.02, np.sqrt(0.0002)
    panels = len(state)
    panel_w, height = 400, 480
    # Every frame uses the same metric scale. The specimen's axial displacement
    # interpolates the two solved nodes; no claimed 3D transverse deformation.
    low = min(-0.004, float(state[:, :, 0].min())-0.004)
    high = max(length+float(tool.max())+0.005,
               length+float(state[:, :, 2].max())+0.005)
    scale = min(350/(high-low), 160/width)
    def y(z):
        return 390-(z-low)*scale
    result = []
    limits = (min(0, float(forces.min())), float(forces.max()))
    for index in np.linspace(0, len(times)-1, frames).round().astype(int):
        image = Image.new("RGB", (panels*panel_w, height), (242, 241, 237))
        draw = ImageDraw.Draw(image)
        for panel in range(panels):
            cx = panel_w*(panel+0.5)
            x0, x1 = cx-width*scale/2, cx+width*scale/2
            bottom = float(state[panel, index, 0])
            top = length+float(state[panel, index, 2])
            draw.rectangle((x0, y(top), x1, y(bottom)), fill=(168, 156, 136), outline=(79, 75, 67), width=2)
            # Fine axial material lines follow solved extension, at true scale.
            for z in np.linspace(bottom, top, 33)[1:-1]:
                draw.line((x0+1, y(z), x1-1, y(z)), fill=(143, 134, 119), width=1)
            for x in np.linspace(x0, x1, 13)[1:-1]:
                draw.line((x, y(top)+1, x, y(bottom)-1), fill=(155, 145, 127), width=1)
            z_tool = length+tool[index]
            draw.rectangle((cx-width*scale*0.8, y(z_tool)-9,
                            cx+width*scale*0.8, y(z_tool)), fill=(72, 75, 74))
            # A fixed base appears only for clamped probes.
            if np.all(state[panel, :, 0] == 0):
                draw.rectangle((cx-width*scale, y(0), cx+width*scale, y(0)+10), fill=(102, 103, 98))
            force_inset(image, times, forces[panel], times[index],
                        (panel*panel_w+24, 405, (panel+1)*panel_w-24, 465), limits=limits)
        result.append(image)
    output = Path(output)
    result[0].save(output, save_all=True, append_images=result[1:], duration=round(1000/fps), loop=0)
    result[len(result)//2].save(output.with_suffix(".png"))
    return output


def render_active_identification(output):
    output = Path(output)
    report = json.loads((output/"report.json").read_text(encoding="utf-8"))
    if (output/"ambiguity_release.npz").exists():
        render_trace(output/"ambiguity_release.npz", output/"ambiguity_release.gif")
        render_trace(output/"ambiguity_probe.npz", output/"ambiguity_probe.gif")
    for i in range(len(report["first_trial_task_probes"])):
        render_trace(output/f"probe_{i+1}.npz", output/f"probe_{i+1}.gif")
