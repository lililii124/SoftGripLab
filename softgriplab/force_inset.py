"""Compact force–time traces drawn from recorded forces, without text overlays."""
import numpy as np
from PIL import Image, ImageDraw


def force_inset(image, times, forces, now, box, *, limits=None):
    times, forces = np.asarray(times), np.asarray(forces)
    if times.ndim != 1 or forces.shape != times.shape or not np.isfinite(forces).all():
        raise ValueError("force inset requires finite recorded scalar forces")
    left, top, right, bottom = box
    panel = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(panel)
    draw.rectangle(box, fill=(245, 244, 240, 235))
    x0, x1, y0, y1 = left + 12, right - 12, top + 12, bottom - 12
    low, high = (
        limits
        if limits is not None
        else (min(0.0, forces.min()), max(0.0, forces.max()))
    )
    pad = max((high - low) * 0.12, 1e-6)
    low, high = low - pad, high + pad
    span = max(times[-1] - times[0], 1e-12)

    def point(t, f):
        return (
            x0 + (t - times[0]) / span * (x1 - x0),
            y1 - (f - low) / (high - low) * (y1 - y0),
        )

    zero_y = point(times[0], 0)[1]
    draw.line((x0, y0, x0, y1), fill=(177, 177, 170, 255), width=1)
    draw.line((x0, zero_y, x1, zero_y), fill=(177, 177, 170, 255), width=1)
    selected = times <= now
    trace = [point(t, f) for t, f in zip(times[selected], forces[selected])]
    current = point(
        float(np.clip(now, times[0], times[-1])), float(np.interp(now, times, forces))
    )
    trace.append(current)
    if len(trace) > 1:
        draw.line(trace, fill=(65, 66, 62, 255), width=2)
    draw.line((current[0], y0, current[0], y1), fill=(190, 190, 181, 220), width=1)
    draw.ellipse(
        (current[0] - 3, current[1] - 3, current[0] + 3, current[1] + 3),
        fill=(65, 66, 62, 255),
    )
    image.paste(panel, (0, 0), panel)
