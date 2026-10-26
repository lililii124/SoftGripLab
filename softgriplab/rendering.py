"""Neutral rendering of solver output. No prescribed specimen deformation."""
import numpy as np
from PIL import Image, ImageDraw
from .animations import font

BACKGROUND = "#f5f4f0"
INK = "#292a28"
SECONDARY = "#666760"


def robot_frame(rgb, detail, time_s, phase, forces, gaps):
    frame = Image.new("RGB", (960, 560), BACKGROUND)
    frame.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 65))
    frame.paste(Image.fromarray(detail.astype(np.uint8)), (560, 65))
    d = ImageDraw.Draw(frame)
    d.text((22, 12), "Panda / frictional pick-and-place", font=font(24), fill=INK)
    d.text((23, 43), f"{time_s:4.1f} s   {phase}", font=font(14), fill=SECONDARY)
    d.text((579, 43), "Finger contact / close view", font=font(14), fill=SECONDARY)
    d.line((560, 65, 560, 495), fill="#d4d2ca")
    d.text((22, 511), "Motor-driven arm and fingers", font=font(16), fill=INK)
    d.text(
        (22, 535), "Rigid object / no attachment joint", font=font(13), fill=SECONDARY
    )
    d.text(
        (578, 507),
        f"Normal force   L {forces[0]:5.2f} N   R {forces[1]:5.2f} N",
        font=font(15),
        fill=INK,
    )
    measured = [f"{g * 1000:+.2f}" if np.isfinite(g) else "open" for g in gaps]
    d.text(
        (578, 535),
        f"Contact distance (mm)   {measured[0]} / {measured[1]}",
        font=font(13),
        fill=SECONDARY,
    )
    return frame
