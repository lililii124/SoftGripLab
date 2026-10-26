"""Neutral rendering of solver output. No prescribed specimen deformation."""
import numpy as np
from PIL import Image, ImageDraw
from .animations import font

BACKGROUND = "#f5f4f0"
INK = "#292a28"
SECONDARY = "#666760"


def robot_frame(rgb, detail, time_s, phase, forces, gaps):
    frame = Image.new("RGB", (960, 480), BACKGROUND)
    frame.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 45))
    frame.paste(Image.fromarray(detail.astype(np.uint8)), (560, 45))
    d = ImageDraw.Draw(frame)
    d.text((22, 12), "Panda · Pick and place", font=font(20), fill=INK)
    d.text((579, 12), "Contact", font=font(18), fill=SECONDARY)
    return frame
