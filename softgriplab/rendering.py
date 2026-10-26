"""Neutral rendering of solver output. No prescribed specimen deformation."""
import numpy as np
from PIL import Image, ImageDraw
from .animations import font

BACKGROUND = "#f5f4f0"
INK = "#292a28"
SECONDARY = "#666760"


def robot_frame(rgb, detail, time_s, phase, forces, gaps):
    frame = Image.new("RGB", (960, 430), BACKGROUND)
    frame.paste(Image.fromarray(rgb.astype(np.uint8)), (0, 0))
    frame.paste(Image.fromarray(detail.astype(np.uint8)), (560, 0))
    return frame
