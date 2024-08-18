import colorsys
from jaxtyping import UInt8
import numpy as np

def mod_color(i: int, n: int, lightness: float = 0.5, saturation: float = 0.8) -> tuple[int, int, int]:
    """Color per every `i mod n`, evenly spaced around the hue circle"""
    hue = i / n
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return tuple(int(x*255) for x in rgb) # type: ignore

def black(width: int, height: int) -> UInt8[np.ndarray, 'height width 3']:
    return np.zeros((height, width, 3), dtype=np.uint8)