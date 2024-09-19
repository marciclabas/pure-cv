import colorsys
from jaxtyping import UInt8
import numpy as np
import pure_cv as vc
import cv2

def mod_color(i: int, n: int, lightness: float = 0.5, saturation: float = 0.8) -> tuple[int, int, int]:
    """Color per every `i mod n`, evenly spaced around the hue circle"""
    hue = i / n
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return tuple(int(x*255) for x in rgb) # type: ignore

def black(width: int, height: int) -> UInt8[np.ndarray, 'height width 3']:
    return np.zeros((height, width, 3), dtype=np.uint8)

def blend_texture(img: vc.Img, texture: vc.Img, alpha=0.7) -> vc.Img:
  """Alpha blend a texture on top of an image"""
  h, w = img.shape[:2]
  texture = cv2.resize(texture, (w, h))
  return cv2.addWeighted(img.astype(float), alpha, texture.astype(float), 1 - alpha, 0).astype(np.uint8)