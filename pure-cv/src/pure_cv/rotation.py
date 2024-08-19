from typing import Literal
import cv2 as cv
from .types_ import Rotation
from .types_ import Img

def code(rot: Rotation) -> int:
  if rot == 90:
    return cv.ROTATE_90_COUNTERCLOCKWISE
  if rot == -90:
    return cv.ROTATE_90_CLOCKWISE
  if rot == 180:
    return cv.ROTATE_180
  else:
    raise ValueError(f'Invalid rotation {rot}')
  
def rotate(img: Img, rotation: Rotation | Literal[0]) -> Img:
  if rotation == 0:
    return img
  return cv.rotate(img, code(rotation))