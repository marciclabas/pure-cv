from typing import overload, Callable
import cv2 as cv
import numpy as np
import ramda as R
from .types_ import Img2D, Img

@R.curry
def imdecode(img, flags = cv.IMREAD_UNCHANGED) -> Img:
  return cv.imdecode(img, flags)

def imread(path: str) -> Img:
  """Read in RGB (instead of BGR)"""
  img = cv.imread(path)
  return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def grayscale(img: Img) -> Img2D:
  """`Array[h, w, ...] -> Array[h, w]`"""
  return img if len(np.shape(img)) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def threshold(img: Img) -> Img2D:
  """`Array[h, w, ...] -> Array[h, w]` with values in `{0, 255}` only"""
  gray_img = grayscale(img)
  return cv.adaptiveThreshold(~gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -3)


@overload
def cvtColor(src: Img, code: int) -> Img: ...
@overload
def cvtColor(*, code: int) -> Callable[[Img], Img]: ...
@R.curry
def cvtColor(src: Img, /, code: int) -> Img: # type: ignore
  return cv.cvtColor(src, code)

@overload
def dilate(src: Img, /, kernel: cv.Mat, iterations: int) -> Img: ...
@overload
def dilate(*, kernel: cv.Mat, iterations: int) -> Callable[[Img], Img]: ...

@R.curry
def dilate(src: Img, /, kernel: cv.Mat, iterations: int) -> Img: # type: ignore
  return cv.dilate(src, kernel=kernel, iterations=iterations)

@overload
def erode(src: Img, /, kernel: cv.Mat, iterations: int) -> Img: ...
@overload
def erode(*, kernel: cv.Mat, iterations: int) -> Callable[[Img], Img]: ...

@R.curry
def erode(src: Img, /, kernel: cv.Mat, iterations: int) -> Img: # type: ignore
  return cv.erode(src, kernel=kernel, iterations=iterations)