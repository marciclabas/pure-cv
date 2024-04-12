import cv2 as cv
import numpy as np
import ramda as R

Img2D = cv.Mat
Img3D = cv.Mat
Img = Img2D
Mat3x3 = cv.Mat
_N = int
_2 = int

@R.curry
def imdecode(img, flags = cv.IMREAD_UNCHANGED) -> Img:
    return cv.imdecode(img, flags)

def imread(path: str) -> Img:
    """Read in RGB"""
    img = cv.imread(path)
    return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def grayscale(img: Img) -> Img2D:
    """`Array[h, w, ...] -> Array[h, w]`"""
    return img if len(np.shape(img)) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def threshold(img: Img) -> Img2D:
    """`Array[h, w, ...] -> Array[h, w]` with values in `{0, 255}` only"""
    gray_img = grayscale(img)
    return cv.adaptiveThreshold(~gray_img, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 15, -3)

@R.curry
def cvtColor(src: Img, code: int) -> Img:
    return cv.cvtColor(src, code)

@R.curry
def dilate(src: Img, kernel: cv.Mat, iterations: int) -> Img:
    return cv.dilate(src, kernel=kernel, iterations=iterations)

@R.curry
def erode(src: Img, kernel: cv.Mat, iterations: int) -> Img:
    return cv.erode(src, kernel=kernel, iterations=iterations)