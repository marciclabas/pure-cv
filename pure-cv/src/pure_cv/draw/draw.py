import cv2 as cv
import numpy as np
import ramda as R

N = int; _1 = int; _2 = int; _4 = int

Img = cv.Mat
Img3D = cv.Mat

@R.curry
def vertices(
    vertices: np.ndarray,
    img: Img,
    radius: int = 10,
    color: int | tuple[int] = (255, 0, 0)
):
    if isinstance(color, tuple):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(np.shape(img)) == 2 else np.copy(img)
        
    output = img.copy()
    for [x, y] in vertices:
        cv.circle(output, (int(x), int(y)), radius=radius, color=color, thickness=-1)
        
    return output

@R.curry
def vlines(xs: list[float], img: Img, color = (255, 0, 0), thickness = 3):
    height = img.shape[0]
    ls = np.array([[[x, 0, x, height-1]] for x in xs], dtype=np.int32)
    return lines(ls, img, color, thickness)

@R.curry
def hlines(ys: list[float], img: Img, color = (255, 0, 0), thickness = 3):
    width = img.shape[1]
    ls = np.array([[[0, y, width-1, y]] for y in ys], dtype=np.int32)
    return lines(ls, img, color, thickness)

@R.curry
def lines(
    lines: tuple[N, _1, _4],
    img: Img,
    color = (255, 0, 0),
    thickness = 3
):
    out = img.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv.line(out, [x1, y1], [x2, y2], color=color, thickness=thickness)
        
    return out

@R.curry
def contours(
    contours: list[np.ndarray[tuple[N, _1, _2]]],
    img: Img,
    color: int | tuple[int] = (255, 0, 0),
    thickness = 3
) -> Img:
    if isinstance(color, tuple):
        outimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(np.shape(img)) == 2 else np.copy(img)
    else:
        outimg = np.copy(img) if len(np.shape(img)) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
    return cv.drawContours(outimg, [np.array(c, dtype=np.int32) for c in contours], -1, color, thickness)