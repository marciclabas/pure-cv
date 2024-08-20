from typing import Sequence, overload, Callable
from jaxtyping import Shaped
import cv2 as cv
import numpy as np
import ramda as R
import pure_cv as vc

@overload
def vertices(img: vc.Img, /, vertices: vc.Vertices, *, radius: int = 10, color: int | tuple[int, int, int] = (255, 0, 0)) -> vc.Img: ...
@overload
def vertices(*, vertices: vc.Vertices, radius: int = 10, color: int | tuple[int, int, int] = (255, 0, 0)) -> Callable[[vc.Img], vc.Img]: ...

@R.curry
def vertices( # type: ignore
  img: vc.Img, /, vertices: vc.Vertices, radius: int = 10, color: int | tuple[int, int, int] = (255, 0, 0)
) -> vc.Img:
    if isinstance(color, tuple):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(np.shape(img)) == 2 else np.copy(img)
    output = img.copy()
    for [x, y] in vertices:
        cv.circle(output, (int(x), int(y)), radius=radius, color=color, thickness=-1) # type: ignore
    return output


@overload
def vlines(img: vc.Img, /, xs: list[float], *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> vc.Img: ...
@overload
def vlines(*, xs: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[vc.Img], vc.Img]: ...

@R.curry
def vlines( # type: ignore
    img: vc.Img, /, xs: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> vc.Img:
    height = img.shape[0]
    ls = np.array([[[x, 0, x, height-1]] for x in xs], dtype=np.int32)
    return lines(img, ls, color=color, thickness=thickness)

@overload
def hlines(img: vc.Img, /, ys: list[float], *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> vc.Img: ...
@overload
def hlines(*, ys: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[vc.Img], vc.Img]: ...

@R.curry
def hlines( # type: ignore
    img: vc.Img, /, ys: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> vc.Img:
    width = img.shape[1]
    ls = np.array([[[0, y, width-1, y]] for y in ys], dtype=np.int32)
    return lines(img, ls, color=color, thickness=thickness)

Segment = Shaped[np.ndarray, '1 4']
Segments = Shaped[np.ndarray, 'N 1 4'] | Sequence[Segment]

@overload
def lines(img: vc.Img, /, lines: Segments, *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> vc.Img: ...
@overload
def lines(*, lines: Segments, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[vc.Img], vc.Img]: ...
@R.curry
def lines( # type: ignore
  img: vc.Img, /, lines: Segments, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> vc.Img: # type: ignore
  out = img.copy()
  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(out, [x1, y1], [x2, y2], color=color, thickness=thickness) # type: ignore
  return out


@overload
def contours(img: vc.Img, /, contours: vc.Contours, *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> vc.Img: ...
@overload
def contours(*, contours: vc.Contours, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[vc.Img], vc.Img]: ...
@R.curry
def contours( # type: ignore
  img: vc.Img, /, contours: vc.Contours, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> vc.Img:
  if isinstance(color, tuple):
    outimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(np.shape(img)) == 2 else np.copy(img)
  else:
    outimg = np.copy(img) if len(np.shape(img)) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
  return cv.drawContours(outimg, [np.array(c, dtype=np.int32) for c in contours], -1, color, thickness) # type: ignore

@overload
def bboxes(img: vc.Img, /, bboxes: vc.BBoxes, *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> vc.Img: ...
@overload
def bboxes(*, bboxes: vc.BBoxes, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[vc.Img], vc.Img]: ...
@R.curry
def bboxes( # type: ignore
  img: vc.Img, /, bboxes: vc.BBoxes, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> vc.Img:
  cnts = vc.bbox2contour(bboxes)
  return contours(img, contours=cnts, color=color, thickness=thickness)
  