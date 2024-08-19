from typing import Sequence, overload, Callable
from jaxtyping import Shaped
import cv2 as cv
import numpy as np
import ramda as R
from ..types_ import Img

Vec2 = Shaped[np.ndarray, '2']
Vertices = Shaped[np.ndarray, 'N 2'] | Sequence[Vec2]

@overload
def vertices(img: Img, /, vertices: Vertices, *, radius: int = 10, color: int | tuple[int, int, int] = (255, 0, 0)) -> Img: ...
@overload
def vertices(*, vertices: Vertices, radius: int = 10, color: int | tuple[int, int, int] = (255, 0, 0)) -> Callable[[Img], Img]: ...

@R.curry
def vertices( # type: ignore
  img: Img, /, vertices: Vertices, radius: int = 10, color: int | tuple[int, int, int] = (255, 0, 0)
) -> Img:
    if isinstance(color, tuple):
        img = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(np.shape(img)) == 2 else np.copy(img)
    output = img.copy()
    for [x, y] in vertices:
        cv.circle(output, (int(x), int(y)), radius=radius, color=color, thickness=-1) # type: ignore
    return output


@overload
def vlines(img: Img, /, xs: list[float], *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Img: ...
@overload
def vlines(*, xs: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[Img], Img]: ...

@R.curry
def vlines( # type: ignore
    img: Img, /, xs: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> Img:
    height = img.shape[0]
    ls = np.array([[[x, 0, x, height-1]] for x in xs], dtype=np.int32)
    return lines(img, ls, color=color, thickness=thickness)

@overload
def hlines(img: Img, /, ys: list[float], *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Img: ...
@overload
def hlines(*, ys: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[Img], Img]: ...

@R.curry
def hlines( # type: ignore
    img: Img, /, ys: list[float], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> Img:
    width = img.shape[1]
    ls = np.array([[[0, y, width-1, y]] for y in ys], dtype=np.int32)
    return lines(img, ls, color=color, thickness=thickness)

Segment = Shaped[np.ndarray, '1 4']
Segments = Shaped[np.ndarray, 'N 1 4'] | Sequence[Segment]

@overload
def lines(img: Img, /, lines: Segments, *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Img: ...
@overload
def lines(*, lines: Segments, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[Img], Img]: ...
@R.curry
def lines( # type: ignore
  img: Img, /, lines: Segments, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> Img: # type: ignore
  out = img.copy()
  for line in lines:
    x1, y1, x2, y2 = line[0]
    cv.line(out, [x1, y1], [x2, y2], color=color, thickness=thickness) # type: ignore
  return out


@overload
def contours(img: Img, /, contours: Sequence[Shaped[np.ndarray, 'N 2']], *, color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Img: ...
@overload
def contours(*, contours: Sequence[Shaped[np.ndarray, 'N 2']], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3) -> Callable[[Img], Img]: ...
@R.curry
def contours( # type: ignore
  img: Img, /, contours: Sequence[Shaped[np.ndarray, 'N 2']], color: int | tuple[int, int, int] = (255, 0, 0), thickness: int = 3
) -> Img:
  if isinstance(color, tuple):
    outimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR) if len(np.shape(img)) == 2 else np.copy(img)
  else:
    outimg = np.copy(img) if len(np.shape(img)) == 2 else cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
  return cv.drawContours(outimg, [np.array(c, dtype=np.int32) for c in contours], -1, color, thickness) # type: ignore

