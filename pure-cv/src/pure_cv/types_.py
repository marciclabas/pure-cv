from typing_extensions import Literal, Annotated, TypedDict, Sequence, TypeAlias
from nudantic import NdArray

Img2D = Annotated[NdArray, 'W H']
Img3D = Annotated[NdArray, 'W H C']
Img: TypeAlias = Img2D | Img3D | Annotated[NdArray, '*']

Contour = Annotated[NdArray, 'C 2']
"""`(x1, y1), ...`"""
Contours = Annotated[NdArray, 'N C 2']
"""`[(x1, y1), ...], ...]`"""

BBox = Annotated[NdArray, '2 2']
"""`(x1, y1), (x2, y2)`"""

BBoxes = Annotated[NdArray, 'N 2 2']
"""`[((x1, y1), (x2, y2)), ...]`"""

Vec2 = tuple[float, float] | Sequence[float] | Annotated[NdArray, '2']
Vertices = Annotated[NdArray, 'N 2'] | Sequence[Vec2]

Corners = Annotated[NdArray, '4 2']
"""`tl, tr, br, bl`"""

Segment = Annotated[NdArray, '1 4'] | Sequence[Sequence[float]]
Segments = Annotated[NdArray, 'N 1 4'] | Sequence[Segment]

Mat3 = Annotated[NdArray, '3 3']

class Rect(TypedDict):
  tl: Vec2
  size: Vec2

class Pads(TypedDict, total=False):
  l: float; r: float; t: float; b: float

Rotation = Literal[90, 180, -90]