from typing_extensions import Literal, Annotated, TypedDict, TypeAlias, TYPE_CHECKING
if TYPE_CHECKING:
  import numpy as np

Img2D = Annotated['np.ndarray', 'W H']
Img3D = Annotated['np.ndarray', 'W H C']
Img: TypeAlias = Img2D | Img3D | Annotated['np.ndarray', '*']

Contour = Annotated[list, 'C 2']
"""`(x1, y1), (x2, y2), ...`"""
Contours = Annotated[list, 'N C 2']
"""`[[x1, y1], [x2, y2], ...], ...]`"""

BBox = Annotated['np.ndarray', '4']
"""`x1, y1, x2, y2`"""

Vec2 = tuple[float, float]

class Corners(TypedDict):
  tl: Vec2
  tr: Vec2
  br: Vec2
  bl: Vec2

Rotation = Literal[90, 180, -90]