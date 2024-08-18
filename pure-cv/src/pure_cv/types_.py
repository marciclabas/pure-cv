from typing import Literal, Annotated, TypeAlias, TYPE_CHECKING
if TYPE_CHECKING:
  import numpy as np

Img2D = Annotated['np.ndarray', 'W H']
Img3D = Annotated['np.ndarray', 'W H C']
Img: TypeAlias = 'Img2D | Img3D | np.ndarray'

Contour = Annotated['np.ndarray', 'N 2']
"""`(x1, y1), (x2, y2), ...`"""
BBox = Annotated['np.ndarray', '4']
"""`x1, y1, x2, y2`"""

Rotation = Literal[90, 180, -90]