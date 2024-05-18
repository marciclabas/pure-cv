from typing import TypeAlias
from jaxtyping import Shaped
import numpy as np

Img2D: TypeAlias = Shaped[np.ndarray, "w h"]
Img3D: TypeAlias = Shaped[np.ndarray, "w h c"]
Img: TypeAlias = Img2D | Img3D | np.ndarray