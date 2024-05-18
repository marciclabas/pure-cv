from jaxtyping import UInt8
import numpy as np

def black(width: int, height: int) -> UInt8[np.ndarray, 'height width 3']:
    return np.zeros((height, width, 3), dtype=np.uint8)