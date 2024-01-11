import numpy as np

_3 = int; W = int; H = int

def black(width: int, height: int) -> np.ndarray[tuple[H, W, _3], np.uint8]:
    return np.zeros((height, width, 3), dtype=np.uint8)