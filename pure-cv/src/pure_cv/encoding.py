from typing import Literal
import base64
import numpy as np
import cv2 as cv
import numpy as np
import ramda as R

@R.curry
def encode(img: cv.Mat, format: Literal[".png", ".jpg", ".webp"]) -> bytes: 
    _, encoded = cv.imencode(format, img)
    return encoded.tobytes()

def decode(img: bytes, flags = cv.IMREAD_UNCHANGED) -> cv.Mat:
    arr = np.frombuffer(img, np.uint8)
    return cv.imdecode(arr, flags)

@R.curry
def b64encode(img: cv.Mat, format: str = ".png", url_safe: bool = True) -> str:
    """Encode `img` into a Base64 string"""
    _, encoded = cv.imencode(format, img)
    b64 = base64.urlsafe_b64encode(encoded) if url_safe else base64.b64encode(encoded)
    return b64.decode("utf-8")

def b64decode(b64img: str) -> cv.Mat:
    b = base64.urlsafe_b64decode(b64img)
    arr = np.frombuffer(b, np.uint8)
    return cv.imdecode(arr, -1)
