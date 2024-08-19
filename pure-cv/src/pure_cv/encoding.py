from typing import Literal, overload, Callable
import base64
import numpy as np
import cv2 as cv
import numpy as np
import ramda as R
from .types_ import Img

@overload
def encode(img: Img, format: Literal[".png", ".jpg", ".webp"]) -> bytes: ...
@overload
def encode(*, format: Literal[".png", ".jpg", ".webp"]) -> Callable[[Img], bytes]: ...

@R.curry
def encode(img: Img, format: Literal[".png", ".jpg", ".webp"]) -> bytes: # type: ignore
    _, encoded = cv.imencode(format, img)
    return encoded.tobytes()

@overload
def b64encode(img: Img, format: Literal[".png", ".jpg", ".webp"] = ".png", url_safe: bool = True) -> str: ...
@overload
def b64encode(*, format: Literal[".png", ".jpg", ".webp"] = ".png", url_safe: bool = True) -> Callable[[Img], str]: ...

@R.curry
def b64encode(img: Img, format: Literal[".png", ".jpg", ".webp"] = ".png", url_safe: bool = True) -> str: # type: ignore
    """Encode `img` into a Base64 string"""
    _, enc = cv.imencode(format, img)
    encoded = bytes(enc)
    b64 = base64.urlsafe_b64encode(encoded) if url_safe else base64.b64encode(encoded)
    return b64.decode("utf-8")

def decode(img: bytes, flags = cv.IMREAD_UNCHANGED) -> Img:
    arr = np.frombuffer(img, np.uint8)
    return cv.imdecode(arr, flags)

def b64decode(b64img: str) -> Img:
    b = base64.urlsafe_b64decode(b64img)
    arr = np.frombuffer(b, np.uint8)
    return cv.imdecode(arr, -1)
