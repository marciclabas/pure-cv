from typing import Callable, overload
import cv2 as cv
import ramda as R
from .img_types import Img

@overload
def rescale_h(img: Img, /, height: int) -> Img: ...
@overload
def rescale_h(*, height: int) -> Callable[[Img], Img]: ...

@R.curry
def rescale_h(img: Img, /, height: int) -> Img: # type: ignore
    """Rescales the image to `height`"""
    h, w = img.shape[:2]
    width = int(height * w / h)
    return cv.resize(img, (width, height))

@overload
def descale_h(img: Img, /, target_height: int) -> Img: ...
@overload
def descale_h(*, target_height: int) -> Callable[[Img], Img]: ...

@R.curry
def descale_h(img: Img, /, target_height: int) -> Img: # type: ignore
    """Rescales the image to `height = min(img.height, target_height)`"""
    cur_height, _ = img.shape[:2]
    return rescale_h(img, target_height) if target_height < cur_height else img
