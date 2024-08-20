from typing import Callable, overload
import cv2 as cv
import ramda as R
from .types_ import Img

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
def rescale_w(img: Img, /, width: int) -> Img: ...
@overload
def rescale_w(*, width: int) -> Callable[[Img], Img]: ...

@R.curry
def rescale_w(img: Img, /, width: int) -> Img: # type: ignore
    """Rescales the image to `width`"""
    h, w = img.shape[:2]
    height = int(width * h / w)
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

@overload
def descale_w(img: Img, /, target_width: int) -> Img: ...
@overload
def descale_w(*, target_width: int) -> Callable[[Img], Img]: ...
@R.curry
def descale_w(img: Img, /, target_width: int) -> Img: # type: ignore
    """Rescales the image to `width = min(img.width, target_width)`"""
    _, cur_width = img.shape[:2]
    return rescale_w(img, target_width) if target_width < cur_width else img


@overload
def descale_max(img: Img, /, target_max: int) -> Img: ...
@overload
def descale_max(*, target_max: int) -> Callable[[Img], Img]: ...
@R.curry
def descale_max(img: Img, /, target_max: int) -> Img: # type: ignore
    """Rescales the image's largest dimension to `min(img.height, img.width, target_max)`"""
    h, w = img.shape[:2]
    return rescale_h(img, target_max) if h > w else rescale_w(img, target_max)
