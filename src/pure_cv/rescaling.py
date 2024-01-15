import cv2 as cv
import ramda as R

@R.curry
def rescale_h(img: cv.Mat, height: int) -> cv.Mat:
    """Rescales the image to `height`"""
    h, w = img.shape[:2]
    width = int(height * w/h)
    return cv.resize(img, (width, height))

@R.curry
def descale_h(img: cv.Mat, target_height: int) -> cv.Mat:
    """Rescales the image to `height = min(img.height, target_height)`"""
    cur_height, _ = img.shape[:2]
    return rescale_h(img, target_height) if target_height < cur_height else img
