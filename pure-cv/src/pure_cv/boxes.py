from typing_extensions import Unpack
import numpy as np
import pure_cv as vc

def bbox2contour(bboxes: vc.BBox | vc.BBoxes) -> vc.Contour | vc.Contours:
  """Convert a bounding box to a 4-corner contour. Supports vectorized operation"""
  X1 = bboxes[..., 0, 0] # (N,)
  Y1 = bboxes[..., 0, 1] # (N,)
  X2 = bboxes[..., 1, 0] # (N,)
  Y2 = bboxes[..., 1, 1] # (N,)
  TLs = np.stack([X1, Y1], axis=-1) # (N, 2)
  TRs = np.stack([X2, Y1], axis=-1) # (N, 2)
  BRs = np.stack([X2, Y2], axis=-1) # (N, 2)
  BLs = np.stack([X1, Y2], axis=-1) # (N, 2)
  return np.stack([TLs, TRs, BRs, BLs], axis=-2) # (N, 4, 2)

def contour2bbox(contours: vc.Contour | vc.Contours) -> vc.BBox | vc.BBoxes:
  """Convert a 4-corner contour to a bounding box. Supports vectorized operation"""
  xs = contours[..., 0] # (N,)
  ys = contours[..., 1] # (N,)
  X1 = np.min(xs, axis=-1) # (N,)
  Y1 = np.min(ys, axis=-1) # (N,)
  X2 = np.max(xs, axis=-1) # (N,)
  Y2 = np.max(ys, axis=-1) # (N,)
  TL = np.stack([X1, Y1], axis=-1) # (N, 2)
  BR = np.stack([X2, Y2], axis=-1) # (N, 2)
  return np.stack([TL, BR], axis=-2) # (N, 2, 2)

# def roi_old(
#   img: vc.Img, contour: vc.Contour, *,
#   l = 0.1, r = 0.1, t = 0.15, b = 0.25
# ) -> vc.Img:
#   x, y, w, h = cv.boundingRect(np.array(contour, dtype=np.float32)) # just in case, other types fail!
#   top = max(int(y - t*h), 0)
#   bot = int(y + (1+b)*h)
#   left = max(int(x - l*w), 0)
#   right = int(x + (1+r)*w)
#   return img[top:bot, left:right]

def roi(
  img: vc.Img, contour: vc.Contour, *,
  l = 0.1, r = 0.1, t = 0.15, b = 0.25
):
  """Extract a 4-corner contour by persective correcting it to a rectangle"""
  return vc.corners.correct(img, contour, l=l, r=r, t=t, b=b)


def extract_contours(img: vc.Img, contours: vc.Contours, **pads: Unpack[vc.Pads]) -> list[vc.Img]:
  """Extract contours from an image"""
  return [roi(img, contour, **pads) for contour in contours]