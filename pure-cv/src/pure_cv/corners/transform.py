from typing_extensions import Unpack
from jaxtyping import Shaped
import numpy as np
import cv2 as cv
import pure_cv as vc

def cartesian2homogeneous(points: Shaped[np.ndarray, '* 2']) -> Shaped[np.ndarray, '* 3']:
  """Convert cartesian points to homogeneous points"""
  ones = np.ones((*points.shape[:-1], 1))
  return np.concatenate([points, ones], axis=-1)

def homogeneous2cartesian(points: Shaped[np.ndarray, '* 3']) -> Shaped[np.ndarray, '* 2']:
  """Convert homogeneous points to cartesian points"""
  return points[..., :-1] / points[..., -1:]

def warp(points: Shaped[np.ndarray, '* 2'], homography: vc.Mat3) -> Shaped[np.ndarray, '* 2']:
  """Warp points using a homography"""
  M = np.array(homography)
  h = cartesian2homogeneous(points)
  h2 = h @ M.T
  return homogeneous2cartesian(h2)

def pad(corners: vc.Corners, *, padx: float, pady: float) -> vc.Corners:
  """Pad perspective corners, by:
  1. Move the centroid to 0
  2. Scale by `[1+padx, 1+pady]`
  3. Move back to original position
  """
  cs = np.array(corners)
  centroid = cs.mean(axis=0)
  return (cs - centroid) * np.array([1+padx, 1+pady]) + centroid

# def homography(corners: vc.Corners):
#   """Compute homography to map corners to a rectangle. Returns `homography, (width, height)`"""
#   tl, tr, br, bl = np.array(corners)
#   w = int((np.linalg.norm(tl-tr) + np.linalg.norm(bl-br)) / 2)
#   h = int((np.linalg.norm(tl-bl) + np.linalg.norm(tr-br)) / 2)
#   src = np.array([tl, tr, br, bl])
#   dst = np.array([[0, 0], [w, 0], [w, h], [0, h]])
#   return cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)), (w, h)

def homography(corners: vc.Corners, *, l=0., r=0., t=0., b=0.):
  """Compute homography to map corners to a rectangle. Returns `homography, (width, height)`
  - `l, r, t, b`: relative paddings to add around the rectangle
  """
  tl, tr, br, bl = np.array(corners)
  detected_w = int((np.linalg.norm(tl-tr) + np.linalg.norm(bl-br)) / 2)
  detected_h = int((np.linalg.norm(tl-bl) + np.linalg.norm(tr-br)) / 2)
  pad_left = int(detected_w*l); pad_right = int(detected_w*r)
  pad_top = int(detected_h*t); pad_bot = int(detected_h*b)
  w = int(detected_w + pad_left + pad_right)
  h = int(detected_h + pad_top + pad_bot)
  src = np.array([tl, tr, br, bl])
  dst = np.array([[pad_left, pad_top], [w-pad_right, pad_top], [w-pad_right, h-pad_bot], [pad_left, h-pad_bot]])
  return cv.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32)), (w, h)

def correct(img: vc.Img, corners: vc.Corners, **pads: Unpack[vc.Pads]) -> vc.Img:
  """Correct image perspective (from absolute corners)
  - `l, r, t, b`: relative paddings to add around the rectangle (default to 0)
  """
  M, (w, h) = homography(corners, **pads)
  return cv.warpPerspective(img, M, (w, h)) 

def unwarp_contours(contours: vc.Contours, corners: vc.Corners) -> vc.Contours:
  """Re-coord contours w.r.t. to the original image
  - `corners`: perspective corners of the original image, yielding the corrected image
  - `contours`: contours w.r.t. the corrected image
  """
  M, _ = homography(corners)
  Minv = np.linalg.inv(M)
  return warp(contours, Minv)