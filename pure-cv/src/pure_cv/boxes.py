import numpy as np
import pure_cv as vc

def bbox2contour(bbox: vc.BBox) -> vc.Contour:
  """Convert a bounding box to a 4-corner contour"""
  x1, y1, x2, y2 = bbox
  return np.array([(x1, y1), (x2, y1), (x2, y2), (x1, y2)])

def contour2bbox(contour: vc.Contour) -> vc.BBox:
  """Convert a 4-corner contour to a bounding box"""
  xs, ys = contour.T
  return np.array([xs.min(), ys.min(), xs.max(), ys.max()])