import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from .types_ import Img

def show(
  *inputs: Img | tuple[str, Img],
  ncols: int = 3, height: float = 5, width: float = 5,
  axis = False, cmap = "gray"
) -> Figure:
  """Shows a set of images or (title, image) pairs"""

  def show_input(ax: Axes, input: Img | tuple[str, Img]):
    match input:
      case title, img:
        ax.imshow(img, cmap=cmap)
        ax.set_title(title)
      case img:
        ax.imshow(img, cmap=cmap)
    ax.axis(axis)
    
  n = len(inputs)
  if n == 0:
    return Figure()
  if n == 1:
    fig, ax = plt.subplots(figsize=(width, height))
    show_input(ax, inputs[0])
  elif n <= ncols:
    fig, ax = plt.subplots(1, n, figsize=(5*n, height))
    for i, input in enumerate(inputs):
      show_input(ax[i], input)
  elif ncols == 1:
    nrows = n
    fig, ax = plt.subplots(nrows, 1, figsize=(5, height*nrows))
    for i, input in enumerate(inputs):
      show_input(ax[i], input)
  else:
    nrows = math.ceil(n / ncols)
    fig, ax = plt.subplots(nrows, ncols, figsize=(5*ncols, height*nrows))
    for i, input in enumerate(inputs):
      show_input(ax[i // ncols, i % ncols], input)
  
  plt.close(fig)
  return fig