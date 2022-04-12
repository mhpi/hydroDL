# from . import time
# from . import grid
# from .interp import interpNan
# import numpy as np
#
#
# def index2d(ind, ny, nx):
#     iy = np.floor(ind / nx)
#     ix = np.floor(ind % nx)
#     return int(iy), int(ix)
#
#
# def fillNan(mat, mask):
#     temp = mat.copy()
#     temp[~mask] = np.nan
#     return temp
