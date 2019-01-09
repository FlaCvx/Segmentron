import dicom
import os
import numpy
from matplotlib import pyplot, cm
import numpy as np


def show_liver_slice(slice, ArrayDicom):
    pyplot.figure(dpi=500)
    pyplot.axes().set_aspect('equal', 'datalim')
    pyplot.set_cmap(pyplot.gray())
    x = np.arange(0, ArrayDicom.shape[0], 1)
    y = np.arange(0, ArrayDicom.shape[1], 1)
    #pyplot.pcolormesh(x, y, numpy.flipud(ArrayDicom[:, :, slice]))
    pyplot.pcolormesh(x, y, ArrayDicom[:, :, slice])
