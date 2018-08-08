
import numpy as np


class statError(object):
    def __init__(self, *, pred, target):
        ngrid, nt = pred.shape
        # RMSE
        self.RMSE = np.sqrt(np.nanmean((pred-target)**2, axis=1))
        # ubRMSE
        predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
        targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
        predAnom = pred-predMean
        targetAnom = target-targetMean
        self.ubRMSE = np.sqrt(np.nanmean((predAnom-targetAnom)**2, axis=1))


class statSigma(object):
    def __init__(self, *, dataMC, dataSigma):
        self.sigmaMC_mat = np.std(dataMC, axis=2)
        self.sigmaX_mat = dataSigma
        self.sigmaMC = np.nanmean(self.sigmaMC_mat, axis=1)
        self.sigmaX = np.nanmean(self.sigmaX_mat, axis=1)
