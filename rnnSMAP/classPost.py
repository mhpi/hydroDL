
import numpy as np
import scipy
import time


class statError(object):
    def __init__(self, *, pred, target):
        ngrid, nt = pred.shape
        # Bias
        self.Bias = np.nanmean(pred-target, axis=1)
        # RMSE
        self.RMSE = np.sqrt(np.nanmean((pred-target)**2, axis=1))
        # ubRMSE
        predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
        targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
        predAnom = pred-predMean
        targetAnom = target-targetMean
        self.ubRMSE = np.sqrt(np.nanmean((predAnom-targetAnom)**2, axis=1))
        # rho
        rho = np.full(ngrid, np.nan)
        for k in range(0, ngrid):
            x = pred[k, :]
            y = target[k, :]
            ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
            if ind.shape[0] > 0:
                xx = x[ind]
                yy = y[ind]
                rho[k] = scipy.stats.pearsonr(xx, yy)[0]
        self.rho = rho


class statSigma(object):
    def __init__(self, *, dataMC, dataSigma, dataSigmaBatch):
        if dataMC is not None:
            self.sigmaMC_mat = np.std(dataMC, axis=2)
            self.sigmaMC = np.sqrt(np.nanmean(self.sigmaMC_mat**2, axis=1))
        if dataSigma is not None:
            self.sigmaX_mat = dataSigma
            self.sigmaX = np.sqrt(np.nanmean(self.sigmaX_mat**2, axis=1))
        if dataMC is not None and dataSigma is not None:
            self.sigma_mat = np.sqrt(self.sigmaMC_mat**2+self.sigmaX_mat**2)
            self.sigma = np.sqrt(np.mean(self.sigma_mat**2, axis=1))
        # if dataSigmaBatch is not None:
        #     self.sigma_mat = np.sqrt(np.nanmean(dataSigmaBatch**2, axis=2))
        #     self.sigma = np.sqrt(np.nanmean(self.sigma_mat**2, axis=1))


class statConf(object):
    def __init__(self, *, statSigma, dataPred, dataTarget, dataMC, rmBias=False):
        u = dataPred
        y = dataTarget
        if rmBias is True:
            [ng, nt] = u.shape
            b = np.nanmean(u, axis=1)-np.nanmean(y, axis=1)
            u = u-np.tile(b[:, None], [1, nt])
        # z = np.nanmean(dataMC, axis=2)
        # sigmaLst = ['sigmaMC', 'sigmaX', 'sigma']

        if hasattr(statSigma, 'sigmaX_mat'):
            s = getattr(statSigma, 'sigmaX_mat')
            conf = scipy.special.erf(-np.abs(y-u)/s/np.sqrt(2))+1
            setattr(self, 'conf_sigmaX', conf)

        if hasattr(statSigma, 'sigma_mat'):
            s = getattr(statSigma, 'sigma_mat')
            conf = scipy.special.erf(-np.abs(y-u)/s/np.sqrt(2))+1
            setattr(self, 'conf_sigma', conf)

        # if hasattr(statSigma, 'sigmaComb_mat'):
        #     s = getattr(statSigma, 'sigmaComb_mat')
        #     conf = scipy.special.erf(np.abs(y-u)/s/np.sqrt(2))
        #     setattr(self, 'conf_sigmaComb', conf)

        if hasattr(statSigma, 'sigmaMC_mat'):
            n = dataMC.shape[2]
            # yR = np.stack((y, 2*u-y), axis=2)
            # yRsort = np.sort(yR, axis=2)
            # bMat1 = dataMC <= np.tile(yRsort[:, :, 0:1], [1, 1, n])
            # n1 = np.count_nonzero(bMat1, axis=2)
            # bMat2 = dataMC >= np.tile(yRsort[:, :, 1:2], [1, 1, n])
            # n2 = np.count_nonzero(bMat2, axis=2)
            # conf = (n1+n2)/n
            # conf[np.isnan(y)] = np.nan

            # y = dataMC[:, :, 0]
            dmat = np.tile(np.abs(y-u)[:, :, None], [1, 1, n])
            dmatMC = np.abs(dataMC-np.tile(u[:, :, None], [1, 1, n]))
            bMat = dmatMC >= dmat
            n1 = np.count_nonzero(bMat, axis=2)
            conf = n1/n
            conf[np.isnan(y)] = np.nan

            # m1 = np.concatenate((y[:, :, None], dataMC), axis=2)
            # m2 = np.concatenate(((2*u-y)[:, :, None], dataMC), axis=2)
            # # rm1 = np.argsort(m1)[:, :, 0]
            # rm1 = np.where(np.argsort(m1) == 0)[2].reshape(y.shape[0],y.shape[1])
            # # rm2 = np.argsort(m2)[:, :, 0]
            # rm2 = np.where(np.argsort(m2) == 0)[2].reshape(y.shape[0], y.shape[1])
            # conf = 1-np.abs(rm1-rm2)/n
            # conf[np.isnan(y)] = np.nan
            setattr(self, 'conf_sigmaMC', conf)


class statProb(object):
    def __init__(self, *, statSigma, dataPred, dataTarget, dataMC):
        u = dataPred
        y = dataTarget
        z = np.nanmean(dataMC, axis=2)
        # sigmaLst = ['sigmaMC', 'sigmaX', 'sigma']

        if hasattr(statSigma, 'sigmaX_mat'):
            s = getattr(statSigma, 'sigmaX_mat')
            # prob = scipy.special.erf(np.abs(y-u)/s/np.sqrt(2))
            prob = scipy.stats.norm.pdf((y-u)/s)
            setattr(self, 'prob_sigmaX', prob)

        if hasattr(statSigma, 'sigma_mat'):
            s = getattr(statSigma, 'sigma_mat')
            # prob = scipy.special.erf(np.abs(y-z)/s/np.sqrt(2))
            prob = scipy.stats.norm.pdf((y-u)/s)
            setattr(self, 'prob_sigma', prob)

        if hasattr(statSigma, 'sigmaComb_mat'):
            s = getattr(statSigma, 'sigmaComb_mat')
            # prob = scipy.special.erf(np.abs(y-u)/s/np.sqrt(2))
            prob = scipy.stats.norm.pdf((y-u)/s)
            setattr(self, 'prob_sigmaComb', prob)

        if hasattr(statSigma, 'sigmaMC_mat'):
            n = dataMC.shape[2]
            m = np.concatenate((y[:, :, None], dataMC), axis=2)
            rm = np.argsort(m)[:, :, 0]
            prob = 1-np.abs(2*rm-n)/n
            prob[np.isnan(y)] = np.nan
            setattr(self, 'prob_sigmaMC', prob)


class statNorm(object):
    def __init__(self, *, statSigma, dataPred, dataTarget):
        u = dataPred
        y = dataTarget
        sigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
        for sigmaStr in sigmaLst:
            if hasattr(statSigma, sigmaStr+'_mat'):
                s = getattr(statSigma, sigmaStr+'_mat')
                yNorm = (y-u)/s
                setattr(self, 'yNorm_' + sigmaStr, yNorm)
