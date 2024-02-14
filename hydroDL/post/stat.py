import numpy as np
import scipy.stats
from hydroDL.master.master import calFDC

keyLst = ["Bias", "RMSE", "ubRMSE", "Corr"]

def statError(pred, target):
    ngrid, nt = pred.shape
    # Bias
    Bias = np.nanmean(pred - target, axis=1)
    # RMSE
    RMSE = np.sqrt(np.nanmean((pred - target)**2, axis=1))
    # ubRMSE
    predMean = np.tile(np.nanmean(pred, axis=1), (nt, 1)).transpose()
    targetMean = np.tile(np.nanmean(target, axis=1), (nt, 1)).transpose()
    predAnom = pred - predMean
    targetAnom = target - targetMean
    ubRMSE = np.sqrt(np.nanmean((predAnom - targetAnom)**2, axis=1))
    # FDC metric
    predFDC = calFDC(pred)
    targetFDC = calFDC(target)
    FDCRMSE = np.sqrt(np.nanmean((predFDC - targetFDC) ** 2, axis=1))
    # rho R2 NSE
    Corr = np.full(ngrid, np.nan)
    CorrSp = np.full(ngrid, np.nan)
    R2 = np.full(ngrid, np.nan)
    NSE = np.full(ngrid, np.nan)
    PBiaslow = np.full(ngrid, np.nan)
    PBiashigh = np.full(ngrid, np.nan)
    absPBiaslow = np.full(ngrid, np.nan)
    absPBiashigh = np.full(ngrid, np.nan)
    absPBiasother = np.full(ngrid, np.nan)
    PBias = np.full(ngrid, np.nan)
    PBiasother = np.full(ngrid, np.nan)
    KGE = np.full(ngrid, np.nan)
    KGE12 = np.full(ngrid, np.nan)
    RMSElow = np.full(ngrid, np.nan)
    RMSEhigh = np.full(ngrid, np.nan)
    RMSEother = np.full(ngrid, np.nan)
    for k in range(0, ngrid):
        x = pred[k, :]
        y = target[k, :]
        ind = np.where(np.logical_and(~np.isnan(x), ~np.isnan(y)))[0]
        if ind.shape[0] > 0:
            xx = x[ind]
            yy = y[ind]
            # percent bias
            PBias[k] = np.sum(xx - yy) / np.sum(yy) * 100

            # FHV the peak flows bias 2%
            # FLV the low flows bias bottom 30%, log space
            pred_sort = np.sort(xx)
            target_sort = np.sort(yy)
            indexlow = round(0.3 * len(pred_sort))
            indexhigh = round(0.98 * len(pred_sort))
            lowpred = pred_sort[:indexlow]
            highpred = pred_sort[indexhigh:]
            otherpred = pred_sort[indexlow:indexhigh]
            lowtarget = target_sort[:indexlow]
            hightarget = target_sort[indexhigh:]
            othertarget = target_sort[indexlow:indexhigh]

            # PBiaslow[k] = np.sum(lowpred - lowtarget) / np.sum(lowtarget) * 100
            # PBiashigh[k] = np.sum(highpred - hightarget) / np.sum(hightarget) * 100
            # PBiasother[k] = np.sum(otherpred - othertarget) / np.sum(othertarget) * 100

            PBiaslow[k] = np.sum((lowpred - lowtarget)) / (np.sum(lowtarget) + 0.0001) * 100
            PBiashigh[k] = np.sum((highpred - hightarget)) / np.sum(hightarget) * 100
            PBiasother[k] = np.sum((otherpred - othertarget)) / np.sum(othertarget) * 100
            absPBiaslow[k] = np.sum(abs(lowpred - lowtarget)) / (np.sum(lowtarget) + 0.0001) * 100
            absPBiashigh[k] = np.sum(abs(highpred - hightarget)) / np.sum(hightarget) * 100
            absPBiasother[k] = np.sum(abs(otherpred - othertarget)) / np.sum(othertarget) * 100

            RMSElow[k] = np.sqrt(np.nanmean((lowpred - lowtarget)**2))
            RMSEhigh[k] = np.sqrt(np.nanmean((highpred - hightarget)**2))
            RMSEother[k] = np.sqrt(np.nanmean((otherpred - othertarget)**2))

            if ind.shape[0] > 1:
                # Theoretically at least two points for correlation
                Corr[k] = scipy.stats.pearsonr(xx, yy)[0]
                CorrSp[k] = scipy.stats.spearmanr(xx, yy)[0]
                yymean = yy.mean()
                yystd = np.std(yy)
                xxmean = xx.mean()
                xxstd = np.std(xx)
                KGE[k] = 1 - np.sqrt((Corr[k]-1)**2 + (xxstd/yystd-1)**2 + (xxmean/yymean-1)**2)
                KGE12[k] = 1 - np.sqrt((Corr[k] - 1) ** 2 + ((xxstd*yymean)/ (yystd*xxmean) - 1) ** 2 + (xxmean / yymean - 1) ** 2)
                SST = np.sum((yy-yymean)**2)
                SSReg = np.sum((xx-yymean)**2)
                SSRes = np.sum((yy-xx)**2)
                R2[k] = 1-SSRes/SST
                NSE[k] = 1-SSRes/SST

    outDict = dict(Bias=Bias, RMSE=RMSE, ubRMSE=ubRMSE, Corr=Corr, CorrSp=CorrSp, R2=R2, NSE=NSE,
                   FLV=PBiaslow, FHV=PBiashigh, PBias=PBias, PBiasother=PBiasother, AFLV=absPBiaslow,
                   AFHV=absPBiashigh, AFMV=absPBiasother, KGE=KGE, KGE12=KGE12,
                   lowRMSE=RMSElow, highRMSE=RMSEhigh, midRMSE=RMSEother)
    return outDict

