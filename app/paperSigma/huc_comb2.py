import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotBox')
doOpt.append('plotConf')
doOpt.append('plotBin')


hucLst = ['01020405', '12131518', '01021518', '04051213']
testName = '03060708091011141617_v2f1'

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'huc_spatial')

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(hucLst)):
        trainName = hucLst[k]+'_v2f1'
        out = trainName+'_y15_Forcing'

        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2015])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statErrLst.append(statErr)
        statSigma = ds.statCalSigma(field='LSTM')
        statSigmaLst.append(statSigma)
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statConfLst.append(statConf)

#################################################
if 'plotConf' in doOpt:
    strConfLst = ['conf_sigmaMC', 'conf_sigmaX', 'conf_sigma']
    titleLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
    fig, axes = plt.subplots(ncols=len(strConfLst), figsize=(12, 4))

    for k in range(0, len(strConfLst)):
        plotLst = list()
        for iHuc in range(0, len(hucLst)):
            temp = getattr(statConfLst[iHuc], strConfLst[k])
            plotLst.append(temp)
        rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[k], cLst='myrgcb',
            legendLst=hucLst)
        axes[k].set_title(titleLst[k])
    saveFile = os.path.join(saveFolder, 'hucComb_conf.png')
    fig.show()
    fig.savefig(saveFile, dpi=600)

#################################################
if 'plotBox' in doOpt:
    strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
    strErrLst = ['RMSE', 'ubRMSE']
    dataTp = (statSigmaLst, statErrLst)
    attrTp = (strSigmaLst, strErrLst)
    titleTp = ('Sigma', 'Error')
    saveFileTp = ('boxSigma', 'boxErr')

    for iP in range(0, len(dataTp)):
        statLst = dataTp[iP]
        attrLst = attrTp[iP]
        data = list()
        for k in range(0, len(hucLst)):
            stat = statLst[k]
            tempLst = list()
            for strS in attrLst:
                tempLst.append(getattr(stat, strS))
            data.append(tempLst)
        labelS = attrLst
        titleStr = 'Temporal Test ' + titleTp[iP]
        fig = rnnSMAP.funPost.plotBox(
            data, labelC=hucLst, labelS=labelS, figsize=(8, 6),
            title=titleStr)
        saveFile = os.path.join(saveFolder, +'hucComb_'+saveFileTp[iP])
        fig.savefig(saveFile, dpi=100)


#################################################
# plot confidence figure
if 'plotBin' in doOpt:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for iHuc in range(0, len(hucLst)):
        sigmaMC = getattr(statSigmaLst[iHuc], 'sigmaMC')
        # sigmaMC = getattr(statErrLst[iHuc], 'ubRMSE')
        sigma = getattr(statSigmaLst[iHuc], 'sigma')
        ubRMSE = getattr(statErrLst[iHuc], 'Bias')
        confMat = getattr(statConfLst[iHuc], 'conf_sigma')
        nbin = 20
        xbin = np.percentile(sigmaMC, range(0, 101, int(100/nbin)))
        xbinMean = (xbin[0:nbin]+xbin[1:nbin+1])/2
        corrLst = list()
        distLst = list()
        for k in range(0, nbin):
            ind = (sigmaMC > xbin[k]) & (sigmaMC <= xbin[k+1])
            conf = rnnSMAP.funPost.flatData(confMat[ind, :])
            yRank = np.arange(len(conf))/float(len(conf)-1)
            dist = np.sqrt(((conf - yRank) ** 2).mean())
            corr = scipy.stats.pearsonr(sigma[ind], ubRMSE[ind])[0]
            corrLst.append(corr)
            distLst.append(dist)
        ax.plot(xbinMean, distLst, marker='*',
                color=cLst[iHuc], label=hucTitleLst[iHuc])
    # ax.set_ylabel('correlation')
    ax.set_ylabel('Distance')
    ax.set_xlabel('sigmaMC')
    ax.legend()
    fig.show()
    saveFile = os.path.join(saveFolder, 'CONUS_sigmaMCbin')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
