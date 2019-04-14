import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


hucLst = ['04051118', '03101317', '02101114',
          '01020304', '02030406', '14151617']
hucTitleLst = ['HUC 04+05+11+18', 'HUC 03+10+13+17', 'HUC 02+10+11+14',
               'HUC 02+03+04+06', 'HUC 01+02+03+04', 'HUC 14+15+16+17']

# hucLst = ['04051118', '03101317', '14151617', '02030406']
# hucTitleLst = ['HUC 04+05+11+18', 'HUC 03+10+13+17',
#                'HUC 14+15+16+17', 'HUC 02+03+04+06']
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
cLst = 'myrgcb'


doOpt = []
doOpt.append('test')
doOpt.append('plotBin')
# doOpt.append('plotProb')

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(hucLst)):
        trainName = hucLst[k]+'_v2f1'
        out = trainName+'_y15_Forcing_dr60'
        testName = 'ex_'+hucLst[k]+'_v2f1'

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
# plot confidence figure
if 'plotBin' in doOpt:
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    for iHuc in range(0, len(hucLst)):
        sigmaMC = getattr(statSigmaLst[iHuc], 'sigmaMC_mat')
        sigmaX = getattr(statSigmaLst[iHuc], 'sigmaX_mat')
        dataBin = sigmaMC/sigmaX
        # dataBin = sigmaX
        sigma = getattr(statSigmaLst[iHuc], 'sigma')
        ubRMSE = getattr(statErrLst[iHuc], 'ubRMSE')
        confMat = getattr(statConfLst[iHuc], 'conf_sigma')
        nbin = 10
        xbin = np.percentile(dataBin, range(0, 101, int(100/nbin)))
        xbinMean = (xbin[0:nbin]+xbin[1:nbin+1])/2
        corrLst = list()
        distLst = list()
        for k in range(0, nbin):
            ind = (dataBin > xbin[k]) & (dataBin <= xbin[k+1])
            conf = rnnSMAP.funPost.flatData(confMat[ind])
            if k == 0:
                print(iHuc, len(conf))
            yRank = np.arange(len(conf))/float(len(conf)-1)
            dist = np.abs(conf - yRank).max()
            distLst.append(dist)
        ax.plot(xbinMean, distLst, marker='*',
                color=cLst[iHuc], label=hucLst[iHuc])
    ax.set_ylabel(r'd($p_{mc}$, 1-to-1)')
    ax.set_xlabel(r'$\sigma_{mc}$ / $\sigma_{x}$')
    # ax.set_xlabel(r'$\sigma_{x}$')
    ax.legend()
    fig.show()
    # saveFile = os.path.join(saveFolder, 'CONUS_sigmaRatioBin')
    saveFile = os.path.join(saveFolder, 'CONUS_sigmaMCBin')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
