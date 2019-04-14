import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import matplotlib

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
doOpt.append('test')
doOpt.append('plotMap')
doOpt.append('plotVS')

trainName = 'CONUSv2f1'
testName = 'CONUSv2f1'
strSigmaLst = ['sigma']
strErrLst = ['ubRMSE', 'RMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'paper')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

#################################################
if 'test' in doOpt:
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']

    out = trainName+'_y15_Forcing'
    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=[2016,2017])
    ds.readData(var='SMAP_AM', field='SMAP')
    ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
    statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
    statSigma = ds.statCalSigma(field='LSTM')

#################################################
if 'plotMap' in doOpt:
    cRangeErr = [0, 0.08]
    for s in strErrLst:
        grid = ds.data2grid(data=getattr(statErr, s))
        saveFile = os.path.join(saveFolder, 'map_'+testName+'_'+s)
        titleStr = s
        fig = rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, cRange=cRangeErr, title=titleStr, showFig=False)
        fig.savefig(saveFile)
    for s in strSigmaLst:
        grid = ds.data2grid(data=getattr(statSigma, s))
        saveFile = os.path.join(saveFolder, 'map_'+testName+'_'+s)
        titleStr = s
        if s == 'sigmaMC':
            cRangeSigma = [0, 0.04]
        else:
            cRangeSigma = [0, 0.08]
        fig = rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, cRange=cRangeSigma, title=titleStr, showFig=False)
        fig.savefig(saveFile, dpi=1000)

#################################################
if 'plotVS' in doOpt:
    for iS in range(0, len(strSigmaLst)):
        for iE in range(0, len(strErrLst)):
            strE = strErrLst[iE]
            strS = strSigmaLst[iS]
            y = getattr(statErr, strE)
            x = getattr(statSigma, strS)
            # ub = np.percentile(y, 95)
            # lb = np.percentile(y, 5)
            # ind = np.logical_and(y >= lb, y <= ub)
            # x = x[ind]
            # y = y[ind]
            # ax = axes[iE]
            fig, ax = plt.subplots(figsize=(8, 6))
            rnnSMAP.funPost.plotVS(x, y, ax=ax, doRank=False)
            # rnnSMAP.funPost.plot121Line(ax)
            ax.set_ylabel(strE)
            ax.set_xlabel(strS)
            fig.suptitle(strS+' vs '+strE)
            saveFile = os.path.join(saveFolder, strS+'2'+strE+'_'+testName)
            fig.savefig(saveFile, dpi=1000)
            plt.close(fig)
