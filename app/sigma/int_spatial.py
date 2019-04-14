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
# doOpt.append('plotMap')
# doOpt.append('plotBox')
doOpt.append('plotVS')

trainNameLst = ['CONUSv2f1']
testNameLst = ['CONUSv2f2']
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'interval_spatial')

#################################################
if 'test' in doOpt:
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']

    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for k in range(0, len(trainNameLst)):
        out = trainNameLst[k]+'_y15_Forcing'
        testName = testNameLst[k]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2015])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)

#################################################
if 'plotMap' in doOpt:
    cRangeErr = [0, 0.1]

    for k in range(0, len(trainNameLst)):
        trainName = trainNameLst[k]
        ds = dsLst[k]
        statErr = statErrLst[k]
        statSigma = statSigmaLst[k]
        for s in strErrLst:
            grid = ds.data2grid(data=getattr(statErr, s))
            saveFile = os.path.join(saveFolder, 'map_'+trainName+'_'+s)
            titleStr = 'temporal '+s+' '+trainName
            fig = rnnSMAP.funPost.plotMap(
                grid, crd=ds.crdGrid, cRange=cRangeErr, title=titleStr, showFig=False)
            fig.savefig(saveFile)
        for s in strSigmaLst:
            grid = ds.data2grid(data=getattr(statSigma, s))
            saveFile = os.path.join(saveFolder, 'map_'+trainName+'_'+s)
            titleStr = 'temporal '+s+' '+trainName
            if s == 'sigmaMC':
                cRangeSigma = [0, 0.03]
            else:
                cRangeSigma = [0, 0.06]
            fig = rnnSMAP.funPost.plotMap(
                grid, crd=ds.crdGrid, cRange=cRangeSigma, title=titleStr, showFig=False)
            fig.savefig(saveFile)


#################################################
if 'plotBox' in doOpt:
    data = list()
    for k in range(0, len(trainNameLst)):
        statSigma = statSigmaLst[k]
        tempLst = list()
        for strS in strSigmaLst:
            tempLst.append(getattr(statSigma, strS))
        data.append(tempLst)
    fig = rnnSMAP.funPost.plotBox(
        data, labelC=trainNameLst, labelS=strSigmaLst, title='Temporal Test CONUS')
    saveFile = os.path.join(saveFolder, 'boxPlot_sigma')

#################################################
if 'plotVS' in doOpt:
    for k in range(0, len(trainNameLst)):
        trainName = trainNameLst[k]
        fig, axes = plt.subplots(
            len(strErrLst), len(strSigmaLst), figsize=(8, 6))
        statErr = statErrLst[k]
        statSigma = statSigmaLst[k]
        for iS in range(0, len(strSigmaLst)):
            for iE in range(0, len(strErrLst)):
                strS = strSigmaLst[iS]
                strE = strErrLst[iE]
                y = getattr(statErr, strE)
                x = getattr(statSigma, strS)
                # ub = np.percentile(y, 95)
                # lb = np.percentile(y, 5)
                # ind = np.logical_and(y >= lb, y <= ub)
                # x = x[ind]
                # y = y[ind]
                ax = axes[iE, iS]
                rnnSMAP.funPost.plotVS(x, y, ax=ax, doRank=False)
                # rnnSMAP.funPost.plot121Line(ax)
                if iS == 0:
                    ax.set_ylabel(strE)
                if iE == len(strErrLst)-1:
                    ax.set_xlabel(strS)
        fig.suptitle('Temporal '+trainName)
        saveFile = os.path.join(saveFolder, 'vsPlot_'+trainName)
        fig.savefig(saveFile)
        plt.close(fig)
        y = getattr(statSigma, 'sigmaMC')
        x = getattr(statSigma, 'sigmaX')
        fig = rnnSMAP.funPost.plotVS(x, y)
        saveFile = os.path.join(saveFolder, 'vsPlotSigma_'+trainName)
        fig.savefig(saveFile)
