import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intend to test if hidden size affact sigma performance

doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'int_multiInput')

trainName = 'CONUSv4f1'
outLst = ['CONUSv4f1_y15_Forcing_noAPCP',
          'CONUSv4f1_y15_Forcing',
          'CONUSv4f1_y15_soilM']
caseLst = ['noAPCP', 'Forcing', 'soilM']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']


#################################################
if 'test' in doOpt:
    trainName = 'CONUSv4f1'
    testName = 'CONUSv4f1'
    strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']    
    strErrLst = ['RMSE', 'ubRMSE']

    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for out in outLst:
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100,
                    field='LSTM', testBatch=100)
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)

#################################################
if 'plotMap' in doOpt:
    cRangeErr = [0, 0.1]
    cRangeSigma = [0, 0.04]
    for k in range(0, len(caseLst)):
        ds = dsLst[k]
        statErr = statErrLst[k]
        statSigma = statSigmaLst[k]
        for s in strErrLst:
            grid = ds.data2grid(data=getattr(statErr, s))
            saveFile = os.path.join(saveFolder, 'map'+s+'_'+caseLst[k])
            titleStr = 'temporal '+s+' '+caseLst[k]
            fig = rnnSMAP.funPost.plotMap(
                grid, crd=ds.crdGrid, cRange=cRangeErr, title=titleStr, showFig=False)
            fig.savefig(saveFile)
        for s in strSigmaLst:
            grid = ds.data2grid(data=getattr(statSigma, s))
            saveFile = os.path.join(saveFolder, 'map'+s+'_'+caseLst[k])
            titleStr = 'temporal '+s+' '+trainName
            fig = rnnSMAP.funPost.plotMap(
                grid, crd=ds.crdGrid, cRange=cRangeSigma, title=titleStr, showFig=False)
            fig.savefig(saveFile)


#################################################
if 'plotBox' in doOpt:
    dataSigma = list()
    for k in range(0, len(caseLst)):
        statSigma = statSigmaLst[k]
        tempLst = list()
        for strS in strSigmaLst:
            tempLst.append(getattr(statSigma, strS))
        dataSigma.append(tempLst)
    fig = rnnSMAP.funPost.plotBox(
        dataSigma, labelC=caseLst, labelS=strSigmaLst, title='Temporal Test CONUS')
    saveFile = os.path.join(saveFolder, 'boxSigma')
    fig.savefig(saveFile)

    dataErr = list()
    for k in range(0, len(caseLst)):
        statErr = statErrLst[k]
        tempLst = list()
        for strE in strErrLst:
            tempLst.append(getattr(statErr, strE))
        dataErr.append(tempLst)
    fig = rnnSMAP.funPost.plotBox(
        dataErr, labelC=caseLst, labelS=strErrLst, title='Temporal Test Noise')
    saveFile = os.path.join(saveFolder, 'boxErr')
    fig.savefig(saveFile)

#################################################
if 'plotVS' in doOpt:
    for k in range(0, len(caseLst)):
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
                ax = axes[iE, iS]
                rnnSMAP.funPost.plotVS(x, y, ax=ax, doRank=False)
                if iS == 0:
                    ax.set_ylabel(strE)
                if iE == len(strErrLst)-1:
                    ax.set_xlabel(strS)
        fig.suptitle('Temporal '+caseLst[k])
        saveFile = os.path.join(saveFolder, 'vsPlot_'+caseLst[k])
        fig.savefig(saveFile)
        plt.close(fig)
