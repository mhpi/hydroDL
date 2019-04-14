import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
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
doOpt.append('plotBox')
# doOpt.append('plotVS')

trainName = 'CONUSv2f1'
testName = trainName
epochLst = [100, 200, 300, 400, 500]
strEpochLst = ['ep100', 'ep200', 'ep300', 'ep400', 'ep500']
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'int_epoch')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rnnSMAP.kPath['DB_L3_NA'],
        rootOut=rnnSMAP.kPath['Out_L3_NA'],
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu', model='cudnn',
        loss='mse'
    )
    cudaIdLst = [2]
    trainLst = ['CONUSv2f1']
    for k in range(0, len(trainLst)):
        trainName = trainLst[k]
        opt['train'] = trainName
        opt['out'] = trainName+'_y15_Forcing'
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

#################################################
if 'test' in doOpt:
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']

    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    out = trainName+'_y15_Forcing'
    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
    ds.readData(var='SMAP_AM', field='SMAP')
    for k in range(0, len(epochLst)):
        fieldName = 'LSTM'+strEpochLst[k]
        ds.readPred(rootOut=rootOut, out=out, drMC=100,
                    epoch=epochLst[k], field=fieldName)
        statErr = ds.statCalError(predField=fieldName, targetField='SMAP')
        statSigma = ds.statCalSigma(field=fieldName)

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)


#################################################
if 'plotBox' in doOpt:
    data = list()
    for k in range(0, len(epochLst)):
        statSigma = statSigmaLst[k]
        tempLst = list()
        for strS in strSigmaLst:
            tempLst.append(getattr(statSigma, strS))
        data.append(tempLst)
    fig = rnnSMAP.funPost.plotBox(
        data, labelC=strEpochLst, labelS=strSigmaLst, title='Temporal Test CONUS')
    saveFile = os.path.join(saveFolder, 'boxPlot_sigma')

#################################################
if 'plotVS' in doOpt:
    for k in range(0, len(epochLst)):
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
                ub = np.percentile(y, 95)
                lb = np.percentile(y, 5)
                ind = np.logical_and(y >= lb, y <= ub)
                x = x[ind]
                y = y[ind]
                ax = axes[iE, iS]
                rnnSMAP.funPost.plotVS(x, y, ax=ax, doRank=False)
                # rnnSMAP.funPost.plot121Line(ax)
                if iS == 0:
                    ax.set_ylabel(strE)
                if iE == len(strErrLst)-1:
                    ax.set_xlabel(strS)
        fig.suptitle('Temporal '+trainName+strEpochLst[k])
        saveFile = os.path.join(saveFolder, 'vsPlot_'+trainName+strEpochLst[k])
        fig.savefig(saveFile)
        plt.close(fig)
        y = getattr(statSigma, 'sigmaMC')
        x = getattr(statSigma, 'sigmaX')
        fig = rnnSMAP.funPost.plotVS(x, y)
        saveFile = os.path.join(
            saveFolder, 'vsPlotSigma_'+trainName+strEpochLst[k])
        fig.savefig(saveFile)
