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
doOpt.append('train')
# doOpt.append('test')
# doOpt.append('plotBox')
# doOpt.append('plotVS')
# doOpt.append('plotVS2')


hucLst = ['04051118', '03101317', '02101114',
          '01020304', '02030406', '14151617']
# hucLst = ['12131518', '01021518', '04051213']
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'huc_spatial')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        dr=0.6, modelOpt='relu',
        model='cudnn', loss='sigma'
    )
    cudaIdLst = [0, 1, 2, 0, 1, 2]
    for k in range(0, len(hucLst)):
        trainName = hucLst[k]+'_v2f1'
        opt['train'] = trainName
        opt['out'] = trainName+'_y15_Forcing_dr60'
        runTrainLSTM.runCmdLine(opt=opt, cudaID=k % 3, screenName=opt['out'])


#################################################
if 'test' in doOpt:
    dsTuple = ([], [])
    labelS = ['HUC-exHUC', 'CONUS-exHUC']
    for k in range(0, len(hucLst)):
        trainName = hucLst[k]+'_v2f1'
        out = trainName+'_y15_soilM'
        outLst = [trainName+'_y15_soilM',
                  'CONUSv2f1_y15_soilM']
        testNameLst = ['ex_'+hucLst[k]+'_v2f1',
                       'ex_'+hucLst[k]+'_v2f1']

        for kk in range(0, len(dsTuple)):
            out = outLst[kk]
            testName = testNameLst[kk]
            ds = rnnSMAP.classDB.DatasetPost(
                rootDB=rootDB, subsetName=testName, yrLst=[2015])
            ds.readData(var='SMAP_AM', field='SMAP')
            ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
            dsTuple[kk].append(ds)

    statErrTuple = ([], [])
    statSigmaTuple = ([], [])
    statErrTuple2 = ([], [])
    statSigmaTuple2 = ([], [])
    for k in range(0, len(hucLst)):
        for kk in range(0, len(dsTuple)):
            ds = dsTuple[kk][k]
            statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
            statSigma = ds.statCalSigma(field='LSTM')
            statErrTuple[kk].append(statErr)
            statSigmaTuple[kk].append(statSigma)

#################################################
if 'plotBox' in doOpt:
    dataSigmaX = list()
    dataSigmaMC = list()
    dataErr = list()
    dataErr2 = list()
    for k in range(0, len(hucLst)):
        tempSigmaX = []
        tempSigmaMC = []
        tempErr = []
        tempErr2 = []
        for kk in range(0, len(dsTuple)):
            tempSigmaX.append(statSigmaTuple[kk][k].sigmaX)
            tempSigmaMC.append(statSigmaTuple[kk][k].sigmaMC)
            tempErr.append(statErrTuple[kk][k].ubRMSE)
            tempErr2.append(statErrTuple[kk][k].RMSE)
        dataSigmaX.append(tempSigmaX)
        dataSigmaMC.append(tempSigmaMC)
        dataErr.append(tempErr)
        dataErr2.append(tempErr2)
    plotTup = (dataSigmaX, dataSigmaMC, dataErr, dataErr2)
    plotStr = ('sigmaX', 'sigmaMC', 'ubRMSE', 'RMSE')
    for k in range(0, len(plotTup)):
        fig = rnnSMAP.funPost.plotBox(
            plotTup[k], labelC=hucLst, labelS=labelS,
            title=plotStr[k]+' Spatial Extrapolation')
        saveFile = os.path.join(saveFolder, 'box_'+plotStr[k])
        fig.show()
        fig.savefig(saveFile)

#################################################
# Plot - regression
if 'plotVS' in doOpt:
    fig1, axes1 = plt.subplots(
        nrows=len(dsTuple), ncols=len(hucLst), figsize=[10, 8])
    fig2, axes2 = plt.subplots(
        nrows=len(dsTuple), ncols=len(hucLst), figsize=[10, 8])
    fig3, axes3 = plt.subplots(
        nrows=len(dsTuple), ncols=len(hucLst), figsize=[10, 8])
    strErr = 'ubRMSE'
    for iFig in [1, 2, 3]:
        for k in range(0, len(hucLst)):
            for kk in range(0, len(dsTuple)):
                y = getattr(statErrTuple[kk][k], strErr)
                if iFig == 1:
                    x = statSigmaTuple[kk][k].sigmaX
                    ax = axes1[kk, k]
                if iFig == 2:
                    x = statSigmaTuple[kk][k].sigmaMC
                    ax = axes2[kk, k]
                if iFig == 3:
                    x = statSigmaTuple[kk][k].sigma
                    ax = axes3[kk, k]
                corr = np.corrcoef(x, y)[0, 1]
                pLr = np.polyfit(x, y, 1)
                xLr = np.array([np.min(x), np.max(x)])
                yLr = np.poly1d(pLr)(xLr)
                ax.plot(x, y, 'b*')
                ax.plot(xLr, yLr, 'k-')
                ax.set_title('corr='+'{:.2f}'.format(corr))
                if k == 0:
                    ax.set_ylabel(labelS[kk])
                if kk == len(dsTuple)-1:
                    ax.set_xlabel(hucLst[k])
    fig1.suptitle('sigmaX vs '+strErr)
    saveFile = os.path.join(saveFolder, 'sigmaX2'+strErr)
    fig1.savefig(saveFile)
    fig2.suptitle('sigmaMC vs '+strErr)
    saveFile = os.path.join(saveFolder, 'sigmaMC2'+strErr)
    fig2.savefig(saveFile)
    fig3.suptitle('sigma vs '+strErr)
    saveFile = os.path.join(saveFolder, 'sigma2'+strErr)
    fig3.savefig(saveFile)


#################################################
if 'plotVS2' in doOpt:
    fig, axes = plt.subplots(nrows=len(dsTuple), ncols=len(hucLst))
    for k in range(0, len(hucLst)):
        for kk in range(0, len(dsTuple)):
            y = statErrTuple[kk][k].RMSE
            x = [statSigmaTuple[kk][k].sigmaX, statSigmaTuple[kk][k].sigmaMC]
            yLr = rnnSMAP.funPost.regLinear(y, x)
            pLr = np.polyfit(x, y, 1)
            xLr = np.array([np.min(x), np.max(x)])
            yLr = np.poly1d(pLr)(xLr)
            ax.plot(x, y, 'b*')
            ax.plot(xLr, yLr, 'k-')
            ax.set_title('corr='+'{:.2f}'.format(corr))
            if k == 0:
                ax.set_ylabel(labelS[kk])
            if kk == len(dsTuple)-1:
                ax.set_xlabel(hucLst[k])
    fig1.suptitle('sigmaX vs RMSE')
    fig1.show()
    fig2.suptitle('sigmaMC vs RMSE')
    fig2.show()
