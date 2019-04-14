import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import scipy
import pylab
import scipy.stats as stats

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
# doOpt.append('train')
doOpt.append('test')
doOpt.append('plotConf')
doOpt.append('plotNorm')
doOpt.append('plotScale')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
# doOpt.append('plotVS')

trainName = 'CONUSv2f1'
testName = 'CONUSv2f1'
out = trainName+'_y15_Forcing'
yrLst = [[2015], [2016], [2017]]
caseStrLst = ['y15', 'y16', 'y17']
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'interval_temporal')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rnnSMAP.kPath['DB_L3_NA'],
        rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
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
    statConfLst = list()
    statNormLst = list()
    for k in range(0, len(caseStrLst)):
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yrLst[k])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
        statNorm = rnnSMAP.classPost.statNorm(
            statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
        statNormLst.append(statNorm)


#################################################
# plot confidence
if 'plotConf' in doOpt:
    fig, axes = plt.subplots(ncols=len(caseStrLst), figsize=(12, 6))
    for k in range(0, len(caseStrLst)):
        statConf = statConfLst[k]
        plotLst = [statConf.conf_sigmaX,
                   statConf.conf_sigmaMC,
                   statConf.conf_sigma]
        legendLst = ['simgaX', 'sigmaMC', 'sigmaComb']
        rnnSMAP.funPost.plotCDF(plotLst, ax=axes[k], legendLst=legendLst)
        axes[k].set_title(caseStrLst[k])
    fig.show()


#################################################
# plot norm distribution
if 'plotNorm' in doOpt:
    fig, axes = plt.subplots(ncols=len(caseStrLst), figsize=(12, 6))
    for k in range(0, len(caseStrLst)):
        statNorm = statNormLst[k]
        plotLst = [statNorm.yNorm_sigmaX,
                   statNorm.yNorm_sigmaMC,
                   statNorm.yNorm_sigma]
        legendLst = ['simgaX', 'sigmaMC', 'sigmaComb']
        rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[k], legendLst=legendLst, ref='norm')
        axes[k].set_xlim([-5, 5])
        fig.show()

#################################################
# plot scale result
if 'plotScale' in doOpt:
    fig, axes = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))

    for k in range(0, len(caseStrLst)):
        statConf = statConfLst[k]
        plotLstConf = [statConf.conf_sigmaX,
                       statConf.conf_sigmaMC,
                       statConf.conf_sigma]
        statNorm = statNormLst[k]
        plotLstNorm = [statNorm.yNorm_sigmaX,
                       statNorm.yNorm_sigmaMC,
                       statNorm.yNorm_sigma]
        legendLst = ['simgaX', 'sigmaMC', 'sigmaComb']
        s = statSigmaLst[k].sigmaMC_mat
        y = dsLst[k].SMAP
        u = dsLst[k].LSTM
        cmap = plt.cm.jet
        cLst = cmap(np.linspace(0, 1, 7))
        if caseStrLst[k] is 'y15':
            sF15 = rnnSMAP.funPost.scaleSigma(s, u, y)
            s1 = s*sF15
            conf1, yNorm1 = rnnSMAP.funPost.reCalSigma(s1, u, y)
            s2 = np.sqrt(np.square(s1)+np.square(statSigmaLst[k].sigma_mat))
            conf2, yNorm2 = rnnSMAP.funPost.reCalSigma(s2, u, y)
            plotLstConf.extend([conf1, conf2])
            plotLstNorm.extend([yNorm1, yNorm2])
            legendLst.extend(['sigmaMC_scaleY15', 'sigmaComb_scaleY15'])
        if caseStrLst[k] is 'y16':
            sF16 = rnnSMAP.funPost.scaleSigma(s, u, y)
            s1 = s*sF15
            conf1, yNorm1 = rnnSMAP.funPost.reCalSigma(s1, u, y)
            s2 = s*sF16
            conf2, yNorm2 = rnnSMAP.funPost.reCalSigma(s2, u, y)
            s3 = np.sqrt(np.square(s1)+np.square(statSigmaLst[k].sigma_mat))
            conf3, yNorm3 = rnnSMAP.funPost.reCalSigma(s3, u, y)
            s4 = np.sqrt(np.square(s2)+np.square(statSigmaLst[k].sigma_mat))
            conf4, yNorm4 = rnnSMAP.funPost.reCalSigma(s4, u, y)
            plotLstConf.extend([conf1, conf2, conf3, conf4])
            plotLstNorm.extend([yNorm1, yNorm2, yNorm3, yNorm4])
            legendLst.extend(['sigmaMC_scaleY15', 'sigmaComb_scaleY15',
                              'sigmaMC_scaleY16', 'sigmaComb_scaleY16'])
        if caseStrLst[k] is 'y17':
            sF17 = rnnSMAP.funPost.scaleSigma(s, u, y)
            s1 = s*sF15
            conf1, yNorm1 = rnnSMAP.funPost.reCalSigma(s1, u, y)
            s2 = s*sF16
            conf2, yNorm2 = rnnSMAP.funPost.reCalSigma(s2, u, y)
            s3 = np.sqrt(np.square(s1)+np.square(statSigmaLst[k].sigma_mat))
            conf3, yNorm3 = rnnSMAP.funPost.reCalSigma(s3, u, y)
            s4 = np.sqrt(np.square(s2)+np.square(statSigmaLst[k].sigma_mat))
            conf4, yNorm4 = rnnSMAP.funPost.reCalSigma(s4, u, y)
            plotLstConf.extend([conf1, conf2, conf3, conf4])
            plotLstNorm.extend([yNorm1, yNorm2, yNorm3, yNorm4])
            legendLst.extend(['sigmaMC_scaleY15', 'sigmaComb_scaleY15',
                              'sigmaMC_scaleY16', 'sigmaComb_scaleY16'])
        rnnSMAP.funPost.plotCDF(
            plotLstConf, ax=axes[0, k], legendLst=legendLst, cLst=cLst)
        rnnSMAP.funPost.plotCDF(
            plotLstNorm, ax=axes[1, k], legendLst=legendLst, ref='norm', cLst=cLst)
        axes[1, k].set_xlim([-5, 5])
    fig.show()

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
    rnnSMAP.funPost.plotBox(
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
