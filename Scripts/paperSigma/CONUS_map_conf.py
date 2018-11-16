import os
import rnnSMAP
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

figTitleLst = ['Training', 'Temporal Test', 'Spatial Test']
figNameLst = ['train', 'temporal', 'spatial']

for iFig in range(0, 3):
    # iFig = 0
    figTitle = figTitleLst[iFig]
    if iFig == 0:
        testName = 'CONUSv2f1'
        yr = [2015]
    if iFig == 1:
        testName = 'CONUSv2f1'
        yr = [2016, 2017]
    if iFig == 2:
        testName = 'CONUSv2f2'
        yr = [2015]

    trainName = 'CONUSv2f1'
    out = trainName+'_y15_Forcing'
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    caseStrLst = ['sigmaMC', 'sigmaX', 'sigma']
    nCase = len(caseStrLst)
    saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

    #################################################
    # test
    predField = 'LSTM'
    targetField = 'SMAP'

    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=yr)
    ds.readData(var='SMAP_AM', field='SMAP')
    ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
    statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
    statSigma = ds.statCalSigma(field='LSTM')
    statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
    statNorm = rnnSMAP.classPost.statNorm(
        statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)

    #################################################
    # setup figure
    fig = plt.figure(figsize=[10, 6])
    gs = gridspec.GridSpec(
        2, 2, width_ratios=[1.2, 1], height_ratios=[1, 1])

    dataErr = getattr(statErr, 'ubRMSE')
    dataSigma = getattr(statSigma, 'sigma')
    # dataSigma = np.nanmean(statConf.conf_sigma, axis=1)
    cRange = [0, 0.1]

    # plot map RMSE
    ax = fig.add_subplot(gs[0, 0])
    grid = ds.data2grid(data=dataErr)
    titleStr = 'ubRMSE of '+figTitle
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                            cRange=cRange, title=titleStr)

    # plot map sigma
    ax = fig.add_subplot(gs[1, 0])
    grid = ds.data2grid(data=dataSigma)
    titleStr = 'Sigma of '+figTitle
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                            cRange=cRange, title=titleStr)

    # plot map sigma vs RMSE
    ax = fig.add_subplot(gs[0:, 1])
    ax.set_aspect('equal', 'box')
    y = dataErr
    x = dataSigma
    rnnSMAP.funPost.plotVS(x, y, ax=ax)

    fig.show()
    saveFile = os.path.join(saveFolder, 'map_'+figNameLst[iFig])
    fig.savefig(saveFile, dpi=1200)
