import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

trainName = 'CONUSv2f1'
out = trainName+'_y15_Forcing'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

doOpt = []
# doOpt.append('loadData')
doOpt.append('plotConf')
# doOpt.append('plotProb')

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statNormLst = list()

    for k in range(0, 3):
        # k = 0
        if k == 0:
            testName = 'CONUSv2f1'
            yr = [2015]
        if k == 1:
            testName = 'CONUSv2f1'
            yr = [2016, 2017]
        if k == 2:
            testName = 'CONUSv2f2'
            yr = [2015]

        predField = 'LSTM'
        targetField = 'SMAP'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statNorm = rnnSMAP.classPost.statNorm(
            statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)
        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statNormLst.append(statNorm)

#################################################
# plot confidence figure
if 'plotConf' in doOpt:
    figTitleLst = ['Training', 'Temporal Test', 'Spatial Test']
    fig, axes = plt.subplots(
        ncols=len(figTitleLst), figsize=(12, 4), sharey=True)
    sigmaStrLst = ['sigmaX', 'sigma']
    for iFig in range(0, 3):
        statNorm = statNormLst[iFig]
        figTitle = figTitleLst[iFig]
        plotLst = list()
        for k in range(0, len(sigmaStrLst)):
            plotLst.append(getattr(statNorm, 'yNorm_'+sigmaStrLst[k]))
        legendLst = [r'$norm_{x}$', r'$norm_{comb}$']
        _, _, out = rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[iFig], legendLst=legendLst, cLst='grbm', ref='norm',
            xlabel='Predicting Probablity', ylabel=None, showDiff=False)
        axes[iFig].set_title(figTitle)
        axes[iFig].set_xlim([-3, 3])
        print(out['rmseLst'])
        if iFig == 0:
            axes[iFig].set_ylabel('True Probablity')
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'CONUS_dist')
    fig.savefig(saveFile, dpi=300)
    fig.savefig(saveFile+'.eps')
