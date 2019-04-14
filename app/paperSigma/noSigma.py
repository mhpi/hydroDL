import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats
import matplotlib

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


#################################################
# intervals temporal see if sigma output affact sigmaMC
doOpt = []
# doOpt.append('train')
doOpt.append('test')
doOpt.append('plotMap')
# doOpt.append('plotMapAll')

rootOutLst = [rnnSMAP.kPath['Out_L3_NA'], rnnSMAP.kPath['OutSigma_L3_NA']]
testName = 'CONUSv2f1'
yr = [2017]
nCase = len(rootOutLst)

saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 6})

#################################################
if 'test' in doOpt:
    rootDB = rnnSMAP.kPath['DB_L3_NA']

    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(rootOutLst)):
        rootOut = rootOutLst[k]
        out = 'CONUSv2f1_y15_Forcing_dr60'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)

if 'plotMap' in doOpt:
    fig, ax = plt.subplots(figsize=[12, 6])
    diff = getattr(statErrLst[0], 'Bias')-getattr(statErrLst[1], 'Bias')
    diff[diff <= 0] = np.nan
    cRange = [-0.00, 0.005]
    grid = ds.data2grid(data=diff)
    titleStr = r'ubRMSE(w/o $\sigma_x$) - ubRMSE(w/ $\sigma_x$)'
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=ax,
                            cRange=cRange, title=titleStr)
    fig.show()
    saveFile = os.path.join(saveFolder, 'map_noSigma')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')

if 'plotMapAll' in doOpt:
    fig, axes = plt.subplots(1, 3, figsize=[12, 4])

    diff = getattr(statErrLst[0], 'ubRMSE')
    cRange = [0, 0.05]
    grid = ds.data2grid(data=diff)
    titleStr = r'ubRMSE(w/o $\sigma_x$)'
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=axes[0],
                            cRange=cRange, title=titleStr)

    diff = getattr(statErrLst[1], 'ubRMSE')
    cRange = [0, 0.05]
    grid = ds.data2grid(data=diff)
    titleStr = r'ubRMSE(w/ $\sigma_x$)'
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=axes[1],
                            cRange=cRange, title=titleStr)

    diff = getattr(statErrLst[0], 'ubRMSE')-getattr(statErrLst[1], 'ubRMSE')
    cRange = [-0.005, 0.005]
    grid = ds.data2grid(data=diff)
    titleStr = r'ubRMSE(w/o $\sigma_x$) - ubRMSE(w/ $\sigma_x$)'
    rnnSMAP.funPost.plotMap(grid, crd=ds.crdGrid, ax=axes[2],
                            cRange=cRange, title=titleStr)
    fig.show()
    saveFile = os.path.join(saveFolder, 'map_noSigma')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
