import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from rnnSMAP import runTestLSTM
import shapefile
import time
import imp
import math
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# train on one HUC and test on CONUS. look at map of sigma

doOpt = []
# doOpt.append('test')
# doOpt.append('loadData')
# doOpt.append('plotMapMC')
doOpt.append('plotMapPaper')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma', 'huc_single_map')
strSigmaLst = ['sigmaX', 'sigmaMC']
strErrLst = ['Bias', 'ubRMSE']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
yrLst = [2017]

hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS'
shapeLst = shapefile.Reader(hucShapeFile).shapes()
shapeHucLst = shapefile.Reader(hucShapeFile).records()

matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 12})


#################################################
# test
if 'test' in doOpt:
    # for k in range(0, 18):
    for k in [2, 10, 14, 15]:
        k = k-1
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
        testName = 'CONUSv2f1'
        out = trainName+'_y15_Forcing_dr60'
        runTestLSTM.runCmdLine(
            rootDB=rootDB, rootOut=rootOut, out=out, testName=testName,
            yrLst=yrLst, cudaID=k % 3, screenName=out)
        # if k % 3 == 2:
        # time.sleep(1000)

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    # for k in range(0, 18):
    for k in [2, 10, 14, 15]:
        k = k-1
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
        testName = 'CONUSv2f1'
        out = trainName+'_y15_Forcing_dr60'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yrLst)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)

#################################################
if 'plotMapMC' in doOpt:
    a = 0
    for k in [2, 10, 14, 15]:
        k = k-1
        statSigma = statSigmaLst[a]
        statErr = 'sigmaMC'
        fig, ax = plt.subplots(figsize=[6, 3])
        data = statSigma.sigmaMC
        grid = ds.data2grid(data=data)
        titleStr = r'$\sigma_{mc}$' + ' from HUC%02d model' % (k+1)
        rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, ax=ax, title=titleStr,
            shape=shapeLst[k])
        plt.tight_layout()
        fig.show()
        saveFile = os.path.join(saveFolder, 'map_sigmaMC_%02d' % (k+1))
        fig.savefig(saveFile)
        a = a+1

#################################################
if 'plotMapPaper' in doOpt:
    a = 0
    figNum = ['(a)', '(b)', '(c)', '(d)']
    fig, axes = plt.subplots(2, 2, figsize=[12, 7])
    for k in [2, 10, 14, 15]:
        k = k-1
        statSigma = statSigmaLst[a]
        statErr = 'sigmaMC'
        data = statSigma.sigmaMC
        grid = ds.data2grid(data=data)
        titleStr = figNum[a]+' '+r'$\sigma_{mc}$' + ' from HUC%02d model' % (k+1)
        ax = axes[math.floor(a/2), a % 2]
        rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, ax=ax, title=titleStr,
            shape=shapeLst[k])
        a = a+1
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'map_sigmaMC')
    fig.savefig(saveFile)
    fig.savefig(saveFile+'.eps')
