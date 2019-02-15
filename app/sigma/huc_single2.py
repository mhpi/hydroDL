import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
from rnnSMAP import runTestLSTM
import shapefile
import time
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# train on one HUC and test on CONUS. look at map of sigma

doOpt = []
# doOpt.append('test')
# doOpt.append('loadData')
doOpt.append('plotMapMC')
# doOpt.append('plotMap')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'huc_single2')
strSigmaLst = ['sigmaX', 'sigmaMC']
strErrLst = ['Bias', 'ubRMSE']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
yrLst = [2016, 2017]

hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS'
shapeLst = shapefile.Reader(hucShapeFile).shapes()


#################################################
# test
if 'test' in doOpt:
    for k in range(0, 18):
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
        testName = 'CONUSv2f1'
        out = trainName+'_y15_Forcing'
        runTestLSTM.runCmdLine(
            rootDB=rootDB, rootOut=rootOut, out=out, testName=testName,
            yrLst=yrLst, cudaID=k % 3, screenName=out)
        if k % 3 == 2:
            time.sleep(1000)

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for k in range(0, 1):
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
        testName = 'CONUSv2f1'
        out = trainName+'_y15_Forcing'
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
    for k in range(0, 1):
        statSigma = statSigmaLst[k]
        statErr = 'sigmaMC'
        fig, ax = plt.subplots(figsize=[12, 4])
        data = statSigma.sigmaMC
        grid = ds.data2grid(data=data)
        titleStr = r'$\sigma_{mc}$' + 'for model trained on HUC%02d' % (k+1)
        rnnSMAP.funPost.plotMap(
            grid, crd=ds.crdGrid, ax=ax, title=titleStr,
            shape=shapeLst[k])
        fig.show()
        saveFile = os.path.join(saveFolder, 'map_sigma_%02d' % (k+1))
        fig.savefig(saveFile)


#################################################
if 'plotMap' in doOpt:
    strSigmaLst = ['sigmaX', 'sigmaMC']
    strErrLst = ['Bias', 'ubRMSE']
    # plot map RMSE
    # for k in range(0, 18):
    for k in range(0, 1):
        statSigma = statSigmaLst[k]
        statErr = statErrLst[k]
        fig, axes = plt.subplots(ncols=len(strSigmaLst), figsize=[12, 4])
        for kk in range(0, len(strSigmaLst)):
            strSigma = strSigmaLst[kk]
            data = getattr(statSigma, strSigma)
            grid = ds.data2grid(data=data)
            titleStr = strSigma + 'of model trained on HUC%02d' % (k+1)
            rnnSMAP.funPost.plotMap(
                grid, crd=ds.crdGrid, ax=axes[kk], title=titleStr,
                shape=shapeLst[k])
        # fig.show()
        saveFile = os.path.join(saveFolder, 'map_sigma_%02d' % (k+1))
        fig.savefig(saveFile)

        fig, axes = plt.subplots(ncols=len(strErrLst), figsize=[12, 4])
        for kk in range(0, len(strErrLst)):
            strErr = strErrLst[kk]
            data = getattr(statErr, strErr)
            grid = ds.data2grid(data=data)
            titleStr = strErr + 'of model trained on HUC%02d' % (k+1)
            rnnSMAP.funPost.plotMap(
                grid, crd=ds.crdGrid, ax=axes[kk], title=titleStr)
        # fig.show()
        saveFile = os.path.join(saveFolder, 'map_error_%02d' % (k+1))
        fig.savefig(saveFile)
