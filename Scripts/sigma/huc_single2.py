import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import huc_single_test
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intend to test huc vs huc

doOpt = []
doOpt.append('loadData')
# doOpt.append('crdMap')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

testHuc = 17
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'huc_single2')
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']

#################################################
# load data and plot map
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for k in range(0, 18):
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
        testName = 'hucn1_'+str(testHuc).zfill(2)+'_v2f1'
        out = trainName+'_y15_Forcing'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016,2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)

errLst = np.ndarray([18])
for k in range(0, 18):
    errLst[k] = np.median(statErrLst[k].RMSE)
indRank = np.argsort(errLst)
hucStrRank = ["%i" % (x+1) for x in indRank]

#################################################
if 'plotBox' in doOpt:
    dataTp = (statSigmaLst, statErrLst)
    attrTp = (strSigmaLst, strErrLst)
    titleTp = ('Sigma', 'Error')
    saveFileTp = ('boxSigma_soilM', 'boxErr_soilM')

    for iP in range(0, len(dataTp)):
        statLst = dataTp[iP]
        attrLst = attrTp[iP]
        data = list()
        for k in range(0, 18):
            ind = indRank[k]
            stat = statLst[ind]
            tempLst = list()
            for strS in attrLst:
                tempLst.append(getattr(stat, strS))
            data.append(tempLst)
        labelS = attrLst
        fig = rnnSMAP.funPost.plotBox(
            data, labelC=hucStrRank, labelS=labelS, figsize=(12, 8),
            title='Temporal Test on huc'+str(testHuc)+' '+titleTp[iP])
        saveFile = os.path.join(saveFolder, str(testHuc)+'_'+saveFileTp[iP])
        fig.savefig(saveFile)
