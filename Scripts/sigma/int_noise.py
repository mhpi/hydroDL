import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import numpy as np
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# noise affact on sigmaX (or sigmaMC)
doOpt = []
doOpt.append('train')
# doOpt.append('test')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
# doOpt.append('plotVS')

# noiseNameLst = ['5e4', '1e3', '2e3', '5e3',
#                 '1e2', '2e2', '5e2', '1e1', '2e1', '5e1']

noiseNameLst = ['1e2', '2e2', '3e2', '4e2', '5e2',
                '6e2', '7e2', '8e2', '9e2', '1e1']
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'int_noise')
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015, varC='varConstLst_Noah',
        dr=0.6, modelOpt='relu', model='cudnn',
        loss='sigma'
    )
    trainName = 'CONUSv4f1'
    opt['train'] = trainName
    cudaIdLst = np.tile([0, 1, 2], 10)

    for k in range(0, len(noiseNameLst)):
        opt['target'] = 'SMAP_AM_sn'+noiseNameLst[k]
        opt['var'] = 'varLst_Forcing'
        opt['out'] = opt['train']+'_y15_Forcing_dr06_sn'+noiseNameLst[k]
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statErrLst2 = list()
    statSigmaLst2 = list()
    for k in range(0, len(noiseNameLst)):
        testName = 'CONUSv4f1'
        # targetName = 'SMAP_AM_sn'+noiseNameLst[k]
        targetName = 'SMAP_AM'
        out = 'CONUSv4f1_y15_soilM_sn'+noiseNameLst[k]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var=targetName, field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)

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
        for k in range(0, len(noiseNameLst)):
            stat = statLst[k]
            tempLst = list()
            for strS in attrLst:
                if strS == 'sigmaMC':
                    tempLst.append(getattr(stat, strS)*2)
                else:
                    tempLst.append(getattr(stat, strS))
            data.append(tempLst)
        labelS = attrLst
        fig = rnnSMAP.funPost.plotBox(
            data, labelC=noiseNameLst, labelS=labelS,
            title='Temporal Test ' + titleTp[iP])
        saveFile = os.path.join(saveFolder, saveFileTp[iP])
        fig.savefig(saveFile)
