import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import numpy as np
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()
import matplotlib

#################################################
# noise affact on sigmaX (or sigmaMC)
doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')

noiseOpt = 'SMAP'

noiseNameLst = ['5e2', '1e1', '2e1', '3e1', '4e1', '5e1']
noiseNameLstPlot = ['0.05', '0.1', '0.2', '0.3', '0.4', '0.5']
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'noise_red')
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu', model='cudnn',
        loss='sigma'
    )
    trainName = 'CONUSv4f1'
    opt['train'] = trainName
    cudaIdLst = np.tile([0, 1, 2], 10)

    for k in range(5, len(noiseNameLst)):
        # opt['target'] = 'SMAP_AM_rn'+noiseNameLst[k]        
        opt['var'] = 'varLst_Forcing'
        opt['out'] = opt['train']+'_y15_Forcing_rn'+noiseNameLst[k]
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for k in range(0, len(noiseNameLst)):
        testName = 'CONUSv4f1'
        if noiseOpt == 'SMAP':
            # targetName = 'SMAP_AM_rn'+noiseNameLst[k]
            targetName = 'SMAP_AM'
            out = 'CONUSv4f1_y15_Forcing_rn'+noiseNameLst[k]
        if noiseOpt == 'APCP':
            targetName = 'SMAP_AM'
            out = 'CONUSv4f1_y15_APCP_rn'+noiseNameLst[k]

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
    if noiseOpt == 'SMAP':
        saveFileTp = ('boxSigma_rnSMAP', 'boxErr_rnSMAP')
    elif noiseOpt == 'APCP':
        saveFileTp = ('boxSigma_rnAPCP', 'boxErr_rnAPCP')

    for iP in range(0, len(dataTp)):
        statLst = dataTp[iP]
        attrLst = attrTp[iP]
        data = list()
        for k in range(0, len(noiseNameLst)):
            stat = statLst[k]
            tempLst = list()
            for strS in attrLst:
                # tempLst.append(getattr(stat, strS))            
                if strS == 'sigmaMC':
                    tempLst.append(getattr(stat, strS))
                else:
                    tempLst.append(getattr(stat, strS))
            data.append(tempLst)
        labelS = attrLst
        fig = rnnSMAP.funPost.plotBox(
            data, labelC=noiseNameLstPlot, labelS=labelS,
            title='Temporal Test ' + titleTp[iP])
        saveFile = os.path.join(saveFolder, saveFileTp[iP])
        fig.savefig(saveFile, dpi=1000)
