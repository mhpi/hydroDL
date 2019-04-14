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
doOpt.append('plotBox')

noiseOpt = 'SMAP'

noiseNameLst = [None, '5e2', '1e1', '2e1', '3e1', '4e1', '5e1']
noiseNameLstPlot = ['0', '0.05', '0.1', '0.2', '0.3', '0.4', '0.5']
strSigmaLst = ['sigmaX', 'sigmaMC']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'int_noise_red')
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
        opt['target'] = 'SMAP_AM_rn'+noiseNameLst[k]
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
        
        # targetName = 'SMAP_AM'
        if noiseNameLst[k] is not None:
            targetName = 'SMAP_AM_rn'+noiseNameLst[k]
            out = 'CONUSv4f1_y15_Forcing_rn'+noiseNameLst[k]
        else:
            targetName = 'SMAP_AM'
            out = 'CONUSv4f1_y15_Forcing'

        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016,2017])
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
    saveFileTp = ('boxSigma', 'boxErr')

    for iP in range(0, len(dataTp)):
        dataLst = dataTp[iP]
        attrLst = attrTp[iP]

        for strS in attrLst:
            plotLst = list()
            statRef = getattr(dataLst[0], strS)
            for data in dataLst:
                stat = getattr(data, strS)
                plotLst.append(stat/statRef)
            fig = rnnSMAP.funPost.plotBox(
                plotLst, labelC=noiseNameLstPlot, labelS=None,
                title='Temporal Test ' + strS)
            saveFile = os.path.join(saveFolder, 'box_'+strS)
            fig.savefig(saveFile, dpi=300)
