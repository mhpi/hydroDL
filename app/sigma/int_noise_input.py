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
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')

trainName = 'CONUSv4f1'
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
varLst = ['varLst_Forcing',
          #   'varLst_Forcing_noSPFH',
          'varLst_APCP_rn1e1',
          'varLst_APCP_rn2e1',
          'varLst_APCP_rn5e1',
          'varLst_APCP_rn1e0',
          'varLst_APCP_rn2e0',
          'varLst_Forcing_noAPCP']
outLst = [trainName+'_y15_Forcing',
          #   trainName+'_y15_Forcing_noSPFH',
          trainName+'_y15_APCP_rn1e1',
          trainName+'_y15_APCP_rn2e1',
          trainName+'_y15_APCP_rn5e1',
          trainName+'_y15_APCP_rn1e0',
          trainName+'_y15_APCP_rn2e0',
          trainName+'_y15_Forcing_noAPCP', ]
labelLst = ['old', '1e1', '2e1', '5e1', '1e0', '2e0', 'noAPCP']

strSigmaLst = ['sigmaX', 'sigmaMC']
# strSigmaLst = ['sigmaMC']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'noise_red')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu', model='cudnn',
        loss='mse', train='CONUSv4f1'
    )
    cudaIdLst = np.tile([0, 1, 2], 10)
    for k in range(0, len(outLst)):
        opt['target'] = 'SMAP_AM'
        opt['var'] = varLst[k]
        opt['out'] = outLst[k]
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statErrLst2 = list()
    statSigmaLst2 = list()
    for k in range(0, len(outLst)):
        testName = 'CONUSv4f1'
        targetName = 'SMAP_AM'
        out = outLst[k]

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
    saveFileTp = ('boxSigma_rnAPCP', 'boxErr_rnAPCP')

    for iP in range(0, len(dataTp)):
        statLst = dataTp[iP]
        attrLst = attrTp[iP]
        data = list()
        for k in range(0, len(outLst)):
            stat = statLst[k]
            tempLst = list()
            for strS in attrLst:
                tempLst.append(getattr(stat, strS))
            data.append(tempLst)
        fig = rnnSMAP.funPost.plotBox(
            data, labelC=labelLst, labelS=attrLst,
            title='Temporal Test ' + titleTp[iP])

        # for strS in attrLst:
        #     tempLst = list()
        #     for k in range(0, len(outLst)):
        #         stat = statLst[k]
        #         tempLst.append(getattr(stat, strS))
        #     data.append(tempLst)
        # fig = rnnSMAP.funPost.plotBox(
        #     data, labelC=attrLst, labelS=labelLst,
        #     title='Temporal Test ' + titleTp[iP])

        saveFile = os.path.join(saveFolder, saveFileTp[iP])
        fig.savefig(saveFile)
        fig.show()
