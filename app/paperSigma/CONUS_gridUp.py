import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# if sigmaMC goes down as datapoint increase
doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotBox')
# doOpt.append('plotBoxErr')
doOpt.append('plotConf')

trainNameLst = ['CONUSv2f1', 'CONUSv4f1', 'CONUSv8f1', 'CONUSv16f1']
testName = 'CONUSv16f1'

strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['ubRMSE', 'RMSE']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

# drLst = np.arange(0.1, 1, 0.1)
# drStrLst = ["%02d" % (x*100) for x in drLst]

# for kk in range(0, len(drStrLst)):
#     drStr = drStrLst[kk]
drStr = '60'
#################################################
if 'test' in doOpt:
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']

    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(trainNameLst)):
        # out = trainNameLst[k]+'_y15_Forcing'+'_dr'+drStr
        out = trainNameLst[k]+'_y15_Forcing'
        testName = testName
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100,
                    field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)


#################################################
if 'plotBox' in doOpt:
    data = list()
    for k in range(0, len(trainNameLst)):
        statSigma = statSigmaLst[k]
        tempLst = list()
        for strS in strSigmaLst:
            tempLst.append(getattr(statSigma, strS))
        data.append(tempLst)
    fig = rnnSMAP.funPost.plotBox(
        data, labelC=trainNameLst, labelS=strSigmaLst)
    fig.show()
    saveFile = os.path.join(saveFolder, 'gridUp_sigma')

#################################################
if 'plotBoxErr' in doOpt:
    data = list()
    for k in range(0, len(trainNameLst)):
        statErr = statErrLst[k]
        tempLst = list()
        for strE in strErrLst:
            tempLst.append(getattr(statErr, strE))
        data.append(tempLst)
    fig = rnnSMAP.funPost.plotBox(
        data, labelC=trainNameLst, labelS=strErrLst)
    fig.show()
    saveFile = os.path.join(saveFolder, 'gridUp_err')


#################################################
# plot confidence figure
if 'plotConf' in doOpt:
    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))

    sigmaStrLst = ['sigmaMC', 'sigmaX', 'sigma']

    for iS in range(0, len(sigmaStrLst)):
        tempLst = list()
        for k in range(0, len(trainNameLst)):
            tempLst.append(getattr(statConfLst[k], 'conf_'+sigmaStrLst[iS]))
        rnnSMAP.funPost.plotCDF(
            tempLst, ax=axes[iS], legendLst=trainNameLst, showDiff=False)
        axes[iS].set_title(sigmaStrLst[iS])
    fig.show()
    saveFile=os.path.join(saveFolder, 'gridUp_conf.png')
    fig.savefig(saveFile, dpi=100)
