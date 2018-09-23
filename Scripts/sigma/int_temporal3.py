import os
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
doOpt.append('plotVS')

trainName = 'CONUSv2f1'
testName = trainName

strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'int_temporal3')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rnnSMAP.kPath['DB_L3_NA'],
        rootOut=rnnSMAP.kPath['Out_L3_NA'],
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu', model='cudnn',
        loss='mse'
    )
    cudaIdLst = [2]
    trainLst = ['CONUSv2f1']
    for k in range(0, len(trainLst)):
        trainName = trainLst[k]
        opt['train'] = trainName
        opt['out'] = trainName+'_y15_Forcing'
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

#################################################
if 'test' in doOpt:
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    out = trainName+'_y15_Forcing'
    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
    ds.readData(var='SMAP_AM', field='SMAP')
    ds.readPred(field='LSTM1', out=out, drMC=100,
                rootOut=rnnSMAP.kPath['Out_L3_NA'],)
    ds.readPred(out=out, drMC=100, field='LSTM2',
                rootOut=rnnSMAP.kPath['OutSigma_L3_NA'])
    statErr1 = ds.statCalError(predField='LSTM1', targetField='SMAP')
    statErr2 = ds.statCalError(predField='LSTM2', targetField='SMAP')
    statSigma1 = ds.statCalSigma(field='LSTM1')
    statSigma2 = ds.statCalSigma(field='LSTM2')

#################################################
if 'plotVS' in doOpt:
    sigmaMC = statSigma1.sigmaMC
    sigmaX = statSigma2.sigmaX
    sigma = np.sqrt(sigmaX**2+sigmaMC**2)
    sigmaLst = [sigmaX, sigmaMC, sigma]
    x = statSigma2.sigmaMC_mat
    y = ds.LSTM2-ds.SMAP
    fig = rnnSMAP.funPost.plotVS(x.flatten(), y.flatten(), doRank=False)
    rnnSMAP.funPost.plot121Line(fig.axes[0])
    fig.show()


#################################################
if 'plotBox' in doOpt:
    data = list()

    tempLst = list()
    tempLst.append(statSigma1.sigmaMC)
    tempLst.append(statSigma2.sigmaMC)
    data.append(tempLst)

    tempLst = list()
    tempLst.append(statErr1.ubRMSE)
    tempLst.append(statErr2.ubRMSE)
    data.append(tempLst)

    fig = rnnSMAP.funPost.plotBox(
        data, labelC=['sigmaMC', 'ubRMSE'], labelS=['wo sigma', 'w sigma'], title='Temporal Test CONUS')
