
import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

# Train
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu', model='cudnn',
    loss='sigma'
)
cudaIdLst = [0, 0, 0, 1, 2]
trainLst = ['CONUSv16f1', 'CONUSv8f1', 'CONUSv4f1', 'CONUSv2f1', 'CONUS']
for k in range(0, len(trainLst)):
    trainName = trainLst[k]
    opt['train'] = trainName
    opt['out'] = trainName+'_y15_soilM'
    # runTrainLSTM.runCmdLine(opt=opt, cudaID=cudaIdLst[k], screenName=trainName)

#################################################
# Test
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
trainNameLst = ['CONUSv2f1', 'CONUSv4f1', 'CONUSv8f1', 'CONUSv16f1']
dsLst = list()
statErrLst = list()
statSigmaLst = list()
for k in range(0, len(trainNameLst)):
    out = trainNameLst[k]+'_y15_soilM'
    testName = trainNameLst[k]
    ds = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
    ds.readData(var='SMAP_AM', field='SMAP')
    ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
    statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
    statSigma = ds.statCalSigma(field='LSTM')
    dsLst.append(ds)
    statErrLst.append(statErr)
    statSigmaLst.append(statSigma)

labelS = ['sigmaX', 'sigmaMC']
# Plot box
data = list()
for k in range(0, len(trainNameLst)):
    data.append([statSigmaLst[k].sigmaX, statSigmaLst[k].sigmaMC])
rnnSMAP.funPost.plotBox(data, labelC=trainNameLst,
                        labelS=labelS, title='Temporal Test CONUS')

#################################################
# Plot - regression
fig, axes = plt.subplots(2, ncols=len(trainNameLst))
for iFig in [0, 1]:
    for k in range(0, len(trainNameLst)):
        y = statErrLst[k].RMSE
        if iFig == 0:
            x = statSigmaLst[k].sigmaX
            ax = axes[iFig, k]
        elif iFig == 1:
            x = statSigmaLst[k].sigmaMC
            ax = axes[iFig, k]
        corr = np.corrcoef(x, y)[0, 1]
        pLr = np.polyfit(x, y, 1)
        xLr = np.array([np.min(x), np.max(x)])
        yLr = np.poly1d(pLr)(xLr)
        ax.plot(x, y, 'b*')
        ax.plot(xLr, yLr, 'k-')
        ax.set_title('corr='+'{:.2f}'.format(corr))
        if k == 0:
            ax.set_ylabel(labelS[iFig])
        if iFig == 1:
            ax.set_xlabel(trainNameLst[k])
fig.suptitle('Temporal Test CONUS - RMSE')
fig.show()
