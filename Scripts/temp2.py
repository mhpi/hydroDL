import rnnSMAP
from rnnSMAP import runTrainLSTM
import imp
import numpy as np
import scipy
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
imp.reload(rnnSMAP)

opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu', model='cudnn',
    loss='sigma'
)
cudaIdLst = [1, 2]
trainLst = ['CONUSv4f1', 'CONUSv2f1']
for k in range(0, len(trainLst)):
    trainName = trainLst[k]
    opt['train'] = trainName
    opt['out'] = trainName+'_y15_soilM3'
    # runTrainLSTM.runCmdLine(
    #     opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])
    # rnnSMAP.funLSTM.trainLSTM(opt)


trainName = 'CONUSv4f1'
testName = trainName
outName = trainName+'_y15_soilM3'
# outName = trainName+'_y15_soilM_sn1e1'


# outName = 'cudnn_sigma3'

rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
ds = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
ds.readData(var='SMAP_AM', field='SMAP')
ds.readPred(rootOut=rootOut, out=outName, drMC=100,
            field='LSTM', testBatch=100)
statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
statSigma = ds.statCalSigma(field='LSTM')

#
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']
fig, axes = plt.subplots(
    len(strErrLst), len(strSigmaLst), figsize=(8, 6))
for iS in range(0, len(strSigmaLst)):
    for iE in range(0, len(strErrLst)):
        strS = strSigmaLst[iS]
        strE = strErrLst[iE]
        y = getattr(statErr, strE)
        x = getattr(statSigma, strS)
        # ub = np.percentile(y, 95)
        # lb = np.percentile(y, 5)
        # ind = np.logical_and(y >= lb, y <= ub)
        # x = x[ind]
        # y = y[ind]
        ax = axes[iE, iS]
        rnnSMAP.funPost.plotVS(x, y, ax=ax, doRank=False)
        rnnSMAP.funPost.plot121Line(ax)
        if iS == 0:
            ax.set_ylabel(strE)
        if iE == len(strErrLst)-1:
            ax.set_xlabel(strS)
fig.suptitle('Temporal '+trainName)
fig.show()

fig2 = rnnSMAP.funPost.plotVS(statSigma.sigmaX, statSigma.sigmaMC,
                             figsize=(7, 5), xlabel='sigmaX', ylabel='sigmaMC')
fig2.show()
#
# fig = rnnSMAP.funPost.plotBox([statErr.ubRMSE, statErr.RMSE])
