import rnnSMAP
# from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

hucLst = ['04051118', '03101317', '02101114',
          '01020304', '02030406', '14151617']

#################################################
# Training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu',
    model='cudnn', loss='sigma'
)
cudaIdLst = [0, 1, 2, 0, 1, 2]
for k in range(0, len(hucLst)):
    trainName = hucLst[k]+'_v2f1'
    opt['train'] = trainName
    opt['out'] = trainName+'_y15_soilM'
    print(trainName)
    # runTrainLSTM.runCmdLine(opt=opt, cudaID=cudaIdLst[k],
    # screenName=trainName)

#################################################
# Test
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
dsTuple = ([], [])
labelS = ['HUC-HUC', 'CONUS-HUC']
for k in range(0, len(hucLst)):
    trainName = hucLst[k]+'_v2f1'
    out = trainName+'_y15_soilM'
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

    outLst = [trainName+'_y15_soilM',
              'CONUSv2f1_y15_soilM']
    testNameLst = [hucLst[k]+'_v2f1',
                   hucLst[k]+'_v2f1']

    for kk in range(0, len(dsTuple)):
        out = outLst[kk]
        testName = testNameLst[kk]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsTuple[kk].append(ds)


#################################################
# Statistic
statErrTuple = ([], [])
statSigmaTuple = ([], [])
for k in range(0, len(hucLst)):
    for kk in range(0, len(dsTuple)):
        ds = dsTuple[kk][k]
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statErrTuple[kk].append(statErr)
        statSigmaTuple[kk].append(statSigma)

#################################################
# Plot - boxplot
dataSigmaX = list()
dataSigmaMC = list()
dataErr = list()
for k in range(0, len(hucLst)):
    tempSigmaX = []
    tempSigmaMC = []
    tempErr = []
    for kk in range(0, len(dsTuple)):
        tempSigmaX.append(statSigmaTuple[kk][k].sigmaX)
        tempSigmaMC.append(statSigmaTuple[kk][k].sigmaMC)
        tempErr.append(statErrTuple[kk][k].ubRMSE)
    dataSigmaX.append(tempSigmaX)
    dataSigmaMC.append(tempSigmaMC)
    dataErr.append(tempErr)

rnnSMAP.funPost.plotBox(dataSigmaX, labelC=hucLst,
                        labelS=labelS,
                        title='SigmaX Temporal Extrapolation')
rnnSMAP.funPost.plotBox(dataSigmaMC, labelC=hucLst,
                        labelS=labelS,
                        title='SigmaMC Temporal Extrapolation')
rnnSMAP.funPost.plotBox(dataErr, labelC=hucLst,
                        labelS=labelS,
                        title='ubRMSE Temporal Extrapolation')

#################################################
# Plot - regression
fig1, axes1 = plt.subplots(nrows=len(dsTuple), ncols=len(hucLst))
fig2, axes2 = plt.subplots(nrows=len(dsTuple), ncols=len(hucLst))
    for k in range(0, len(hucLst)):
        for kk in range(0, len(dsTuple)):
            y = statErrTuple[kk][k].ubRMSE
            if iFig == 1:
                x = statSigmaTuple[kk][k].sigmaX
                ax = axes1[kk, k]
            elif iFig == 2:
                x = statSigmaTuple[kk][k].sigmaMC
                ax = axes2[kk, k]
            corr = np.corrcoef(x, y)[0, 1]
            pLr = np.polyfit(x, y, 1)
            xLr = np.array([np.min(x), np.max(x)])
            yLr = np.poly1d(pLr)(xLr)
            ax.plot(x, y, 'b*')
            ax.plot(xLr, yLr, 'k-')
            ax.set_title('corr='+'{:.2f}'.format(corr))
            if k == 0:
                ax.set_ylabel(labelS[kk])
            if kk == len(dsTuple)-1:
                ax.set_xlabel(hucLst[k])
fig1.suptitle('sigmaX vs ubRMSE')
fig1.show()
fig2.suptitle('sigmaMC vs ubRMSE')
fig2.show()
