import hydroDL
import os
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
from hydroDL.post import plot, stat
from hydroDL import utils
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 100
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'closeLoop')
ty1 = [20150406, 20160406]
ty2 = [20160406, 20170406]

doLst = list()
doLst.append('train')
doLst.append('test')
doLst.append('post')

if 'train' in doLst:
    # load data
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1]
    ny = 1

    # train
    for k in range(5):
        sd = utils.time.t2dt(ty1[0]) - dt.timedelta(days=1 + k)
        ed = utils.time.t2dt(ty1[1]) - dt.timedelta(days=1 + k)
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
        model = rnn.LstmCloseModel(
            nx=nx + 1, ny=ny, hiddenSize=64, fillObs=True)
        lossFun = crit.RmseLoss()
        model = train.trainModel(
            model, (x, obs), y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'LSTM-DA-' + str(k + 1)
        train.saveModel(outFolder, model, nEpoch, modelName=modelName)

if 'test' in doLst:
    # load data
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1]
    ny = 1
    yT = df.getData(varT='SMAP_AM', doNorm=False, rmNan=False)
    yT = yT[:, :, 0]

    # test
    ypLst = list()
    modelName = 'LSTM'
    model = train.loadModel(outFolder, 100, modelName=modelName)
    yP = train.testModel(model, x, batchSize=100).squeeze()
    ypLst.append(
        dbCsv.transNorm(yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
    for k in range(5):
        sd = utils.time.t2dt(ty2[0]) - dt.timedelta(days=1 + k)
        ed = utils.time.t2dt(ty2[1]) - dt.timedelta(days=1 + k)
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=rootDB, subset='CONUSv4f1', tRange=[sd, ed])
        obs = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
        modelName = 'LSTM-DA-' + str(k + 1)
        model = train.loadModel(outFolder, nEpoch, modelName=modelName)
        yP = train.testModel(model, (x, obs), batchSize=100).squeeze()
        ypLst.append(
            dbCsv.transNorm(
                yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

if 'post' in doLst:
    statDictLst = list()
    for k in range(0, len(ypLst)):
        statDictLst.append(stat.statError(ypLst[k], yT))
    keyLst = ['RMSE', 'ubRMSE', 'Bias', 'Corr']
    caseLst = ['LSTM', 'DA1', 'DA2', 'DA3', 'DA4', 'DA5']

    # # plot box
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            temp.append(statDictLst[k][statStr])
        dataBox.append(temp)
    fig = plot.plotBoxFig(dataBox, keyLst, caseLst, sharey=False)
    fig.show()

    # plot time series
    t = utils.time.t2dtLst(ty2[0], ty2[1])
    fig, axes = plt.subplots(5, 1, figsize=(12, 8))
    for k in range(5):
        iGrid = np.random.randint(0, 412)
        yPlot = [ypLst[i][iGrid, :] for i in range(len(ypLst))]
        yPlot.append(yT[iGrid, :])
        if k == 0:
            plot.plotTS(
                t,
                yPlot,
                ax=axes[k],
                cLst='bgrk',
                legLst=['LSTM', 'Close', 'CloseDA', 'SMAP'])
        else:
            plot.plotTS(t, yPlot, ax=axes[k], cLst='bgrk')
    fig.show()

    # plot map
    fig, axes = plt.subplots(2, 2, figsize=(8, 5))
    crd = df.getGeo()
    dataLst = list()
    for k in range(len(statDictLst)):
        data = statDictLst[k]['RMSE']
        dataLst.append(data)
    dataLst.append(statDictLst[0]['RMSE'] - statDictLst[2]['RMSE'])
    for k in range(len(dataLst)):
        data = dataLst[k]
        grid, uy, ux = utils.grid.array2grid(data, crd)
        cRange = [0, 0.1] if k != 3 else [-0.02, 0.02]
        plot.plotMap(
            grid,
            crd=[uy, ux],
            ax=axes[k % 2][int(k / 2)],
            cRange=cRange,
            title=caseLst[k])
    fig.show()

# interactive map

# # print error box
# postMat = np.ndarray([len(ypLst), len(keyLst)])
# for iS in range(len(keyLst)):
#     statStr = keyLst[iS]
#     for k in range(len(ypLst)):
#         err = np.nanmean(statDictLst[k][statStr])
#         print('{} of {} = {:.5f}'.format(statStr, caseLst[k], err))
#         postMat[k, iS] = err
# np.set_printoptions(precision=4, suppress=True)
# print(postMat)
