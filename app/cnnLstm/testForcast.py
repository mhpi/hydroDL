import hydroDL
import os
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
from hydroDL import post
from hydroDL import utils
import numpy as np

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 300
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'cnnCond')
ty1 = [20150501, 20160501]
tyc1 = [20150401, 20160501]
ty2 = [20160501, 20170501]
tyc2 = [20160401, 20170501]

doLst = list()
doLst.append('trainCnn')
doLst.append('trainLstm')
doLst.append('testCnn')
doLst.append('testLstm')
doLst.append('post')

if 'trainLstm' in doLst:
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1]
    ny = 1
    model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
    lossFun = crit.RmseLoss()
    model = train.trainModel(
        model, x, y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
    modelName = 'lstmForcast'
    train.saveModel(outFolder, model, nEpoch, modelName=modelName)

if 'trainCnn' in doLst:
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=tyc1)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    yc = np.copy(y)
    yc[:, :, 0] = utils.interpNan(yc[:, :, 0], mode='pre')
    nx = x.shape[-1]
    ny = 1
    for opt in range(1, 3):
        model = rnn.LstmCnnForcast(
            nx=nx, ny=ny, ct=30, hiddenSize=64, cnnSize=16, opt=opt)
        lossFun = crit.RmseLoss()
        model = train.trainModel(
            model, (x, yc), y, lossFun, nEpoch=nEpoch, miniBatch=[100, 60])
        modelName = 'cnnForcast' + str(opt)
        train.saveModel(outFolder, model, nEpoch, modelName=modelName)

ypLst = list()
df = hydroDL.data.dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
yT = df.getData(varT='SMAP_AM', doNorm=False, rmNan=False).squeeze()

if 'testLstm' in doLst:
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    model = train.loadModel(outFolder, nEpoch, modelName='lstmForcast')
    yP = train.testModel(model, x).squeeze()
    ypLst.append(
        dbCsv.transNorm(yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

if 'testCnn' in doLst:
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=tyc2)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    yc = np.copy(y)
    yc[:, :, 0] = utils.interpNan(yc[:, :, 0], mode='pre')

    for opt in range(1, 3):
        modelName = 'cnnForcast' + str(opt)
        model = train.loadModel(outFolder, nEpoch, modelName=modelName)
        yP = train.testModel(model, x, z=yc, batchSize=100).squeeze()
        ypLst.append(
            dbCsv.transNorm(
                yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

if 'post' in doLst:
    statDictLst = list()
    for k in range(0, len(ypLst)):
        statDictLst.append(post.statError(ypLst[k], yT))

    statStrLst = ['RMSE', 'ubRMSE', 'Bias', 'Corr']
    caseLst = ['LSTM', 'CNNforcast-opt1', 'CNNforcast-opt2']
    # caseLst = ['LSTM', 'CNN-opt2', 'CNN-opt3']
    postMat = np.ndarray([len(ypLst), len(statStrLst)])
    for iS in range(len(statStrLst)):
        statStr = statStrLst[iS]
        for k in range(len(ypLst)):
            err = np.nanmean(statDictLst[k][statStr])
            print('{} of {} = {:.5f}'.format(statStr, caseLst[k], err))
            postMat[k, iS] = err
    np.set_printoptions(precision=4, suppress=True)
    print(postMat)
