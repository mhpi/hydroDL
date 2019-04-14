import hydroDL
import os
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
from hydroDL import post
from hydroDL import utils
import numpy as np

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 5
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'cnnCond')
ty1 = [20150402, 20160401]
ty2 = [20160401, 20170401]
ty12 = [20150402, 20170401]
ty3 = [20170401, 20180401]

doLst = list()
# doLst.append('trainCnn')
# doLst.append('trainLstm')
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
    modelName = 'lstm_y1'
    train.saveModel(outFolder, model, nEpoch, modelName=modelName)

if 'trainCnn' in doLst:
    dfc = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    xc = dfc.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    yc = dfc.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    yc[:, :, 0] = utils.interpNan(yc[:, :, 0])
    c = np.concatenate((yc, xc), axis=2)
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    nx = x.shape[-1]
    ny = 1
    for opt in range(1, 4):
        model = rnn.LstmCnnCond(
            nx=nx, ny=ny, ct=365, hiddenSize=64, cnnSize=32, opt=opt)
        lossFun = crit.RmseLoss()
        model = train.trainModel(model, (x, c), y, lossFun, nEpoch=nEpoch, miniBatch=[100, 30])
        modelName = 'cnn' + str(opt) + '_y1y1'
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
    model = train.loadModel(outFolder, nEpoch, modelName='lstm_y1')
    yP = train.testModel(model, x).squeeze()
    ypLst.append(
        dbCsv.transNorm(yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))
if 'testCnn' in doLst:
    dfc = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
    xc = dfc.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    yc = dfc.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
    yc[:, :, 0] = utils.interpNan(yc[:, :, 0])
    z = np.concatenate((yc, xc), axis=2)
    df = hydroDL.data.dbCsv.DataframeCsv(
        rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
    x = df.getData(
        varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
    for opt in range(1, 4):
        modelName = 'cnn' + str(opt) + '_y1y1'
        model = train.loadModel(outFolder, nEpoch, modelName=modelName)
        yP = train.testModel(model, x, z=z).squeeze()
        ypLst.append(
            dbCsv.transNorm(
                yP, rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False))

if 'post' in doLst:
    statDictLst = list()
    for k in range(0, len(ypLst)):
        statDictLst.append(post.statError(ypLst[k], yT))

    statStrLst = ['RMSE', 'ubRMSE', 'Bias', 'Corr']
    caseLst = ['LSTM', 'CNN-opt1', 'CNN-opt2', 'CNN-opt3']
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
