import os.path
import hydroDL
from . import option
from collections import OrderedDict
import numpy as np


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return mDict


def readMasterFile(out):
    mFile = os.path.join(out, 'master.json')
    mDict = option.loadOpt(mFile)
    return mDict


def writeMasterFile(mDict):
    out = mDict['out']
    if not os.path.isdir(out):
        os.mkdir(out)
    mFile = os.path.join(out, 'master.json')
    option.saveOpt(mDict, mFile)


def loadModel(out, epoch=None):
    if epoch is None:
        mDict = readMasterFile(out)
        epoch = mDict['train']['nEpoch']
    model = hydroDL.model.train.loadModel(out, epoch)
    return model


def namePred(out, tRange, subset, doMC=False, suffix=None):
    fileName = subset + '_' + str(tRange[0]) + '_' + str(tRange[1]) + '.npy'
    if suffix is not None:
        fileName = fileName + '_' + suffix
    filePath = os.path.join(out, fileName)
    return filePath


def train(mDict, overwrite=False):
    if mDict is str:
        mDict = readMasterFile(mDict)
    out = mDict['out']
    optData = mDict['data']
    optModel = mDict['model']
    optLoss = mDict['loss']
    optTrain = mDict['train']
    # data
    if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData['path'],
            subset=optData['subset'],
            tRange=optData['tRange'])
        x = df.getData(
            varT=optData['varT'],
            varC=optData['varC'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0])
        y = df.getData(
            varT=optData['target'],
            doNorm=optData['doNorm'][1],
            rmNan=optData['rmNan'][1])
        nx = x.shape[-1]
    # loss
    if eval(optLoss['name']) is hydroDL.model.crit.SigmaLoss:
        lossFun = hydroDL.model.crit.SigmaLoss(prior=optLoss['prior'])
        if optModel['ny'] != 2:
            print('updated ny by sigma loss')
            optModel['ny'] = 2
    elif eval(optLoss['name']) is hydroDL.model.crit.RmseLoss:
        lossFun = hydroDL.model.crit.RmseLoss()
        if optModel['ny'] != 1:
            print('updated ny by rmse loss')
            optModel['ny'] = 1
    # model
    if eval(optModel['name']) is hydroDL.model.rnn.CudnnLstmModel:
        if optModel['nx'] != nx:
            print('updated nx by input data')
            optModel['nx'] = nx
        model = hydroDL.model.rnn.CudnnLstmModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'])

    mFile = os.path.join(out, 'master.json')
    if os.path.isfile(mFile) and overwrite is False:
        Exception('trained model exist')
    else:
        writeMasterFile(mDict)
        model = hydroDL.model.train.trainModel(
            model,
            x,
            y,
            lossFun,
            nEpoch=optTrain['nEpoch'],
            miniBatch=optTrain['miniBatch'],
            saveEpoch=optTrain['saveEpoch'],
            saveFolder=out)


def test(out,
         *,
         tRange,
         subset,
         doMC=False,
         suffix=None,
         batchSize=None,
         epoch=None,
         reTest=False):
    mDict = readMasterFile(out)

    optData = mDict['data']
    if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData['path'], subset=subset, tRange=tRange)
        x = df.getData(
            varT=optData['varT'],
            varC=optData['varC'],
            doNorm=optData['doNorm'][0],
            rmNan=optData['rmNan'][0])

    fileName = namePred(out, tRange, subset, doMC, suffix)
    if os.path.isfile(fileName) and reTest is False:
        pred = np.load(fileName)
    else:
        model = loadModel(out, epoch=epoch)
        pred = hydroDL.model.train.testModel(model, x, batchSize=batchSize)
        if optData['doNorm'][1] is True:
            if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
                pred = hydroDL.data.dbCsv.transNorm(
                    pred,
                    rootDB=optData['path'],
                    fieldName=optData['target'],
                    fromRaw=False)
                # np.load(fileName, pred)
    return pred
