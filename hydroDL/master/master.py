import os.path
import hydroDL
from . import option
from collections import OrderedDict


def wrapMaster(loc, optData, optModel, optLoss, optTrain):
    masterDict = OrderedDict(
        loc=loc, data=optData, model=optModel, loss=optLoss, train=optTrain)
    return masterDict


def readMasterFile(loc):
    masterFile = os.path.join(loc, 'master.json')
    masterDict = option.loadOpt(masterFile)
    return masterDict


def writeMasterFile(masterDict):
    loc = masterDict['loc']
    if not os.path.isdir(loc):
        os.mkdir(loc)
    masterFile = os.path.join(loc, 'master.json')
    option.saveOpt(masterDict, masterFile)


def runMaster(masterDict, overwrite=False):
    if masterDict is str:
        masterDict = readMasterFile(masterDict)
    loc = masterDict['loc']
    optData = masterDict['data']
    optModel = masterDict['model']
    optLoss = masterDict['loss']
    optTrain = masterDict['train']
    if eval(optData['name']) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData['path'],
            subsetName=optData['subset'],
            tRange=optData['dateRange'])
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
    if eval(optModel['name']) is hydroDL.model.rnn.CudnnLstmModel:
        if optModel['nx'] != nx:
            print('updated nx by input data')
            optModel['nx'] = nx
        model = hydroDL.model.rnn.CudnnLstmModel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            hiddenSize=optModel['hiddenSize'])
    if eval(optLoss['name']) is hydroDL.model.crit.SigmaLoss:
        lossFun = hydroDL.model.crit.SigmaLoss(prior=optLoss['prior'])

    masterFile = os.path.join(loc, 'master.json')
    if os.path.isfile(masterFile) and overwrite is False:
        Exception('trained model exist')
    else:
        writeMasterFile(masterDict)
        model = hydroDL.model.train.trainModel(
            model,
            x,
            y,
            lossFun,
            nEpoch=optTrain['nEpoch'],
            miniBatch=optTrain['miniBatch'],
            saveEpoch=optTrain['saveEpoch'],
            saveFolder=loc)
