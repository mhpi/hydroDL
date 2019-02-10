# This module include database classes

import numpy as np
from . import funDB
from . import funLSTM
from . import classPost


class Dataset(object):
    r"""Base class Database SMAP

        Arguments:
            rootDB:
            subsetName:
    """

    def __init__(self, rootDB, subsetName, yrLst):
        self.rootDB = rootDB
        self.subsetName = subsetName
        rootName, crd, indSub, indSkip = funDB.readDBinfo(
            rootDB=rootDB, subsetName=subsetName)
        self.crd = crd
        (gridY, gridX, indY, indX) = funDB.crd2grid(crd[:, 0], crd[:, 1])
        self.crdGrid = (gridY, gridX)
        self.crdGridInd = np.stack((indY, indX), axis=1)

        self.indSub = indSub
        self.indSkip = indSkip
        self.rootName = rootName

        self.yrLst = yrLst
        self.time = funDB.readDBtime(
            rootDB=self.rootDB, rootName=self.rootName, yrLst=yrLst)

    def __repr__(self):
        return 'later'


class DatasetLSTM(Dataset):
    r"""Base class Database SMAP
    Arguments:
    """

    def __init__(self, *, rootDB, subsetName, yrLst,
                 var=('varConstLst_Noah', 'varConstLst_Noah'),
                 targetName='SMAP_AM'):
        super().__init__(rootDB, subsetName, yrLst)

        # input variable
        if isinstance(var[0], list):
            self.varInputTs = var[0]
        else:
            self.varInputTs = funDB.readVarLst(
                rootDB=self.rootDB, varLst=var[0])
        if isinstance(var[1], list):
            self.varInputConst = var[1]
        else:
            self.varInputConst = funDB.readVarLst(
                rootDB=self.rootDB, varLst=var[1])
        self.varInput = self.varInputTs + self.varInputConst

        # target variable
        self.varTarget = targetName

    def readTarget(self):
        nt = len(self.time)
        ngrid = len(self.indSub)
        data = funDB.readDataTS(
            rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
            indSkip=self.indSkip, yrLst=self.yrLst, fieldName=self.varTarget,
            nt=nt, ngrid=ngrid)
        stat = funDB.readStat(rootDB=self.rootDB, fieldName=self.varTarget)
        dataNorm = (data-stat[2])/stat[3]
        return dataNorm

    def readInput(self):
        nt = len(self.time)
        ngrid = len(self.indSub)
        nvar = len(self.varInput)
        data = np.ndarray([ngrid, nt, nvar])
        dataNorm = np.ndarray([ngrid, nt, nvar])
        stat = np.ndarray([4, nvar])

        # time series
        k = 0
        for var in self.varInputTs:
            dataTemp = funDB.readDataTS(
                rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
                indSkip=self.indSkip, yrLst=self.yrLst, fieldName=var,
                nt=nt, ngrid=ngrid)
            statTemp = funDB.readStat(rootDB=self.rootDB, fieldName=var)
            dataNormTemp = (dataTemp-statTemp[2])/statTemp[3]
            data[:, :, k] = dataTemp
            dataNorm[:, :, k] = dataNormTemp
            stat[:, k] = statTemp
            k = k+1

        # const
        for var in self.varInputConst:
            dataTemp = funDB.readDataConst(
                rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
                indSkip=self.indSkip, yrLst=self.yrLst, fieldName=var,
                ngrid=ngrid)
            statTemp = funDB.readStat(
                rootDB=self.rootDB, fieldName=var, isConst=True)
            dataNormTemp = (dataTemp-statTemp[2])/statTemp[3]
            data[:, :, k] = np.repeat(np.reshape(
                dataTemp, [ngrid, 1]), nt, axis=1)
            dataNorm[:, :, k] = np.repeat(np.reshape(
                dataNormTemp, [ngrid, 1]), nt, axis=1)
            stat[:, k] = statTemp
            k = k+1

        dataNorm[np.where(np.isnan(dataNorm))] = 0
        return dataNorm


class DatasetPost(Dataset):
    r"""Base class Database SMAP
    Arguments:
    """

    def __init__(self, *, rootDB, subsetName, yrLst):
        super().__init__(rootDB, subsetName, yrLst)

    def readData(self, *, var, field=None):
        nt = len(self.time)
        ngrid = len(self.indSub)
        data = funDB.readDataTS(
            rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
            indSkip=self.indSkip, yrLst=self.yrLst, fieldName=var,
            nt=nt, ngrid=ngrid)
        stat = funDB.readStat(rootDB=self.rootDB, fieldName=var)
        if field is None:
            field = var
        setattr(self, field, data)
        setattr(self, field+'_stat', stat)

    def readPred(self, *, rootOut, out, drMC=0, field='LSTM', testBatch=0,
                 reTest=False, epoch=None):
        bPred = funLSTM.checkPred(out=out, rootOut=rootOut,
                                  test=self.subsetName, drMC=drMC, epoch=epoch,
                                  syr=self.yrLst[0], eyr=self.yrLst[-1])
        if reTest is True:
            bPred = False
        if bPred is False:
            print('running test')
            funLSTM.testLSTM(out=out, rootOut=rootOut, test=self.subsetName,
                             syr=self.yrLst[0], eyr=self.yrLst[-1], drMC=drMC,
                             testBatch=testBatch, epoch=epoch)
        dataPred, dataSigma, dataPredBatch, dataSigmaBatch = funLSTM.readPred(
            out=out, rootOut=rootOut, test=self.subsetName, epoch=epoch,
            syr=self.yrLst[0], eyr=self.yrLst[-1], drMC=drMC, reReadMC=reTest)

        if drMC == 0:
            # setattr(self, field, dataPred)
            setattr(self, field+'_Sigma', dataSigma)
        else:
            # dataPredMean = np.mean(dataPredBatch, axis=2)
            # dataPredMean = dataPredBatch[:, :, 0]
            # setattr(self, field, dataPredMean)
            dataSigmaMean = np.sqrt(np.mean(dataSigmaBatch**2, axis=2))
            setattr(self, field+'_Sigma', dataSigmaMean)
        setattr(self, field, dataPred)
        setattr(self, field+'_MC', dataPredBatch)
        setattr(self, field+'_SigmaMC', dataSigmaBatch)

    def data2grid(self, *, field=None, data=None):
        if field is None and data is None:
            raise Exception('no input to data2grid')
        if field is not None and data is not None:
            raise Exception('repeat input to data2grid')
        if field is not None:
            data = getattr(self, field)
        elif data.shape[0] != self.crd.shape[0]:
            raise Exception('data is of wrong size')

        indY = self.crdGridInd[:, 0]
        indX = self.crdGridInd[:, 1]
        ny = len(self.crdGrid[0])
        nx = len(self.crdGrid[1])
        if data.ndim == 2:
            nt = data.shape[1]
            grid = np.full([ny, nx, nt], np.nan)
            grid[indY, indX, :] = data
        elif data.ndim == 1:
            grid = np.full([ny, nx], np.nan)
            grid[indY, indX] = data
        # setattr(self, field+'_grid', grid)
        return grid

    def statCalError(self, *, predField='LSTM', targetField='SMAP'):
        pred = getattr(self, predField)
        target = getattr(self, targetField)
        statError = classPost.statError(pred=pred, target=target)
        # setattr(self, 'statErr_'+predField+'_'+targetField, statError)
        return statError

    def statCalSigma(self, *, field='LSTM'):
        # dataPred = getattr(self, field)
        dataPredBatch = getattr(self, field+'_MC')
        dataSigma = getattr(self, field+'_Sigma')
        dataSigmaBatch = getattr(self, field+'_SigmaMC')
        # dataSigma = np.sqrt(np.mean(dataSigmaBatch**2, axis=2))
        statSigma = classPost.statSigma(
            dataMC=dataPredBatch, dataSigma=dataSigma,
            dataSigmaBatch=dataSigmaBatch)
        setattr(self, 'statSigma_'+field, statSigma)
        return statSigma

    def statCalConf(self, *, predField='LSTM', targetField='SMAP', rmBias=False):
        dataPred = getattr(self, predField)
        dataTarget = getattr(self, targetField)
        dataMC = getattr(self, predField+'_MC')
        if hasattr(self, 'statSigma_'+predField):
            statSigma = getattr(self, 'statSigma_'+predField)
        else:
            statSigma = self.statCalSigma(field=predField)
        statConf = classPost.statConf(
            statSigma=statSigma, dataPred=dataPred, dataTarget=dataTarget,
            dataMC=dataMC, rmBias=rmBias)
        return statConf

    def statCalProb(self, *, predField='LSTM', targetField='SMAP'):
        dataPred = getattr(self, predField)
        dataTarget = getattr(self, targetField)
        dataMC = getattr(self, predField+'_MC')
        if hasattr(self, 'statSigma_'+predField):
            statSigma = getattr(self, 'statSigma_'+predField)
        else:
            statSigma = self.statCalSigma(field=predField)
        statProb = classPost.statProb(
            statSigma=statSigma, dataPred=dataPred, dataTarget=dataTarget,
            dataMC=dataMC)
        return statProb
