# This module include database classes

import os
import numpy as np
import pandas as pd
import datetime as dt
from . import funDB
from . import funLSTM
import time


class Dataset(object):
    r"""Base class Database SMAP

        Arguments:
            rootDB:
            subsetName:
    """

    def __init__(self, rootDB, subsetName, yrLst):
        self.__rootDB = rootDB
        self.__subsetName = subsetName
        rootName, crd, indSub, indSkip = funDB.readDBinfo(
            rootDB=rootDB, subsetName=subsetName)
        self.__crd = crd
        self.__indSub = indSub
        self.__indSkip = indSkip
        self.__rootName = rootName

        self.__yrLst = yrLst
        self.__time = funDB.readDBtime(
            rootDB=self.rootDB, rootName=self.rootName, yrLst=yrLst)

    @property
    def rootName(self):
        return self.__rootName

    @property
    def crd(self):
        return self.__crd

    @property
    def indSub(self):
        return self.__indSub

    @property
    def rootDB(self):
        return self.__rootDB

    @property
    def subsetName(self):
        return self.__subsetName

    @property
    def indSkip(self):
        return self.__indSkip

    @property
    def nGrid(self):
        return len(self.crd)

    @property
    def time(self):
        return self.__time

    @property
    def yrLst(self):
        return self.__yrLst

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

    def readPred(self, *, rootOut, out, drMC=0, var='pred'):
        bPred = funLSTM.checkPred(out=out, rootOut=rootOut, test=self.subsetName,
                          syr=self.yrLst[0], eyr=self.yrLst[-1], drMC=drMC)
        if bPred is False:
            print('running test')
            funLSTM.testLSTM(out=out, rootOut=rootOut, test=self.subsetName,
                             syr=self.yrLst[0], eyr=self.yrLst[-1], drMC=drMC)
        dataPred, dataSigma, dataPredBatch, dataSigmaBatch = funLSTM.readPred(
            out=out, rootOut=rootOut, test=self.subsetName, syr=self.yrLst[0], eyr=self.yrLst[-1], drMC=drMC)
        setattr(self, var, dataPred)
        setattr(self, var+'Sigma', dataSigma)
        setattr(self, var+'MC', dataPredBatch)
        setattr(self, var+'MC', dataSigmaBatch)
