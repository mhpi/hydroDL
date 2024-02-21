""" 
read and extract data from CSV database.
This module allows you to read time series inputs/forcings and define subsets
to read from.
"""
import os
import numpy as np
import pandas as pd
import time
import datetime as dt
import hydroDL.utils as utils
from hydroDL.dataset import Dataframe, DataModel
import hydroDL

################################################
# The definitions between ### are for convenience only.
# You don't need them unless you are calling them from outside,
# e.g., "dbCsv.xxx", or if you call DataModelCsv without supplying
# actual arguments
# This block shouldn't be here.
# We will phase out these definitions from this file gradually.

varTarget = ["SMAP_AM"]
# ===== SMAP varForcing =====
varForcing = [
    "APCP_FORA",
    "DLWRF_FORA",
    "DSWRF_FORA",
    "TMP_2_FORA",
    "SPFH_2_FORA",
    "VGRD_10_FORA",
    "UGRD_10_FORA",
]

varConst = [
    "Bulk",
    "Capa",
    "Clay",
    "NDVI",
    "Sand",
    "Silt",
    "flag_albedo",
    "flag_extraOrd",
    "flag_landcover",
    "flag_roughness",
    "flag_vegDense",
    "flag_waterbody",
]

varSoilM = [
    "APCP_FORA",
    "DLWRF_FORA",
    "DSWRF_FORA",
    "TMP_2_FORA",
    "SPFH_2_FORA",
    "VGRD_10_FORA",
    "UGRD_10_FORA",
    "SOILM_0-10_NOAH",
]

varForcingGlobal = ["GPM", "Wind", "Tair", "Psurf", "Qair", "SWdown", "LWdown"]
varSoilmGlobal = [
    "SoilMoi0-10",
    "GPM",
    "Wind",
    "Tair",
    "Psurf",
    "Qair",
    "SWdown",
    "LWdown",
]
varConstGlobal = [
    "Bulk",
    "Capa",
    "Clay",
    "NDVI",
    "Sand",
    "Silt",
    "flag_albedo",
    "flag_extraOrd",
    "flag_landcover",
    "flag_roughness",
    "flag_vegDense",
    "flag_waterbody",
]

################################################


def t2yrLst(tArray):
    t1 = tArray[0].astype(object)
    t2 = tArray[-1].astype(object)
    y1 = t1.year
    y2 = t2.year
    if t1 < dt.date(y1, 4, 1):
        y1 = y1 - 1
    if t2 < dt.date(y2, 4, 1):
        y2 = y2 - 1
    yrLst = list(range(y1, y2 + 1))
    tDb = utils.time.tRange2Array([dt.date(y1, 4, 1), dt.date(y2 + 1, 4, 1)])
    return yrLst, tDb


def readDBinfo(*, rootDB, subset):
    subsetFile = os.path.join(rootDB, "Subset", subset + ".csv")
    print(subsetFile)
    dfSubset = pd.read_csv(subsetFile, dtype=np.int64, header=0)
    rootName = dfSubset.columns.values[0]
    indSub = dfSubset.values.flatten()

    crdFile = os.path.join(rootDB, rootName, "crd.csv")
    crdRoot = pd.read_csv(crdFile, dtype=float, header=None).values

    indAll = np.arange(0, crdRoot.shape[0], dtype=np.int64)
    if np.array_equal(indSub, np.array([-1])):
        indSub = indAll
        indSkip = None
    else:
        indSub = indSub - 1
        indSkip = np.delete(indAll, indSub)
    crd = crdRoot[indSub, :]
    return rootName, crd, indSub, indSkip


def readSubset(*, rootDB, subset):
    subsetFile = os.path.join(rootDB, "Subset", subset + ".csv")
    print("reading subset " + subsetFile)
    dfSubset = pd.read_csv(subsetFile, dtype=np.int64, header=0)
    rootName = dfSubset.columns.values[0]
    indSub = dfSubset.values.flatten()
    return rootName, indSub


def readDBtime(*, rootDB, rootName, yrLst):
    tnum = np.empty(0, dtype=np.datetime64)
    for yr in yrLst:
        timeFile = os.path.join(rootDB, rootName, str(yr), "timeStr.csv")
        temp = (
            pd.read_csv(timeFile, dtype=str, header=None)
            .astype(np.datetime64)
            .values.flatten()
        )
        tnum = np.concatenate([tnum, temp], axis=0)
    return tnum


def readVarLst(*, rootDB, varLst):
    varFile = os.path.join(rootDB, "Variable", varLst + ".csv")
    varLst = pd.read_csv(varFile, header=None, dtype=str).values.flatten().tolist()
    return varLst


def readDataTS(*, rootDB, rootName, indSub, indSkip, yrLst, fieldName):
    tnum = readDBtime(rootDB=rootDB, rootName=rootName, yrLst=yrLst)
    nt = len(tnum)
    ngrid = len(indSub)

    # read data
    data = np.zeros([ngrid, nt])
    k1 = 0
    for yr in yrLst:
        t1 = time.time()
        dataFile = os.path.join(rootDB, rootName, str(yr), fieldName + ".csv")
        dataTemp = pd.read_csv(
            dataFile, dtype=float, skiprows=indSkip, header=None
        ).values
        k2 = k1 + dataTemp.shape[1]
        data[:, k1:k2] = dataTemp
        k1 = k2
        print("read " + dataFile, time.time() - t1)
    data[np.where(data == -9999)] = np.nan
    return data


def readDataConst(*, rootDB, rootName, indSub, indSkip, fieldName):
    # read data
    dataFile = os.path.join(rootDB, rootName, "const", fieldName + ".csv")
    data = pd.read_csv(
        dataFile, dtype=float, skiprows=indSkip, header=None
    ).values.flatten()
    data[np.where(data == -9999)] = np.nan
    return data


def readStat(*, rootDB, fieldName, isConst=False):
    if isConst is False:
        statFile = os.path.join(rootDB, "Statistics", fieldName + "_stat.csv")
    else:
        statFile = os.path.join(
            rootDB, "Statistics", "const_" + fieldName + "_stat.csv"
        )
    stat = pd.read_csv(statFile, dtype=float, header=None).values.flatten()
    return stat


def transNorm(data, *, rootDB, fieldName, fromRaw=True, isConst=False):
    stat = readStat(rootDB=rootDB, fieldName=fieldName, isConst=isConst)
    if fromRaw is True:
        dataOut = (data - stat[2]) / stat[3]
    else:
        dataOut = data * stat[3] + stat[2]
    return dataOut


def transNormSigma(data, *, rootDB, fieldName, fromRaw=True):
    stat = readStat(rootDB=rootDB, fieldName=fieldName, isConst=False)
    if fromRaw is True:
        dataOut = np.log((data / stat[3]) ** 2)
    else:
        dataOut = np.sqrt(np.exp(data)) * stat[3]
    return dataOut


class DataframeCsv(Dataframe):
    def __init__(self, rootDB, *, subset, tRange):
        super(DataframeCsv, self).__init__()
        self.rootDB = rootDB
        self.subset = subset
        rootName, crd, indSub, indSkip = readDBinfo(rootDB=rootDB, subset=subset)
        self.lat = crd[:, 0]
        self.lon = crd[:, 1]
        self.indSub = indSub
        self.indSkip = indSkip
        self.rootName = rootName
        self.time = utils.time.tRange2Array(tRange)

    def getDataTs(self, varLst, *, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        yrLst, tDb = t2yrLst(self.time)
        indDb, ind = utils.time.intersect(tDb, self.time)
        nt = len(tDb)
        ngrid = len(self.indSub)
        nvar = len(varLst)
        data = np.ndarray([ngrid, nt, nvar])

        # time series
        for k in range(nvar):
            dataTemp = readDataTS(
                rootDB=self.rootDB,
                rootName=self.rootName,
                indSub=self.indSub,
                indSkip=self.indSkip,
                yrLst=yrLst,
                fieldName=varLst[k],
            )
            if doNorm is True:
                dataTemp = transNorm(dataTemp, rootDB=self.rootDB, fieldName=varLst[k])
            data[:, :, k] = dataTemp
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        dataOut = data[:, indDb, :]
        return dataOut

    def getDataConst(self, varLst, *, doNorm=True, rmNan=True):
        if type(varLst) is str:
            varLst = [varLst]
        ngrid = len(self.indSub)
        nvar = len(varLst)
        data = np.ndarray([ngrid, nvar])
        for k in range(nvar):
            dataTemp = readDataConst(
                rootDB=self.rootDB,
                rootName=self.rootName,
                indSub=self.indSub,
                indSkip=self.indSkip,
                fieldName=varLst[k],
            )
            if doNorm is True:
                dataTemp = transNorm(
                    dataTemp, rootDB=self.rootDB, fieldName=varLst[k], isConst=True
                )
            data[:, k] = dataTemp
        if rmNan is True:
            data[np.where(np.isnan(data))] = 0
        return data


class DataModelCsv(DataModel):
    def __init__(
        self,
        *,
        # rootDB=hydroDL.pathSMAP["DB_L3_Global"],
        rootDB=None,
        subset="CONUSv4f1",
        varT=varForcing,
        varC=varConst,
        target="SMAP_AM",
        tRange=[20150401, 20160401],
        doNorm=[True, True],
        rmNan=[True, False],
        daObs=0
    ):
        super(DataModelCsv, self).__init__()
        df = DataframeCsv(rootDB=rootDB, subset=subset, tRange=tRange)

        self.x = df.getDataTs(varLst=varT, doNorm=doNorm[0], rmNan=rmNan[0])
        self.y = df.getDataTs(varLst=target, doNorm=doNorm[1], rmNan=rmNan[1])
        self.c = df.getDataConst(varLst=varC, doNorm=doNorm[0], rmNan=rmNan[0])

    def getData(self):
        return self.x, self.y, self.c
