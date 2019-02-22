import numpy as np
import os


class dataset(object):
    r"""Base class of SMAP database
        Arguments:
            rootDB:
            subsetName:
    """

    def __init__(self, rootDB, subsetName, yrLst):
        self.rootDB = rootDB
        self.subsetName = subsetName
        rootName, crd, indSub, indSkip = readDBinfo(
            rootDB=rootDB, subsetName=subsetName)
        self.crd = crd
        (gridY, gridX, indY, indX) = crd2grid(crd[:, 0], crd[:, 1])
        self.crdGrid = (gridY, gridX)
        self.crdGridInd = np.stack((indY, indX), axis=1)

        self.indSub = indSub
        self.indSkip = indSkip
        self.rootName = rootName

        self.yrLst = yrLst
        self.time = readDBtime(
            rootDB=self.rootDB, rootName=self.rootName, yrLst=yrLst)

    def __repr__(self):
        return 'later'


class datasetLSTM(dataset):
    r""" a dataset that can be feed into training functions
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
            self.varInputTs = readVarLst(
                rootDB=self.rootDB, varLst=var[0])
        if isinstance(var[1], list):
            self.varInputConst = var[1]
        else:
            self.varInputConst = readVarLst(
                rootDB=self.rootDB, varLst=var[1])
        self.varInput = self.varInputTs + self.varInputConst

        # target variable
        self.varTarget = targetName

    def readTarget(self):
        nt = len(self.time)
        ngrid = len(self.indSub)
        data = readDataTS(
            rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
            indSkip=self.indSkip, yrLst=self.yrLst, fieldName=self.varTarget,
            nt=nt, ngrid=ngrid)
        stat = readStat(rootDB=self.rootDB, fieldName=self.varTarget)
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
            dataTemp = readDataTS(
                rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
                indSkip=self.indSkip, yrLst=self.yrLst, fieldName=var,
                nt=nt, ngrid=ngrid)
            statTemp = readStat(rootDB=self.rootDB, fieldName=var)
            dataNormTemp = (dataTemp-statTemp[2])/statTemp[3]
            data[:, :, k] = dataTemp
            dataNorm[:, :, k] = dataNormTemp
            stat[:, k] = statTemp
            k = k+1

        # const
        for var in self.varInputConst:
            dataTemp = readDataConst(
                rootDB=self.rootDB, rootName=self.rootName, indSub=self.indSub,
                indSkip=self.indSkip, yrLst=self.yrLst, fieldName=var,
                ngrid=ngrid)
            statTemp = readStat(
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


def readDBinfo(*, rootDB, subsetName):
    subsetFile = os.path.join(rootDB, 'Subset', subsetName+'.csv')
    print(subsetFile)
    dfSubset = pd.read_csv(subsetFile, dtype=np.int64, header=0)
    rootName = dfSubset.columns.values[0]
    indSub = dfSubset.values.flatten()

    crdFile = os.path.join(rootDB, rootName, 'crd.csv')
    crdRoot = pd.read_csv(crdFile, dtype=np.float, header=None).values

    indAll = np.arange(0, crdRoot.shape[0], dtype=np.int64)
    if np.array_equal(indSub, np.array([-1])):
        indSub = indAll
        indSkip = None
    else:
        indSub = indSub-1
        indSkip = np.delete(indAll, indSub)
    crd = crdRoot[indSub, :]
    return rootName, crd, indSub, indSkip


def readDBtime(*, rootDB, rootName, yrLst):
    tnum = np.empty(0, dtype=np.datetime64)
    for yr in yrLst:
        timeFile = os.path.join(
            rootDB, rootName, str(yr), 'timeStr.csv')
        temp = pd.read_csv(timeFile, dtype=str, header=None).astype(
            np.datetime64).values.flatten()
        tnum = np.concatenate([tnum, temp], axis=0)
    return tnum


def readVarLst(*, rootDB, varLst):
    varFile = os.path.join(rootDB, 'Variable', varLst+'.csv')
    varLst = pd.read_csv(varFile, header=None,
                         dtype=str).values.flatten().tolist()
    return varLst


def readDataTS(*, rootDB, rootName, indSub, indSkip, yrLst, fieldName,
               nt=-1, ngrid=-1):
    if nt == -1:
        tnum = readDBtime(rootDB, rootName, yrLst)
        nt = len(tnum)

    if ngrid == -1:
        ngrid = len(indSub)

    # read data
    data = np.zeros([ngrid, nt])
    k1 = 0
    for yr in yrLst:
        t1 = time.time()
        dataFile = os.path.join(rootDB, rootName, str(yr), fieldName+'.csv')
        dataTemp = pd.read_csv(
            dataFile, dtype=np.float, skiprows=indSkip, header=None).values
        k2 = k1+dataTemp.shape[1]
        data[:, k1:k2] = dataTemp
        k1 = k2
        print('read '+dataFile, time.time()-t1)
    data[np.where(data == -9999)] = np.nan
    return data


def readDataConst(*, rootDB, rootName, indSub, indSkip,
                  yrLst, fieldName, ngrid=-1):
    if ngrid == -1:
        ngrid = len(indSub)

    # read data
    dataFile = os.path.join(rootDB, rootName, 'const', fieldName+'.csv')
    data = pd.read_csv(dataFile, dtype=np.float,
                       skiprows=indSkip, header=None).values.flatten()
    data[np.where(data == -9999)] = np.nan
    return data


def readStat(*, rootDB, fieldName, isConst=False):
    if isConst is False:
        statFile = os.path.join(rootDB, 'Statistics', fieldName+'_stat.csv')
    else:
        statFile = os.path.join(rootDB, 'Statistics',
                                'const_'+fieldName+'_stat.csv')
    stat = pd.read_csv(statFile, dtype=np.float, header=None).values.flatten()
    return stat


def crd2grid(y, x):
    ux, indX0, indX = np.unique(x, return_index=True, return_inverse=True)
    uy, indY0, indY = np.unique(y, return_index=True, return_inverse=True)

    minDx = np.min(ux[1:]-ux[0:-1])
    minDy = np.min(uy[1:]-uy[0:-1])
    maxDx = np.max(ux[1:]-ux[0:-1])
    maxDy = np.max(uy[1:]-uy[0:-1])
    if maxDx > minDx*2:
        print('skipped rows')
    #     indMissX=np.where((ux[1:]-ux[0:-1])>minDx*2)[0]
    #     insertX=(ux[indMissX+1]+ux[indMissX])/2
    #     ux=np.insert(ux,indMissX,insertX)
    if maxDy > minDy*2:
        print('skipped coloums')
    #     indMissY=np.where((uy[1:]-uy[0:-1])>minDy*2)
    #     raise Exception('skipped coloums or rows')

    uy = uy[::-1]
    ny = len(uy)
    indY = ny-1-indY
    return (uy, ux, indY, indX)
