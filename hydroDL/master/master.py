import os
import hydroDL
from collections import OrderedDict
import numpy as np
import json
from hydroDL import utils
from hydroDL.model import crit, rnn
from hydroDL.model.train import trainModel
from hydroDL.model.test import testModel
import datetime as dt
import pandas as pd
import random
import torch
import time


def wrapMaster(out, optData, optModel, optLoss, optTrain):
    mDict = OrderedDict(
        out=out, data=optData, model=optModel, loss=optLoss, train=optTrain
    )
    return mDict


def readMasterFile(out):
    mFile = os.path.join(out, "master.json")
    with open(mFile, "r") as fp:
        mDict = json.load(fp, object_pairs_hook=OrderedDict)
    print("read master file " + mFile)
    return mDict


def writeMasterFile(mDict):
    out = mDict["out"]
    if not os.path.isdir(out):
        os.makedirs(out)
    mFile = os.path.join(out, "master.json")
    with open(mFile, "w") as fp:
        json.dump(mDict, fp, indent=4)
    print("write master file " + mFile)
    return out


def loadModel(outFolder, epoch, modelName="model"):
    modelFile = os.path.join(outFolder, modelName + "_Ep" + str(epoch) + ".pt")
    model = torch.load(modelFile)
    return model


def namePred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
    mDict = readMasterFile(out)
    if (
        "name" in mDict["data"].keys()
        and mDict["data"]["name"] == "hydroDL.data.camels.DataframeCamels"
    ):
        target = ["Streamflow"]
    else:
        target = mDict["data"]["target"]
    if type(target) is not list:
        target = [target]
    nt = len(target)
    lossName = mDict["loss"]["name"]
    if epoch is None:
        epoch = mDict["train"]["nEpoch"]
    if type(subset) is list:
        # if list, name as the number of subset list
        subset = str(len(subset))
    fileNameLst = list()
    for k in range(nt):
        testName = "_".join([subset, str(tRange[0]), str(tRange[1]), "ep" + str(epoch)])
        fileName = "_".join([testName, target[k]])
        fileNameLst.append(fileName)
        if lossName == "hydroDL.model.crit.SigmaLoss":
            fileName = "_".join([testName, target[k], "SigmaX"])
            fileNameLst.append(fileName)
    if doMC is not False:
        mcFileNameLst = list()
        for fileName in fileNameLst:
            fileName = "_".join([testName, target[k], "SigmaMC" + str(doMC)])
            mcFileNameLst.append(fileName)
        fileNameLst = fileNameLst + mcFileNameLst

    # sum up to file path list
    filePathLst = list()
    for fileName in fileNameLst:
        if suffix is not None:
            fileName = fileName + "_" + suffix
        filePath = os.path.join(out, fileName + ".csv")
        filePathLst.append(filePath)
    return filePathLst


# def readPred(out, tRange, subset, epoch=None, doMC=False, suffix=None):
#     mDict = readMasterFile(out)
#     dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
#     for k in range(len(filePathLst)):
#         filePath = filePathLst[k]
#         dataPred[:, :, k] = pd.read_csv(
#             filePath, dtype=float, header=None).values
#     isSigmaX = False
#     if mDict['loss']['name'] == 'hydroDL.model.crit.SigmaLoss':
#         isSigmaX = True
#         pred = dataPred[:, :, ::2]
#         sigmaX = dataPred[:, :, 1::2]
#     else:
#         pred = dataPred


def mvobs(data, mvday, rmNan=True):
    obslen = data.shape[1] - mvday + 1  # The length of training daily data
    ngage = data.shape[0]
    mvdata = np.full((ngage, obslen, 1), np.nan)
    for ii in range(obslen):
        tempdata = data[:, ii : ii + mvday, :]
        tempmean = np.nanmean(tempdata, axis=1)
        mvdata[:, ii, 0] = tempmean[:, 0]
    if rmNan is True:
        mvdata[np.where(np.isnan(mvdata))] = 0
    return mvdata


def calFDC(data):
    # data = Ngrid * Nday
    Ngrid, Nday = data.shape
    FDC100 = np.full([Ngrid, 100], np.nan)
    for ii in range(Ngrid):
        tempdata0 = data[ii, :]
        tempdata = tempdata0[~np.isnan(tempdata0)]
        # deal with no data case for some gages
        if len(tempdata) == 0:
            tempdata = np.full(Nday, 0)
        # sort from large to small
        temp_sort = np.sort(tempdata)[::-1]
        # select 100 quantile points
        Nlen = len(tempdata)
        ind = (np.arange(100) / 100 * Nlen).astype(int)
        FDCflow = temp_sort[ind]
        if len(FDCflow) != 100:
            raise Exception("unknown assimilation variable")
        else:
            FDC100[ii, :] = FDCflow

    return FDC100


def loadData(optData, readX=True, readY=True):
    if eval(optData["name"]) is hydroDL.data.dbCsv.DataframeCsv:
        df = hydroDL.data.dbCsv.DataframeCsv(
            rootDB=optData["rootDB"], subset=optData["subset"], tRange=optData["tRange"]
        )
        if readY is True:
            y = df.getDataTs(
                varLst=optData["target"],
                doNorm=optData["doNorm"][1],
                rmNan=optData["rmNan"][1],
            )
        else:
            y = None

        if readX is True:
            x = df.getDataTs(
                varLst=optData["varT"],
                doNorm=optData["doNorm"][0],
                rmNan=optData["rmNan"][0],
            )
            c = df.getDataConst(
                varLst=optData["varC"],
                doNorm=optData["doNorm"][0],
                rmNan=optData["rmNan"][0],
            )
            if optData["daObs"] > 0:
                nday = optData["daObs"]
                sd = utils.time.t2dt(optData["tRange"][0]) - dt.timedelta(days=nday)
                ed = utils.time.t2dt(optData["tRange"][1]) - dt.timedelta(days=nday)
                df = hydroDL.data.dbCsv.DataframeCsv(
                    rootDB=optData["rootDB"], subset=optData["subset"], tRange=[sd, ed]
                )
                obs = df.getDataTs(
                    varLst=optData["target"],
                    doNorm=optData["doNorm"][1],
                    rmNan=optData["rmNan"][1],
                )
                x = (x, obs)
        else:
            x = None
            c = None
    elif eval(optData["name"]) is hydroDL.data.camels.DataframeCamels:
        df = hydroDL.data.camels.DataframeCamels(
            subset=optData["subset"], tRange=optData["tRange"]
        )
        x = df.getDataTs(
            varLst=optData["varT"],
            doNorm=optData["doNorm"][0],
            rmNan=optData["rmNan"][0],
        )
        y = df.getDataObs(
            doNorm=optData["doNorm"][1],
            rmNan=optData["rmNan"][1],
            basinnorm=optData["basinNorm"],
        )
        c = df.getDataConst(
            varLst=optData["varC"],
            doNorm=optData["doNorm"][0],
            rmNan=optData["rmNan"][0],
            SAOpt=optData["SAOpt"],
        )
        if "addVar" in optData.keys():
            addName = optData["addVar"]
            if addName == "Lstm":
                sac = df.getSAC(
                    basinnorm=optData["basinNorm"],
                    doNorm=optData["doNorm"][1],
                    rmNan=optData["rmNan"][0],
                )
                x = np.concatenate([x, sac], axis=2)
                print("SAC output is used")
            if addName == "hourprecip":
                hourp = df.getHour(
                    doNorm=optData["doNorm"][0], rmNan=optData["rmNan"][0]
                )
                x = (x, hourp)
                print("hourly precip is used")
            if addName == "SMAP":
                smapinv = df.getSMAP()
                x = (x, smapinv)
                print("smap inv is used")
            if addName == "SMAPFDC":
                smapdata = df.getCSV(
                    doNorm=True, rmNan=False, readRange=[20150401, 20200401]
                )
                smapdata = np.squeeze(smapdata)  # dim Ngrid*Nday
                smapinv = calFDC(smapdata)
                x = (x, smapinv)
                print("smap FDC inv is used")
        if c.size == 0:
            c = None
        # # judge if do SA analysis to attributes
        # if 'SAOpt' in optData.keys():
        #     if optData['SAOpt'] is not None:
        #         SAname, SAfac = optData['SAOpt']
        #         varList = optData['varC']
        #         # find the index of target constant
        #         indVar = varList.index(SAname)
        #         c[:, indVar] = c[:, indVar]*(1+SAfac)
        # judge if need local calibration kernel
        if "lckernel" in optData.keys():
            if optData["lckernel"] is not None:
                hisRange = optData["lckernel"]  # history record trange
                df = hydroDL.data.camels.DataframeCamels(
                    subset=optData["subset"], tRange=hisRange
                )
                if "fdcopt" in optData.keys():
                    if optData["fdcopt"] is not False:
                        if type(optData["fdcopt"]) is list:
                            # Use the FDC of specified gages
                            df = hydroDL.data.camels.DataframeCamels(
                                subset=optData["fdcopt"], tRange=hisRange
                            )
                        # calculate FDC
                        dadata = df.getDataObs(doNorm=optData["doNorm"][1], rmNan=False)
                        dadata = np.squeeze(dadata)  # dim Ngrid*Nday
                        if "dailymig" in optData.keys() and optData["dailymig"] is True:
                            dadata[np.where(np.isnan(dadata))] = 0
                            print("Daily time series was directly migrated")
                        else:
                            dadata = calFDC(dadata)
                            print("FDC was calculated and used!")
                    else:
                        dadata = df.getDataObs(doNorm=optData["doNorm"][1], rmNan=True)
                        dadata = np.squeeze(dadata)  # dim Ngrid*Nday
                        print("Local calibration kernel is used with raw data!")
                else:
                    dadata = df.getDataObs(doNorm=optData["doNorm"][1], rmNan=True)
                    dadata = np.squeeze(dadata)  # dim Ngrid*Nday
                    print("Local calibration kernel is used with raw data!")
                x = (x, dadata)
            else:
                print("Local calibration kernel is shut down!")

        if type(optData["daObs"]) is int:
            ndaylst = [optData["daObs"]]
        elif type(optData["daObs"]) is list:
            ndaylst = optData["daObs"]
        else:
            raise Exception("unknown datatype for daobs")

        # judge if multiple day assimilation or if needing assimilation
        if ndaylst[0] > 0 or len(ndaylst) > 1:
            if optData["damean"] is False:
                tRangePre = [19790101, 20150101]  # largest trange
                tLstPre = utils.time.tRange2Array(tRangePre)
                df = hydroDL.data.camels.DataframeCamels(
                    subset=optData["subset"], tRange=tRangePre
                )
                dadataPre = df.getDataObs(doNorm=optData["doNorm"][1], rmNan=True)
            dadata = np.full((x.shape[0], x.shape[1], len(ndaylst)), np.nan)
            for ii in range(len(ndaylst)):
                nday = ndaylst[ii]
                if optData["damean"] is False:
                    sd = utils.time.t2dt(optData["tRange"][0]) - dt.timedelta(days=nday)
                    ed = utils.time.t2dt(optData["tRange"][1]) - dt.timedelta(days=nday)
                    timese = utils.time.tRange2Array([sd, ed])
                    C, ind1, ind2 = np.intersect1d(timese, tLstPre, return_indices=True)
                    if optData["davar"] == "streamflow":
                        obs = dadataPre[:, ind2, :]
                    elif optData["davar"] == "precipitation":
                        df = hydroDL.data.camels.DataframeCamels(
                            subset=optData["subset"], tRange=[sd, ed]
                        )
                        obs = df.getDataTs(
                            varLst=["prcp"], doNorm=optData["doNorm"][0], rmNan=True
                        )
                    else:
                        raise Exception("unknown assimilation variable")

                else:
                    if optData["dameanopt"] == 0:  # previous moving avergae da
                        sd = utils.time.t2dt(optData["tRange"][0]) - dt.timedelta(
                            days=nday
                        )
                        ed = utils.time.t2dt(optData["tRange"][1]) - dt.timedelta(
                            days=1
                        )
                        df = hydroDL.data.camels.DataframeCamels(
                            subset=optData["subset"], tRange=[sd, ed]
                        )
                        if optData["davar"] == "streamflow":
                            obsday = df.getDataObs(
                                doNorm=optData["doNorm"][1], rmNan=False
                            )
                        elif optData["davar"] == "precipitation":
                            obsday = df.getDataTs(
                                varLst=["prcp"],
                                doNorm=optData["doNorm"][0],
                                rmNan=False,
                            )
                        else:
                            raise Exception("unknown assimilation variable")
                        obs = mvobs(obsday, mvday=nday, rmNan=True)
                    # 1: regular mean DA for test temporialy; 2:add weight
                    elif optData["dameanopt"] > 0:
                        sd = utils.time.t2dt(optData["tRange"][0]) - dt.timedelta(
                            days=nday
                        )
                        ed = utils.time.t2dt(optData["tRange"][1]) - dt.timedelta(
                            days=1
                        )
                        Nint = int((ed - sd) / dt.timedelta(days=nday))
                        ed = sd + Nint * dt.timedelta(days=nday)
                        df = hydroDL.data.camels.DataframeCamels(
                            subset=optData["subset"], tRange=[sd, ed]
                        )
                        if optData["davar"] == "streamflow":
                            obsday = df.getDataObs(
                                doNorm=optData["doNorm"][1], rmNan=False
                            )
                        elif optData["davar"] == "precipitation":
                            obsday = df.getDataTs(
                                varLst=["prcp"],
                                doNorm=optData["doNorm"][0],
                                rmNan=False,
                            )
                        else:
                            raise Exception("unknown assimilation variable")
                        obsday = np.reshape(obsday, (obsday.shape[0], -1, nday))
                        # obsmean = np.nanmean(obsday, axis=2)
                        obsmean = obsday[:, :, -1]  # test regular single observation
                        obsmean = np.tile(obsmean, nday).reshape(-1, nday, Nint)
                        obs = np.transpose(obsmean, (0, 2, 1)).reshape(
                            obsday.shape[0], nday * Nint, 1
                        )
                        endindex = x.shape[1]
                        obs = obs[:, 0:endindex, :]
                        obs[np.where(np.isnan(obs))] = 0
                dadata[:, :, ii] = obs.squeeze()
            x = (x, dadata)
            # test DI(3)-A hypothesis
            # x = np.concatenate((x, dadata[:, :, 0:3]), axis=2)
            # if len(ndaylst) >3:
            #     x = (x, dadata[:, :, 3:])
            # regular mean DA for test temporialy, add weight dimension
            if optData["dameanopt"] == 2:
                winput = (nday + 1 - np.arange(1, nday + 1)) / nday
                winput = np.tile(winput, Nint)[0:endindex]
                winput = np.tile(winput, (obsday.shape[0], 1))
                winput = np.expand_dims(winput, axis=2)
                x[0] = np.concatenate([x[0], winput], axis=2)

    else:
        raise Exception("unknown database")

    return df, x, y, c


def train(mDict):
    if mDict is str:
        mDict = readMasterFile(mDict)
    out = mDict["out"]
    optData = mDict["data"]
    optModel = mDict["model"]
    optLoss = mDict["loss"]
    optTrain = mDict["train"]

    # fix the random seed
    if optTrain["seed"] is None:
        # generate random seed
        randomseed = int(np.random.uniform(low=0, high=1e6))
        optTrain["seed"] = randomseed
        print("random seed updated!")
    else:
        randomseed = optTrain["seed"]

    random.seed(randomseed)
    torch.manual_seed(randomseed)
    np.random.seed(randomseed)
    torch.cuda.manual_seed(randomseed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # data
    df, x, y, c = loadData(optData)
    # x: ngage*nday*nvar
    # y: ngage*nday*nvar
    # c: ngage*nvar
    # temporal test, fill obs nan using LSTM forecast
    # temp = x[:,:,-1, None]
    # y[np.isnan(y)] = temp[np.isnan(y)]

    if c is None:
        if type(x) is tuple:
            nx = x[0].shape[-1]
        else:
            nx = x.shape[-1]
    else:
        if type(x) is tuple:
            nx = x[0].shape[-1] + c.shape[-1]
        else:
            nx = x.shape[-1] + c.shape[-1]
    ny = y.shape[-1]

    # loss
    if eval(optLoss["name"]) is hydroDL.model.crit.SigmaLoss:
        lossFun = crit.SigmaLoss(prior=optLoss["prior"])
        optModel["ny"] = ny * 2
    elif eval(optLoss["name"]) is hydroDL.model.crit.RmseLoss:
        lossFun = crit.RmseLoss()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.NSELoss:
        lossFun = crit.NSELoss()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.NSELosstest:
        lossFun = crit.NSELosstest()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.MSELoss:
        lossFun = crit.MSELoss()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.RmseLossCNN:
        lossFun = crit.RmseLossCNN()
        optModel["ny"] = ny
    elif eval(optLoss["name"]) is hydroDL.model.crit.ModifyTrend1:
        lossFun = crit.ModifyTrend1()
        optModel["ny"] = ny

    # model
    if optModel["nx"] != nx:
        print("updated nx by input data")
        optModel["nx"] = nx
    if eval(optModel["name"]) is hydroDL.model.rnn.CudnnLstmModel:
        if type(x) is tuple:
            x = np.concatenate([x[0], x[1]], axis=2)
            if c is None:
                nx = x.shape[-1]
            else:
                nx = x.shape[-1] + c.shape[-1]
            optModel["nx"] = nx
            print("Concatenate input and obs, update nx by obs")
        model = rnn.CudnnLstmModel(
            nx=optModel["nx"], ny=optModel["ny"], hiddenSize=optModel["hiddenSize"]
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.CpuLstmModel:
        model = rnn.CpuLstmModel(
            nx=optModel["nx"], ny=optModel["ny"], hiddenSize=optModel["hiddenSize"]
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.LstmCloseModel:
        model = rnn.LstmCloseModel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            hiddenSize=optModel["hiddenSize"],
            fillObs=True,
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.AnnModel:
        model = rnn.AnnCloseModel(
            nx=optModel["nx"], ny=optModel["ny"], hiddenSize=optModel["hiddenSize"]
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.AnnCloseModel:
        model = rnn.AnnCloseModel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            hiddenSize=optModel["hiddenSize"],
            fillObs=True,
        )
    elif eval(optModel["name"]) is hydroDL.model.cnn.LstmCnn1d:
        convpara = optModel["convNKSP"]
        model = hydroDL.model.cnn.LstmCnn1d(
            nx=optModel["nx"],
            ny=optModel["ny"],
            rho=optModel["rho"],
            nkernel=convpara[0],
            kernelSize=convpara[1],
            stride=convpara[2],
            padding=convpara[3],
        )
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLSTMmodel:
        daobsOption = optData["daObs"]
        if type(daobsOption) is list:
            if len(daobsOption) - 3 >= 7:
                # using 1dcnn only when number of obs larger than 7
                optModel["nobs"] = len(daobsOption)
                convpara = optModel["convNKS"]
                model = rnn.CNN1dLSTMmodel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    nobs=optModel["nobs"] - 3,
                    hiddenSize=optModel["hiddenSize"],
                    nkernel=convpara[0],
                    kernelSize=convpara[1],
                    stride=convpara[2],
                    poolOpt=optModel["poolOpt"],
                )
                print("CNN1d Kernel is used!")
            else:
                if type(x) is tuple:
                    x = np.concatenate([x[0], x[1]], axis=2)
                    nx = x.shape[-1] + c.shape[-1]
                    optModel["nx"] = nx
                    print("Concatenate input and obs, update nx by obs")
                model = rnn.CudnnLstmModel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    hiddenSize=optModel["hiddenSize"],
                )
                optModel["name"] = "hydroDL.model.rnn.CudnnLstmModel"
                print("Too few obserservations, not using cnn kernel")
        else:
            raise Exception("CNN kernel used but daobs option is not obs list")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLSTMInmodel:
        # daobsOption = optData['daObs']
        daobsOption = list(range(24))
        if type(daobsOption) is list:
            if len(daobsOption) - 3 >= 7:
                # using 1dcnn only when number of obs larger than 7
                optModel["nobs"] = len(daobsOption)
                convpara = optModel["convNKS"]
                model = rnn.CNN1dLSTMInmodel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    # nobs=optModel['nobs']-3,
                    nobs=24,  # temporary test
                    hiddenSize=optModel["hiddenSize"],
                    nkernel=convpara[0],
                    kernelSize=convpara[1],
                    stride=convpara[2],
                    poolOpt=optModel["poolOpt"],
                )
                print("CNN1d Kernel is used!")
            else:
                if type(x) is tuple:
                    x = np.concatenate([x[0], x[1]], axis=2)
                    nx = x.shape[-1] + c.shape[-1]
                    optModel["nx"] = nx
                    print("Concatenate input and obs, update nx by obs")
                model = rnn.CudnnLstmModel(
                    nx=optModel["nx"],
                    ny=optModel["ny"],
                    hiddenSize=optModel["hiddenSize"],
                )
                optModel["name"] = "hydroDL.model.rnn.CudnnLstmModel"
                print("Too few obserservations, not using cnn kernel")
        else:
            raise Exception("CNN kernel used but daobs option is not obs list")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLCmodel:
        # LCrange = optData['lckernel']
        # tLCLst = utils.time.tRange2Array(LCrange)
        if len(x[1].shape) == 2:
            # for LC-FDC
            optModel["nobs"] = x[1].shape[-1]
        elif len(x[1].shape) == 3:
            # for LC-SMAP--get time step
            optModel["nobs"] = x[1].shape[1]
        convpara = optModel["convNKS"]
        model = rnn.CNN1dLCmodel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            nobs=optModel["nobs"],
            hiddenSize=optModel["hiddenSize"],
            nkernel=convpara[0],
            kernelSize=convpara[1],
            stride=convpara[2],
            poolOpt=optModel["poolOpt"],
        )
        print("CNN1d Local calibartion Kernel is used!")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CNN1dLCInmodel:
        LCrange = optData["lckernel"]
        tLCLst = utils.time.tRange2Array(LCrange)
        optModel["nobs"] = x[1].shape[-1]
        convpara = optModel["convNKS"]
        model = rnn.CNN1dLCInmodel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            nobs=optModel["nobs"],
            hiddenSize=optModel["hiddenSize"],
            nkernel=convpara[0],
            kernelSize=convpara[1],
            stride=convpara[2],
            poolOpt=optModel["poolOpt"],
        )
        print("CNN1d Local calibartion Kernel is used!")
    elif eval(optModel["name"]) is hydroDL.model.rnn.CudnnInvLstmModel:
        # optModel['ninv'] = x[1].shape[-1]
        optModel["ninv"] = x[1].shape[-1] + c.shape[-1]  # Test the inv using attributes
        model = rnn.CudnnInvLstmModel(
            nx=optModel["nx"],
            ny=optModel["ny"],
            hiddenSize=optModel["hiddenSize"],
            ninv=optModel["ninv"],
            nfea=optModel["nfea"],
            hiddeninv=optModel["hiddeninv"],
        )
        print("LSTMInv model is used!")
    # train
    if optTrain["saveEpoch"] > optTrain["nEpoch"]:
        optTrain["saveEpoch"] = optTrain["nEpoch"]

    # train model
    writeMasterFile(mDict)
    model = trainModel(
        model,
        x,
        y,
        c,
        lossFun,
        nEpoch=optTrain["nEpoch"],
        miniBatch=optTrain["miniBatch"],
        saveEpoch=optTrain["saveEpoch"],
        saveFolder=out,
    )


def test(
    out,
    *,
    tRange,
    subset,
    doMC=False,
    suffix=None,
    batchSize=None,
    epoch=None,
    reTest=False,
    basinnorm=True,
    savePath=None,
    SAOpt=None,
    FDCgage=None,
    dailymig=False,
    closedLoop=None
):
    mDict = readMasterFile(out)

    optData = mDict["data"]
    optData["subset"] = subset
    optData["tRange"] = tRange
    optData["basinNorm"] = basinnorm
    if "damean" not in optData.keys():
        optData["damean"] = False
    if "dameanopt" not in optData.keys():
        optData["dameanopt"] = 0
    if "davar" not in optData.keys():
        optData["davar"] = "streamflow"
    elif type(optData["davar"]) is list:
        optData["davar"] = "".join(optData["davar"])
    if SAOpt is not None:
        optData["SAOpt"] = SAOpt
    elif "SAOpt" not in optData.keys():
        optData["SAOpt"] = None
    if (FDCgage is not None) and (type(FDCgage) is list):
        # Specify to use the FDC of these gages
        optData["fdcopt"] = FDCgage
    optData["dailymig"] = dailymig
    # generate file names and run model
    if savePath is None:
        # default file path
        filePathLst = namePred(
            out, tRange, subset, epoch=epoch, doMC=doMC, suffix=suffix
        )
    else:
        if type(savePath) is not list:
            savePath = [savePath]
        filePathLst = savePath
    print("output files:", filePathLst)
    for filePath in filePathLst:
        if not os.path.isfile(filePath):
            reTest = True
    if reTest is True:
        print("Runing new results")
        if closedLoop is not None:
            # closed loop experiments
            df, x, obs, c = loadData(optData)
            nLoop = closedLoop + 10
            # create Nan observations for closedLoop forecast
            initobs = x[1]  # integrated obs
            initfor = x[0]
            lenObs = initobs.shape[1]
            fixIndex = np.arange(0, lenObs, closedLoop)
            firstobs = np.full(initobs.shape, 0.0)
            firstobs[:, fixIndex, :] = initobs[:, fixIndex, :]
            x = (initfor, firstobs)  # first integrated observations
            model = loadModel(out, epoch=epoch)
            tempPath = filePathLst[0][:-4]  # remove '.csv'
            filePathLst = list()
            for iloop in range(0, nLoop):
                # model = loadModel(out, epoch=epoch)
                tempfilePath = (
                    tempPath
                    + "_"
                    + "NLOOP"
                    + str(closedLoop)
                    + "_"
                    + "loop"
                    + str(iloop)
                    + ".csv"
                )
                filePathLst.append(tempfilePath)
                testModel(
                    model,
                    x,
                    c,
                    batchSize=batchSize,
                    filePathLst=[tempfilePath],
                    doMC=doMC,
                )
                # read predicted results
                dataPred = pd.read_csv(tempfilePath, dtype=float, header=None).values
                updateData = np.full(initobs.shape, 0.0)
                updateData[:, 1:, 0] = dataPred[:, 0:-1]
                updateData[:, fixIndex, :] = initobs[:, fixIndex, :]
                x = (initfor, updateData)  # update the integrated observations

        else:
            df, x, obs, c = loadData(optData)
            model = loadModel(out, epoch=epoch)
            t0 = time.time()
            testModel(
                model, x, c, batchSize=batchSize, filePathLst=filePathLst, doMC=doMC
            )
            print("testing time is {}".format(time.time() - t0))

    else:
        print("Loaded previous results")
        df, x, obs, c = loadData(optData, readX=False)

    # load previous result - readPred
    mDict = readMasterFile(out)
    dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
    for k in range(len(filePathLst)):
        filePath = filePathLst[k]
        dataPred[:, :, k] = pd.read_csv(filePath, dtype=float, header=None).values
    isSigmaX = False
    if mDict["loss"]["name"] == "hydroDL.model.crit.SigmaLoss" or doMC is not False:
        isSigmaX = True
        pred = dataPred[:, :, ::2]
        sigmaX = dataPred[:, :, 1::2]
    else:
        pred = dataPred

    if optData["doNorm"][1] is True:
        if eval(optData["name"]) is hydroDL.data.dbCsv.DataframeCsv:
            target = optData["target"]
            if type(optData["target"]) is not list:
                target = [target]
            nTar = len(target)
            for k in range(nTar):
                pred[:, :, k] = hydroDL.data.dbCsv.transNorm(
                    pred[:, :, k],
                    rootDB=optData["rootDB"],
                    fieldName=target[k],
                    fromRaw=False,
                )
                obs[:, :, k] = hydroDL.data.dbCsv.transNorm(
                    obs[:, :, k],
                    rootDB=optData["rootDB"],
                    fieldName=target[k],
                    fromRaw=False,
                )
                if isSigmaX is True:
                    sigmaX[:, :, k] = hydroDL.data.dbCsv.transNormSigma(
                        sigmaX[:, :, k],
                        rootDB=optData["rootDB"],
                        fieldName=target[k],
                        fromRaw=False,
                    )
        elif eval(optData["name"]) is hydroDL.data.camels.DataframeCamels:
            nvar = pred.shape[-1]
            targetstr = []
            for ii in range(nvar):
                targetstr.append("usgsFlow")
            if nvar != len(targetstr):
                raise Exception("wrong target variable number")
            pred = hydroDL.data.camels.transNorm(pred, targetstr, toNorm=False)
            obs = hydroDL.data.camels.transNorm(obs, "usgsFlow", toNorm=False)
            if basinnorm is True:
                if type(subset) is list:
                    gageid = np.array(subset)
                elif type(subset) is str:
                    gageid = subset
                for ii in range(nvar):
                    pred[:, :, ii] = hydroDL.data.camels.basinNorm(
                        pred[:, :, ii], gageid=gageid, toNorm=False
                    )
                obs = hydroDL.data.camels.basinNorm(obs, gageid=gageid, toNorm=False)
    if isSigmaX is True:
        return df, pred, obs, sigmaX
    else:
        return df, pred, obs  # pred: ngage*nday*nvar
