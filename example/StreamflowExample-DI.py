import sys

sys.path.append('../')
from hydroDL import master, utils
from hydroDL.master import default
from hydroDL.master.master import loadModel, wrapMaster, writeMasterFile
from hydroDL.master.master import readMasterFile
# from hydroDL.post import plot, stat
from hydroDL.post import stat, plot
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.model.test import testModel
from hydroDL.model.train import trainModel
from hydroDL.model.rnn import CudnnLstmModel, CpuLstmModel
from hydroDL.model.crit import RmseLoss, NSELossBatch

import numpy as np
import pandas as pd
import os
import torch
import random
import datetime as dt
import json

#checking version of torch and cuda, and checking if cuda is available
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
# Options for different interface
interfaceOpt = 1
# ==1 default, the recommended and more interpretable version with clear data and training flow. We improved the
# original one to explicitly load and process data, set up model and loss, and train the model.
# ==0, the original "pro" version to train jobs based on the defined configuration dictionary.
# Results are very similar for two options.

flow_regime = 0
# 0: low flow expert
# 1: high flow expert

# Options for training and testing
# 0: train base model without DI
# 1: train DI model
# 0,1: do both base and DI model
# 2: test trained models
Action = [2]
gpuid = 6
torch.cuda.set_device(gpuid)

# Set hyperparameters
EPOCH = 300
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 1  # save model for every "saveEPOCH" epochs
Ttrain = [19801001, 19951001] # Training period
forType = 'daymet'
trainBuff = 365
loadTrain = True
subset_train = 'All'  #give the list of basins to train on or else fix 'All' to use all

# Fix random seed
seedid = 111111
random.seed(seedid)
torch.manual_seed(seedid)
np.random.seed(seedid)
torch.cuda.manual_seed(seedid)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Change the seed to have different runnings.
# We use the mean discharge of 6 runnings with different seeds to account for randomness and report results

# Define root directory of database and output
# Modify this based on your own location of CAMELS dataset.
# Following the data download instruction in README file, you should organize the folders like
# 'your/path/to/Camels/basin_timeseries_v1p2_metForcing_obsFlow' and 'your/path/to/Camels/camels_attributes_v2.0'
# Then 'rootDatabase' here should be 'your/path/to/Camels'
# You can also define the database directory in hydroDL/__init__.py by modifying pathCamels['DB'] variable
rootDatabase = os.path.join(os.sep,"scratch", "Camels")  # CAMELS dataset root directory: /scratch/Camels
camels.initcamels(flow_regime=flow_regime, rootDB=rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(
    os.sep, "data", "kas7897", "lstm_tuning", "hydroDL", "output", "rnnStreamflow"
)
# rootOut = os.path.join(os.path.sep, 'data', 'kas7897', 'dPLHBVrelease', 'output', 'rnnStreamflow')
# Root directory to save training results: /data/rnnStreamflow
# rootOut = "./output/streamflow/"

# Root directory to save training results: /data/rnnStreamflow
if forType == 'daymet':
    varF = ['dayl', 'prcp', 'srad', 'tmean', 'vp']
else:
    varF = ['dayl', 'prcp', 'srad', 'tmax', 'vp']
# Define all the configurations into dictionary variables
# three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
# be used to directly train the model when interfaceOpt == 0

# define dataset
# default module stores default configurations, using update to change the config
attrLst = [ 'p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq','high_prec_dur',
               'low_prec_freq','low_prec_dur', 'elev_mean', 'slope_mean', 'area_gages2', 'frac_forest', 'lai_max',
               'lai_diff', 'gvf_max', 'gvf_diff', 'dom_land_cover_frac', 'dom_land_cover', 'root_depth_50',
               'soil_depth_pelletier', 'soil_depth_statsgo', 'soil_porosity', 'soil_conductivity',
               'max_water_content', 'sand_frac', 'silt_frac', 'clay_frac', 'geol_1st_class', 'glim_1st_class_frac',
               'geol_2nd_class', 'glim_2nd_class_frac', 'carbonate_rocks_frac', 'geol_porostiy', 'geol_permeability']
optData = default.optDataCamels
optData = default.update(
    optData, varT=varF, varC=attrLst, tRange=Ttrain, forType=forType, subset=subset_train)  # Update the training period

# if (interfaceOpt == 1) and (2 not in Action):
if (interfaceOpt == 1) and (loadTrain is True):
    # load training data explicitly for the interpretable interface. Notice: if you want to apply our codes to your own
    # dataset, here is the place you can replace data.
    # read data from original CAMELS dataset
    # df: CAMELS dataframe; x: forcings[nb,nt,nx]; y: streamflow obs[nb,nt,ny]; c:attributes[nb,nc]
    # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
    # ny: number of target variables, nc: number of constant attributes
    df = camels.DataframeCamels(subset=optData["subset"], tRange=optData["tRange"], forType=forType)
    x = df.getDataTs(varLst=optData["varT"], doNorm=False, rmNan=False, flow_regime=flow_regime)
    y = df.getDataObs(doNorm=False, rmNan=False, basinnorm=False, flow_regime=flow_regime)
    # transform discharge from ft3/s to mm/day and then divided by mean precip to be dimensionless.
    # output = discharge/(area*mean_precip)
    # this can also be done by setting the above option "basinnorm = True" for df.getDataObs()
    y_temp = camels.basinNorm(y, optData["subset"], toNorm=True)
    c = df.getDataConst(varLst=optData["varC"], doNorm=False, rmNan=False, flow_regime=flow_regime)

    # process, do normalization and remove nan
    series_data = np.concatenate([x, y_temp], axis=2)
    seriesvarLst = varF + ["runoff"]
    # calculate statistics for norm and saved to a dictionary
    statDict = camels.getStatDic(
        flow_regime=flow_regime,
        attrLst=attrLst,
        attrdata=c,
        seriesLst=seriesvarLst,
        seriesdata=series_data,
    )
    # normalize
    attr_norm = camels.transNormbyDic(c, attrLst, statDict, toNorm=True, flow_regime=flow_regime)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(
        series_data, seriesvarLst, statDict, toNorm=True, flow_regime=flow_regime
    )

    # prepare the inputs
    xTrain = series_norm[:, :, :-1]  # forcing, not include obs
    xTrain[np.isnan(xTrain)] = 0.0
    yTrain = np.expand_dims(series_norm[:, :, -1], 2)
    attrs = attr_norm


# define model and update configure
if torch.cuda.is_available():
    optModel = default.optLstm
else:
    optModel = default.update(default.optLstm, name="hydroDL.model.rnn.CpuLstmModel")
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
# define loss function
if flow_regime==0:
    optLoss = default.optLossRMSE
elif flow_regime==1:
    optLoss = default.optLossNSEBatch

# define training options
optTrain = default.update(
    default.optTrainCamels,
    miniBatch=[BATCH_SIZE, RHO],
    nEpoch=EPOCH,
    saveEpoch=saveEPOCH,
    seed=seedid,
    trainBuff=trainBuff
)

# define output folder for model results
exp_name = f"CAMELSDemo"
exp_disp = "TestRun"
save_path = os.path.join(
    exp_name,
    exp_disp,
    "epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}_trainBuff{}_flowregime{}".format(
        optTrain["nEpoch"],
        optTrain["miniBatch"][0],
        optTrain["miniBatch"][1],
        optModel["hiddenSize"],
        optData["tRange"][0],
        optData["tRange"][1],
        optTrain['trainBuff'],
        flow_regime,
    ),
)


# Train the base model without data integration
if 0 in Action:
    out = os.path.join(rootOut, save_path, "All")  # output folder to save results
    # Wrap up all the training configurations to one dictionary in order to save into "out" folder
    masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)
    if interfaceOpt == 1:  # use the more interpretable version interface
        nx = xTrain.shape[-1] + attrs.shape[-1]  # update nx, nx = nx + nc
        ny = yTrain.shape[-1]
        # load model for training
        if torch.cuda.is_available():
            model = CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        else:
            model = CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        optModel = default.update(optModel, nx=nx, ny=ny)
        # the loaded model should be consistent with the 'name' in optModel Dict above for logging purpose
        if flow_regime==0:
            lossFun = RmseLoss()
        elif flow_regime==1:
            lossFun = NSELossBatch(np.nanstd(yTrain, axis=1))
        # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose
        # update and write the dictionary variable to out folder for logging and future testing
        masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)
        writeMasterFile(masterDict)
        # log statistics
        statFile = os.path.join(out, "statDict.json")
        with open(statFile, "w") as fp:
            json.dump(statDict, fp, indent=4)
        # train model
        model = trainModel(
            model,
            xTrain,
            yTrain,
            attrs,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out,
            bufftime=trainBuff
        )
    elif interfaceOpt == 0:  # directly train the model using dictionary variable
        master.train(masterDict)


# Train DI model
if 1 in Action:
    nDayLst = [1, 3]
    for nDay in nDayLst:
        # nDay: previous Nth day observation to integrate
        # update parameter "daObs" for data dictionary variable
        optData = default.update(default.optDataCamels, daObs=nDay)
        # define output folder for DI models
        out = os.path.join(rootOut, save_path, "All-DI" + str(nDay))
        masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)
        if interfaceOpt == 1:
            # optData['daObs'] != 0, load previous observation data to integrate
            sd = utils.time.t2dt(optData["tRange"][0]) - dt.timedelta(days=nDay)
            ed = utils.time.t2dt(optData["tRange"][1]) - dt.timedelta(days=nDay)
            dfdi = camels.DataframeCamels(subset=optData["subset"], tRange=[sd, ed])
            datatemp = dfdi.getDataObs(
                doNorm=False, rmNan=False, basinnorm=True
            )  # 'basinnorm=True': output = discharge/(area*mean_precip)
            # normalize data
            dadata = camels.transNormbyDic(datatemp, "runoff", statDict, toNorm=True)
            dadata[np.where(np.isnan(dadata))] = 0.0

            xIn = np.concatenate([xTrain, dadata], axis=2)
            nx = xIn.shape[-1] + attrs.shape[-1]  # update nx, nx = nx + nc
            ny = yTrain.shape[-1]
            # load model for training
            if torch.cuda.is_available():
                model = CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            else:
                model = CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            optModel = default.update(optModel, nx=nx, ny=ny)
            lossFun = RmseLoss()
            # update and write dictionary variable to out folder for logging and future testing
            masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)
            writeMasterFile(masterDict)
            # log statistics
            statFile = os.path.join(out, "statDict.json")
            with open(statFile, "w") as fp:
                json.dump(statDict, fp, indent=4)
            # train model
            model = trainModel(
                model,
                xIn,
                yTrain,
                attrs,
                lossFun,
                nEpoch=EPOCH,
                miniBatch=[BATCH_SIZE, RHO],
                saveEpoch=saveEPOCH,
                saveFolder=out,
            )
        elif interfaceOpt == 0:
            master.train(masterDict)

# Test models
if 2 in Action:
    testTrainBuff = True
    loadtrainBuff = 0
    TestEPOCH = 300  # choose the model to test after trained "TestEPOCH" epoches
    # generate a folder name list containing all the tested model output folders
    caseLst = ["All"]
    if optData["daObs"] > 0:
        nDayLst = [1, 3]  # which DI models to test: DI(1), DI(3)
        for nDay in nDayLst:
            caseLst.append("All-DI" + str(nDay))
    outLst = [
        os.path.join(rootOut, save_path, x) for x in caseLst
    ]
    # outLst = [os.path.join(rootOut, save_path)]
    # outLst includes all the directories to test
    subset = "All"  # 'All': use all the CAMELS gages to test; Or pass the gage list
    tRange = [19951001, 20101001]
    TestBuff = xTrain.shape[1] - loadtrainBuff    # Testing period
    testBatch = 15  # do batch forward to save GPU memory
    predLst = list()
    for out in outLst:
        if interfaceOpt == 1:  # use the more interpretable version interface
            # load testing data
            mDict = readMasterFile(out)
            optData = mDict["data"]
            df = camels.DataframeCamels(subset=subset, tRange=tRange,forType=optData['forType'])
            x = df.getDataTs(varLst=optData["varT"], doNorm=False, rmNan=False, flow_regime=flow_regime)
            obs = df.getDataObs(doNorm=False, rmNan=False, basinnorm=False, flow_regime=flow_regime)
            c = df.getDataConst(varLst=optData["varC"], doNorm=False, rmNan=False, flow_regime=flow_regime)

            # do normalization and remove nan
            # load the saved statDict to make sure using the same statistics as training data
            statFile = os.path.join(out, "statDict.json")
            with open(statFile, "r") as fp:
                statDict = json.load(fp)
            seriesvarLst = optData["varT"]
            attrLst = optData["varC"]
            attr_norm = camels.transNormbyDic(c, attrLst, statDict, toNorm=True, flow_regime=flow_regime)
            attr_norm[np.isnan(attr_norm)] = 0.0
            xTest = camels.transNormbyDic(x, seriesvarLst, statDict, toNorm=True, flow_regime=flow_regime)
            xTest[np.isnan(xTest)] = 0.0
            attrs = attr_norm

            if testTrainBuff is True:
                xTestBuff = xTrain[:, -TestBuff:, :]
                xTest = np.concatenate([xTestBuff, xTest], axis=1)

            if optData["daObs"] > 0:
                # optData['daObs'] != 0, load previous observation data to integrate
                nDay = optData["daObs"]
                sd = utils.time.t2dt(tRange[0]) - dt.timedelta(days=nDay)
                ed = utils.time.t2dt(tRange[1]) - dt.timedelta(days=nDay)
                dfdi = camels.DataframeCamels(subset=subset, tRange=[sd, ed])
                datatemp = dfdi.getDataObs(
                    doNorm=False, rmNan=False, basinnorm=True
                )  # 'basinnorm=True': output = discharge/(area*mean_precip)
                # normalize data
                dadata = camels.transNormbyDic(
                    datatemp, "runoff", statDict, toNorm=True
                )
                dadata[np.where(np.isnan(dadata))] = 0.0
                xIn = np.concatenate([xTest, dadata], axis=2)

            else:
                xIn = xTest

            # load and forward the model for testing
            testmodel = loadModel(out, epoch=TestEPOCH)
            filePathLst = master.master.namePred(
                out, tRange, "All", epoch=TestEPOCH
            )  # prepare the name of csv files to save testing results
            testModel(
                testmodel, xIn, c=attrs, batchSize=testBatch, filePathLst=filePathLst
            )
            # read out predictions
            # dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
            dataPred = np.ndarray([xTest.shape[0], xTest.shape[1], len(filePathLst)])

            for k in range(len(filePathLst)):
                filePath = filePathLst[k]
                dataPred[:, :, k] = pd.read_csv(
                    filePath, dtype=float, header=None
                ).values
            # transform back to the original observation
            temppred = camels.transNormbyDic(dataPred, "runoff", statDict, toNorm=False, flow_regime=flow_regime)
            pred = camels.basinNorm(temppred, subset, toNorm=False)

        elif interfaceOpt == 0:  # only for models trained by the pro interface
            df, pred, obs = master.test(
                out,
                tRange=tRange,
                subset=subset,
                batchSize=testBatch,
                basinnorm=True,
                epoch=TestEPOCH,
                reTest=True,
            )

        # change the units ft3/s to m3/s
        if testTrainBuff is True:
            obs = obs[:, 0:, :] * 0.0283168
        else:
            obs = obs[:, TestBuff:, :] * 0.0283168
        # obs = obs * 0.0283168
        pred = pred[:, TestBuff:, :] * 0.0283168
        # pred = pred * 0.0283168

        # prediction and obs to mm/day
        obs = camels.basinTrans(obs, subset)
        pred = camels.basinTrans(pred, subset)
        predLst.append(pred)  # the prediction list for all the models

    # calculate statistic metrics
    statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]

    # Show boxplots of the results
    plt.rcParams["font.size"] = 14
    keyLst = ["Bias", "NSE", "KGE", "FLV", "FHV", "AFLV", "AFHV", "highRMSE", "lowRMSE"]
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            data = statDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)
    print(
        "Bias (mm/day), NSE,KGE, FLV, FHV, AFLV, AFHV, highRMSE (mm/day), lowRMSE (mm/day)",
        np.nanmedian(dataBox[0][0]), np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]), np.nanmedian(dataBox[4][0]), np.nanmedian(dataBox[5][0]),
        np.nanmedian(dataBox[6][0]), np.nanmedian(dataBox[7][0]), np.nanmedian(dataBox[8][0])
    )
    labelname = ["LSTM"]
    if optData["daObs"] > 0:
        for nDay in nDayLst:
            labelname.append("DI(" + str(nDay) + ")")
    # xlabel = ["Bias ($\mathregular{m^3}$/s)", "NSE", "FLV(%)", "FHV(%)"]
    xlabel = keyLst
    fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(12, 5))
    fig.patch.set_facecolor("white")
    fig.show()
    plt.savefig(os.path.join(rootOut, save_path, "Boxplot.png"), dpi=500)

    # Plot timeseries and locations
    plt.rcParams["font.size"] = 12
    # get Camels gages info
    gageinfo = camels.gageDict
    gagelat = gageinfo["lat"]
    gagelon = gageinfo["lon"]
    # randomly select 7 gages to plot
    gageindex = np.random.randint(0, 671, size=7).tolist()
    plat = gagelat[gageindex]
    plon = gagelon[gageindex]
    t = utils.time.tRange2Array(tRange)
    fig, axes = plt.subplots(4, 2, figsize=(12, 10), constrained_layout=True)
    axes = axes.flat
    npred = 2  # plot the first two prediction: Base LSTM and DI(1)
    subtitle = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(k)", "(l)"]
    txt = ["a", "b", "c", "d", "e", "f", "g", "h", "k"]
    ylabel = "Flow rate ($\mathregular{m^3}$/s)"
    for k in range(len(gageindex)):
        iGrid = gageindex[k]
        yPlot = [obs[iGrid, :]]
        for y in predLst[0:npred]:
            yPlot.append(y[iGrid, :])
        # get the NSE value of LSTM and DI(1) model
        NSE_LSTM = str(round(statDictLst[0]["NSE"][iGrid], 2))
        if optData["daObs"] > 0:
            NSE_DI1 = str(round(statDictLst[1]["NSE"][iGrid], 2))
        else:
            NSE_DI1 = 'nil'
        # plot time series
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst="kbrmg",
            markerLst="---",
            legLst=["USGS", "LSTM: " + NSE_LSTM, "DI(1): " + NSE_DI1],
            title=subtitle[k],
            linespec=["-", ":", ":"],
            ylabel=ylabel,
        )
    # plot gage location
    plot.plotlocmap(
        plat,
        plon,
        ax=axes[-1],
        baclat=gagelat,
        baclon=gagelon,
        title=subtitle[-1],
        txtlabel=txt,
    )
    fig.patch.set_facecolor("white")
    fig.show()
    plt.savefig(os.path.join(rootOut, save_path, "Timeseries.png"), dpi=500)

    # Plot NSE spatial patterns
    gageinfo = camels.gageDict
    gagelat = gageinfo["lat"]
    gagelon = gageinfo["lon"]
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), constrained_layout=True)
    axs = axs.flat
    data = statDictLst[0]["NSE"]
    plot.plotMap(
        data,
        ax=axs[0],
        lat=gagelat,
        lon=gagelon,
        title="(a) LSTM",
        cRange=[0.0, 1.0],
        shape=None,
    )
    if optData["daObs"] > 0:
        data = statDictLst[1]["NSE"]
        plot.plotMap(
            data,
            ax=axs[1],
            lat=gagelat,
            lon=gagelon,
            title="(b) DI(1)",
            cRange=[0.0, 1.0],
            shape=None,
        )
        deltaNSE = statDictLst[1]["NSE"] - statDictLst[0]["NSE"]
        plot.plotMap(
            deltaNSE, ax=axs[2], lat=gagelat, lon=gagelon, title="(c) Delta NSE", shape=None
        )
    fig.show()
    plt.savefig(os.path.join(rootOut, save_path, "NSEPattern.png"), dpi=500)
