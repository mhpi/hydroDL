import sys
sys.path.append('../')
from hydroDL import master, utils
from hydroDL.master import default
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train

import numpy as np
import os
import torch

# Options for different interface
interfaceOpt = 1
# ==1 is the more interpretable version, explicitly load data, model and loss, and train the model.
# ==0 is the "pro" version, efficiently train different models based on the defined dictionary variables.
# the results are identical.

# Options for training and testing
# 0: train base model without DI
# 1: train DI model
# 0,1: do both at the same time
# 2: test trained models
Action = [0,1]
# gpuid = 0
# torch.cuda.set_device(gpuid)

# Set hyperparameters
EPOCH = 300
BATCH_SIZE = 100
RHO = 365
HIDDENSIZE = 256
saveEPOCH = 20 # save model for every "saveEPOCH" epochs
Ttrain = [19851001, 19951001]  # Training period
seedid = 111111     # fix the random seed to make reproducible, use 111111 as example

# Define root directory of database and output
# Modify this based on your own location
rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory: /scratch/Camels
rootOut = os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # Model output root directory: /data/rnnStreamflow
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

# Define all the configurations into dictionary variables
# three purposes using these dictionaries. 1. saved as configuration logging file. 2. for future testing. 3. can also
# be used to directly train the model when interfaceOpt == 0
# define dataset
optData = default.optDataCamels
optData = default.update(optData, tRange=Ttrain)  # Update the training period
# define model and update parameters
if torch.cuda.is_available():
    optModel = default.optLstm
else:
    optModel = default.update(
        default.optLstm,
        name='hydroDL.model.rnn.CpuLstmModel')
optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
# define loss function
optLoss = default.optLossRMSE
# define training options
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seedid)
# define output folder for model results
exp_name = 'CAMELSDemo'
exp_disp = 'ReleaseRun'
save_path = os.path.join(exp_name, exp_disp, \
            'epochs{}_batch{}_rho{}_hiddensize{}_Tstart{}_Tend{}'.format(optTrain['nEpoch'], optTrain['miniBatch'][0],
                                                                          optTrain['miniBatch'][1],
                                                                          optModel['hiddenSize'],
                                                                          optData['tRange'][0], optData['tRange'][1]))
out = os.path.join(rootOut, save_path, 'All-85-95') # output folder to save results
# Wrap up all the training configurations to one dictionary in order to save into "out" folder
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

# Train the base model without data integration
if 0 in Action:
    if interfaceOpt == 1:  # use the more interpretable version interface
        # load data
        df, x, y, c = master.loadData(optData)  # df: CAMELS dataframe; x: forcings; y: streamflow obs; c:attributes
        # main outputs of this step are numpy ndArrays: x[nb,nt,nx], y[nb,nt, ny], c[nb,nc]
        # nb: number of basins, nt: number of time steps (in Ttrain), nx: number of time-dependent forcing variables
        # ny: number of target variables, nc: number of constant attributes
        nx = x.shape[-1] + c.shape[-1]  # update nx, nx = nx + nc
        ny = y.shape[-1]
        # load model for training
        if torch.cuda.is_available():
            model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        else:
            model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
        optModel = default.update(optModel, nx=nx, ny=ny)
        # the loaded model should be consistent with the 'name' in optModel Dict above for logging purpose
        lossFun = crit.RmseLoss()
        # the loaded loss should be consistent with the 'name' in optLoss Dict above for logging purpose
        # update and write the dictionary variable to out folder for logging and future testing
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # train model
        model = train.trainModel(
            model,
            x,
            y,
            c,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out)
    elif interfaceOpt==0: # directly train the model using dictionary variable
        master.train(masterDict)


# Train DI model
if 1 in Action:
    nDayLst = [1,3]
    for nDay in nDayLst:
        # nDay: previous Nth day observation to integrate
        # update parameter "daObs" for data dictionary variable
        optData = default.update(default.optDataCamels, daObs=nDay)
        # define output folder for DI models
        out = os.path.join(rootOut, save_path, 'All-85-95-DI' + str(nDay))
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        if interfaceOpt==1:
            # load data
            df, x, y, c = master.loadData(optData)
            # optData['daObs'] != 0, return a tuple to x, x[0]:forcings x[1]: integrated observations
            x = np.concatenate([x[0], x[1]], axis=2)
            nx = x.shape[-1] + c.shape[-1]
            ny = y.shape[-1]
            # load model for training
            if torch.cuda.is_available():
                model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            else:
                model = rnn.CpuLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)
            optModel = default.update(optModel, nx=nx, ny=ny)
            lossFun = crit.RmseLoss()
            # update and write dictionary variable to out folder for logging and future testing
            masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
            master.writeMasterFile(masterDict)
            # train model
            model = train.trainModel(
                model,
                x,
                y,
                c,
                lossFun,
                nEpoch=EPOCH,
                miniBatch=[BATCH_SIZE, RHO],
                saveEpoch=saveEPOCH,
                saveFolder=out)
        elif interfaceOpt==0:
            master.train(masterDict)

# Test models
if 2 in Action:
    TestEPOCH = 300 # choose the model to test after trained "TestEPOCH" epoches
    # generate a folder name list containing all the tested model output folders
    caseLst = ['All-85-95']
    nDayLst = [1, 3]
    for nDay in nDayLst:
        caseLst.append('All-85-95-DI' + str(nDay))
    outLst = [os.path.join(rootOut, save_path, x) for x in caseLst]
    subset = 'All'  # 'All': use all the CAMELS gages to test; Or pass the gage list
    tRange = [19951001, 20051001]  # Testing period
    predLst = list()
    for out in outLst:
        df, pred, obs = master.test(out, tRange=tRange, subset=subset, basinnorm=True, epoch=TestEPOCH, reTest=True)
        # change the units ft3/s to m3/s
        obs = obs * 0.0283168
        pred = pred * 0.0283168
        predLst.append(pred) # the prediction list for all the models

    # calculate statistic metrics
    statDictLst = [stat.statError(x.squeeze(), obs.squeeze()) for x in predLst]

    # Show boxplots of the results
    plt.rcParams['font.size'] = 14
    keyLst = ['Bias', 'NSE', 'FLV', 'FHV']
    dataBox = list()
    for iS in range(len(keyLst)):
        statStr = keyLst[iS]
        temp = list()
        for k in range(len(statDictLst)):
            data = statDictLst[k][statStr]
            data = data[~np.isnan(data)]
            temp.append(data)
        dataBox.append(temp)
    labelname = ['LSTM']
    for nDay in nDayLst:
        labelname.append('DI(' + str(nDay) + ')')
    xlabel = ['Bias ($\mathregular{m^3}$/s)', 'NSE', 'FLV(%)', 'FHV(%)']
    fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(12, 5))
    fig.patch.set_facecolor('white')
    fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "Boxplot.png"), dpi=500)

    # Plot timeseries and locations
    plt.rcParams['font.size'] = 12
    # get Camels gages info
    gageinfo = camels.gageDict
    gagelat = gageinfo['lat']
    gagelon = gageinfo['lon']
    # randomly select 7 gages to plot
    gageindex = np.random.randint(0, 671, size=7).tolist()
    plat = gagelat[gageindex]
    plon = gagelon[gageindex]
    t = utils.time.tRange2Array(tRange)
    fig, axes = plt.subplots(4,2, figsize=(12,10), constrained_layout=True)
    axes = axes.flat
    npred = 2  # plot the first two prediction: Base LSTM and DI(1)
    subtitle = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)', '(g)', '(h)', '(k)', '(l)']
    txt = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'k']
    ylabel = 'Flow rate ($\mathregular{m^3}$/s)'
    for k in range(len(gageindex)):
        iGrid = gageindex[k]
        yPlot = [obs[iGrid, :]]
        for y in predLst[0:npred]:
            yPlot.append(y[iGrid, :])
        # get the NSE value of LSTM and DI(1) model
        NSE_LSTM = str(round(statDictLst[0]['NSE'][iGrid], 2))
        NSE_DI1 = str(round(statDictLst[1]['NSE'][iGrid], 2))
        # plot time series
        plot.plotTS(
            t,
            yPlot,
            ax=axes[k],
            cLst='kbrmg',
            markerLst='---',
            legLst=['USGS', 'LSTM: '+NSE_LSTM, 'DI(1): '+NSE_DI1], title=subtitle[k], linespec=['-',':',':'], ylabel=ylabel)
    # plot gage location
    plot.plotlocmap(plat, plon, ax=axes[-1], baclat=gagelat, baclon=gagelon, title=subtitle[-1], txtlabel=txt)
    fig.patch.set_facecolor('white')
    fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "/Timeseries.png"), dpi=500)

    # Plot NSE spatial patterns
    gageinfo = camels.gageDict
    gagelat = gageinfo['lat']
    gagelon = gageinfo['lon']
    nDayLst = [1, 3]
    fig, axs = plt.subplots(3,1, figsize=(8,8), constrained_layout=True)
    axs = axs.flat
    data = statDictLst[0]['NSE']
    plot.plotMap(data, ax=axs[0], lat=gagelat, lon=gagelon, title='(a) LSTM', cRange=[0.0, 1.0], shape=None)
    data = statDictLst[1]['NSE']
    plot.plotMap(data, ax=axs[1], lat=gagelat, lon=gagelon, title='(b) DI(1)', cRange=[0.0, 1.0], shape=None)
    deltaNSE = statDictLst[1]['NSE'] - statDictLst[0]['NSE']
    plot.plotMap(deltaNSE, ax=axs[2], lat=gagelat, lon=gagelon, title='(c) Delta NSE', shape=None)
    fig.show()
    # plt.savefig(os.path.join(rootOut, save_path, "/NSEPattern.png"), dpi=500)
