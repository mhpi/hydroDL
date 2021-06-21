import sys
sys.path.append('../')
from hydroDL import master
from hydroDL.master import default
from hydroDL.data import camels
from hydroDL.model import rnn, crit, train


import json
import os
import numpy as np
import torch
import random


# Options for different interface
interfaceOpt = 1
# ==1 default, the improved and more interpretable version. It's easier to see the data flow, model setup and training
# process. Recommended for most users.
# ==0 the original "pro" version we used to run heavy jobs for the paper. It was later improved for clarity to obtain option 1.
# Results are very similar for two options and have little difference in computational performance.

Action = [1, 2]
# Using Action options to control training different models
# 1: Train Base LSTM PUR Models without integrating any soft info
# 2: Train CNN-LSTM to integrate FDCs

# Hyperparameters
EPOCH = 300
BATCH_SIZE=100
RHO=365
HIDDENSIZE=256
saveEPOCH = 10 # save model for every "saveEPOCH" epochs
Ttrain=[19851001, 19951001] # training period
LCrange = [19851001, 19951001]

# Define root directory of database and output
# Modify this based on your own location of CAMELS dataset
# Following the data download instruction in README file, you should organize the folders like
# 'your/path/to/Camels/basin_timeseries_v1p2_metForcing_obsFlow' and 'your/path/to/Camels/camels_attributes_v2.0'
# Then 'rootDatabase' here should be 'your/path/to/Camels'
# You can also define the database directory in hydroDL/__init__.py by modifying pathCamels['DB'] variable

rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # Model output root directory

# define random seed
# seedid = [159654, 109958, 257886, 142365, 229837, 588859] # six seeds randomly generated using np.random.uniform
seedid = 159654
random.seed(seedid)
torch.manual_seed(seedid)
np.random.seed(seedid)
torch.cuda.manual_seed(seedid)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Fix seed for training, change it to have different runnings with different seeds
# We use the mean discharge of 6 runnings with different seeds to account for randomness


# directory to save training results
exp_name='PUR'
exp_disp='Testrun'
save_path = os.path.join(exp_name, exp_disp, str(seedid))

# Divide CAMELS dataset into 7 PUR regions
gageinfo = camels.gageDict
hucinfo = gageinfo['huc']
gageid = gageinfo['id']
# get the id list of each region
regionID = list()
regionNum = list()
regionDivide = [ [1,2], [3,6], [4,5,7], [9,10], [8,11,12,13], [14,15,16,18], [17] ] # seven regions
for ii in range(len(regionDivide)):
    tempcomb = regionDivide[ii]
    tempregid = list()
    for ih in tempcomb:
        tempid = gageid[hucinfo==ih].tolist()
        tempregid = tempregid + tempid
    regionID.append(tempregid)
    regionNum.append(len(tempregid))

# Only for interfaceOpt=0 using multiple GPUs, not used here
# cid = 0   # starting GPU id
# gnum = 6  # how many GPUs you have

# Region withheld as testing target. Take region 1 as an example.
# Change this to 1,2,..,7 to run models for all 7 PUR regions in CONUS.
testRegion = 1

iexp = testRegion - 1  # index
TestLS = regionID[iexp] # basin ID list for testing, should be withheld for training
TrainLS = list(set(gageid.tolist()) - set(TestLS)) # basin ID for training
gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

# prepare the training dataset
optData = default.optDataCamels
optData = default.update(optData, tRange=Ttrain, subset=TrainLS, lckernel=None, fdcopt=False)
climateList = camels.attrLstSel + ['p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq',
                                    'high_prec_dur','low_prec_freq','low_prec_dur']
# climateList = ['slope_mean', 'area_gages2', 'frac_forest', 'soil_porosity', 'max_water_content']
# climateList = []
optData = default.update(optData, varT=camels.forcingLst, varC= climateList)
# varT: forcing used for training   varC: attributes used for training

# The above controls what attributes used for training, change varC for input-selection-ensemble
# for 5 attributes model: climateList = ['slope_mean', 'area_gages2', 'frac_forest', 'soil_porosity', 'max_water_content']
# for no-attribute model: varC = []
# the input-selection ensemble represents using the mean prediction of full, 5-attr and no-attr models,
# in total the mean of 3(different attributes)*6(different random seeds) = 18 models

if interfaceOpt == 1:
# read data from CAMELS dataset
    df = camels.DataframeCamels(
        subset=optData['subset'], tRange=optData['tRange'])
    x = df.getDataTs(
        varLst=optData['varT'],
        doNorm=False,
        rmNan=False)
    y = df.getDataObs(
        doNorm=False,
        rmNan=False,
        basinnorm=True)
    # "basinnorm = True" will call camels.basinNorm() on the original discharge data. This will transform discharge
    # from ft3/s to mm/day and then divided by mean precip to be dimensionless. output = discharge/(area*mean_precip)
    c = df.getDataConst(
        varLst=optData['varC'],
        doNorm=False,
        rmNan=False)

    # process, do normalization and remove nan
    series_data = np.concatenate([x, y], axis=2)
    seriesvarLst = camels.forcingLst + ['runoff']
    # calculate statistics for normalization and save to a dictionary
    statDict = camels.getStatDic(attrLst=climateList, attrdata=c, seriesLst=seriesvarLst, seriesdata=series_data)
    # normalize
    attr_norm = camels.transNormbyDic(c, climateList, statDict, toNorm=True)
    attr_norm[np.isnan(attr_norm)] = 0.0
    series_norm = camels.transNormbyDic(series_data, seriesvarLst, statDict, toNorm=True)

    # prepare the inputs
    xTrain = series_norm[:,:,:-1] # forcing, not include obs
    xTrain[np.isnan(xTrain)]  = 0.0
    yTrain = np.expand_dims(series_norm[:,:,-1], 2)

    if attr_norm.size == 0: # [], no-attribute case
        attrs = None
        Nx = xTrain.shape[-1]
    else:
        # with attributes
        attrs=attr_norm
        Nx = xTrain.shape[-1] + attrs.shape[-1]
    Ny = yTrain.shape[-1]

# define loss function
optLoss = default.optLossRMSE
lossFun = crit.RmseLoss()
# configuration for training
optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seedid)
hucdic = 'Reg-'+str(iexp+1)+'-Num'+str(regionNum[iexp])

if 1 in Action:
# Train base LSTM PUR model
    out = os.path.join(rootOut, save_path, hucdic,'Reg-85-95-Sub-Full')
    # out = os.path.join(rootOut, save_path, hucdic,'Reg-85-95-Sub-5attr')
    # out = os.path.join(rootOut, save_path, hucdic,'Reg-85-95-Sub-Noattr')
    if not os.path.isdir(out):
        os.makedirs(out)
    # log training gage information
    gageFile = os.path.join(out, 'gage.json')
    with open(gageFile, 'w') as fp:
        json.dump(gageDic, fp, indent=4)
    # define model config
    optModel = default.update(default.optLstm, name='hydroDL.model.rnn.CudnnLstmModel', hiddenSize=HIDDENSIZE)

    if interfaceOpt == 1:
        # define, load and train model
        optModel = default.update(optModel, nx=Nx, ny=Ny)
        model = rnn.CudnnLstmModel(nx=optModel['nx'], ny=optModel['ny'], hiddenSize=optModel['hiddenSize'])
        # Wrap up all the training configurations to one dictionary in order to save into "out" folder
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # log statistics
        statFile = os.path.join(out, 'statDict.json')
        with open(statFile, 'w') as fp:
            json.dump(statDict, fp, indent=4)
        # Train the model
        trainedModel = train.trainModel(
            model,
            xTrain,
            yTrain,
            attrs,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out)

    if interfaceOpt == 0:
        # Only need to pass the wrapped configuration dict 'masterDict' for training
        # nx, ny will be automatically updated later
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.train(masterDict)

        ## Not used here.
        ## A potential way to run batch jobs simultaneously in background through multiple GPUs and Linux screens.
        ## To use this, must manually set the "pathCamels['DB']" in hydroDL/__init__.py as your own root path of CAMELS data.
        ## Use the following master.runTrain() instead of the above master.train().
        # master.runTrain(masterDict, cudaID=cid % gnum, screen='test-'+str(cid))
        # cid = cid + 1

if 2 in Action:
# Train CNN-LSTM PUR model to integrate FDCs
    # LCrange defines from which period to get synthetic FDC
    LCTstr = str(LCrange[0]) + '-' + str(LCrange[1])
    out = os.path.join(rootOut, save_path, hucdic, 'Reg-85-95-Sub-Full-FDC' + LCTstr)
    # out = os.path.join(rootOut, save_path, hucdic, 'Reg-85-95-Sub-5attr-FDC' + LCTstr)
    # out = os.path.join(rootOut, save_path, hucdic, 'Reg-85-95-Sub-Noattr-FDC' + LCTstr)
    if not os.path.isdir(out):
        os.makedirs(out)
    gageFile = os.path.join(out, 'gage.json')
    with open(gageFile, 'w') as fp:
        json.dump(gageDic, fp, indent=4)

    optData = default.update(default.optDataCamels, tRange=Ttrain, subset=TrainLS,
                             lckernel=LCrange, fdcopt=True)
    # define model
    convNKS = [(10, 5, 1), (5, 3, 3), (1, 1, 1)]
    # CNN parameters for 3 layers: [(Number of kernels 10,5,1), (kernel size 5,3,3), (stride 1,1,1)]
    optModel = default.update(default.optCnn1dLstm, name='hydroDL.model.rnn.CNN1dLCmodel',
                              hiddenSize=HIDDENSIZE, convNKS=convNKS, poolOpt=[2, 2, 1])  # use CNN-LSTM model

    if interfaceOpt == 1:
        # load data and create synthetic FDCs as inputs
        dffdc = camels.DataframeCamels(subset=optData['subset'], tRange=optData['lckernel'])
        datatemp = dffdc.getDataObs(
            doNorm=False, rmNan=False, basinnorm=True)
        # normalize data
        dadata = camels.transNormbyDic(datatemp, 'runoff', statDict, toNorm=True)
        dadata = np.squeeze(dadata)  # dim Nbasin*Nday
        fdcdata = master.master.calFDC(dadata)
        print('FDC was calculated and used!')
        xIn = (xTrain, fdcdata)

        # load model
        Nobs = xIn[1].shape[-1]
        optModel = default.update(optModel, nx=Nx, ny=Ny, nobs=Nobs) # update input dims
        convpara = optModel['convNKS']

        model = rnn.CNN1dLCmodel(
            nx=optModel['nx'],
            ny=optModel['ny'],
            nobs=optModel['nobs'],
            hiddenSize=optModel['hiddenSize'],
            nkernel=convpara[0],
            kernelSize=convpara[1],
            stride=convpara[2],
            poolOpt=optModel['poolOpt'])
        print('CNN1d Local calibartion Kernel is used!')

        # Wrap up all the training configurations to one dictionary in order to save into "out" folder
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.writeMasterFile(masterDict)
        # log statistics
        statFile = os.path.join(out, 'statDict.json')
        with open(statFile, 'w') as fp:
            json.dump(statDict, fp, indent=4)

        # Train the model
        trainedModel = train.trainModel(
            model,
            xIn,  # need to well defined
            yTrain,
            attrs,
            lossFun,
            nEpoch=EPOCH,
            miniBatch=[BATCH_SIZE, RHO],
            saveEpoch=saveEPOCH,
            saveFolder=out)

    if interfaceOpt == 0:
        # Only need to pass the wrapped configuration 'masterDict' for training
        # nx, ny, nobs will be automatically updated later
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.train(masterDict)  # train model

        # master.runTrain(masterDict, cudaID=cid % gnum, screen='test-'+str(cid))
        # cid = cid + 1
