import sys
sys.path.append('../')
from hydroDL import master
from hydroDL.master import default
from hydroDL.data import camels


import json
import os


Action = [1, 2]
# Using Action options to control training different models
# 1: Train Base LSTM PUR Models without integrating any soft info
# 2: Train CNN-LSTM to integrate FDCs

# Hyperparameters
EPOCH = 300
BATCH_SIZE=100
RHO=365
HIDDENSIZE=256
saveEPOCH = 20 # save model for every "saveEPOCH" epochs
Ttrain=[19851001, 19951001] # training period
LCrange = [19851001, 19951001]

# Define root directory of database and output
# Modify this based on your own location of CAMELS dataset
# You can also define the database directory in hydroDL/__init__.py by modifying pathCamels variable
rootDatabase = os.path.join(os.path.sep, 'mnt', 'sdb', 'Data', 'Camels')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnStreamflow')  # Model output root directory


# define random seed
# seedid = [159654, 109958, 257886, 142365, 229837, 588859]
seedid = 588859
# seed for training, change it to have different runnings with different seeds. We use the mean discharge of 6 runnings
# to account for randomness

# directory to save training results
exp_name='PUR'
exp_disp='Firstrun'
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

# cid = 0   # starting GPU id, not used here
# gnum = 6  # how many GPUs you have

# for iHUC in range(1, len(regionID)): # loop through 7 regions, once with one region withheld as testing target
for iHUC in range(1, 2): #  Take region 1 as an example
    iexp = iHUC - 1  #index
    TestLS = regionID[iexp] # basin ID list for testing, shoud be withheld for training
    TrainLS = list(set(gageid.tolist()) - set(TestLS)) # basin ID for training
    gageDic = {'TrainID': TrainLS, 'TestID': TestLS}

    # prepare the training dataset
    optData = default.optDataCamels
    optData = default.update(optData, tRange=Ttrain, subset=TrainLS, lckernel=None, fdcopt=False)
    climateList =  camels.attrLstSel + ['p_mean','pet_mean','p_seasonality','frac_snow','aridity','high_prec_freq',
                                        'high_prec_dur','low_prec_freq','low_prec_dur']
    # climateList = ['slope_mean', 'area_gages2', 'frac_forest', 'soil_porosity', 'max_water_content']
    optData = default.update(optData, varC= climateList)
    # optData = default.update(optData, varC= [])

    # The above controls what attributes used for training, change varC for input-selection-ensemble
    # for 5 attributes model: climateList = ['slope_mean', 'area_gages2', 'frac_forest', 'soil_porosity', 'max_water_content']
    # for no-attribute model: directly set varC = []
    # the input-selection ensemble represents using the mean prediction of full, 5-attr and no-attr models

    optModel = default.update(default.optLstm, hiddenSize=HIDDENSIZE)
    optLoss = default.optLossRMSE
    optTrain = default.update(default.optTrainCamels, miniBatch=[BATCH_SIZE, RHO], nEpoch=EPOCH, saveEpoch=saveEPOCH, seed=seedid)
    hucdic = 'Reg-'+str(iexp+1)+'-Num'+str(regionNum[iexp])

    if 1 in Action:
    # Train base LSTM PUR model
        out = os.path.join(rootOut, save_path, hucdic,'Reg-85-95-Sub-Full')
        # out = os.path.join(rootOut, save_path, hucdic,'Reg-85-95-Sub-5attr')
        # out = os.path.join(rootOut, save_path, hucdic,'Reg-85-95-Sub-Noattr')
        if not os.path.isdir(out):
            os.makedirs(out)
        gageFile = os.path.join(out, 'gage.json')
        with open(gageFile, 'w') as fp:
            json.dump(gageDic, fp, indent=4)
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.train(masterDict) # train model

        ## A potential way to run batch jobs through multiple GPUs and Linux screens
        ## Need to define database path "pathCamels" well in hydroDL/__init__.py
        ## not used here
        # master.runTrain(masterDict, cudaID=cid % gnum, screen='test-'+str(cid))
        # cid = cid + 1



    if 2 in Action:
    # Train CNN-LSTM PUR model to integrate FDCs
        optData = default.update(default.optDataCamels, tRange=Ttrain, subset=TrainLS,
                                 lckernel=LCrange, fdcopt=True)
        # LCrange defines from which period to get synthetic FDC
        convNKS = [(10, 5, 1), (5, 3, 3), (1, 1, 1)] # CNN parameters
        optModel = default.update(default.optCnn1dLstm, name='hydroDL.model.rnn.CNN1dLCmodel',
                                  hiddenSize=HIDDENSIZE, convNKS=convNKS, poolOpt=[2,2,1]) # use CNN-LSTM model
        LCTstr = str(LCrange[0]) + '-' + str(LCrange[1])
        out = os.path.join(rootOut, save_path, hucdic, 'Reg-85-95-Sub-Full-FDC' + LCTstr)
        # out = os.path.join(rootOut, save_path, hucdic, 'Reg-85-95-Sub-5attr-FDC' + LCTstr)
        # out = os.path.join(rootOut, save_path, hucdic, 'Reg-85-95-Sub-Noattr-FDC' + LCTstr)
        if not os.path.isdir(out):
            os.makedirs(out)
        gageFile = os.path.join(out, 'gage.json')
        with open(gageFile, 'w') as fp:
            json.dump(gageDic, fp, indent=4)
        masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
        master.train(masterDict)  # train model

        # master.runTrain(masterDict, cudaID = cid % gnum, screen='test-LC' + str(cid))
        # cid = cid + 1


