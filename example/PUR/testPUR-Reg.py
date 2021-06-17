import sys
sys.path.append('../')
from hydroDL import master
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels
from hydroDL.master import loadModel
from hydroDL.model import train

import numpy as np
import pandas as pd
import torch
import json
import os
import random


interfaceOpt = 1
# set it the same as which you used to train your model
# ==1 default, the recommended and more interpretable version
# ==0 the "pro" version

# Define root directory of database and output
# Modify this based on your own location of CAMELS dataset and saved models

rootDatabase = os.path.join(os.path.sep, 'scratch', 'Camels')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict

rootOut = os.path.join(os.path.sep, 'data', 'rnnStreamflow')  # Model output root directory

# The directory you defined in training to save the model under the above rootOut
exp_name='PUR'
exp_disp='Testrun'
save_path = os.path.join(rootOut, exp_name, exp_disp)

random.seed(159654)
# this random is only used for fractional FDC scenarios and
# to sample which basins in the target region have FDCs.

# same as training, get the 7 regions basin ID for testing
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


# test trained models
testgpuid = 0 # which gpu used for testing
torch.cuda.set_device(testgpuid)

# # test models ran by different random seeds.
# # final reported results are the mean of predictions from different seeds
# seedid = [159654, 109958, 257886, 142365, 229837, 588859]
seedid = [159654] # take this seed as an example

gageinfo = camels.gageDict
gageid = gageinfo['id']
subName = 'Sub'
tRange = [19951001, 20051001] # testing periods
testEpoch = 300 # test the saved model after trained how many epochs
FDCMig = True  # option for Fractional FDC experiments, migrating 1/3 or 1/10 FDCs to all basins in the target region
FDCrange = [19851001, 19951001]
FDCstr = str(FDCrange[0])+'-'+str(FDCrange[1])

expName = 'Full' # testing full-attribute model as an example. Could be 'Full', 'Noattr', '5attr'
caseLst = ['-'.join(['Reg-85-95', subName, expName])] # PUR: 'Reg-85-95-Sub-Full'
caseLst.append('-'.join(['Reg-85-95', subName, expName, 'FDC']) + FDCstr) # PUR with FDC: 'Reg-85-95-Sub-Full-FDC' + LCTstr

# samFrac = [1/3, 1/10] # Fractional FDCs available in the target region
samFrac = [1/3]
migOptLst = [False, False] # the list indicating if it's migration experiment for each one in caseLst
if FDCMig == True:
    for im in range(len(samFrac)):
        caseLst.append('-'.join(['Reg-85-95', subName, expName, 'FDC']) + FDCstr)
        migOptLst.append(True)
# caseLst summarizes all the experiment directories needed for testing

# Get the randomly sampled basin ID in the target region that have FDCs and save to file for the future use
# the sampled basin should be the same for different ensemble members
# this part only needs to be ran once to generate sample file. The following section will read the saved file
sampleIndLst = list()
for ir in range(len(regionID)):
    testBasin = regionID[ir]
    sampNum = round(1/3 * len(testBasin))  # or 1/10
    sampleIn = random.sample(range(0, len(testBasin)), sampNum)
    sampleIndLst.append(sampleIn)
samLstFile = os.path.join(save_path, 'samp103Lst.json')  # or 'samp110Lst.json'
with open(samLstFile, 'w') as fp:
    json.dump(sampleIndLst, fp, indent=4)


# Load the sample Ind
indfileLst = ['samp103Lst.json']  # ['samp103Lst.json', 'samp110Lst.json']
sampleInLstAll = list()
for ii in range(len(indfileLst)):
    samLstFile = os.path.join(save_path, indfileLst[ii])
    with open(samLstFile, 'r') as fp:
        tempind = json.load(fp)
    sampleInLstAll.append(tempind)

for iEns in range(len(seedid)): #test trained models with different seeds
    tempseed = seedid[iEns]
    predtempLst = []
    regcount = 0

    # for iT in range(len(regionID)): # test all the 7 regions
    for iT in range(0, 1): # take region 1 as an example
        testBasin = regionID[iT] # testing basins
        testInd = [gageid.tolist().index(x) for x in testBasin]
        trainBasin = list(set(gageid.tolist()) - set(testBasin))
        trainInd = [gageid.tolist().index(x) for x in trainBasin]
        testregdic = 'Reg-'+str(iT+1)+'-Num'+str(regionNum[iT])

        # Migrate FDC for fractional experiment based on the nearest distance
        if FDCMig == True:
            FDCList = []
            testlat = gageinfo['lat'][testInd]
            testlon = gageinfo['lon'][testInd]
            for iF in range(len(samFrac)):
                sampleInLst = sampleInLstAll[iF]
                samplelat = testlat[sampleInLst[iT]]
                samplelon = testlon[sampleInLst[iT]]
                nearID = list()
                # calculate distances to the gages with FDC available
                # and identify using the FDC of which gage for each test basin
                for ii in range(len(testlat)):
                    dist = np.sqrt((samplelat-testlat[ii])**2 + (samplelon-testlon[ii])**2)
                    nearID.append(np.argmin(dist))
                FDCLS = gageid[testInd][sampleInLst[iT]][nearID].tolist()
                FDCList.append(FDCLS)

        outLst = [os.path.join(save_path, str(tempseed), testregdic, x) for x in caseLst]
        # all the directories to test in this list

        icount = 0
        imig = 0
        for out in outLst:
            # testing sequence: LSTM, LSTM with FDC, LSTM with fractional FDC migration
            if interfaceOpt == 1:
                # load testing data
                mDict = master.readMasterFile(out)
                optData = mDict['data']
                df = camels.DataframeCamels(
                    subset=testBasin, tRange=tRange)
                x = df.getDataTs(
                    varLst=optData['varT'],
                    doNorm=False,
                    rmNan=False)
                obs = df.getDataObs(
                    doNorm=False,
                    rmNan=False,
                    basinnorm=False)
                c = df.getDataConst(
                    varLst=optData['varC'],
                    doNorm=False,
                    rmNan=False)

                # do normalization and remove nan
                # load the saved statDict
                statFile = os.path.join(out, 'statDict.json')
                with open(statFile, 'r') as fp:
                    statDict = json.load(fp)
                seriesvarLst = optData['varT']
                climateList = optData['varC']
                attr_norm = camels.transNormbyDic(c, climateList, statDict, toNorm=True)
                attr_norm[np.isnan(attr_norm)] = 0.0
                xTest = camels.transNormbyDic(x, seriesvarLst, statDict, toNorm=True)
                xTest[np.isnan(xTest)] = 0.0
                if attr_norm.size == 0:  # [], no-attribute case
                    attrs = None
                else:
                    attrs = attr_norm


                if optData['lckernel'] is not None:
                    if migOptLst[icount] is True:
                        # the case migrating FDCs
                        dffdc = camels.DataframeCamels(subset=FDCList[imig], tRange=optData['lckernel'])
                        imig = imig+1
                    else:
                        dffdc = camels.DataframeCamels(subset=testBasin, tRange=optData['lckernel'])
                    datatemp = dffdc.getDataObs(
                        doNorm=False, rmNan=False, basinnorm=True)
                    # normalize data
                    dadata = camels.transNormbyDic(datatemp, 'runoff', statDict, toNorm=True)
                    dadata = np.squeeze(dadata)  # dim Ngrid*Nday
                    fdcdata = master.master.calFDC(dadata)
                    print('FDC was calculated and used!')
                    xIn = (xTest, fdcdata)
                else:
                    xIn = xTest

                # load and forward the model for testing
                testmodel = loadModel(out, epoch=testEpoch)
                filePathLst = master.master.namePred(
                    out, tRange, 'All', epoch=testEpoch) # prepare the name of csv files to save testing results
                train.testModel(
                    testmodel, xIn, c=attrs, filePathLst=filePathLst)
                # read out predictions
                dataPred = np.ndarray([obs.shape[0], obs.shape[1], len(filePathLst)])
                for k in range(len(filePathLst)):
                    filePath = filePathLst[k]
                    dataPred[:, :, k] = pd.read_csv(
                        filePath, dtype=np.float, header=None).values
                # transform back to the original observation
                temppred = camels.transNormbyDic(dataPred, 'runoff', statDict, toNorm=False)
                pred = camels.basinNorm(temppred, np.array(testBasin), toNorm=False)

            elif interfaceOpt == 0:
                if migOptLst[icount] is True:
                    # for FDC migration case
                    df, pred, obs = master.test(out, tRange=tRange, subset=testBasin, basinnorm=True, epoch=testEpoch,
                                                reTest=True, FDCgage=FDCList[imig])
                    imig = imig + 1
                else:
                    # for other ordinary cases
                    df, pred, obs = master.test(out, tRange=tRange, subset=testBasin, basinnorm=True, epoch=testEpoch,
                                                reTest=True)

            ## change the units ft3/s to m3/s
            obs = obs*0.0283168
            pred = pred*0.0283168

            # concatenate results in different regions to one array
            # and save the array of different experiments to a list
            if regcount == 0:
                predtempLst.append(pred)
            else:
                predtempLst[icount] = np.concatenate([predtempLst[icount], pred], axis=0)
            icount = icount + 1
        if regcount == 0:
            obsAll = obs
        else:
            obsAll = np.concatenate([obsAll, obs], axis=0)
        regcount = regcount+1

    # concatenate results of different seeds to the third dim of array
    if iEns == 0:
        predLst = predtempLst
    else:
        for ii in range(len(outLst)):
            predLst[ii] = np.concatenate([predLst[ii], predtempLst[ii]], axis=2)
    # predLst: List of all experiments with shape: Ntime*Nbasin*Nensemble

# get the ensemble mean from simulations of different seeds
ensLst = []
for ii in range(len(outLst)):
    temp = np.nanmean(predLst[ii], axis=2, keepdims=True)
    ensLst.append(temp)


# plot boxplots for different experiments
statDictLst = [stat.statError(x.squeeze(), obsAll.squeeze()) for x in ensLst]
keyLst=['NSE', 'KGE'] # which metric to show
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(statDictLst)):
        data = statDictLst[k][statStr]
        data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)

plt.rcParams['font.size'] = 14
labelname = ['PUR', 'PUR-FDC', 'PUR-1/3FDC']
xlabel = ['NSE', 'KGE']
fig = plot.plotBoxFig(dataBox, xlabel, labelname, sharey=False, figsize=(6, 5))
fig.patch.set_facecolor('white')
fig.show()

# save evaluation results
outpath = os.path.join(save_path, 'TestResults', expName)
if not os.path.isdir(outpath):
    os.makedirs(outpath)

EnsEvaFile = os.path.join(outpath, 'EnsEva'+str(testEpoch)+'.npy')
np.save(EnsEvaFile, statDictLst)

obsFile = os.path.join(outpath, 'obs.npy')
np.save(obsFile, obsAll)

predFile = os.path.join(outpath, 'pred'+str(testEpoch)+'.npy')
np.save(predFile, predLst)
