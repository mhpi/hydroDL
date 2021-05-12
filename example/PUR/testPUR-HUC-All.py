import sys
sys.path.append('../')
from hydroDL import master, utils
from hydroDL.post import plot, stat
import matplotlib.pyplot as plt
from hydroDL.data import camels

import numpy as np
import os
import torch
from collections import OrderedDict
import json
import os
import random

# Define root directory of database and output
# Modify this based on your own location of CAMELS dataset and saved models
rootDatabase = os.path.join(os.path.sep, 'mnt', 'sdb', 'Data', 'Camels')  # CAMELS dataset root directory
camels.initcamels(rootDatabase)  # initialize three camels module-scope variables in camels.py: dirDB, gageDict, statDict
rootOut = os.path.join(os.path.sep, 'mnt', 'sdb', 'rnnStreamflow')  # Model output root directory

# The directory you defined in training to save the model under the above rootOut
exp_name='PUR'
exp_disp='Firstrun'
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


# test original model
testgpuid = 0 # which gpu used for testing
torch.cuda.set_device(testgpuid)
# seedid = [159654, 109958, 257886, 142365, 229837, 588859] # test simulations ran by different seeds
seedid = [588859] # take this seed as an example

gageinfo = camels.gageDict
gageid = gageinfo['id']
subName = 'Sub'
tRange = [19951001, 20051001] # testing periods
testEpoch = 300 # test the saved model after trained how many epochs
FDCMig = True  # option for Fractional FDC experiments, migrating 1/3 or 1/10 FDCs to all basins in the target region
FDCrange = [19851001, 19951001]
FDCstr = str(FDCrange[0])+'-'+str(FDCrange[1])

expName = 'Full' # using full-attribute model as an example, 'Full', 'Noattr', '5attr'
caseLst = ['-'.join(['Reg-85-95', subName, expName])] # PUR: 'Reg-85-95-Sub-Full'
caseLst.append('-'.join(['Reg-85-95', subName, expName, 'FDC']) + FDCstr) # PUR with FDC: 'Reg-85-95-Sub-Full-FDC' + LCTstr

# samFrac = [1/3, 1/10] # Fractional FDCs available in the target region
samFrac = [1/3]

if FDCMig == True:
    for im in range(len(samFrac)):
        caseLst.append('-'.join(['Reg-85-95', subName, expName, 'FDC']) + FDCstr)
# caseLst summarizes all the experiment directories needed for testing

# Get the random sampled basin ID in the target region that have FDCs and save to file for the future use
# the sampled basin should be the same for different ensembles
# only need to be ran once to generate sample file
sampleIndLst = list()
for ir in range(len(regionID)):
    testBasin = regionID[ir]
    sampNum = round(1/3 * len(testBasin))  # or 1/10
    sampleIn = random.sample(range(0, len(testBasin)), sampNum)
    sampleIndLst.append(sampleIn)
samLstFile = os.path.join(save_path, 'samp103Lst.json')
with open(samLstFile, 'w') as fp:
    json.dump(sampleIndLst, fp, indent=4)

# Load the sample Ind
indfileLst = ['samp103Lst.json']
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
        testBasin = regionID[iT]
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
                # calculate distances to the gages with FDC availble
                # and identify using the FDC of which gage for each basin
                for ii in range(len(testlat)):
                    dist = np.sqrt((samplelat-testlat[ii])**2 + (samplelon-testlon[ii])**2)
                    nearID.append(np.argmin(dist))
                FDCLS = gageid[testInd][sampleInLst[iT]][nearID].tolist()
                FDCList.append(FDCLS)

        outLst = [os.path.join(save_path, str(tempseed), testregdic, x) for x in caseLst]
        # all the directories to test in this list

        subset = testBasin # testing basins
        icount = 1
        for out in outLst:
            # testing sequence: LSTM, LSTM with FDC, LSTM with fractional FDC migration
            if FDCMig == True and icount > 2:
                # FDC migrate testing always in the last test.
                df, pred, obs = master.test(out, tRange=tRange, subset=subset, basinnorm=True, epoch=testEpoch,
                                            reTest=True, FDCgage=FDCList[icount-3])
            else:
                df, pred, obs = master.test(out, tRange=tRange, subset=subset, basinnorm=True, epoch=testEpoch,
                                            reTest=True)

            ## change the units ft3/s to m3/s
            obs = obs*0.0283168
            pred = pred*0.0283168

            # concatenate results in different regions to one array
            # and save the array of different experiments to a list
            if regcount == 0:
                predtempLst.append(pred)
            else:
                predtempLst[icount-1] = np.concatenate([predtempLst[icount-1], pred], axis=0)
            icount = icount + 1
        if regcount == 0:
            obsAll = obs
        else:
            obsAll = np.concatenate([obsAll, obs], axis=0)
        regcount = regcount+1

    # concatenate results of different seeds to the third dim
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
outpath = os.path.join(save_path, 'TestResults')
if not os.path.isdir(outpath):
    os.makedirs(outpath)

EnsEvaFile = os.path.join(outpath, 'EnsEva'+str(testEpoch)+'.npy')
np.save(EnsEvaFile, statDictLst)

obsFile = os.path.join(outpath, 'obs.npy')
np.save(obsFile, obsAll)

predFile = os.path.join(outpath, 'pred'+str(testEpoch)+'.npy')
np.save(predFile, predLst)
