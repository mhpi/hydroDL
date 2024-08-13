import pickle
import pandas as pd
import numpy as np
import os
import json
import random
import torch
import sys
sys.path.append('../../')
from hydroDL.model import test

from hydroDL.data import scale
from hydroDL.master.master import loadModel
from hydroDL.post import stat

randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

traingpuid = 1
torch.cuda.set_device(traingpuid)


rootDB_s=f'/mnt/sdb/yxs275/check_code/SWE_data/'
modelpath = "/mnt/sdb/yxs275/snow_hydroDL/output/"

DateRange=['2001-01-01', '2019-12-31']
testDateRange=['2016-01-01', '2019-12-31']


var_x_list =  ['pr_gridMET', 'tmmn_gridMET', 'tmmx_gridMET', 'srad_gridMET', 'vs_gridMET', 'th_gridMET',
                 'sph_gridMET', 'rmin_gridMET', 'rmax_gridMET']

attributeLst = ['lat','mean_elev', 'mean_slope', 'aspect',
                  'dom_land_cover', 'dom_land_cover_frac', 'forest_fraction']

targetLst = ['SWE']

##Hyperparameters
EPOCH = 600
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 50
HIDDENSIZE = 256
trainBuff = 365


### Read data:
# load forcing and target data
time_range = pd.date_range(DateRange[0], DateRange[-1], freq='d')
startyear = time_range[0].year
endyear = time_range[-1].year
for year in range(startyear,endyear+1):
    for fid, foring_ in enumerate(var_x_list+targetLst):

        foring_data = pd.read_csv(rootDB_s+'/'+str(year)+'/' + foring_ + '.csv', header=None, )
        foring_data = np.expand_dims(foring_data, axis = -1)
        if fid==0:
            xTrain_year = foring_data
        else:
            xTrain_year = np.concatenate((xTrain_year,foring_data), axis=-1)



    if year== startyear:
        xTrain = xTrain_year
    else:
        xTrain = np.concatenate((xTrain,xTrain_year), axis=1)

# load attributes
for aid, attribute_ in enumerate(attributeLst) :
    attribute_data = pd.read_csv(rootDB_s+'/const/' + attribute_ + '.csv',  header=None, )
    if aid==0:
        attribute = attribute_data
    else:
        attribute = np.concatenate((attribute,attribute_data), axis=-1)

##Select the training data
testing_time  = pd.date_range(testDateRange[0], testDateRange[-1], freq='d')

index_start = time_range.get_loc(testing_time[0])
index_end = time_range.get_loc(testing_time[-1]) + 1

xTrain = xTrain[:,index_start:index_end]
target = xTrain[:,:,len(var_x_list):]
## Calculate the statistics and normalize the data
stat_dict={}
for fid, forcing_item in enumerate(var_x_list+targetLst) :
        stat_dict[forcing_item] = scale.cal_stat(xTrain[:,:,fid])

for aid, attribute_item in enumerate (attributeLst):
    stat_dict[attribute_item] = scale.cal_stat(attribute[:,aid])


xTrain_norm = scale.trans_norm(
    xTrain, var_x_list+targetLst, stat_dict, to_norm=True
)

xTrain_norm[xTrain_norm!=xTrain_norm]  = 0

attribute_norm = scale.trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0


forcing_train_norm = xTrain_norm[:,:,:len(var_x_list)]
target_train_norm = xTrain_norm[:,:,len(var_x_list):]

## Load model
rootOut = modelpath+'/LSTM_SWE_temp/'
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{trainBuff}") # output folder to save results

with open(out + '/scaler_stat.json') as f:
    stat_dict = json.load(f)

## test the model
testepoch = EPOCH ## Can check other epochs too
model_path = out
print("Load model from ", model_path)
testmodel = loadModel(model_path, epoch=testepoch)

testbatch =200

filePathLst = [out+f"/SWE_norm.csv"]

testmodel.inittime = 0


test.testModel(
    testmodel, forcing_train_norm, c=attribute_norm, batchSize=testbatch, filePathLst=filePathLst)

dataPred = pd.read_csv(  out+f"/SWE_norm.csv", dtype=np.float32, header=None).values
dataPred = np.expand_dims(dataPred, axis=-1)

yPred = scale.trans_norm(
    dataPred,
    targetLst,
    stat_dict,
    to_norm=False,
)

evaDict = [stat.statError(yPred[:,:,0], target[:,:,0])]

evaDictLst = evaDict
keyLst = ['NSE', 'RMSE','Bias', 'Corr']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("LSTM model for SWE prediction: NSE, RMSE,Bias, Corr: ",
      np.nanmedian(dataBox[0][0]),
      np.nanmedian(dataBox[1][0]), np.nanmedian(dataBox[2][0]), np.nanmedian(dataBox[3][0]))



pred_df = pd.DataFrame(yPred[:,:,0].transpose(), index=testing_time)

yearly_max_pred = pred_df.resample('AS-OCT').max()

obs_df = pd.DataFrame(target[:,:,0].transpose(), index=testing_time)
yearly_max_obs = obs_df.resample('AS-OCT').max()


yearly_max_pred = yearly_max_pred[(yearly_max_pred.index >= f'{testing_time[0].year}-10-01') & (yearly_max_pred.index < f'{testing_time[-1].year}-09-30')]
yearly_max_obs = yearly_max_obs[(yearly_max_obs.index >= f'{testing_time[0].year}-10-01') & (yearly_max_obs.index <f'{testing_time[-1].year}-10-01')]




evaDict = [stat.statError(yearly_max_pred.values.transpose(), yearly_max_obs.values.transpose())]


evaDictLst = evaDict
keyLst = ['absBias']
dataBox = list()
for iS in range(len(keyLst)):
    statStr = keyLst[iS]
    temp = list()
    for k in range(len(evaDictLst)):
        data = evaDictLst[k][statStr]
        #data = data[~np.isnan(data)]
        temp.append(data)
    dataBox.append(temp)


print("LSTM SWE annual dMax ",
      np.nanmedian(dataBox[0][0]))
