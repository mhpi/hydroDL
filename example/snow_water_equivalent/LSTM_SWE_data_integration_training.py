########
### MHPI hydroDL LSTM code for snow water equivalent
### This code contains deep learning code (Long short-term memory) used to model snow water equivalent

### Dear Code User,

### Thank you for using our LSTM model. We are glad to see our code being
### used to advance research in our field.

### As you use our code, we kindly request that you review and cite
### the relevant papers above that were used to develop the code.
### By doing so, you will help ensure that the contributions of
### the researchers who developed the underlying algorithms
### are properly recognized and appreciated.

### We appreciate your cooperation in this matter and would be happy to
### assist you in finding the appropriate sources to cite.
### If you have any questions or concerns, please do not hesitate to contact us.

### Thank you for your support!
### If you have any question for this release, please contact Yalan Song (yxs275@psu.edu), or Chaopeng Shen(cshen@engr.psu.edu)

### If this work is useful to you, please cite
### Song, Y., Tsai, W.P., Gluck, J., Rhoades, A., Zarzycki, C., McCrary, R., Lawson, K. and Shen, C., 2024. LSTM-based data integration to improve snow water equivalent prediction and diagnose error sources. Journal of Hydrometeorology, 25(1), pp.223-237.


import pandas as pd
import numpy as np
import os
import json
import random
import torch
import sys
sys.path.append('../../')
from hydroDL.model import crit, train
from hydroDL.model import rnn as rnn
from hydroDL.data import scale

randomseed = 111111
random.seed(randomseed)
torch.manual_seed(randomseed)
np.random.seed(randomseed)
torch.cuda.manual_seed(randomseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

traingpuid = 0
torch.cuda.set_device(traingpuid)
device = torch.cuda.current_device()

##Please contact us if you need the training data
rootDB_s=f'/mnt/sdb/yxs275/check_code/KFOLD_inputs/SNOTEL_filter_data_1988/'

DateRange=['2001-01-01', '2019-12-31']
TrainingDateRange=['2001-01-01', '2015-12-31']




var_x_list =  ['pr_gridMET', 'tmmn_gridMET', 'tmmx_gridMET', 'srad_gridMET', 'vs_gridMET', 'th_gridMET',
                 'sph_gridMET', 'rmin_gridMET', 'rmax_gridMET']

attributeLst = ['lat','mean_elev', 'mean_slope', 'aspect',
                  'dom_land_cover', 'dom_land_cover_frac', 'forest_fraction']



targetLst = ['SWE']

#Data integration laggaed day
##Can be 30 or 7
DI_day = 30
## Data integration variable
##Can be ['SWE'] or ['SCF_MODIS10A1F']
DI_varibale = ['SWE']

### Read data:
# load forcing and target data
time_range = pd.date_range(DateRange[0], DateRange[-1], freq='d')
startyear = time_range[0].year
endyear = time_range[-1].year
for year in range(startyear,endyear+1):
    for fid, foring_ in enumerate(var_x_list+DI_varibale+targetLst):

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
training_time  = pd.date_range(TrainingDateRange[0], TrainingDateRange[-1], freq='d')

index_start = time_range.get_loc(training_time[0])
index_end = time_range.get_loc(training_time[-1]) + 1

xTrain = xTrain[:,index_start:index_end]

## Calculate the statistics and normalize the data
stat_dict={}
for fid, forcing_item in enumerate(var_x_list+DI_varibale+targetLst) :
        stat_dict[forcing_item] = scale.cal_stat(xTrain[:,:,fid])

for aid, attribute_item in enumerate (attributeLst):
    stat_dict[attribute_item] = scale.cal_stat(attribute[:,aid])


xTrain_norm = scale.trans_norm(
    xTrain, var_x_list+DI_varibale+targetLst, stat_dict, to_norm=True
)

xTrain_norm[xTrain_norm!=xTrain_norm]  = 0

attribute_norm = scale.trans_norm(attribute, list(attributeLst), stat_dict, to_norm=True)
attribute_norm[attribute_norm!=attribute_norm] = 0


forcing_train_norm = xTrain_norm[:,:,:len(var_x_list)]
integrated_var_norm = xTrain_norm[:,:,len(var_x_list):len(var_x_list)+len(DI_varibale)]
target_train_norm = xTrain_norm[:,:,-len(targetLst):]

forcing_train_norm_combined =  np.concatenate((forcing_train_norm[:,DI_day:,:],integrated_var_norm[:,:integrated_var_norm.shape[1]-DI_day,:]), axis=-1)

target_train_norm = target_train_norm[:,DI_day:,:]
##Hyperparameters

EPOCH = 600 # total epoches to train the mode
BATCH_SIZE = 100
RHO = 365
saveEPOCH = 50
HIDDENSIZE = 256
trainBuff = 365

nx = forcing_train_norm_combined.shape[-1] + attribute_norm.shape[-1]  # update nx, nx = nx + nc
ny =len(targetLst)

# load model for training
model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=HIDDENSIZE)

# loss function : MSE loss, used in Song et al, 2024
lossFun = crit.MSELoss()

# loss function : NSE loss
#lossFun = crit.NSELossBatch(np.nanstd(target_train_norm, axis=1 ),device =device)

rootOut = "/mnt/sdb/yxs275/snow_hydroDL/output/"+'/LSTM_SWE_temp'+f'_DI_{DI_varibale[0]}_{DI_day}'
if os.path.exists(rootOut) is False:
    os.mkdir(rootOut)
out = os.path.join(rootOut, f"exp_EPOCH{EPOCH}_BS{BATCH_SIZE}_RHO{RHO}_HS{HIDDENSIZE}_trainBuff{trainBuff}") # output folder to save results
if os.path.exists(out) is False:
    os.mkdir(out)

with open(out+'/scaler_stat.json','w') as f:
    json.dump(stat_dict, f)


# training the model
model = train.trainModel(
    model,
    forcing_train_norm_combined,
    target_train_norm,
    attribute_norm,
    lossFun,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=saveEPOCH,
    saveFolder=out,
    bufftime=trainBuff
)
