import os
import torch
import numpy as np

import sys 
sys.path.append('..')

from hydroDL.master.master import loadModel
from hydroDL.model.crit import RmseLoss
from hydroDL.model.rnn import CudnnLstmModel as LSTM
from hydroDL.model.rnn import CpuLstmModel as LSTM_CPU
from hydroDL.model.train import trainModel
from hydroDL.model.test import testModel
from hydroDL.post.stat import statError as cal_metric
from hydroDL.data.load_csv import LoadCSV
from hydroDL.utils.norm import re_folder, trans_norm

# set configuration
output_s = "./output/quick_start/"  # output path
csv_path_s = "./demo_data/"  # demo data path
all_date_list = ["2015-04-01", "2017-03-31"]  # demo data time period
train_date_list = ["2015-04-01", "2016-03-31"]  # training period
# time series variables list
var_time_series = ["VGRD_10_FORA", "DLWRF_FORA", "UGRD_10_FORA", "DSWRF_FORA", "TMP_2_FORA", "SPFH_2_FORA", "APCP_FORA", ]
# constant variables list
var_constant = ["flag_extraOrd", "Clay", "Bulk", "Sand", "flag_roughness", "flag_landcover", "flag_vegDense", "Silt", "NDVI",
         "flag_albedo", "flag_waterbody", "Capa", ]
# target variable list
target = ["SMAP_AM"]

# generate output folder
re_folder(output_s)

# hyperparameter
EPOCH = 100
BATCH_SIZE = 50
RHO = 30
HIDDEN_SIZE = 256

# load your datasets
"""
You can change it with your data. The data structure is as follows:
x_train (forcing data, e.g. precipitation, temperature ...): [pixels, time, features] 
c_train (constant data, e.g. soil properties, land cover ...): [pixels, features]
target (e.g. soil moisture, streamflow ...): [pixels, time, 1]

Data type: numpy.float
We have normalized the raw data. 
example:
    If the data size is "[pixels, time, features]" or "[pixels, features]", the statistics for 10% to 90% of the data are calculated as follows:
    
    from hydroDL.utils.norm import cal_statistics
    stat_list = cal_statistics(data, re_extreme=True, percent=10)
    [left_p10, left_p90, mean, std] = stat_list
"""
train_csv = LoadCSV(csv_path_s, train_date_list, all_date_list)
x_train = train_csv.load_time_series(var_time_series)  # data size: [pixels, time, features]
c_train = train_csv.load_constant(var_constant, convert_time_series=False)  # [pixels, features]
y_train = train_csv.load_time_series(target, remove_nan=False)  # [pixels, time, 1]

# define model and loss function
loss_fn = RmseLoss()  # loss function
# select model: GPU or CPU
if torch.cuda.is_available():
    LSTM = LSTM
else:
    LSTM = LSTM_CPU
model = LSTM(nx=len(var_time_series) + len(var_constant), ny=len(target), hiddenSize=HIDDEN_SIZE)

# training the model
last_model = trainModel(
    model,
    x_train,
    y_train,
    c_train,
    loss_fn,
    nEpoch=EPOCH,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=1,
    saveFolder=output_s,
)

# validation the result
# load validation datasets
val_date_list = ["2016-04-01", "2017-03-31"]  # validation period
# load your data. same as training data
val_csv = LoadCSV(csv_path_s, val_date_list, all_date_list)
x_val = val_csv.load_time_series(var_time_series)
c_val = val_csv.load_constant(var_constant, convert_time_series=False)
y_val = val_csv.load_time_series(target, remove_nan=False)

val_epoch = EPOCH # Select the epoch for testing

# load the model
test_model = loadModel(output_s, epoch=val_epoch)

# set the path to save result
save_csv = os.path.join(output_s, "predict.csv")

# validation
pred_val = testModel(test_model, x_val, c_val, batchSize=len(x_train), filePathLst=[save_csv],)

# select the metrics
metrics_list = ["Bias", "RMSE", "ubRMSE", "Corr"]
pred_val = pred_val.numpy()
# denormalization
pred_val = trans_norm(pred_val, csv_path_s, var_s=target[0], from_raw=False)
y_val = trans_norm(y_val, csv_path_s, var_s=target[0], from_raw=False)
pred_val, y_val = np.squeeze(pred_val), np.squeeze(y_val)
metrics_dict = cal_metric(pred_val, y_val)  # calculate the metrics
metrics = ["Median {}: {:.4f}".format(x, np.nanmedian(metrics_dict[x])) for x in metrics_list]
print("Epoch {}: {}".format(val_epoch, metrics))