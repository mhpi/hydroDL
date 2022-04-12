import os
import torch
import random
import numpy as np
from hydroDL.model.crit import RmseLoss
from hydroDL.data.load_csv import LoadCSV
from hydroDL.master.master import loadModel
from hydroDL.model.rnn import CudnnLstmModel as LSTM
from hydroDL.model.rnn import CpuLstmModel as LSTM_CPU
from hydroDL.model.train_multi import trainModel, testModel

root_DB = "./multiscale/coarse_resolution"  # location of fine-resolution datasets
root_fine = "./multiscale/fine_resolution"  # location of coarse-resolution datasets
root_site = "./multiscale/insitu"  # location of in-situ datasets
root_DB = "/data/jxl6499/local/hydroDL_zenodo/example/multiscale/coarse_resolution"  # location of fine-resolution datasets
root_fine = "/data/jxl6499/local/hydroDL_zenodo/example/multiscale/fine_resolution"  # location of coarse-resolution datasets
root_site = "/data/jxl6499/local/hydroDL_zenodo/example/multiscale/insitu"  # location of in-situ datasets
all_csv_range = ["2016-01-01", "2019-12-31"]  # date for all datasets in the folder
train_range = ["2016-01-01", "2018-12-31"]  # date for training period
var_time_series = ["APCP", "TMP", "DLWRF", "DSWRF", "SPFH", "PRES"]  # forcing datasets
var_constant = ["Albedo", "landcover", "Capa", "NDVI", "T_BULK_DEN", "T_SILT", "T_SAND",
                "T_CLAY", ]  # constant datasets
target = ["SM"]  # target datasets

BATCH_SIZE = 50
RHO = 30
EPOCHS = 40
SAVE_EPOCH = 1
SEED = 42
HIDDEN_SIZE = 256

out = os.path.join("output", "multiscale")  # save location

# Specify the GPU for training
if torch.cuda.is_available():
    LSTM = LSTM
    torch.cuda.set_device(0)

else:
    LSTM = LSTM_CPU


# Set random seeds to guarantee the same result
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


set_seed(SEED)

# get in-situ datasets
load_insitu = LoadCSV(root_site, train_range, all_csv_range)
y_site = load_insitu.load_time_series(target, do_norm=True, remove_nan=False)
x_site = load_insitu.load_time_series(var_time_series, do_norm=True, remove_nan=True)
c_site = load_insitu.load_constant(var_constant, do_norm=True, remove_nan=True, convert_time_series=False)
nx_site = x_site.shape[-1] + c_site.shape[-1]
ny_site = y_site.shape[-1]

# get forcing datasets
load_fine = LoadCSV(root_fine, train_range, all_csv_range)
x_fine = load_fine.load_time_series(var_time_series, do_norm=True, remove_nan=True)
c_fine = load_fine.load_constant(var_constant, do_norm=True, remove_nan=True, convert_time_series=False)
nx_fine = x_fine.shape[-1] + c_fine.shape[-1]

# get target datasets
load_coarse = LoadCSV(root_DB, train_range, all_csv_range)
y_coarse = load_coarse.load_time_series(target, do_norm=True, remove_nan=False)
ny_coarse = y_coarse.shape[-1]

# define model and loss function
loss_fn = RmseLoss()
model = LSTM(nx=nx_fine, ny=ny_coarse, hiddenSize=HIDDEN_SIZE)
opt = {"root_DB": root_DB, "target": target, "out": out}

# training the model
model = trainModel(
    model,
    (x_fine, y_coarse, c_fine),
    (x_site, y_site, c_site),
    loss_fn,
    nEpoch=EPOCHS,
    miniBatch=[BATCH_SIZE, RHO],
    saveEpoch=SAVE_EPOCH,
    opt=opt,
)

# testing the results
test_range = ["2019-01-01", "2019-12-31"]  # date for testing
TEST_EPOCH = 43  # select the epoch for testing

# get in-situ datasets
load_insitu = LoadCSV(root_site, test_range, all_csv_range)
y_site = load_insitu.load_time_series(target, do_norm=True, remove_nan=False)
x_site = load_insitu.load_time_series(var_time_series, do_norm=True, remove_nan=True)
c_site = load_insitu.load_constant(var_constant, do_norm=True, remove_nan=True, convert_time_series=False)

# get forcing datasets
load_fine = LoadCSV(root_fine, test_range, all_csv_range)
x_fine = load_fine.load_time_series(var_time_series, do_norm=True, remove_nan=True)
c_fine = load_fine.load_constant(var_constant, do_norm=True, remove_nan=True, convert_time_series=False)

# get target datasets
load_coarse = LoadCSV(root_DB, test_range, all_csv_range)
y_coarse = load_coarse.load_time_series(target, do_norm=True, remove_nan=False)

# validation
print(TEST_EPOCH + 1)
test_model = loadModel(out, epoch=TEST_EPOCH + 1)

median_dict_in_situ = testModel(
    test_model,
    (x_fine, y_coarse, c_fine),
    (x_site, y_site, c_site),
    batchSize=10,
    filePathLst=[out],
    opt=opt,
)
print(median_dict_in_situ)
