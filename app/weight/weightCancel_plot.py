import os
import pandas as pd
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace
import torch.nn.functional as F

testName = 'CONUSv4f1'
yr = 2015
saveFolder = '/mnt/sdc/rnnSMAP/Result_SMAPgrid/weightDetector/'
cXoutFile = os.path.join(saveFolder, testName+'_yr'+str(yr)+'_cX')
cX1 = dataTemp = pd.read_csv(cXoutFile, dtype=np.float, header=None).values
cHoutFile = os.path.join(saveFolder, testName+'_yr'+str(yr)+'_cH')
cH1 = dataTemp = pd.read_csv(cHoutFile, dtype=np.float, header=None).values

testName = 'CONUSv4f1'
yr = 2017
saveFolder = '/mnt/sdc/rnnSMAP/Result_SMAPgrid/weightDetector/'
cXoutFile = os.path.join(saveFolder, testName+'_yr'+str(yr)+'_cX')
cX2 = dataTemp = pd.read_csv(cXoutFile, dtype=np.float, header=None).values
cHoutFile = os.path.join(saveFolder, testName+'_yr'+str(yr)+'_cH')
cH2 = dataTemp = pd.read_csv(cHoutFile, dtype=np.float, header=None).values

# boxplot
data = (cX1.mean(axis=0), cH1.mean(axis=0), cH2.mean(axis=0), cH2.mean(axis=0))
rnnSMAP.funPost.plotBox(data, labelC=(
    'train count X', 'train count H', 'test count X', 'test count H'))

# map
plotMap(grid, *, crd, lat=None, lon=None, title=None, showFig=True,
        saveFile=None, cRange=None)
