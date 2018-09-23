import rnnSMAP
import numpy as np
import pandas as pd
import torch
import argparse

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


# training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['Out_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1', dr=0.5, modelOpt='relu',
    model='cudnn', loss='sigma', out='CONUSv4f1_y15_Forcing'
)
# rnnSMAP.funLSTM.trainLSTM(opt)

# test
out = opt['out']
testName = 'CONUSv4f1'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['Out_L3_NA']

# training error
ds1 = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2015])  # define dataset
ds1.readData(var='SMAP_AM', field='SMAP')  # read target
ds1.readPred(rootOut=rootOut, out=out, drMC=0, field='LSTM')  # read prediction
statErr1 = ds1.statCalError(
    predField='LSTM', targetField='SMAP')  # calculate error

# test error
ds2 = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
ds2.readData(var='SMAP_AM', field='SMAP')
ds2.readPred(rootOut=rootOut, out=out, drMC=0, field='LSTM')
statErr2 = ds2.statCalError(predField='LSTM', targetField='SMAP')

# plot error
strE = 'RMSE'
dataErr = [getattr(statErr1, strE), getattr(statErr2, strE)]
fig = rnnSMAP.funPost.plotBox(
    dataErr, labelC=['train', 'test'], title='Temporal Test ' + strE)
