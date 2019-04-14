
import rnnSMAP
import numpy as np
import pandas as pd
import torch
import argparse
from rnnSMAP import runTrainLSTM


# training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['Out_L3_NA'],
    syr=2017, eyr=2017,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv16f1', dr=0.5, modelOpt='relu',
    model='cudnn', loss='mse', out='CONUSv16f1_y17_soilM'
)
# runTrainLSTM.runCmdLine(opt=opt, cudaID=1, screenName=opt['out'])

# error
outLst = ['CONUSv16f1_y17_soilM', 'Closed_Loop_LSTM_Model(Raj)']
yrLst = [[2017], [2015, 2016]]
testName = 'CONUSv16f1'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['Out_L3_NA']


plotDataLst = []
strE = 'RMSE'
for out in outLst:
    temp = []
    for yr in yrLst:
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=0, field='LSTM')
        statErr = ds.statCalError(
            predField='LSTM', targetField='SMAP')
        temp.append(getattr(statErr, strE))
    plotDataLst.append(temp)
strE = 'RMSE'

fig = rnnSMAP.funPost.plotBox(
    plotDataLst, labelS=['train', 'test'], labelC=['open loop', 'close loop'], title='Temporal Test ' + strE)
