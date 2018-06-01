import os
import rnnSMAP
import numpy as np
import pandas as pd
import torch

import imp
from argparse import Namespace
from rnnSMAP import *

imp.reload(rnnSMAP)
rnnSMAP.reload()

# train model
opt = rnnSMAP.classLSTM.optLSTM(
    out='CONUSv4f1_pytorch',
    rootDB=rnnSMAP.kPath['DBSMAP_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSMAP_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1'
)
rnnSMAP.funLSTM.trainLSTM(opt)

# test model
out = 'CONUSv4f1_pytorch'
rootOut = kPath['OutSMAP_L3_NA']
testName = 'CONUSv4f1'
syr = 2016
eyr = 2016
rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName, 
    syr=2016, eyr=2016)

rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName, 
    syr=2015, eyr=2015)

rootOut=kPath['OutSMAP_L3_NA']
outName='CONUSv4f1_pytorch'
test='CONUSv4f1'
syr=2015
eyr=2015

outFolder=os.path.join(rootOut,outName)
optTrain=funLSTM.loadOptLSTM(outFolder)
rootDB=optTrain['rootDB']

epoch=None
if epoch is None:
    epoch=optTrain['nEpoch']
saveName = 'test_{}_{}_{}_ep{}.csv'.format(test, str(syr), str(eyr), str(epoch))
outFile=os.path.join(outFolder,saveName)
    
yP = pd.read_csv(outFile, dtype=np.float, header=None).values.transpose()

dataset=classDB.Dataset(rootDB=rootDB,subsetName=test,yrLst=[2015])
dataset.readTarget(loadNorm=True)
yT=dataset.normTarget

tP=torch.from_numpy(yP)
tT=torch.from_numpy(yT)
loc0 = tT != tT
tT[loc0] = tP[loc0]

crit = torch.nn.MSELoss()
loss = crit(tP, tT)