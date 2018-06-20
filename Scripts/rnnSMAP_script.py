import os
import rnnSMAP
import numpy as np
import pandas as pd
import torch

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

# train model
# torch.backends.cudnn.enabled = False 
opt = rnnSMAP.classLSTM.optLSTM(
    out='CONUSv4f1_ptvardr_ut',
    rootDB=rnnSMAP.kPath['DBSMAP_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSMAP_L3_NA'],
    syr=2015, eyr=2015, 
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1',gpu=0,dr=0.5
)
rnnSMAP.funLSTM.trainLSTM(opt)

# test model
out = 'CONUSv4f1_ptvardr_ut'
rootOut = rnnSMAP.kPath['OutSMAP_L3_NA']
testName = 'CONUSv4f1'
syr = 2016
eyr = 2016
rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName, 
    syr=2016, eyr=2017,gpu=0)

rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName, 
    syr=2015, eyr=2015,gpu=0)
