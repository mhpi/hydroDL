import rnnSMAP
import numpy as np
import pandas as pd
import torch
import argparse

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DBSMAP_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSMAP_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1', dr=0.5, modelOpt='relu'
)

opt['model'] = 'cudnn'
opt['out'] = 'cudnn_local3'
rnnSMAP.funLSTM.trainLSTM(opt)

out = opt['out']
rootOut = rnnSMAP.kPath['OutSMAP_L3_NA']
testName = 'CONUSv4f1'
syr = 2016
eyr = 2016
rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName,
    syr=2016, eyr=2017)

rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName,
    syr=2015, eyr=2015)


# opt['model']='torch'
# opt['out']='cudnn_torch'
# rnnSMAP.funLSTM.trainLSTM(opt)
