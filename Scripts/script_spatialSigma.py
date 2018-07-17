import os

import rnnSMAP
import numpy as np
import pandas as pd
import torch
import argparse

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu',
    sigma=1, model='cudnn'
)

trainLst = ['CONUSv16f1', 'CONUSv8f1', 'CONUSv4f1', 'CONUSv2f1', 'CONUS']

for trainName in trainLst:
    print(trainName)
    opt['train'] = trainName
    opt['out'] = trainName+'_y15_soilM'
    testName='CONUSv8f1'
    # rnnSMAP.funLSTM.trainLSTM(opt)

    out = opt['out']
    rootOut = opt['rootOut']

    rnnSMAP.funLSTM.testLSTM(
        out=out, rootOut=rootOut, test=testName,
        syr=2016, eyr=2017,drMC=0)

    rnnSMAP.funLSTM.testLSTM(
        out=out, rootOut=rootOut, test=testName,
        syr=2015, eyr=2015,drMC=0)
