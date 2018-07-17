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
    rootOut=rnnSMAP.kPath['Out_L3_NA'],
    syr=2017, eyr=2017,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUS', dr=0.5, modelOpt='relu',
    sigma=0
)

opt['model'] = 'cudnn'
opt['out'] = 'CONUS_2017_Forcing'
# rnnSMAP.funLSTM.trainLSTM(opt)

out = opt['out']
rootOut = rnnSMAP.kPath['Out_L3_NA']
syr = 2015
eyr = 2016
rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test='CRN',
    syr=syr, eyr=eyr,drMC=0)

rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test='CoreSite',
    syr=syr, eyr=eyr,drMC=0)

