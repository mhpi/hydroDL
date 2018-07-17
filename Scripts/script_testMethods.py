import os

import rnnSMAP
import numpy as np
import pandas as pd
import torch
import argparse

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

parser = argparse.ArgumentParser()
parser.add_argument("-k", "--kk", dest="opt")
args = parser.parse_args()

opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DBSMAP_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSMAP_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1', dr=0.5
)

outLst = ['CONUSv4f1_tied_relu_drXHC',
          'CONUSv4f1_tied_relu_drXH',
          'CONUSv4f1_tied_relu_drC',
          'CONUSv4f1_tied_relu_drW',
          'CONUSv4f1_untied_relu_drXHC',
          'CONUSv4f1_untied_relu_drXH',
          'CONUSv4f1_untied_relu_drC',
          'CONUSv4f1_untied_relu_drW',
          'CONUSv4f1_tied_drHC',
          'CONUSv4f1_tied_drH',
          'CONUSv4f1_tied_drC',
          'CONUSv4f1_tied_drW',
          'CONUSv4f1_untied_drHC',
          'CONUSv4f1_untied_drH',
          'CONUSv4f1_untied_drC',
          'CONUSv4f1_untied_drW'
          ]
drmLst = ['drX+drH+drC', 'drX+drH', 'drC', 'drW',
          'drX+drH+drC', 'drX+drH', 'drC', 'drW',
          'drH+drC', 'drH', 'drC', 'drW',
          'drH+drC', 'drH', 'drC', 'drW']
mopLst = ['tied+relu', 'tied+relu', 'tied+relu', 'tied+relu',
          'relu', 'relu', 'relu', 'relu',
          'tied', 'tied', 'tied', 'tied',
          '', '', '', '']


k = int(args.opt)
# k = 0
# for k in range(0,16):
opt['out'] = outLst[k]
opt['drMethod'] = drmLst[k]
opt['modelOpt'] = mopLst[k]

# rnnSMAP.funLSTM.trainLSTM(opt)

# opt['out'] = 'CONUSv4f1_tied_relu_drW'
# opt['drMethod'] = 'drW'
# opt['modelOpt'] = 'tied+relu'

# rnnSMAP.funLSTM.trainLSTM(opt)

# test model
out = opt['out']
rootOut = rnnSMAP.kPath['OutSMAP_L3_NA']
testName = 'CONUSv4f1'
rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName,
    syr=2016, eyr=2017)

rnnSMAP.funLSTM.testLSTM(
    out=out, rootOut=rootOut, test=testName,
    syr=2015, eyr=2015)
