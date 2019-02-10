import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


#################################################
# intervals temporal see if sigma output affact sigmaMC
doOpt = []
# doOpt.append('train')
doOpt.append('test')
doOpt.append('plotMap')

rootOutLst = [rnnSMAP.kPath['Out_L3_NA'], rnnSMAP.kPath['OutSigma_L3_NA']]
testName = 'CONUSv2f1'
yr = [2017]
nCase = len(rootOutLst)

#################################################
if 'test' in doOpt:
    rootDB = rnnSMAP.kPath['DB_L3_NA']

    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(rootOutLst)):
        rootOut = rootOutLst[k]
        out = 'CONUSv2f1_y15_Forcing_dr60'
        testName = testName
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
