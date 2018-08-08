import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# Training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015, varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu',
    model='cudnn', loss='sigma'
)
for kk in range(0, 2):
    for k in range(0, 18):
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
        opt['train'] = trainName
        if kk == 0:
            opt['var'] = 'varLst_soilM'
            opt['out'] = trainName+'_y15_soilM'
        elif kk == 1:
            opt['var'] = 'varLst_Forcing'
            opt['out'] = trainName+'_y15_Forcing'
        cudaID = k % 3
        print(trainName)
        # runTrainLSTM.runCmdLine(opt=opt, cudaID=cudaID, screenName=opt['out'])

#################################################
# Testing
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
for j in range(0, 18):
    for i in range(0, 18):
        trainName = 'hucn1_'+str(j+1).zfill(2)+'_v2f1'
        testName = 'hucn1_'+str(i+1).zfill(2)+'_v2f1'
        for k in range(0, 2):
            if k == 0:
                out = trainName+'_y15_soilM'
            elif k == 1:
                out = trainName+'_y15_Forcing'
        ds1 = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2015])
        ds1.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        ds2 = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds2.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
