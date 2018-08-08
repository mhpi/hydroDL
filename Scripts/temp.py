import rnnSMAP
from rnnSMAP import runTrainLSTM
import imp
import numpy as np
from mpl_toolkits.basemap import Basemap, cm
import matplotlib.pyplot as plt
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# Training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015, varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu',
    model='cudnn', loss='sigma',
    var='varLst_soilM', train='CONUSv4f1'
)
# runTrainLSTM.runCmdLine(opt=opt, cudaID=2, screenName=opt['out'])

##
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
trainName = 'CONUSv2f1'
testName = 'CONUSv2f1'
out = trainName+'_y15_soilM'
ds = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
ds.readData(var='SMAP_AM', field='SMAP')
ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
ds.data2grid(field='LSTM')
rnnSMAP.funPost.plotMap(
    ds.LSTM_grid[:, :, 100], lat=ds.crdGrid[0], lon=ds.crdGrid[1])
