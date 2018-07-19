
import os
import rnnSMAP
import torch

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
testName = 'CONUSv4f1'
out = 'CONUSv4f1_y15_soilM'


ds=rnnSMAP.classDB.DatasetPost(rootDB=rootDB, subsetName=testName, yrLst=[2016,2017])
ds.readData(var='SMAP_AM',field='SMAP')
ds.readPred(rootOut=rootOut,out=out,drMC=100)

