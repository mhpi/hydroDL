
import rnnSMAP
import torch
import imp
from rnnSMAP import runTrainLSTM
imp.reload(rnnSMAP)
rnnSMAP.reload()

opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'], 
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, loss='sigma',lossPrior='gauss'
)

trainName='CONUSv4f1'
opt['train'] = trainName
opt['out'] = trainName+'_test'
# runTrainLSTM.runCmdLine(opt=opt, cudaID=2, screenName=opt['out'])
rnnSMAP.funLSTM.trainLSTM(opt)

testName='CONUSv4f1'