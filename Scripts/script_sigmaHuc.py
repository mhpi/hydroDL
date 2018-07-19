import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import subprocess

opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu',
    sigma=1, model='cudnn'
)


hucLst = ['04051118', '03101317', '02101114',
          '01020304', '02030406', '14151617']
cudaIdLst=[0,1,2,0,1,2]

for k in range(0,len(hucLst)):
    trainName = hucLst[k]+'_v2f1'
    opt['train'] = trainName
    opt['out'] = trainName+'_1y_soilM'
    testName = 'CONUSv2f1'
    print(trainName)
    runTrainLSTM.runCmdLine(opt=opt, cudaID=cudaIdLst[k], screenName=trainName)

    # out = opt['out']
    # rootOut = opt['rootOut']

    # rnnSMAP.funLSTM.testLSTM(
    #     out=out, rootOut=rootOut, test=testName,
    #     syr=2016, eyr=2017, drMC=100)

    # rnnSMAP.funLSTM.testLSTM(
    #     out=out, rootOut=rootOut, test=testName,
    #     syr=2015, eyr=2015, drMC=100)


