
import rnnSMAP
from rnnSMAP import runTrainLSTM
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

# Train
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rnnSMAP.kPath['DB_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    dr=0.5, modelOpt='relu', model='cudnn',
    loss='sigma'
)
trainName = 'CONUSv4f1'
opt['train'] = trainName

noiseNameLst = ['5e4', '1e3', '2e3', '5e3', '1e2', '2e2', '5e2']

#################################################
# training
cudaIdLst = [0, 1, 2, 1, 2, 1, 2]
for k in range(0, len(noiseNameLst)):
    opt['target'] = 'SMAP_AM_sn'+noiseNameLst[k]
    opt['var'] = 'varLst_soilM'
    opt['out'] = opt['train']+'_y15_soilM_sn'+noiseNameLst[k]
    # runTrainLSTM.runCmdLine(
    #     opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

    opt['var'] = 'varLst_Forcing'
    opt['out'] = opt['train']+'_y15_Forcing_sn'+noiseNameLst[k]
    # runTrainLSTM.runCmdLine(
    #     opt=opt, cudaID=cudaIdLst[k], screenName=opt['out'])

#################################################
# test
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
statErrLst1 = list()
statSigmaLst1 = list()
statErrLst2 = list()
statSigmaLst2 = list()
for k in range(0, len(noiseNameLst)):
    testName = 'CONUSv4f1'
    targetName = 'SMAP_AM_sn'+noiseNameLst[k]
    out = 'CONUSv4f1_y15_soilM_sn'+noiseNameLst[k]
    ds1 = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
    ds1.readData(var=targetName, field='SMAP')
    ds1.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

    out = 'CONUSv4f1_y15_Forcing_sn'+noiseNameLst[k]
    ds2 = rnnSMAP.classDB.DatasetPost(
        rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
    ds2.readData(var=targetName, field='SMAP')
    ds2.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')

    statErr = ds1.statCalError(predField='LSTM', targetField='SMAP')
    statSigma = ds1.statCalSigma(field='LSTM')
    statErrLst1.append(statErr)
    statSigmaLst1.append(statSigma)

    statErr = ds2.statCalError(predField='LSTM', targetField='SMAP')
    statSigma = ds2.statCalSigma(field='LSTM')
    statErrLst2.append(statErr)
    statSigmaLst2.append(statSigma)
