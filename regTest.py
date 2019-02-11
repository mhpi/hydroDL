import rnnSMAP
import datetime
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

"""
Regression code 
"""

# datetime
now = datetime.datetime.now()
datestr = now.strftime('%Y%m%dh%Hm%M')
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
out = 'Regression_'+datestr
dataName = 'CONUSv4f1'

# training
opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rootDB, rootOut=rootOut,
    syr=2015, eyr=2015,
    var='varLst_Forcing', varC='varConstLst_Noah',
    train=dataName, dr=0.5, modelOpt='relu',
    target='SMAP_AM', loss='sigma'
)
opt['out'] = out
rnnSMAP.funLSTM.trainLSTM(opt)


# test
testName = dataName
dsTrain = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2015])  # define dataset
dsTrain.readData(var='SMAP_AM', field='SMAP')  # read target
dsTrain.readPred(rootOut=rootOut, out=out, drMC=0,
                 field='LSTM')  # read prediction

dsTest = rnnSMAP.classDB.DatasetPost(
    rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
dsTest.readData(var='SMAP_AM', field='SMAP')
dsTest.readPred(rootOut=rootOut, out=out, drMC=0, field='LSTM')

# error
statErrTrain = dsTrain.statCalError(
    predField='LSTM', targetField='SMAP')  # calculate error
statErrTest = dsTest.statCalError(predField='LSTM', targetField='SMAP')

# plot 
strE = 'RMSE'
dataErr = [getattr(statErrTrain, strE), getattr(statErrTest, strE)]
fig = rnnSMAP.funPost.plotBox(
    dataErr, labelC=['train', 'test'], title='Temporal Test ' + strE)
fig.savefig('regTest.png')
