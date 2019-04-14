import hydroDL
import os
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train
from hydroDL import post
from hydroDL import utils
import numpy as np

rootDB = hydroDL.pathSMAP['DB_L3_NA']
nEpoch = 100
outFolder = os.path.join(hydroDL.pathSMAP['outTest'], 'cnnCond')
ty1 = [20150402, 20160401]
ty2 = [20160401, 20170401]
ty12 = [20150401, 20170401]
ty3 = [20170401, 20180401]

# load data
dfc = hydroDL.data.dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=ty1)
xc = dfc.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
yc = dfc.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
yc[:, :, 0] = utils.interpNan(yc[:, :, 0])
c = np.concatenate((yc, xc), axis=2)
df = hydroDL.data.dbCsv.DataframeCsv(
    rootDB=rootDB, subset='CONUSv4f1', tRange=ty2)
x = df.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
y = df.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
nx = x.shape[-1]
ny = 1

model = rnn.CnnCondLstm(nx=nx, ny=ny, ct=365, hiddenSize=64, cnnSize=32, opt=3)
lossFun = crit.RmseLoss()
model = train.trainModel(
    model, x, y, lossFun, xc=c, nEpoch=nEpoch, miniBatch=[100, 30])

yOut = train.testModelCnnCond(model, x, y)
# yOut = train.testModel(model, x)
yP = dbCsv.transNorm(
    yOut[:, :, 0], rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False)
yT = dbCsv.transNorm(
    y[:, model.ct:, 0], rootDB=rootDB, fieldName='SMAP_AM', fromRaw=False)
statDict = post.statError(yP, yT)
statDict['RMSE'].mean()
