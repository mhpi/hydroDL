import hydroDL
from hydroDL.data import dbCsv
from hydroDL.model import rnn, crit, train

df1 = hydroDL.data.dbCsv.DataframeCsv(
    rootDB=hydroDL.pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
x1 = df1.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
y1 = df1.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
nx = x1.shape[-1]
ny = 2
model = rnn.CudnnLstmModel(nx=nx, ny=ny, hiddenSize=64)
lossFun = crit.SigmaLoss()
model = hydroDL.model.train.trainModel(
    model, x1, y1, lossFun, nEpoch=5, miniBatch=(30, 100))

df2 = hydroDL.data.dbCsv.DataframeCsv(
    rootDB=hydroDL.pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
x2 = df2.getData(
    varT=dbCsv.varForcing, varC=dbCsv.varConst, doNorm=True, rmNan=True)
y2 = df2.getData(varT='SMAP_AM', doNorm=True, rmNan=False)
yp = train.testModel(model, x2)
