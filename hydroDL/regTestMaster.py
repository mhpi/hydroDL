from hydroDL import pathSMAP, master
import os

optData = master.updateOpt(
    master.default.optDataCsv,
    path=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    dateRange=[20150401, 20160331])
optModel = master.default.optLstm
optLoss = master.default.optLoss
optTrain = master.default.optTrainSMAP
out = os.path.join(pathSMAP['Out_L3_Global'], 'regTest')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)
# master.train(masterDict, overwrite=True)

pred = master.test(
    out, tRange=[20160401, 20170331], subset='CONUSv4f1', epoch=400)
