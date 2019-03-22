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
loc = os.path.join(pathSMAP['Out_L3_Global'], 'regTest')
masterDict = master.wrapMaster(loc, optData, optModel, optLoss, optTrain)
master.runMaster(masterDict, overwrite=True)
