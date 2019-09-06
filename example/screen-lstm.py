from hydroDL import pathSMAP, master
import os

# define training options
out = os.path.join(pathSMAP['Out_L3_NA'], 'RegTest', 'CONUSv4f1_sigma')

optData = master.default.update(
    master.default.optDataCsv,
    rootDB=pathSMAP['DB_L3_NA'],
    subset='CONUSv4f1',
    tRange=[20150401, 20160401],
)
optModel = master.default.optLstm
optLoss = master.default.update(
    master.default.optLoss, name='hydroDL.model.crit.SigmaLoss')
optTrain = master.default.optTrain

masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
master.runTrain(masterDict, cudaID=0, screenName='sigmaTest')
