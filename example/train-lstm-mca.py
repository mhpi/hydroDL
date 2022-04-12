import os
from hydroDL.master import default
from hydroDL.master.master import wrapMaster, train, test

cDir = os.path.dirname(os.path.abspath(__file__))

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB=os.path.join(cDir, "data"),
    subset="CONUSv4f1",
    tRange=[20150401, 20160401],
)
optModel = default.optLstm
optLoss = default.optLossSigma
optTrain = default.update(default.optTrainSMAP, nEpoch=5, saveEpoch=5)
out = os.path.join(cDir, "output", "CONUSv4f1_sigma")
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
train(masterDict)

# test
pred = test(out, tRange=[20160401, 20170401], subset="CONUSv4f1", epoch=5)
