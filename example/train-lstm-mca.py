from hydroDL import pathSMAP, master
import os
from hydroDL.master import default

cDir = os.path.dirname(os.path.abspath(__file__))
cDir = r'/home/kxf227/work/GitHUB/pyRnnSMAP/example/'

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB=os.path.join(cDir, 'data'),
    subset='CONUSv4f1',
    tRange=[20150401, 20160401],
)
optModel = default.optLstm
optLoss = default.optLossSigma
optTrain = default.update(master.default.optTrainSMAP, nEpoch=5, saveEpoch=5)
out = os.path.join(cDir, 'output', 'CONUSv4f1_sigma')
masterDict = master.wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
master.train(masterDict)

# test
pred = master.test(
    out, tRange=[20160401, 20170401], subset='CONUSv4f1')
