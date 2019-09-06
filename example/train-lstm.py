from hydroDL import pathSMAP
from hydroDL.master import default, wrapMaster, train
import os
import torch

cDir = os.path.dirname(os.path.abspath(__file__))

# define training options
optData = default.update(
    default.optDataSMAP,
    rootDB=os.path.join(cDir, 'data'),
    subset='CONUSv4f1',
    tRange=[20150401, 20160401])
if torch.cuda.is_available():
    optModel = default.optLstm
else:
    optModel = default.update(
        default.optLstm,
        name='hydroDL.model.rnn.CpuLstmModel')
optLoss = default.optLossRMSE
optTrain = default.update(default.optTrainSMAP, nEpoch=100)
out = os.path.join(cDir, 'output', 'CONUSv4f1')
masterDict = wrapMaster(out, optData, optModel, optLoss, optTrain)

# train
train(masterDict)
