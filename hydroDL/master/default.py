import hydroDL
from collections import OrderedDict
from hydroDL.data import dbCsv
""" default class options """
optDataCsv = OrderedDict(
    name='hydroDL.data.dbCsv.DataframeCsv',
    path=hydroDL.pathSMAP['DB_L3_Global'],
    subset='Globalv8f1',
    varT=dbCsv.varForcing,
    varC=dbCsv.varConst,
    target='SMAP_AM',
    tRange=[20150401, 20160401],
    doNorm=[True, True],
    rmNan=[True, False])
optLstm = OrderedDict(
    name='hydroDL.model.rnn.CudnnLstmModel',
    nx=len(optDataCsv['varT']) + len(optDataCsv['varC']),
    ny=1,
    hiddenSize=256,
    doReLU=True)
optLoss = OrderedDict(name='hydroDL.model.crit.RmseLoss', prior='gauss')
optTrainSMAP = OrderedDict(miniBatch=[100, 30], nEpoch=500, saveEpoch=100)
