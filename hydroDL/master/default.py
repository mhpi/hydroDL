import hydroDL
from collections import OrderedDict
from hydroDL.data import dbCsv
# SMAP default options
optDataSMAP = OrderedDict(
    name='hydroDL.data.dbCsv.DataframeCsv',
    rootDB=hydroDL.pathSMAP['DB_L3_Global'],
    subset='CONUSv4f1',
    varT=dbCsv.varForcing,
    varC=dbCsv.varConst,
    target=['SMAP_AM'],
    tRange=[20150401, 20160401],
    doNorm=[True, True],
    rmNan=[True, False],
    daObs=0)
optTrainSMAP = OrderedDict(miniBatch=[100, 30], nEpoch=500, saveEpoch=100)

""" model options """
optLstm = OrderedDict(
    name='hydroDL.model.rnn.CudnnLstmModel',
    nx=len(optDataSMAP['varT']) + len(optDataSMAP['varC']),
    ny=1,
    hiddenSize=256,
    doReLU=True)

optLossRMSE = OrderedDict(name='hydroDL.model.crit.RmseLoss', prior='gauss')



def update(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print('skiped ' + key + ': wrong type')
        else:
            print('skiped ' + key + ': not in argument dict')
    return opt
