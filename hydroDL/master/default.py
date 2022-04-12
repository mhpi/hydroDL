import hydroDL
from collections import OrderedDict
from hydroDL.data import dbCsv, camels

# SMAP default options
optDataSMAP = OrderedDict(
    name="hydroDL.data.dbCsv.DataframeCsv",
    rootDB=hydroDL.pathSMAP["DB_L3_Global"],
    subset="CONUSv4f1",
    varT=dbCsv.varForcing,
    varC=dbCsv.varConst,
    target=["SMAP_AM"],
    tRange=[20150401, 20160401],
    doNorm=[True, True],
    rmNan=[True, False],
    daObs=0,
)
optTrainSMAP = OrderedDict(miniBatch=[100, 30], nEpoch=500, saveEpoch=100, seed=42)
# Streamflow default options
optDataCamels = OrderedDict(
    name="hydroDL.data.camels.DataframeCamels",
    subset="All",
    varT=camels.forcingLst,
    varC=camels.attrLstSel,
    target=["Streamflow"],
    tRange=[19900101, 19950101],
    doNorm=[True, True],
    rmNan=[True, False],
    basinNorm=True,
    daObs=0,
    damean=False,
    davar="streamflow",
    dameanopt=0,
    lckernel=None,
    fdcopt=False,
    SAOpt=None,
    addVar=None,
)
# optDataGages = OrderedDict(
#     name='hydroDL.data.gages.DataframeGages',
#     subset='All',
#     varT=gages.forcingLst,
#     varL=gages.LanduseAttr,
#     varC=gages.attrLstSel,
#     target=['Streamflow'],
#     tRange=[19900101, 19950101],
#     doNorm=[True, True],
#     rmNan=[True, False],
#     daObs=0,
#     damean=False,
#     davar='streamflow',
#     dameanopt=0,
#     lckernel=None,
#     fdcopt=False,
#     includeLanduse=False,
#     includeWateruse=False)
optTrainCamels = OrderedDict(miniBatch=[100, 200], nEpoch=100, saveEpoch=50, seed=None)
""" model options """
optLstm = OrderedDict(
    name="hydroDL.model.rnn.CudnnLstmModel",
    nx=len(optDataSMAP["varT"]) + len(optDataSMAP["varC"]),
    ny=1,
    hiddenSize=256,
    doReLU=True,
)
optLstmClose = OrderedDict(
    name="hydroDL.model.rnn.LstmCloseModel",
    nx=len(optDataSMAP["varT"]) + len(optDataSMAP["varC"]),
    ny=1,
    hiddenSize=256,
    doReLU=True,
)
optCnn1dLstm = OrderedDict(
    name="hydroDL.model.rnn.CNN1dLSTMInmodel",
    nx=len(optDataSMAP["varT"]) + len(optDataSMAP["varC"]),
    ny=1,
    nobs=7,
    hiddenSize=256,
    # CNN kernel parameters
    # Nkernel, Kernel Size, Stride
    convNKS=[(10, 5, 1), (3, 3, 3), (2, 2, 1)],
    doReLU=True,
    poolOpt=None,
)
optLstmCnn1d = OrderedDict(
    name="hydroDL.model.cnn.LstmCnn1d",
    nx=len(optDataSMAP["varT"]) + len(optDataSMAP["varC"]) + 1,
    ny=1,
    rho=365 * 10,
    # CNN kernel parameters
    # Nkernel, Kernel Size, Stride
    convNKSP=[(10, 5, 1), (3, 3, 3), (1, 2, 1), (1, 1, 1)],
    doReLU=True,
    poolOpt=None,
)
optPretrain = OrderedDict(
    name="hydroDL.model.rnn.CNN1dLSTMInmodel",
    nx=len(optDataSMAP["varT"]) + len(optDataSMAP["varC"]),
    ny=1,
    nobs=7,
    hiddenSize=256,
    # CNN kernel parameters
    # Nkernel, Kernel Size, Stride
    convNKS=[(10, 5, 1), (3, 3, 3), (2, 2, 1)],
    doReLU=True,
    poolOpt=None,
)
optInvLstm = OrderedDict(
    name="hydroDL.model.rnn.CudnnInvLstmModel",
    nx=len(optDataSMAP["varT"]) + len(optDataSMAP["varC"]),
    ny=1,
    hiddenSize=256,
    ninv=4,
    nfea=10,
    hiddeninv=256,
    doReLU=True,
)


optLossRMSE = OrderedDict(name="hydroDL.model.crit.RmseLoss", prior="gauss")
optLossSigma = OrderedDict(name="hydroDL.model.crit.SigmaLoss", prior="gauss")
optLossNSE = OrderedDict(name="hydroDL.model.crit.NSELosstest", prior="gauss")
optLossMSE = OrderedDict(name="hydroDL.model.crit.MSELoss", prior="gauss")
optLossTrend = OrderedDict(name="hydroDL.model.crit.ModifyTrend1", prior="gauss")
optLossRMSECNN = OrderedDict(name="hydroDL.model.crit.RmseLossCNN", prior="gauss")


def update(opt, **kw):
    for key in kw:
        if key in opt:
            try:
                if key in [
                    "subset",
                    "daObs",
                    "poolOpt",
                    "seed",
                    "lckernel",
                    "SAOpt",
                    "addVar",
                ]:
                    opt[key] = kw[key]
                else:
                    opt[key] = type(opt[key])(kw[key])
            except ValueError:
                print("skiped " + key + ": wrong type")
        else:
            print("skiped " + key + ": not in argument dict")
    return opt


def forceUpdate(opt, **kw):
    for key in kw:
        opt[key] = kw[key]
    return opt
