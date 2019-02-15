import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
# drLst = np.arange(0, 1, 0.1)
drLst = [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]
drStrLst = ["%02d" % (x*100) for x in drLst]

opt = rnnSMAP.classLSTM.optLSTM(
    rootDB=rootDB,
    rootOut=rootOut,
    syr=2015, eyr=2015,
    var='varLst_Forcing', varC='varConstLst_Noah',
    train='CONUSv2f1', dr=0.5, modelOpt='relu',
    model='cudnn', loss='sigma',
)
for k in range(0, len(drLst)):
    opt['dr'] = drLst[k]
    opt['out'] = 'CONUSv2f1_y15_Forcing_dr'+drStrLst[k]
    cudaID = k % 3
    runTrainLSTM.runCmdLine(
        opt=opt, cudaID=cudaID, screenName=opt['out'])
