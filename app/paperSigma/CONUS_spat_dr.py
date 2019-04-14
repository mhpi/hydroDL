import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
from rnnSMAP import runTestLSTM
import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.stats as stats

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
# doOpt.append('train')
# doOpt.append('test')
doOpt.append('loadData')
# doOpt.append('plotComb')
# doOpt.append('plotConf')
# doOpt.append('plotMap')
doOpt.append('plotBox')
doOpt.append('plotConfDist')
# doOpt.append('plotVS')


rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
drLst = np.arange(0.1, 1, 0.1)
drStrLst = ["%02d" % (x*100) for x in drLst]
testName = 'CONUSv2fx2'
yrLst = [2015]
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')
legLst = list()
for dr in drLst:
    legLst.append('dr='+str(dr))

#################################################
if 'train' in doOpt:
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


#################################################
if 'test' in doOpt:
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    for k in range(0, len(drLst)):
        out = 'CONUSv2f1_y15_Forcing_dr'+drStrLst[k]
        cudaID = k % 3
        runTestLSTM.runCmdLine(
            rootDB=rootDB, rootOut=rootOut, out=out, testName=testName,
            yrLst=yrLst, cudaID=cudaID, screenName=out)


#################################################
if 'loadData' in doOpt:
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(drLst)):
        if drLst[k] == 0.5:
            out = 'CONUSv2f1_y15_Forcing'
        else:
            out = 'CONUSv2f1_y15_Forcing_dr'+drStrLst[k]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yrLst)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)

#################################################
if 'plotConf' in doOpt:
    fig, axes = plt.subplots(ncols=3, figsize=(9, 4))

    confXLst = list()
    confMCLst = list()
    confLst = list()
    for k in range(0, len(drLst)):
        statConf = statConfLst[k]
        confXLst.append(statConf.conf_sigmaX)
        confMCLst.append(statConf.conf_sigmaMC)
        confLst.append(statConf.conf_sigma)
    rnnSMAP.funPost.plotCDF(confXLst, ax=axes[0], legendLst=legLst)
    axes[0].set_title('sigmaX')
    rnnSMAP.funPost.plotCDF(confMCLst, ax=axes[1], legendLst=legLst)
    axes[1].set_title('sigmaMC')
    rnnSMAP.funPost.plotCDF(confLst, ax=axes[2], legendLst=legLst)
    axes[2].set_title('sigmaComb')
    plt.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'CONUS_temp_dr_conf.png')
    fig.savefig(saveFile, dpi=100)

#################################################
if 'plotBox' in doOpt:
    data = list()
    # strSigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
    strSigmaLst = []
    # strErrLst = ['ubRMSE', 'Bias']
    strErrLst = ['ubRMSE']
    # labelC = [r'$\sigma_{mc}$', r'$\sigma_{x}$',
    #           r'$\sigma_{comb}$', 'ubRMSE', 'Bias']
    labelC = ['ubRMSE']
    for strSigma in strSigmaLst:
        temp = list()
        for k in range(0, len(drLst)):
            statSigma = statSigmaLst[k]
            temp.append(getattr(statSigma, strSigma))
        data.append(temp)
    for strErr in strErrLst:
        temp = list()
        for k in range(0, len(drLst)):
            statErr = statErrLst[k]
            temp.append(getattr(statErr, strErr))
        data.append(temp)
    fig = rnnSMAP.funPost.plotBox(
        data, labelS=legLst, labelC=labelC,
        colorLst=plt.cm.jet(drLst), figsize=(4, 4), sharey=False)
    # fig.subplots_adjust(wspace=0.5)
    plt.tight_layout()
    saveFile = os.path.join(saveFolder, 'CONUS_spat_dr_box')
    fig.savefig(saveFile, dpi=100)


#################################################
if 'plotConfDist' in doOpt:
    fig, axes = plt.subplots(ncols=2, figsize=(8, 4))
    confLst = list()
    distLst = list()
    for k in range(0, len(drLst)):
        statConf = statConfLst[k]
        confLst.append(statConf.conf_sigma)

        xSort = rnnSMAP.funPost.flatData(statConf.conf_sigma)
        yRank = np.arange(len(xSort))/float(len(xSort)-1)
        # rmse = np.sqrt(((xSort - yRank) ** 2).mean())
        dist = 0
        dbin = 0.01
        for xbin in np.arange(0, 1, dbin):
            ind = (xSort > xbin) & (xSort <= xbin+dbin)
            temp = np.nanmean(np.abs(xSort[ind] - yRank[ind]))
            if not np.isnan(temp):
                dist = dist+temp*dbin
        distLst.append(dist)

    rnnSMAP.funPost.plotCDF(
        confLst, ax=axes[0], legendLst=legLst, showDiff=None,
        xlabel='Predicted Probablity', ylabel='Frequency')
    axes[1].plot(drLst, distLst, marker='*')
    axes[0].set_title(r'$CDF of p_{comb}$')
    axes[1].set_title(r'$d(p_{comb})$')
    axes[1].set_xlabel('Dropout Rate')

    plt.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    saveFile = os.path.join(saveFolder, 'CONUS_spat_dr_conf')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
    fig.show()
