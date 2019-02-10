import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import scipy
import pylab
import scipy.stats as stats

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
# doOpt.append('train')
doOpt.append('test')
doOpt.append('plotConf')
# doOpt.append('plotNorm')
# doOpt.append('plotScale')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
# doOpt.append('plotVS')

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 12})

trainName = 'CONUSv4f1'
testName = 'CONUSv4f2'
yr = [2015]

C1Lst = [2, 3, 4]
C2Lst = [1, 2, 4]
outLst = list()
caseStrLst = list()
outLst.append(trainName+'_y15_Forcing')
caseStrLst.append('no prior')
for j in C1Lst:
    for i in C2Lst:
        outLst.append(trainName+'_y15_Forcing_invGamma_'+str(j)+'_'+str(i))
        caseStrLst.append(r'$\alpha$='+str(j-1)+','+r'$\beta$='+str(i/2))

nCase = len(outLst)
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rnnSMAP.kPath['DB_L3_NA'],
        rootOut=rnnSMAP.kPath['OutSigma_L3_NA'],
        train=trainName,
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu', model='cudnn',
        loss='sigma'
    )
    k = 0
    for j in C1Lst:
        for i in C2Lst:
            opt['out'] = trainName+'_y15_Forcing_invGamma_'+str(j)+'_'+str(i)
            opt['lossPrior'] = 'invGamma+'+str(j)+'+'+str(i)
            runTrainLSTM.runCmdLine(
                opt=opt, cudaID=k % 3, screenName=opt['lossPrior'])
            # rnnSMAP.funLSTM.trainLSTM(opt)
            k = k+1

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    statNormLst = list()
    for k in range(0, nCase):
        out = outLst[k]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
        statNorm = rnnSMAP.classPost.statNorm(
            statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
        statNormLst.append(statNorm)

#################################################
if 'plotConf' in doOpt:
    fig, axes = plt.subplots(ncols=3, figsize=(15, 5))
    confXLst = list()
    confMCLst = list()
    confLst = list()
    for k in range(0, nCase):
        statConf = statConfLst[k]
        confXLst.append(statConf.conf_sigmaX)
        confMCLst.append(statConf.conf_sigmaMC)
        confLst.append(statConf.conf_sigma)

    titleLst = [r'$p_{mc}$', r'$p_{x}$', r'$p_{comb}$']
    strConfLst = ['conf_sigmaMC', 'conf_sigmaX', 'conf_sigma']
    for k in range(0, len(strConfLst)):
        plotLst = list()
        for iCase in range(0, nCase):
            temp = getattr(statConfLst[iCase], strConfLst[k])
            plotLst.append(temp)
        if k == 0:
            rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=caseStrLst, ylabel=None,
                xlabel='Predicting Probablity', showDiff=False)
        else:
            rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=None, ylabel=None,
                xlabel='Predicting Probablity', showDiff=False)
        axes[k].set_title(titleLst[k])
        if k == 0:
            axes[k].set_ylabel('True Probablity')
    saveFile = os.path.join(saveFolder, 'CONUS_invG_spat')
    fig.show()
    plt.tight_layout()
    fig.savefig(saveFile, dpi=300)
    fig.savefig(saveFile+'.eps', dpi=300)
