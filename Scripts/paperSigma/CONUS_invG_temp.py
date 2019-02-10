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
# doOpt.append('test')
# doOpt.append('plotConf')
doOpt.append('plotConfDist')
# doOpt.append('plotInvGammaCDF')

# doOpt.append('plotNorm')
# doOpt.append('plotScale')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
# doOpt.append('plotVS')

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 12})
matplotlib.rcParams.update({'legend.fontsize': 16})


trainName = 'CONUSv4f1'
testName = 'CONUSv4f1'
yr = [2017]

C1Lst = [2, 3, 4]
C2Lst = [1, 2, 4]
outLst = list()
caseStrLst = list()
outLst.append(trainName+'_y15_Forcing')
caseStrLst.append('no prior')
for j in C1Lst:
    for i in C2Lst:
        outLst.append(trainName+'_y15_Forcing_dr60_invGamma_'+str(j)+'_'+str(i))
        caseStrLst.append('a='+str(j-1)+','+'b='+str(i/2))

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
        dr=0.6, modelOpt='relu', model='cudnn',
        loss='sigma'
    )
    k = 0
    for j in C1Lst:
        for i in C2Lst:
            opt['out'] = trainName + \
                '_y15_Forcing_dr60_invGamma_'+str(j)+'_'+str(i)
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
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
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
    outLst = list()
    for k in range(0, len(strConfLst)):
        plotLst = list()
        for iCase in range(0, nCase):
            temp = getattr(statConfLst[iCase], strConfLst[k])
            plotLst.append(temp)
        if k == 0:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=caseStrLst, ylabel=None,
                xlabel='Predicted Probablity', showDiff=False)
        else:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=None, ylabel=None,
                xlabel='Predicted Probablity', showDiff=False)
        outLst.append(out)
        axes[k].set_title(titleLst[k])
        if k == 0:
            axes[k].set_ylabel('Frequency')

    saveFile = os.path.join(saveFolder, 'CONUS_invG_temp')
    fig.show()
    plt.tight_layout()
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')


#################################################
if 'plotConfDist' in doOpt:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # plotLst = list()
    # for k in range(0, len(caseStrLst)):
    #     plotLst.append(getattr(statConfLst[k], 'conf_sigma'))
    # _, _, out = rnnSMAP.funPost.plotCDF(
    #     plotLst, ax=axes[0], legendLst=caseStrLst,
    #     xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)
    # axes[0].set_title(r'CDF($p_{comb}$)')
    # axes[0].set_ylabel('Frequency')

    cLst = 'rgb'
    x = getattr(statConfLst[0], 'conf_sigma')
    xSort = rnnSMAP.funPost.flatData(x)
    yRank = np.arange(len(xSort))/float(len(xSort)-1)
    dd = np.max(np.abs(xSort - yRank))
    ax = axes[0]
    k = 1
    for j in range(0, len(C1Lst)):
        aLst = list()
        bLst = list()
        dLst = list()
        for i in range(0, len(C2Lst)):
            a = C1Lst[j]-1
            b = C2Lst[i]/2
            x = getattr(statConfLst[k], 'conf_sigma')
            xSort = rnnSMAP.funPost.flatData(x)
            yRank = np.arange(len(xSort))/float(len(xSort)-1)
            d = np.max(np.abs(xSort - yRank))
            k = k+1
            aLst.append(a)
            bLst.append(b)
            dLst.append(d)
        ax.plot(bLst, dLst, color=cLst[j], label='a='+str(a), marker='*')
    ax.plot(bLst, [dd, dd, dd], color='k', label='no prior')
    ax.legend(loc='best')
    ax.set_ylabel(r'd($P_{ee}$,1-to-1)')
    ax.set_xlabel('b')
    ax.set_xticks(bLst)
    ax.set_title('(a) Quality of '+r'$\sigma_{comb}$')

    k = 1
    d1Lst = list()
    d2Lst = list()
    ax = axes[1]
    for j in range(0, len(C1Lst)):
        for i in range(0, len(C2Lst)):
            a = C1Lst[j]-1
            b = C2Lst[i]/2
            x = getattr(statSigmaLst[k], 'sigmaX_mat')
            x1 = rnnSMAP.funPost.flatData(x)/0.088578376
            y1 = np.arange(len(x1))/float(len(x1)-1)
            y2 = scipy.stats.invgamma.cdf(x1, a, loc=0, scale=b)
            d1 = np.max(np.abs(y1-y2))

            x = getattr(statConfLst[k], 'conf_sigma')
            xSort = rnnSMAP.funPost.flatData(x)
            yRank = np.arange(len(xSort))/float(len(xSort)-1)
            d2 = np.max(np.abs(xSort - yRank))
            d1Lst.append(d1)
            d2Lst.append(d2)
            k = k+1
    ax.plot(d1Lst, d2Lst, '*')
    ax.set_ylabel(r'd($P_{ee}$,1-to-1)')
    ax.set_xlabel('d(prior,posterior)')
    ax.set_title('(b) Quality Change of '+r'$\sigma_{comb}$')

    plt.tight_layout()
    saveFile = os.path.join(saveFolder, 'invGamma_dist')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
    fig.show()

if 'plotInvGammaCDF' in doOpt:
    matplotlib.rcParams.update({'font.size': 12})
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    k = 1
    d1Lst = list()
    d2Lst = list()
    for j in range(0, len(C1Lst)):
        for i in range(0, len(C2Lst)):
            a = C1Lst[j]-1
            b = C2Lst[i]/2
            x = getattr(statSigmaLst[k], 'sigmaX_mat')
            x1 = rnnSMAP.funPost.flatData(x)/0.088578376
            y1 = np.arange(len(x1))/float(len(x1)-1)
            y2 = scipy.stats.invgamma.cdf(x1, a, scale=b)
            d1 = np.max(np.abs(y1-y2))

            x = getattr(statConfLst[k], 'conf_sigma')
            xSort = rnnSMAP.funPost.flatData(x)
            yRank = np.arange(len(xSort))/float(len(xSort)-1)
            d2 = np.max(np.abs(xSort - yRank))
            d1Lst.append(d1)
            d2Lst.append(d2)
            k = k+1
            axes[j][i].plot(x1, y1, 'b')
            axes[j][i].plot(x1, y2, 'r')
            axes[j][i].set_title('a='+str(a)+'b='+str(b)+'d='+str(d))
    fig.show()
