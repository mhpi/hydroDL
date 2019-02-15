import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# noise affact on sigmaX (or sigmaMC)
doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('plotMap')
doOpt.append('plotErrBox')
# doOpt.append('plotConf')
# doOpt.append('plotConfDist')
# doOpt.append('plotConfLegend')
#
# noiseNameLst = ['0', '5e3', '1e2', '2e2', '5e2', '1e1']

noiseNameLst = ['0', '1e2', '2e2', '3e2', '4e2', '5e2',
                '6e2', '7e2', '8e2', '9e2', '1e1']
noiseLabelLst = ['0', '0.01', '0.02', '0.03', '0.04', '0.05',
                 '0.06', '0.07', '0.08', '0.09', '0.1']
strErrLst = ['RMSE', 'ubRMSE']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma')
rootDB = rnnSMAP.kPath['DB_L3_NA']

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 14})


#################################################
if 'test' in doOpt:
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(noiseNameLst)):
        testName = 'CONUSv4f1'
        if k == 0:
            out = 'CONUSv4f1_y15_Forcing_dr60'
            targetName = 'SMAP_AM'
        else:
            out = 'CONUSv4f1_y15_Forcing_dr06_sn'+noiseNameLst[k]
            targetName = 'SMAP_AM_sn'+noiseNameLst[k]

        rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
        caseStrLst = ['sigmaMC', 'sigmaX', 'sigma']
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2017])
        ds.readData(var=targetName, field='SMAP')
        ds.readPred(out=out, drMC=100, field='LSTM', rootOut=rootOut)

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statErrLst.append(statErr)
        statSigma = ds.statCalSigma(field='LSTM')
        statSigmaLst.append(statSigma)
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
        statConfLst.append(statConf)

#################################################
if 'plotErrBox' in doOpt:
    data = list()
    strErr = 'ubRMSE'
    strSigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
    labelS = [r'$\sigma_{mc}$', r'$\sigma_x$', r'$\sigma_{comb}$', 'ubRMSE']
    for k in range(0, len(noiseNameLst)):
        temp = list()
        for strSigma in strSigmaLst:
            temp.append(getattr(statSigmaLst[k], strSigma))
        temp.append(getattr(statErrLst[k], strErr))
        data.append(temp)

    fig = rnnSMAP.funPost.plotBox(
        data, labelC=noiseLabelLst, figsize=(12, 6), colorLst='rbgk',
        labelS=labelS, title='Error and uncertainty estimates in temporal test')

    # axes[-1].get_legend().remove()
    fig.show()
    saveFile = os.path.join(saveFolder, 'noise_box')
    fig.subplots_adjust(wspace=0.1)
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')

    # figLeg, axLeg = plt.subplots(figsize=(3, 3))
    # leg = axes[-1].get_legend()
    # axLeg.legend(bp['boxes'], labelS, loc='upper right')
    # axLeg.axis('off')
    # figLeg.show()
    # saveFile = os.path.join(saveFolder, 'noise_box_legend')
    # figLeg.savefig(saveFile+'.eps')


#################################################
if 'plotConf' in doOpt:
    strSigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
    titleLst = [r'$p_{mc}$', r'$p_{x}$', r'$p_{comb}$']
    fig, axes = plt.subplots(ncols=len(titleLst),
                             figsize=(12, 4), sharey=True)
    for iFig in range(0, 3):
        plotLst = list()
        for k in range(0, len(noiseNameLst)):
            plotLst.append(getattr(statConfLst[k], 'conf_'+strSigmaLst[iFig]))
        if iFig == 2:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[iFig], legendLst=noiseLabelLst,
                xlabel='Predicted Probablity', ylabel=None, showDiff=True)
        else:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[iFig], legendLst=None,
                xlabel='Predicted Probablity', ylabel=None, showDiff=True)
        axes[iFig].set_title(titleLst[iFig])
        print(out['rmseLst'])
        if iFig == 0:
            axes[iFig].set_ylabel('Frequency')

    saveFile = os.path.join(saveFolder, 'noise_conf')
    plt.tight_layout()
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
    fig.show()

#################################################
if 'plotConfDist' in doOpt:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    strSigmaLst = ['sigmaMC', 'sigmaX']
    titleLst = [r'CDF of $p_{mc}$', r'CDF of $p_{x}$']
    for iFig in range(0, 2):
        plotLst = list()
        for k in range(0, len(noiseNameLst)):
            plotLst.append(getattr(statConfLst[k], 'conf_'+strSigmaLst[iFig]))
        _, _, out = rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[iFig], legendLst=None,
            xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)
        axes[iFig].set_title(titleLst[iFig])
        print(out['rmseLst'])
        if iFig == 0:
            axes[iFig].set_ylabel('Frequency')
    noiseLst = np.arange(0, 0.11, 0.01)
    strSigmaLst = ['sigmaMC', 'sigmaX']
    legLst = [r'$d(p_{mc})$', r'$d(p_{x})$']
    cLst = 'rb'
    axesDist = [axes[2], axes[2].twinx()]
    for iS in range(0, len(strSigmaLst)):
        distLst = list()
        for iN in range(0, len(noiseNameLst)):
            x = getattr(statConfLst[iN], 'conf_'+strSigmaLst[iS])
            # calculate dist of CDF
            xSort = rnnSMAP.funPost.flatData(x)
            yRank = np.arange(len(xSort))/float(len(xSort)-1)
            dist = np.max(np.abs(xSort - yRank))
            distLst.append(dist)
        axesDist[iS].plot(noiseLst, distLst, color=cLst[iS], label=legLst[iS])
        axesDist[iS].tick_params('y', colors=cLst[iS])
    axesDist[0].set_xlabel(r'$\sigma_{noise}$')
    axesDist[0].legend(loc='upper center')
    axesDist[1].legend(loc='lower center')
    axesDist[0].set_title(r'd to $y=x$')
    plt.tight_layout()
    saveFile = os.path.join(saveFolder, 'noise_dist')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
    fig.show()

#################################################
if 'plotConfLegend' in doOpt:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    strSigmaLst = ['sigmaMC', 'sigmaX']
    titleLst = [r'CDF of $p_{mc}$', r'CDF of $p_{x}$']
    plotLst = list()
    for k in range(0, len(noiseNameLst)):
        plotLst.append(getattr(statConfLst[k], 'conf_'+strSigmaLst[iFig]))
    _, _, out = rnnSMAP.funPost.plotCDF(
        plotLst, ax=axes[0], legendLst=noiseLabelLst,
        xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)

    hh, ll = axes[0].get_legend_handles_labels()
    axes[1].legend(hh, ll, borderaxespad=0, loc='lower left', ncol=1)
    axes[1].axis('off')
    saveFile = os.path.join(saveFolder, 'noise_dist_leg')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
    fig.show()
