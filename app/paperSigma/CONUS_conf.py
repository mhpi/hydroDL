import os
import rnnSMAP
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

trainName = 'CONUSv2f1'
out = trainName+'_y15_Forcing_dr60'
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'paperSigma')

doOpt = []
doOpt.append('loadData')
doOpt.append('plotConf')
# doOpt.append('plotBin')
# doOpt.append('plotProb')

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
plt.tight_layout()

#################################################
# load data
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    statProbLst = list()

    for k in range(0, 2):
        if k == 0:  # validation
            testName = 'CONUSv2f1'
            yr = [2016]
        if k == 1:  # temporal test
            testName = 'CONUSv2f1'
            yr = [2017]
        # if k == 2:  # spatial test
        #     testName = 'CONUSv2fx2'
        #     yr = [2015]

        predField = 'LSTM'
        targetField = 'SMAP'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statProb = ds.statCalProb(predField='LSTM', targetField='SMAP')
        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
        statProbLst.append(statProb)

#################################################
# plot confidence figure
if 'plotConf' in doOpt:
    figTitleLst = ['(a) Validation', '(b) Temporal Test']
    fig, axes = plt.subplots(
        ncols=len(figTitleLst), figsize=(12, 6), sharey=True)
    sigmaStrLst = ['sigmaX', 'sigmaMC', 'sigma']
    legendLst = [r'$p_{x}$', r'$p_{mc}$', r'$p_{comb}$']
    for iFig in range(0, 2):
        statConf = statConfLst[iFig]
        figTitle = figTitleLst[iFig]
        plotLst = list()
        for k in range(0, len(sigmaStrLst)):
            plotLst.append(getattr(statConf, 'conf_'+sigmaStrLst[k]))
        
        _, _, out = rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[iFig], legendLst=legendLst, cLst='grbm',
            xlabel='Error Exceedance Probablity', ylabel=None, showDiff='KS')
        axes[iFig].set_title(figTitle)
        print(out['rmseLst'])        
    axes[0].set_ylabel('Frequency')
    # axes[1].get_legend().remove()
    fig.tight_layout()
    fig.show()
    saveFile = os.path.join(saveFolder, 'CONUS_conf')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')

