import os
import rnnSMAP
from rnnSMAP import arunTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import huc_single_test
import imp
import matplotlib
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intend to test huc vs huc

doOpt = []
doOpt.append('loadData')
# doOpt.append('crdMap')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')
doOpt.append('plotConf')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

# hucStrLst = ['02', '05', '18']  # [ref, close, far]
# hucStrLst = ['13', '15', '03']  # [ref, close, far]
# hucStrLst = ['16', '14', '12']  # [ref, close, far]
hucStrLst = ['04', '02', '03']  # [ref, close, far]
drLst = [0.2, 0.5, 0.8]

hucLst = np.asarray(hucStrLst, dtype=int)-1
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma')
caseStr = ''.join(hucStrLst)
hucLegLst = [hucStrLst[0]+' (train)',
             hucStrLst[1]+' (close)',
             hucStrLst[2]+' (far)']
legStrLst = list()
for iHuc in range(0, 3):
    for dr in drLst:
        if iHuc == 0:
            hucStr = 'train'
        if iHuc == 1:
            hucStr = 'close'
        if iHuc == 2:
            hucStr = 'far'
        legStr = hucStrLst[iHuc]+' '+hucStr+' dr='+str(dr)
        legStrLst.append(legStr)


#################################################
# load data and plot map
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(hucLst)):
        testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
        trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
        dsTemp = list()
        statErrTemp = list()
        statSigmaTemp = list()
        statConfTemp = list()
        for dr in drLst:
            if dr == 0.5:
                out = trainName+'_y15_Forcing'
            else:
                drStr = '%02d' % (dr*100)
                out = trainName+'_y15_Forcing_dr'+drStr
            ds = rnnSMAP.classDB.DatasetPost(
                rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
            ds.readData(var='SMAP_AM', field='SMAP')
            ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
            statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
            statSigma = ds.statCalSigma(field='LSTM')
            statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
            dsTemp.append(ds)
            statErrTemp.append(statErr)
            statSigmaTemp.append(statSigma)
            statConfTemp.append(statConf)
        dsLst.append(dsTemp)
        statErrLst.append(statErrTemp)
        statSigmaLst.append(statSigmaTemp)
        statConfLst.append(statConfTemp)

#################################################
if 'plotConf' in doOpt:
    strConfLst = ['conf_sigmaMC', 'conf_sigmaX', 'conf_sigma']
    titleLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
    fig, axes = plt.subplots(ncols=len(strConfLst), figsize=(12, 4))
    cTuple = (plt.cm.Greens(drLst), plt.cm.Reds(drLst), plt.cm.Blues(drLst))
    cLst = np.concatenate(cTuple, axis=0)
    for k in range(0, len(strConfLst)):
        plotLst = list()
        for iHuc in range(0, 3):
            for iDr in range(0, len(drLst)):
                temp = getattr(statConfLst[iHuc][iDr], strConfLst[k])
                plotLst.append(temp)
        rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[k], cLst=cLst, legendLst=legStrLst)
        axes[k].set_title(titleLst[k])
    saveFile = os.path.join(saveFolder, caseStr+'_conf_dr.png')
    fig.show()
    fig.savefig(saveFile, dpi=600)

#################################################
if 'plotBox' in doOpt:
    data = list()
    strSigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
    strErrLst = ['ubRMSE', 'Bias']
    for strSigma in strSigmaLst:
        temp = list()
        for iHuc in range(0, 3):
            for iDr in range(0, len(drLst)):
                statSigma = statSigmaLst[iHuc][iDr]
                temp.append(getattr(statSigma, strSigma))
        data.append(temp)
    for strErr in strErrLst:
        temp = list()
        for iHuc in range(0, 3):
            for iDr in range(0, len(drLst)):
                statErr = statErrLst[iHuc][iDr]
                temp.append(getattr(statErr, strErr))
        data.append(temp)
    labelC = [r'$\sigma_{mc}$', r'$\sigma_{x}$',
              r'$\sigma_{comb}$', 'ubRMSE', 'Bias']
    cTuple = (plt.cm.Greens(drLst), plt.cm.Reds(drLst), plt.cm.Blues(drLst))
    cLst = np.concatenate(cTuple, axis=0)
    fig = rnnSMAP.funPost.plotBox(
        data, labelS=legStrLst, labelC=labelC,
        colorLst=cLst, figsize=(12, 4), sharey=False)
    fig.subplots_adjust(wspace=0.5)
    saveFile = os.path.join(saveFolder, caseStr+'_box_dr')
    fig.savefig(saveFile, dpi=600)
