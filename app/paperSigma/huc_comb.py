import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intervals temporal test
doOpt = []
# doOpt.append('train')
doOpt.append('test')
# doOpt.append('crdMap')
doOpt.append('plotBox')
# doOpt.append('plotConf')


hucLst = ['04051118', '03101317', '02101114',
          '01020304', '02030406', '14151617']
hucTitleLst = ['HUC 04+05+11+18', 'HUC 03+10+13+17', 'HUC 02+10+11+14',
               'HUC 02+03+04+06', 'HUC 01+02+03+04', 'HUC 14+15+16+17']

# hucLst = ['04051118', '03101317', '14151617', '02030406']
# hucTitleLst = ['HUC 04+05+11+18', 'HUC 03+10+13+17',
#                'HUC 14+15+16+17', 'HUC 02+03+04+06']
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma')
matplotlib.rcParams.update({'font.size': 14})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})
matplotlib.rcParams.update({'legend.fontsize': 12})
cLst = 'myrgcb'

#################################################
if 'test' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(hucLst)):
        trainName = hucLst[k]+'_v2f1'
        out = trainName+'_y15_Forcing_dr60'
        testName = 'ex_'+hucLst[k]+'_v2f1'

        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statErrLst.append(statErr)
        statSigma = ds.statCalSigma(field='LSTM')
        statSigmaLst.append(statSigma)
        statConf = ds.statCalConf(
            predField='LSTM', targetField='SMAP', rmBias=True)
        statConfLst.append(statConf)

#################################################
if 'crdMap' in doOpt:
    import shapefile
    hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS.shp'
    hucShape = shapefile.Reader(hucShapeFile)
    fig, axes = plt.subplots(6, 1, figsize=(3, 9))
    for iC in range(0, len(hucLst)):
        # for iC in range(0, 1):
        ax = axes[iC]
        for iHuc in range(0, 18):
            shape = hucShape.shapeRecords()[iHuc]
            parts = shape.shape.parts
            points = shape.shape.points
            ind1 = list(parts)
            ind2 = list(parts)[1:]
            ind2.append(len(points)-1)
            for k in range(0, len(ind1)):
                pp = points[ind1[k]:ind2[k]]
                x = [i[0] for i in pp[::20]]
                y = [i[1] for i in pp[::20]]
                ax.plot(x, y, 'k-', label=None, linewidth=1)
        hucCombStr = hucLst[iC]
        for iHuc in range(0, 4):
            huc = int(hucCombStr[iHuc*2:iHuc*2+2])-1
            shape = hucShape.shapeRecords()[huc]
            parts = shape.shape.parts
            points = shape.shape.points
            ind1 = list(parts)
            ind2 = list(parts)[1:]
            ind2.append(len(points)-1)
            for k in range(0, len(ind1)):
                pp = points[ind1[k]:ind2[k]]
                x = [i[0] for i in pp[::20]]
                y = [i[1] for i in pp[::20]]
                if k == 0:
                    ax.fill(
                        x, y, 'k-', label=hucLst[iC], color=cLst[iC])
                else:
                    ax.fill(x, y, 'k-', color=cLst[iC])
        ax.set_aspect('equal', 'box')
        ax.set_title(hucLst[iC])
        ax.axis('off')
    fig.show()
    fig.tight_layout()
    saveFile = os.path.join(saveFolder, 'hucComb_map')
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')


#################################################
if 'plotConf' in doOpt:
    strConfLst = ['conf_sigmaMC', 'conf_sigmaX', 'conf_sigma']
    titleLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
    fig, axes = plt.subplots(ncols=len(strConfLst), figsize=(12, 4))

    for k in range(0, len(strConfLst)):
        plotLst = list()
        for iHuc in range(0, len(hucLst)):
            temp = getattr(statConfLst[iHuc], strConfLst[k])
            plotLst.append(temp)
        if k < 2:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=None, cLst=cLst,
                xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)
        else:
            _, _, out = rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], legendLst=hucLst, cLst=cLst,
                xlabel=r'$P_{ee}$', ylabel=None, showDiff=False)
        print(out['rmseLst'])
        if k == 0:
            axes[k].set_ylabel('Frequency')
        axes[k].set_title(titleLst[k])
    saveFile = os.path.join(saveFolder, 'hucComb_conf')
    fig.show()
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')


#################################################
if 'plotBox' in doOpt:
    data = list()
    strSigmaLst = ['sigmaMC', 'sigmaMC', 'sigma']
    strErrLst = ['ubRMSE']
    labelC = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$',
              r'$\sigma_{mc} / \sigma_{x}$',
              'ubRMSE']
    for strSigma in strSigmaLst:
        temp = list()
        for k in range(0, len(hucLst)):
            statSigma = statSigmaLst[k]
            temp.append(getattr(statSigma, strSigma))
        data.append(temp)

    temp = list()
    for k in range(0, len(hucLst)):
        statSigma = statSigmaLst[k]
        rate = getattr(statSigma, 'sigmaMC')/getattr(statSigma, 'sigmaX')
        temp.append(rate)
    data.append(temp)

    for strErr in strErrLst:
        temp = list()
        for k in range(0, len(hucLst)):
            statErr = statErrLst[k]
            temp.append(getattr(statErr, strErr))
        data.append(temp)
    fig = rnnSMAP.funPost.plotBox(
        data, labelS=None, labelC=labelC, colorLst=cLst,
        figsize=(12, 4), sharey=False)
    fig.subplots_adjust(wspace=0.5)
    saveFile = os.path.join(saveFolder, 'hucComb_box')
    fig.show()
    fig.savefig(saveFile, dpi=100)
    fig.savefig(saveFile+'.eps')
