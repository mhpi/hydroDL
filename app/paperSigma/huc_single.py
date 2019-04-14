import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
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
doOpt.append('crdMap')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')
# doOpt.append('plotConf')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

for kkk in range(0, 3):
    if kkk == 0:
        hucStrLst = ['16', '14', '12']  # [ref, close, far]
    if kkk == 1:
        hucStrLst = ['13', '15', '03']  # [ref, close, far]
    if kkk == 2:
        hucStrLst = ['02', '05', '18']  # [ref, close, far]

    hucLst = np.asarray(hucStrLst, dtype=int)-1
    saveFolder = os.path.join(
        rnnSMAP.kPath['dirResult'], 'paperSigma')
    caseStr = ''.join(hucStrLst)
    legendLst = [hucStrLst[0]+' (train)',
                 hucStrLst[1]+' (close)',
                 hucStrLst[2]+' (far)']

    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.rcParams.update({'lines.linewidth': 2})
    matplotlib.rcParams.update({'lines.markersize': 10})
    matplotlib.rcParams.update({'legend.fontsize': 16})

    #################################################
    # load data and plot map
    if 'loadData' in doOpt:
        statErrLst = list()
        statSigmaLst = list()
        statConfLst = list()
        for k in range(0, len(hucLst)):
            testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
            trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
            out = trainName+'_y15_Forcing_dr60'
            ds = rnnSMAP.classDB.DatasetPost(
                rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
            ds.readData(var='SMAP_AM', field='SMAP')
            ds.readPred(out=out, drMC=100, field='LSTM',
                        rootOut=rnnSMAP.kPath['OutSigma_L3_NA'])

            statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
            statErrLst.append(statErr)

            statSigma = ds.statCalSigma(field='LSTM')
            statSigmaLst.append(statSigma)
            statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
            statConfLst.append(statConf)

    #################################################
    if 'crdMap' in doOpt:
        import shapefile
        hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS.shp'
        hucShape = shapefile.Reader(hucShapeFile)
        cmap = 'rgb'
        labelLst = ['A (train)', 'B (close)', 'C (far)']
        fig, axes = plt.subplots(2, 1, figsize=(3, 3))
        ax = axes[0]
        for iHuc in range(0, 18):
            shape = hucShape.shapeRecords()[iHuc]
            parts = shape.shape.parts
            points = shape.shape.points
            ind1 = list(parts)
            ind2 = list(parts)[1:]
            ind2.append(len(points)-1)
            for k in range(0, len(ind1)):
                pp = points[ind1[k]:ind2[k]]
                x = [i[0] for i in pp[::10]]
                y = [i[1] for i in pp[::10]]
                ax.plot(x, y, 'k-', label=None, linewidth=1)
        cLst = 'rgb'
        for iHuc in range(0, len(hucLst)):
            shape = hucShape.shapeRecords()[hucLst[iHuc]]
            parts = shape.shape.parts
            points = shape.shape.points
            ind1 = list(parts)
            ind2 = list(parts)[1:]
            ind2.append(len(points)-1)
            for k in range(0, len(ind1)):
                pp = points[ind1[k]:ind2[k]]
                x = [i[0] for i in pp[::10]]
                y = [i[1] for i in pp[::10]]
                if k == 0:
                    ax.fill(
                        x, y, 'k-', label=legendLst[iHuc], color=cLst[iHuc])
                else:
                    ax.fill(x, y, 'k-', color=cLst[iHuc])
        ax.set_aspect('equal', 'box')
        hh, ll = ax.get_legend_handles_labels()
        axes[1].legend(hh, ll)
        axes[1].axis('off')
        axes[0].axis('off')
        fig.show()
        saveFile = os.path.join(saveFolder, caseStr+'_hucMap')
        fig.savefig(saveFile, dpi=100)
        fig.savefig(saveFile+'.eps')

    #################################################
    if 'plotConf' in doOpt:
        strConfLst = ['conf_sigmaMC', 'conf_sigmaX', 'conf_sigma']
        titleLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
        fig, axes = plt.subplots(ncols=len(strConfLst), figsize=(8, 4))

        for k in range(0, len(strConfLst)):
            plotLst = list()
            for iHuc in range(0, 3):
                temp = getattr(statConfLst[iHuc], strConfLst[k])
                plotLst.append(temp)
            rnnSMAP.funPost.plotCDF(
                plotLst, ax=axes[k], cLst='grb', legendLst=legendLst)
            axes[k].set_title(titleLst[k])
        saveFile = os.path.join(saveFolder, caseStr+'_conf.png')
        fig.show()
        fig.savefig(saveFile, dpi=100)

    #################################################
    if 'plotBox' in doOpt:
        data = list()
        # strSigmaLst = ['sigmaMC', 'sigmaX', 'sigma']
        strSigmaLst = ['sigmaMC', 'sigmaX']
        strErrLst = ['ubRMSE']
        labelC = [r'$\sigma_{mc}$', r'$\sigma_{x}$', 'ubRMSE']
        for strSigma in strSigmaLst:
            temp = list()
            for k in range(0, len(hucLst)):
                statSigma = statSigmaLst[k]
                temp.append(getattr(statSigma, strSigma))
            data.append(temp)
        for strErr in strErrLst:
            temp = list()
            for k in range(0, len(hucLst)):
                statErr = statErrLst[k]
                temp.append(getattr(statErr, strErr))
            data.append(temp)
        fig = rnnSMAP.funPost.plotBox(
            data, labelS=None, labelC=labelC,
            colorLst='rgb', figsize=(9, 3), sharey=False)
        fig.subplots_adjust(wspace=0.5)
        fig.tight_layout()
        saveFile = os.path.join(saveFolder, caseStr+'_box')
        fig.savefig(saveFile, dpi=100)
        fig.savefig(saveFile+'.eps')
