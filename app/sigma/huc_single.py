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
# doOpt.append('train')
# doOpt.append('test')
doOpt.append('loadData')
doOpt.append('crdMap')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

hucStrLst = ['02', '05', '18']  # [ref, close, far]
hucLst = np.asarray(hucStrLst, dtype=int)-1
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'hucSingle')
caseStr = ''.join(hucStrLst)
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']

matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams.update({'lines.markersize': 10})

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut,
        syr=2015, eyr=2015, varC='varConstLst_Noah',
        dr=0.5, modelOpt='relu',
        model='cudnn', loss='sigma'
    )
    for kk in range(1, 2):
        for k in range(0, 18):
            trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'
            opt['train'] = trainName
            if kk == 0:
                opt['var'] = 'varLst_soilM'
                opt['out'] = trainName+'_y15_soilM'
            elif kk == 1:
                opt['var'] = 'varLst_Forcing'
                opt['out'] = trainName+'_y15_Forcing'
            cudaID = k % 3
            print(trainName)
            runTrainLSTM.runCmdLine(
                opt=opt, cudaID=cudaID, screenName=opt['out'])

#################################################
if 'test' in doOpt:
    cudaIdLst = np.tile(np.array([0, 1, 2]), 10)
    for k in range(0, 18):
        cudaID = cudaIdLst[k]
        huc_single_test.runTestCmd(
            trainHuc=k, cudaID=cudaID, screenName='huc'+str(k+1).zfill(2))

#################################################
# load data and plot map
if 'loadData' in doOpt:
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    for k in range(0, len(hucLst)):
        testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
        trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
        out = trainName+'_y15_soilM'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        dsLst.append(ds)

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)

#################################################
if 'crdMap' in doOpt:
    import shapefile
    hucShapeFile = '/mnt/sdc/Kuai/Map/HUC/HUC2_CONUS.shp'
    hucShape = shapefile.Reader(hucShapeFile)
    cmap = 'rgb'
    labelLst = ['A (train)', 'B (close)', 'C (far)']
    fig = plt.figure(figsize=(8, 6))
    for shape in hucShape.shapeRecords():
        x = [i[0] for i in shape.shape.points[:]]
        y = [i[1] for i in shape.shape.points[:]]
        plt.plot(x, y, 'k-', label=None)
    for k in range(0, len(hucLst)):
        dataName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=dataName, yrLst=[2016, 2017])
        # plt.plot(ds.crd[:, 1], ds.crd[:, 0], cmap[k] + '*',
        #          label=labelLst[k]+' '+hucStrLst[k])
        plt.plot(ds.crd[:, 1], ds.crd[:, 0], cmap[k] + '*',
                 label=labelLst[k])
    plt.legend(loc='lower left')
    # plt.title(caseStr)
    plt.title('Map of Selected Basins')
    saveFile = os.path.join(saveFolder, caseStr+'_hucMap')
    fig.savefig(saveFile, dpi=1000)


#################################################
if 'plotBox' in doOpt:
    dataTp = (statSigmaLst, statErrLst)
    attrTp = (strSigmaLst, strErrLst)
    titleTp = ('Sigma', 'Error')
    saveFileTp = ('boxSigma_soilM', 'boxErr_soilM')

    for iP in range(0, len(dataTp)):
        statLst = dataTp[iP]
        attrLst = attrTp[iP]
        data = list()
        for k in range(0, len(hucLst)):
            stat = statLst[k]
            tempLst = list()
            for strS in attrLst:
                tempLst.append(getattr(stat, strS))
            data.append(tempLst)
        labelS = attrLst
        # titleStr = 'Temporal Test ' + \
        #     titleTp[iP]+' for model trained on huc'+hucStrLst[0]
        titleStr = 'Temporal Test ' + titleTp[iP]
        fig = rnnSMAP.funPost.plotBox(
            data, labelC=['A', 'B', 'C'], labelS=labelS, figsize=(8, 6),
            title=titleStr)
        saveFile = os.path.join(saveFolder, caseStr+'_'+saveFileTp[iP])
        fig.savefig(saveFile, dpi=1000)

#################################################
if 'plotVS' in doOpt:
    for iE in range(0, len(strErrLst)):
        fig, axes = plt.subplots(
            len(hucLst), len(strSigmaLst), figsize=(10, 10))
        for iS in range(0, len(strSigmaLst)):
            for k in range(0, len(hucLst)):
                statErr = statErrLst[k]
                statSigma = statSigmaLst[k]
                strS = strSigmaLst[iS]
                strE = strErrLst[iE]
                y = getattr(statErr, strE)
                x = getattr(statSigma, strS)
                ub = np.percentile(y, 95)
                lb = np.percentile(y, 5)
                ind = np.logical_and(y >= lb, y <= ub)
                x = x[ind]
                y = y[ind]
                ax = axes[k, iS]
                rnnSMAP.funPost.plotVS(x, y, ax=ax, doRank=False)
                rnnSMAP.funPost.plot121Line(ax)
                if iS == 0:
                    ax.set_ylabel('huc'+hucStrLst[k])
                if k == len(hucLst)-1:
                    ax.set_xlabel(strS)
        fig.suptitle('Temporal '+strE+' on huc'+hucStrLst[0])
        saveFile = os.path.join(saveFolder, caseStr+'_sigmaVs'+strE)
        fig.savefig(saveFile, dpi=1000)
        fig.show()
