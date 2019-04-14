import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import huc_single_test
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

#################################################
# intend to test huc vs huc

doOpt = []
# doOpt.append('train')
# doOpt.append('test')
# doOpt.append('loadData')
doOpt.append('plotBox')


rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'Sigma', 'huc_single3')
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']

#################################################
# load data and plot map
if 'loadData' in doOpt:
    csvFile = '/home/kxf227/work/GitHUB/pyRnnSMAP/Scripts/sigma/hucLst.csv'
    hucMat = np.genfromtxt(csvFile, delimiter=',', dtype=int)
    dsTup = (list(), list())
    statErrTup = (list(), list())
    statSigmaTup = (list(), list())
    hucLst = [[], [], []]
    for k in range(0, 18):
        trainName = 'hucn1_'+str(k+1).zfill(2)+'_v2f1'

        hucLst1 = np.where(hucMat[k] == 1)[0]
        hucLst2 = np.where(hucMat[k] == 3)[0]
        for i in hucLst1:
            for j in hucLst2:
                hucLst[0].append(k)
                hucLst[1].append(i)
                hucLst[2].append(j)
                for kk in [0, 1]:
                    if kk == 0:
                        testName = 'hucn1_'+str(i+1).zfill(2)+'_v2f1'
                    else:
                        testName = 'hucn1_'+str(j+1).zfill(2)+'_v2f1'
                    out = trainName+'_y15_soilM'
                    ds = rnnSMAP.classDB.DatasetPost(
                        rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
                    ds.readData(var='SMAP_AM', field='SMAP')
                    ds.readPred(rootOut=rootOut, out=out,
                                drMC=100, field='LSTM')
                    dsTup[kk].append(ds)
                    statErr = ds.statCalError(
                        predField='LSTM', targetField='SMAP')
                    statSigma = ds.statCalSigma(field='LSTM')
                    statErrTup[kk].append(statErr)
                    statSigmaTup[kk].append(statSigma)
    hucAry = np.array(hucLst)


#################################################
if 'plotBox' in doOpt:
    cLst = 'rb'
    fig, ax = plt.subplots(1, figsize=[8, 8])
    nCase = len(dsTup[0])
    for kk in (0, 1):
        temp = np.ndarray(nCase)
        tempErr = np.ndarray([2, nCase])
        statSigmaLst = statSigmaTup[kk]
        statErrLst = statErrTup[kk]
        temp2 = np.ndarray(nCase)
        for k in range(0, nCase):
            sigma = statSigmaLst[k].sigmaX
            sigma2 = statSigmaLst[k].sigmaMC
            temp[k] = np.mean(sigma)
            temp2[k] = np.mean(sigma2)
            # tempErr[0, k] = np.mean(sigma)-np.percentile(sigma, 25)
            # tempErr[1, k] = np.percentile(sigma, 75)-np.mean(sigma)
        if kk == 0:
            x = temp
            x2 = temp2
        if kk == 1:
            y = temp
            y2 = temp2
        # rnnSMAP.funPost.plotTwinBox(
        #     ax, x, y, xErr, yErr, edgecolor=cLst[j], alpha=0)
    # ind = np.where(xErr < yErr)
    h1 = ax.plot(x, y, 'b*')
    h2 = ax.plot(x2, y2, 'ro', mfc='none')
    ax.set_xlim(0.005, 0.025)
    ax.set_ylim(0.005, 0.025)
    rnnSMAP.funPost.plot121Line(ax)
    # ax.legend(handles=[h1, h2], labels=['sigmaX', 'sigmaMC'])
    ax.legend(['sigmaX', 'sigmaMC'])
    ax.set_xlabel('similar basins')
    ax.set_ylabel('dis-similar basins')

    saveFile = os.path.join(saveFolder, 'hucAllCases')
    fig.savefig(saveFile)
    fig.show()
