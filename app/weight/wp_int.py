import os
import numpy as np
import rnnSMAP
import imp
import matplotlib.pyplot as plt
imp.reload(rnnSMAP)
rnnSMAP.reload()

out = 'CONUSv4f1_y15_Forcing'

saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'weight', 'wp_int')
rootOut = rnnSMAP.kPath['Out_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
dataLst = ['CONUSv4f1', 'CONUSv4f1', 'CONUSv4f2']
caseStr = ['train', 'spatial test', 'temporal test']
syLst = [2015, 2017, 2015]
eyLst = [2015, 2017, 2015]
nCase = len(dataLst)
wOpt = 'wp'
nPerm = 100

doOpt = []
doOpt.append('test')
doOpt.append('plotMap')
doOpt.append('plotBox')
doOpt.append('plotVS')

#################################################
if 'test' in doOpt:
    for k in range(0, nCase):
        testName = dataLst[k]
        syr = syLst[k]
        eyr = eyLst[k]
        cX, cH = rnnSMAP.funWeight.readWeightDector(
            rootOut=rootOut, out=out, test=testName, syr=syr, eyr=eyr,
            wOpt=wOpt, nPerm=nPerm, redo=True)

#################################################
if 'plotMap' in doOpt:
    for k in range(0, nCase):
        testName = dataLst[k]
        syr = syLst[k]
        eyr = eyLst[k]
        cX, cH = rnnSMAP.funWeight.readWeightDector(
            rootOut=rootOut, out=out, test=testName, syr=syr, eyr=eyr,
            wOpt=wOpt)
        rX = cX.sum(axis=2)/cX.shape[2]
        rH = cH.sum(axis=2)/cH.shape[2]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=range(syr, eyr+1))

        nt, ngrid = rX.shape
        indY = ds.crdGridInd[:, 0]
        indX = ds.crdGridInd[:, 1]
        ny = len(ds.crdGrid[0])
        nx = len(ds.crdGrid[1])
        gridX = np.full([ny, nx, nt], np.nan)
        gridH = np.full([ny, nx, nt], np.nan)
        gridX[indY, indX, :] = np.transpose(rX)
        gridH[indY, indX, :] = np.transpose(rH)
        mapFileX = os.path.join(
            saveFolder, 'mapX_'+testName+'_'+str(syr)+'_'+str(eyr))
        mapFileH = os.path.join(
            saveFolder, 'mapH_'+testName+'_'+str(syr)+'_'+str(eyr))
        fig = rnnSMAP.funPost.plotMap(
            gridX.mean(axis=2), crd=ds.crdGrid, saveFile=mapFileX,
            title='weight cancellation input to hidden', showFig=False)
        fig = rnnSMAP.funPost.plotMap(
            gridH.mean(axis=2), crd=ds.crdGrid, saveFile=mapFileH,
            title='weight cancellation hidden to hidden', showFig=False)

#################################################
if 'plotBox' in doOpt:
    dataBox = []
    for k in range(0, nCase):
        testName = dataLst[k]
        syr = syLst[k]
        eyr = eyLst[k]
        cX, cH = rnnSMAP.funWeight.readWeightDector(
            rootOut=rootOut, out=out, test=testName, syr=syr, eyr=eyr,
            wOpt=wOpt)
        rX = cX.sum(axis=2)/cX.shape[2]
        rH = cH.sum(axis=2)/cH.shape[2]
        dataBox.append([rX.mean(axis=0), rH.mean(axis=0)])

    fig = rnnSMAP.funPost.plotBox(
        dataBox, title='box of weight cancellation',
        labelC=['train', 'spatial test', 'temporal test'],
        labelS=['input->hidden', 'hidden->hidden'])
    saveFile = os.path.join(saveFolder, 'boxPlot_CONUSv4')
    fig.savefig(saveFile)

#################################################
# plot MC dropout vs weight cancellation rate
if 'plotVS' in doOpt:
    fig, axes = plt.subplots(nCase, 2, figsize=(8, 10))
    for k in range(0, nCase):
        testName = dataLst[k]
        syr = syLst[k]
        eyr = eyLst[k]
        cX, cH = rnnSMAP.funWeight.readWeightDector(
            rootOut=rootOut, out=out, test=testName, syr=syr, eyr=eyr,
            wOpt=wOpt)
        rX = (cX.sum(axis=2)/cX.shape[2]).transpose()
        rH = (cH.sum(axis=2)/cH.shape[2]).transpose()

        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=range(syr, eyr+1))
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100,
                    field='LSTM', testBatch=100)
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')

        rnnSMAP.funPost.plotVS(rX.mean(axis=1), statSigma.sigmaMC,
                               ax=axes[k, 0], plot121=False, title=caseStr[k])
        rnnSMAP.funPost.plotVS(rH.mean(axis=1), statSigma.sigmaMC,
                               ax=axes[k, 1], plot121=False, title=caseStr[k])

        axes[k, 0].set_ylabel('sigmaMC')
        if k == nCase-1:
            axes[k, 0].set_xlabel('WCR input')
            axes[k, 1].set_xlabel('WCR hidden')
    fig.show()
    saveFile = os.path.join(saveFolder, 'WCRvsSigmaMC_CONUSv4')
    fig.savefig(saveFile)


# fig, axes = plt.subplots()
# rnnSMAP.funPost.plotVS(
#     rX.flatten(), statSigma.sigmaMC_mat.flatten(), ax=axes, plot121=False)
# fig.show()
