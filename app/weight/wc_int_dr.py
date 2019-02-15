import os
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


doOpt = []
# doOpt.append('train')
# doOpt.append('test')
doOpt.append('loadData')
# doOpt.append('plotMap')
doOpt.append('plotBox')
doOpt.append('plotVS')

#################################################
# pre-define options
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['Out_L3_NA']
drLst = [0, 0.2, 0.8]
drStr = ['00', '20', '80']

# test Opt
outLst = ['CONUSv4f1_y15_Forcing_dr00',
          'CONUSv4f1_y15_Forcing_dr20',
          'CONUSv4f1_y15_Forcing',
          'CONUSv4f1_y15_Forcing_dr80']
caseStrLst = ['dr=0', 'dr=0.2', 'dr=0.5', 'dr=0.8']
saveStrLst = ['dr00', 'dr20', 'dr50', 'dr80']
nCase = len(outLst)

testName = 'CONUSv4f1'
yrLst = [2015]
saveSuffix = 'CONUSv4f1_y15'
titleStr = 'Training Set'

# testName = 'CONUSv4f2'
# yrLst = [2015]
# saveSuffix = 'CONUSv4f2_y15'
# titleStr = 'Spatial Test'

# testName = 'CONUSv4f1'
# yrLst = [2017]
# saveSuffix = 'CONUSv4f1_y17'
# titleStr = 'Temporal Test'

wOpt = 'wp'
nPerm = 100
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'weight', wOpt+'_int_dr')
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB,
        rootOut=rootOut,
        syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        train='CONUSv4f1', dr=0.5, modelOpt='relu',
        model='cudnn', loss='mse',
    )
    for k in range(0, len(drLst)):
        opt['dr'] = drLst[k]
        opt['out'] = 'CONUSv4f1_y15_Forcing_dr'+drStr[k]
        cudaID = k % 3
        runTrainLSTM.runCmdLine(
            opt=opt, cudaID=cudaID, screenName=opt['out'])


#################################################
if 'test' in doOpt:
    for k in range(0, nCase):
        out = outLst[k]
        cX, cH = rnnSMAP.funWeight.readWeightDector(
            rootOut=rootOut, out=out, test=testName,
            syr=yrLst[0], eyr=yrLst[-1],
            wOpt=wOpt, nPerm=nPerm, redo=True)

#################################################
if 'loadData' in doOpt:
    cXLst = []
    cHLst = []
    for k in range(0, nCase):
        out = outLst[k]
        cX, cH = rnnSMAP.funWeight.readWeightDector(
            rootOut=rootOut, out=out, test=testName,
            syr=yrLst[0], eyr=yrLst[-1], wOpt=wOpt)
        cXLst.append(cX)
        cHLst.append(cH)

#################################################
if 'plotMap' in doOpt:
    for k in range(0, nCase):
        cX = cXLst[k]
        cH = cHLst[k]
        rX = cX.sum(axis=2)/cX.shape[2]
        rH = cH.sum(axis=2)/cH.shape[2]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yrLst)

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
            saveFolder, 'mapX_'+saveStrLst[k]+'_'+saveSuffix)
        mapFileH = os.path.join(
            saveFolder, 'mapH_'+saveStrLst[k]+'_'+saveSuffix)
        fig = rnnSMAP.funPost.plotMap(
            gridX.mean(axis=2), crd=ds.crdGrid, saveFile=mapFileX,
            title='WCR input to hidden of', showFig=False)
        fig = rnnSMAP.funPost.plotMap(
            gridH.mean(axis=2), crd=ds.crdGrid, saveFile=mapFileH,
            title='WCR hidden to hidden of', showFig=False)

#################################################
if 'plotBox' in doOpt:
    dataBox = []
    for k in range(0, nCase):
        cX = cXLst[k]
        cH = cHLst[k]
        rX = cX.sum(axis=2)/cX.shape[2]
        rH = cH.sum(axis=2)/cH.shape[2]
        dataBox.append([rX.mean(axis=0), rH.mean(axis=0)])

    fig = rnnSMAP.funPost.plotBox(
        dataBox, title='WCR box of '+titleStr,
        labelC=caseStrLst,
        labelS=['input->hidden', 'hidden->hidden'])
    saveFile = os.path.join(saveFolder, 'boxPlot_'+saveSuffix)
    fig.savefig(saveFile)

#################################################
# plot MC dropout vs weight cancellation rate
if 'plotVS' in doOpt:
    fig, axes = plt.subplots(2, nCase, figsize=(12, 6))
    for k in range(0, nCase):
        cX = cXLst[k]
        cH = cHLst[k]
        rX = (cX.sum(axis=2)/cX.shape[2]).transpose()
        rH = (cH.sum(axis=2)/cH.shape[2]).transpose()

        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yrLst)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100,
                    field='LSTM', testBatch=100)
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')

        rnnSMAP.funPost.plotVS(rX.mean(axis=1), statSigma.sigmaMC,
                               ax=axes[0, k], plot121=False, title=caseStrLst[k])
        rnnSMAP.funPost.plotVS(rH.mean(axis=1), statSigma.sigmaMC,
                               ax=axes[1, k], plot121=False, title=caseStrLst[k])

        axes[1, k].set_xlabel('sigmaMC')
        if k == 0:
            axes[0, 0].set_ylabel('WCR input')
            axes[1, 0].set_ylabel('WCR hidden')
    fig.show()
    fig.suptitle('WCR vs sigmaMC ' + titleStr)
    saveFile = os.path.join(saveFolder, 'WCRvsSigmaMC_'+saveSuffix)
    fig.savefig(saveFile)
