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
doOpt.append('test')
doOpt.append('loadData')
# doOpt.append('plotMap')
doOpt.append('plotBox')
# doOpt.append('plotVS')
doOpt.append('plotErr')

#################################################
# pre-define options
rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['Out_L3_NA']
drLst = [0, 0.2, 0.5, 0.8]
drLegLst = ['dr=0', 'dr=0.2', 'dr=0.5', 'dr=0.8']
drSaveLst = ['00', '20', '50', '80']

hucLegLst = ['train', 'close', 'far']
yrLst = [2015]

wOpt = 'wp'
nPerm = 100
saveFolder = os.path.join(rnnSMAP.kPath['dirResult'], 'weight', wOpt+'_huc_dr')
if not os.path.exists(saveFolder):
    os.mkdir(saveFolder)

hucCaseLst = [['16', '14', '12'],
              ['13', '15', '03'],
              ['02', '05', '18']]

#################################################
if 'train' in doOpt:
    opt = rnnSMAP.classLSTM.optLSTM(
        rootDB=rootDB, rootOut=rootOut, syr=2015, eyr=2015,
        var='varLst_Forcing', varC='varConstLst_Noah',
        modelOpt='relu', model='cudnn', loss='mse',
    )
    for k in range(0, len(drLst)):
        for ihuc in [2, 13, 16]:
            trainName = 'hucn1_'+str(ihuc).zfill(2)+'_v2f1'
            opt['train'] = trainName
            opt['dr'] = drLst[k]
            opt['out'] = trainName+'_y15_Forcing_dr'+drSaveLst[k]
            cudaID = k % 3
            runTrainLSTM.runCmdLine(
                opt=opt, cudaID=cudaID, screenName=opt['out'])
            # rnnSMAP.funLSTM.trainLSTM(opt)


for hucSaveLst in hucCaseLst:
# hucSaveLst = hucCaseLst[0]  # [ref, close, far]
    hucLst = np.asarray(hucSaveLst, dtype=int)-1

    #################################################
    # test Opt
    if 'test' in doOpt:
        for i in range(0, len(hucSaveLst)):
            for j in range(0, len(drSaveLst)):
                testName = 'hucn1_' + str(hucLst[i]+1).zfill(2)+'_v2f1'
                trainName = 'hucn1_' + str(hucLst[0]+1).zfill(2)+'_v2f1'
                out = trainName+'_y15_Forcing_dr'+drSaveLst[j]
                cX, cH = rnnSMAP.funWeight.readWeightDector(
                    rootOut=rootOut, out=out, test=testName,
                    syr=yrLst[0], eyr=yrLst[-1],
                    wOpt=wOpt, nPerm=nPerm, redo=True)

    #################################################
    if 'loadData' in doOpt:
        cXLst = []
        cHLst = []
        for i in range(0, len(hucSaveLst)):
            tempX = []
            tempH = []
            for j in range(0, len(drSaveLst)):
                testName = 'hucn1_' + str(hucLst[i]+1).zfill(2)+'_v2f1'
                trainName = 'hucn1_' + str(hucLst[0]+1).zfill(2)+'_v2f1'
                out = trainName+'_y15_Forcing_dr'+drSaveLst[j]
                cX, cH = rnnSMAP.funWeight.readWeightDector(
                    rootOut=rootOut, out=out, test=testName,
                    syr=yrLst[0], eyr=yrLst[-1], wOpt=wOpt)
                tempX.append(cX)
                tempH.append(cH)
            cXLst.append(tempX)
            cHLst.append(tempH)

    #################################################
    if 'plotBox' in doOpt:
        dataBoxX = []
        dataBoxH = []
        for i in range(0, len(hucSaveLst)):
            tempX = []
            tempH = []
            for j in range(0, len(drSaveLst)):
                cX = cXLst[i][j]
                cH = cHLst[i][j]
                rX = cX.sum(axis=2)/cX.shape[2]
                rH = cH.sum(axis=2)/cH.shape[2]
                tempX.append(rX.mean(axis=0))
                tempH.append(rH.mean(axis=0))
            dataBoxX.append(tempX)
            dataBoxH.append(tempH)

        fig = rnnSMAP.funPost.plotBox(
            dataBoxX, title='weight cancellation input->hidden',
            labelC=hucLegLst, labelS=drLegLst)
        saveFile = os.path.join(saveFolder, 'boxPlotX_'+str().join(hucSaveLst))
        fig.savefig(saveFile)

        fig = rnnSMAP.funPost.plotBox(
            dataBoxH, title='weight cancellation hidden->hidden',
            labelC=hucLegLst, labelS=drLegLst)
        saveFile = os.path.join(saveFolder, 'boxPlotH_'+str().join(hucSaveLst))
        fig.savefig(saveFile)

    #################################################
    if 'plotErr' in doOpt:
        plotDataLst = []
        strE = 'RMSE'
        for i in range(0, len(hucSaveLst)):
            temp = []
            for j in range(0, len(drSaveLst)):
                testName = 'hucn1_' + str(hucLst[i]+1).zfill(2)+'_v2f1'
                trainName = 'hucn1_' + str(hucLst[0]+1).zfill(2)+'_v2f1'
                out = trainName+'_y15_Forcing_dr'+drSaveLst[j]

                ds = rnnSMAP.classDB.DatasetPost(
                    rootDB=rootDB, subsetName=testName, yrLst=[2015])
                ds.readData(var='SMAP_AM', field='SMAP')
                ds.readPred(rootOut=rootOut, out=out, drMC=0, field='LSTM')
                statErr = ds.statCalError(
                    predField='LSTM', targetField='SMAP')
                temp.append(getattr(statErr, strE))
            plotDataLst.append(temp)
        strE = 'RMSE'

        fig = rnnSMAP.funPost.plotBox(
            plotDataLst, labelC=hucLegLst, labelS=drLegLst, title='Spatial Test '+strE)
        saveFile = os.path.join(saveFolder, 'box'+strE+'_'+str().join(hucSaveLst))
        fig.savefig(saveFile)
