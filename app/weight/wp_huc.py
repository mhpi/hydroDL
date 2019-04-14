import os
import numpy as np
import matplotlib.pyplot as plt
import rnnSMAP
import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()


hucCaseLst = [['16', '14', '12'],
              ['13', '15', '03'],
              ['02', '05', '18']]
for hucStrLst in hucCaseLst:
    # hucStrLst = ['16', '14', '12']  # [ref, close, far]
    hucLst = np.asarray(hucStrLst, dtype=int)-1

    saveFolder = os.path.join(
        rnnSMAP.kPath['dirResult'], 'weight', 'wp_huc')
    rootOut = rnnSMAP.kPath['OutSigma_L3_NA']
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    nCase = len(hucStrLst)
    caseStr = ['train', 'close', 'far']
    wOpt = 'wp'
    nPerm = 100

    doOpt = []
    doOpt.append('test')
    doOpt.append('plotBox')
    doOpt.append('plotVS')

    if 'test' in doOpt:
        for k in range(0, nCase):
            testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
            trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
            out = trainName+'_y15_Forcing'
            syr = 2016
            eyr = 2017
            cX, cH = rnnSMAP.funWeight.readWeightDector(
                rootOut=rootOut, out=out, test=testName, syr=syr, eyr=eyr,
                wOpt=wOpt, nPerm=nPerm, redo=True)

    #################################################
    # plot box
    if 'plotBox' in doOpt:
        dataBox = []
        for k in range(0, nCase):
            testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
            trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
            out = trainName+'_y15_Forcing'
            syr = 2016
            eyr = 2017
            cX, cH = rnnSMAP.funWeight.readWeightDector(
                rootOut=rootOut, out=out, test=testName, syr=syr, eyr=eyr,
                wOpt=wOpt)
            rX = cX.sum(axis=2)/cX.shape[2]
            rH = cH.sum(axis=2)/cH.shape[2]
            dataBox.append([rX.mean(axis=0), rH.mean(axis=0)])

        fig = rnnSMAP.funPost.plotBox(
            dataBox, title='box of weight cancellation',
            labelC=['train-'+hucStrLst[0], 'close-' +
                    hucStrLst[1], 'far-'+hucStrLst[2]],
            labelS=['input->hidden', 'hidden->hidden'])
        saveFile = os.path.join(saveFolder, 'boxPlot'+str().join(hucStrLst))
        fig.savefig(saveFile)

    #################################################
    # plot MC dropout vs weight cancellation rate
    if 'plotVS' in doOpt:
        fig, axes = plt.subplots(nCase, 2, figsize=(8, 10))
        for k in range(0, nCase):
            testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
            trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
            out = trainName+'_y15_Forcing'
            syr = 2016
            eyr = 2017
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
        saveFile = os.path.join(
            saveFolder, 'WCRvsSigmaMC'+str().join(hucStrLst))
        fig.savefig(saveFile)
