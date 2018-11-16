import os
import rnnSMAP
import matplotlib.pyplot as plt

import imp
imp.reload(rnnSMAP)
rnnSMAP.reload()

figTitleLst = ['Training', 'Temporal Test', 'Spatial Test']
fig, axes = plt.subplots(ncols=len(figTitleLst), figsize=(12, 4))

for iFig in range(0, 3):
    # iFig = 0
    figTitle = figTitleLst[iFig]
    if iFig == 0:
        testName = 'CONUSv2f1'
        yr = [2015]
    if iFig == 1:
        testName = 'CONUSv2f1'
        yr = [2016, 2017]
    if iFig == 2:
        testName = 'CONUSv2f2'
        yr = [2015]

    trainName = 'CONUSv2f1'
    out = trainName+'_y15_Forcing'
    rootDB = rnnSMAP.kPath['DB_L3_NA']
    rootOutLst = [rnnSMAP.kPath['Out_L3_NA'],
                    rnnSMAP.kPath['OutSigma_L3_NA'],
                    rnnSMAP.kPath['OutSigma_L3_NA']]
    caseStrLst = ['sigmaMC', 'sigmaX', 'sigma']
    nCase = len(caseStrLst)
    saveFolder = os.path.join(
        rnnSMAP.kPath['dirResult'], 'paperSigma')

    #################################################
    # test
    predField = 'LSTM'
    targetField = 'SMAP'
    dsLst = list()
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    statNormLst = list()
    for k in range(0, len(caseStrLst)):
        rootOut = rootOutLst[k]
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=yr)
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(rootOut=rootOut, out=out, drMC=100, field='LSTM')
        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statSigma = ds.statCalSigma(field='LSTM')
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
        statNorm = rnnSMAP.classPost.statNorm(
            statSigma=statSigma, dataPred=ds.LSTM, dataTarget=ds.SMAP)

        dsLst.append(ds)
        statErrLst.append(statErr)
        statSigmaLst.append(statSigma)
        statConfLst.append(statConf)
        statNormLst.append(statNorm)

    #################################################
    # plot confidence figure
    plotLst = list()
    for k in range(0, len(caseStrLst)):
        statNorm = statNormLst[k]
        plotLst.append(getattr(statNorm, 'yNorm_'+caseStrLst[k]))
    legendLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
    rnnSMAP.funPost.plotCDF(
        plotLst, ax=axes[iFig], legendLst=legendLst, ref='norm', cLst='grb',
        xlabel='Normalized Prediction')
    axes[iFig].set_title(figTitle)
    axes[iFig].set_xlim([-5, 5])

fig.show()
fig.savefig(saveFolder+'/CONUS_dist.png', dpi=1200)
