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
# doOpt.append('crdMap')
# doOpt.append('plotMap')
# doOpt.append('plotBox')
# doOpt.append('plotVS')
doOpt.append('plotConf')

rootDB = rnnSMAP.kPath['DB_L3_NA']
rootOut = rnnSMAP.kPath['OutSigma_L3_NA']

# hucStrLst = ['02', '05', '18']  # [ref, close, far]
# hucStrLst = ['13', '15', '03']  # [ref, close, far]
hucStrLst = ['16', '14', '12']  # [ref, close, far]

hucLst = np.asarray(hucStrLst, dtype=int)-1
saveFolder = os.path.join(
    rnnSMAP.kPath['dirResult'], 'paperSigma')
caseStr = ''.join(hucStrLst)
strSigmaLst = ['sigmaX', 'sigmaMC', 'sigma']
strErrLst = ['RMSE', 'ubRMSE']

# matplotlib.rcParams.update({'font.size': 16})
# matplotlib.rcParams.update({'lines.linewidth': 2})
# matplotlib.rcParams.update({'lines.markersize': 10})

#################################################
# load data and plot map
if 'loadData' in doOpt:
    statErrLst = list()
    statSigmaLst = list()
    statConfLst = list()
    for k in range(0, len(hucLst)):
        testName = 'hucn1_'+str(hucLst[k]+1).zfill(2)+'_v2f1'
        trainName = 'hucn1_'+str(hucLst[0]+1).zfill(2)+'_v2f1'
        out = trainName+'_y15_Forcing'
        ds = rnnSMAP.classDB.DatasetPost(
            rootDB=rootDB, subsetName=testName, yrLst=[2016, 2017])
        ds.readData(var='SMAP_AM', field='SMAP')
        ds.readPred(out=out, drMC=100, field='LSTM',
                    rootOut=rnnSMAP.kPath['OutSigma_L3_NA'])

        statErr = ds.statCalError(predField='LSTM', targetField='SMAP')
        statErrLst.append(statErr)

        statSigma = ds.statCalSigma(field='LSTM')
        statSigmaLst.append(statSigma.sigmaMC)
        statConf = ds.statCalConf(predField='LSTM', targetField='SMAP')
        statConfLst.append(statConf)


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
if 'plotConf' in doOpt:
    plotLst = list()
    dataTup = (confSigmaMCLst, confSigmaXLst, confSigmaLst)
    titleLst = [r'$\sigma_{mc}$', r'$\sigma_{x}$', r'$\sigma_{comb}$']
    fig, axes = plt.subplots(ncols=len(dataTup), figsize=(12, 4))

    for iFig in range(0, 3):
        plotLst = dataTup[iFig]
        rnnSMAP.funPost.plotCDF(
            plotLst, ax=axes[iFig], cLst='grb',
            legendLst=['A (train)', 'B (close)', 'C (far)'])
        axes[iFig].set_title(titleLst[iFig])
    saveFile = os.path.join(saveFolder, caseStr+'_conf.png')
    fig.show()
    fig.savefig(saveFile, dpi=600)
