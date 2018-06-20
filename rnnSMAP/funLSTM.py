import os
import torch
import time
import numpy as np
import pandas as pd
from argparse import Namespace

from . import classLSTM
from . import classDB
from . import kPath


def loadOptLSTM(outFolder):
    optFile = os.path.join(outFolder, 'opt.txt')
    optTemp = dict()  # type: dict
    with open(optFile, 'r') as ff:
        for line in ff:
            lstTemp = line.strip().split(': ')
            if len(lstTemp) == 1:
                lstTemp = line.strip().split(': ')
                optTemp[lstTemp[0]] = None
            else:
                optTemp[lstTemp[0]] = lstTemp[1]

    opt = classLSTM.optLSTM(**optTemp)
    if opt['rootDB'] is None:
        opt['rootDB'] = kPath.DBSMAP_L3_Global
    if opt['rootOut'] is None:
        opt['rootOut'] = kPath.OutSMAP_L3_Global
    return opt


def saveOptLSTM(outFolder, opt: classLSTM.optLSTM):
    optFile = os.path.join(outFolder, 'opt.txt')
    if os.path.isfile(optFile):
        print('Warning: overwriting existed optFile. Delete manually.')

    with open(optFile, 'w') as ff:
        i = 0
        for key in opt:
            if i != len(opt):
                ff.write(key+': '+str(opt[key])+'\n')
            else:
                ff.write(key+': '+str(opt[key]))
            i = i+1


def trainLSTM(optDict: classLSTM.optLSTM):
    opt = Namespace(**optDict)

    outFolder = os.path.join(opt.rootOut, opt.out)
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    saveOptLSTM(outFolder, optDict)
    runFile = os.path.join(outFolder, 'runFile.csv')

    #############################################
    # load data
    #############################################
    dataset = classDB.Dataset(
        rootDB=opt.rootDB, subsetName=opt.train,
        yrLst=np.arange(opt.syr, opt.eyr+1),
        var=(opt.var, opt.varC), targetName=opt.target)
    dataset.readInput(loadNorm=True)
    dataset.readTarget(loadNorm=True)

    x = dataset.normInput
    y = dataset.normTarget
    ngrid, nt, nx = x.shape
    ny = 1
    y = np.reshape(y, [ngrid, nt, ny])
    rho = opt.rho
    nbatch = opt.nbatch

    #############################################
    # Create Model
    #############################################

    # model = classLSTM.modelLSTM(
    #     nx=nx, ny=ny, hiddenSize=opt.hiddenSize,gpu=opt.gpu)
    # model = classLSTM.modelLSTMcell(
    #     nx=nx, ny=ny, hiddenSize=opt.hiddenSize, gpu=opt.gpu, dr=opt.dr)
    model = classLSTM.modelLSTM_Kuai(
        nx=nx, ny=ny, hiddenSize=opt.hiddenSize, drMethod=opt.drMethod, gpu=opt.gpu)
    model.zero_grad()
    if torch.cuda.is_available():
        model = model.cuda(opt.gpu)

    crit = torch.nn.MSELoss()
    if torch.cuda.is_available():
        crit = crit.cuda(opt.gpu)

    # construct model before optim will automatically make it cuda
    optim = torch.optim.Adadelta(model.parameters())

    xTrain = torch.zeros([rho, nbatch, nx], requires_grad=False)
    yTrain = torch.zeros([rho, nbatch, ny], requires_grad=False)

    #############################################
    # Train Model
    #############################################
    nEpoch = opt.nEpoch
    nIterEpoch = np.ceil(np.log(0.01)/np.log(1-nbatch*rho/ngrid/nt))
    nIterEpoch = nIterEpoch.astype(int)
    saveEpoch = opt.saveEpoch

    iEpoch = 1
    lossEpoch = 0
    timeEpoch = time.time()
    rf = open(runFile, 'a+')
    for iIter in range(1, nEpoch*nIterEpoch+1):
        #############################################
        # Training iteration
        # random a piece of time series
        iGrid = np.random.randint(0, ngrid, [nbatch])
        iT = np.random.randint(0, nt-rho, [nbatch])

        for k in range(nbatch):
            temp = x[iGrid[k]:iGrid[k]+1, np.arange(iT[k], iT[k]+rho), :]
            xTrain[:, k:k+1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
            temp = y[iGrid[k]:iGrid[k]+1, np.arange(iT[k], iT[k]+rho)]
            yTrain[:, k:k+1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))

        if torch.cuda.is_available():
            xTrain = xTrain.cuda(opt.gpu)
            yTrain = yTrain.cuda(opt.gpu)

        yP = model(xTrain)
        loc0 = yTrain != yTrain
        loc1 = yTrain == yTrain
        yT = torch.empty(rho, nbatch, 1).cuda(opt.gpu)
        yT[loc0] = yP[loc0]
        yT[loc1] = yTrain[loc1]
        yT = yT.detach()

        # optim.zero_grad()
        model.zero_grad()
        loss = crit(yP, yT)
        loss.backward()
        optim.step()
        lossEpoch = lossEpoch+loss.item()

        #############################################
        # print result and save model
        if iIter % nIterEpoch == 0:
            if iEpoch % saveEpoch == 0:
                modelFile = os.path.join(outFolder, 'ep'+str(iEpoch)+'.pt')
                torch.save(model, modelFile)
            _ = rf.write(str(lossEpoch/nIterEpoch)+'\n')
            print('Epoch {} Loss {:.3f} time {:.2f}'.format(
                iEpoch, lossEpoch/nIterEpoch, time.time()-timeEpoch))
            lossEpoch = 0
            timeEpoch = time.time()
            iEpoch = iEpoch+1
    rf.close()


def testLSTM(*, out, rootOut, test, syr, eyr,
             epoch=None, gpu=0):
    outFolder = os.path.join(rootOut, out)
    optDict = loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch == None:
        epoch = opt.nEpoch

    #############################################
    # Load Model
    #############################################
    modelFile = os.path.join(outFolder, 'ep'+str(epoch)+'.pt')
    model = torch.load(modelFile)
    model.doDropout = False
    # model.training = False

    #############################################
    # load data
    #############################################
    dataset = classDB.Dataset(
        rootDB=opt.rootDB, subsetName=test,
        yrLst=np.arange(syr, eyr+1),
        var=(opt.var, opt.varC))
    dataset.readInput(loadNorm=True)
    x = dataset.normInput
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    if torch.cuda.is_available():
        xTest = xTest.cuda(gpu)

    #############################################
    # forward model and save prediction
    #############################################
    predName = 'test_{}_{}_{}_ep{}.csv'.format(
        test, str(syr), str(eyr), str(epoch))
    predFile = os.path.join(outFolder, predName)

    yP = model(xTest)
    yPout = yP.detach().cpu().numpy().squeeze()
    print('saving '+predName)
    pd.DataFrame(yPout).to_csv(predFile, header=False, index=False)
