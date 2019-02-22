import os
import sys
import torch
import time
import numpy as np
from argparse import Namespace
from model import opt
import db
import lstmTorch
import lstmKuai
import loss

def trainLSTM(optDict: opt.optLSTM):
    opt = Namespace(**optDict)

    outFolder = os.path.join(opt.rootOut, opt.out)
    if not os.path.isdir(outFolder):
        os.mkdir(outFolder)
    opt.saveOptLSTM(outFolder, optDict)
    runFile = os.path.join(outFolder, 'runFile.csv')
    # logFile = os.path.join(outFolder, 'log')
    # sys.stdout = open(logFile, 'w')

    #############################################
    # load data
    #############################################
    dataset = db.dbSMAP_LSTM(
        rootDB=opt.rootDB, subsetName=opt.train,
        yrLst=np.arange(opt.syr, opt.eyr+1),
        var=(opt.var, opt.varC), targetName=opt.target)
    x = dataset.readInput()
    y = dataset.readTarget()

    ngrid, nt, nx = x.shape
    ny = 1
    y = np.reshape(y, [ngrid, nt, ny])
    rho = opt.rho
    nbatch = opt.nbatch

    #############################################
    # Model
    #############################################

    if opt.loss == 'mse':
        nOut = ny
    elif opt.loss == 'sigma':
        nOut = ny+1

    # model
    modelOpt = opt.modelOpt.split('+')
    tied = 'tied' in modelOpt
    relu = 'relu' in modelOpt
    if opt.model == 'slow':
        model = lstmTorch.localLSTM_slow(
            nx=nx, ny=nOut, hiddenSize=opt.hiddenSize, drMethod=opt.drMethod,
            gpu=opt.gpu, doReLU=relu, doTied=tied)
    elif opt.model == 'torch':
        model = lstmTorch.torchLSTM_cell(
            nx=nx, ny=nOut, hiddenSize=opt.hiddenSize, dr=opt.dr, doReLU=relu)
    elif opt.model == 'cudnn':
        model = lstmTorch.localLSTM_cuDNN(
            nx=nx, ny=nOut, hiddenSize=opt.hiddenSize, dr=opt.dr)

    # loss function
    if opt.loss == 'mse':
        crit = torch.nn.MSELoss()
    elif opt.loss == 'sigma':
        crit = loss.sigmaLoss(prior=opt.lossPrior)

    if opt.gpu > 0:
        crit = crit.cuda()
        model = model.cuda()

    model.zero_grad()
    model.train()

    # construct model before optim will automatically make it cuda
    optim = torch.optim.Adadelta(model.parameters())

    xTrain = torch.zeros([rho, nbatch, nx], requires_grad=False)
    yTrain = torch.zeros([rho, nbatch, ny], requires_grad=False)

    #############################################
    # Training
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
            xTrain = xTrain.cuda()
            yTrain = yTrain.cuda()

        yP = model(xTrain)
        loc0 = yTrain != yTrain
        loc1 = yTrain == yTrain
        yT = torch.empty(rho, nbatch, 1)
        if opt.gpu > 0:
            yT = yT.cuda()
        if opt.loss == 'mse':
            yPtemp = yP
        elif opt.loss == 'sigma':
            yPtemp = yP[:, :, 0:1]
        yT[loc0] = yPtemp[loc0]
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
            rf.write(str(lossEpoch/nIterEpoch)+'\n')
            print('Epoch {} Loss {:.3f} time {:.2f}'.format(
                iEpoch, lossEpoch/nIterEpoch, time.time()-timeEpoch))
            lossEpoch = 0
            timeEpoch = time.time()
            iEpoch = iEpoch+1
    rf.close()
    sys.stdout.close()
