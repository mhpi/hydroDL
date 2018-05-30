import os
import torch
import time
import numpy as np
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
        print('Error: cannot overwrite existed optFile. Delete manually.')
    else:
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

    model = classLSTM.LSTMModel(
        nx=nx, ny=ny, hiddenSize=opt.hiddenSize)
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
    nIterEpoch = int(np.ceil(np.log(0.01)/np.log(1-nbatch*rho/ngrid/nt)))
    saveEpoch = opt.saveEpoch

    runFile = os.path.join(outFolder, 'runFile.csv')

    iEpoch = 1
    lossEpoch = 0
    # timeIter = time.time()
    timeEpoch = time.time()
    for iIter in range(1, nEpoch*nIterEpoch+1):
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
        yT = torch.empty(rho, nbatch, 1).cuda()
        yT[loc0] = yP[loc0]
        yT[loc1] = yTrain[loc1]
        yT = yT.detach()

        optim.zero_grad()
        loss = crit(yP, yT)
        loss.backward()
        optim.step()
        # print('Epoch {} Iter {} Loss {:.3f} time {:.2f}'.format(iEpoch, iIter, loss.data[0]),time.time()-timeIter)
        # timeIter = time.time()
        lossEpoch = lossEpoch+loss.data[0]

        if iIter % nIterEpoch == 0:
            if iEpoch % saveEpoch == 0:
                modelFile = os.path.join(outFolder, 'ep'+str(iEpoch)+'.pt')
                model.save_state_dict(modelFile)
            with open(runFile, 'w') as myfile:
                myfile.write(str(lossEpoch/nIterEpoch)+'\n')
            print('Epoch {} Loss {:.3f} time {:.2f}'.format(
                iEpoch, lossEpoch/nIterEpoch, time.time()-timeEpoch))
            lossEpoch = 0
            timeEpoch = time.time()
            iEpoch = iEpoch+1
