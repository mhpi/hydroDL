import os
import sys
import torch
import time
import shutil
import numpy as np
import pandas as pd
from argparse import Namespace

from . import classLSTM
from . import classDB
# from . import classPost
from . import funDB
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
    # logFile = os.path.join(outFolder, 'log')
    # sys.stdout = open(logFile, 'w')

    #############################################
    # load data
    #############################################
    dataset = classDB.DatasetLSTM(
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
        model = classLSTM.localLSTM_slow(
            nx=nx, ny=nOut, hiddenSize=opt.hiddenSize, drMethod=opt.drMethod,
            gpu=opt.gpu, doReLU=relu, doTied=tied)
    elif opt.model == 'torch':
        model = classLSTM.torchLSTM_cell(
            nx=nx, ny=nOut, hiddenSize=opt.hiddenSize, dr=opt.dr, doReLU=relu)
    elif opt.model == 'cudnn':
        model = classLSTM.localLSTM_cuDNN(
            nx=nx, ny=nOut, hiddenSize=opt.hiddenSize, dr=opt.dr)

    # loss function
    if opt.loss == 'mse':
        crit = torch.nn.MSELoss()
    elif opt.loss == 'sigma':
        crit = classLSTM.sigmaLoss(prior=opt.lossPrior)

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


def testLSTM(*, rootOut, out, test, syr, eyr, epoch=None, drMC=0, testBatch=0):
    outFolder = os.path.join(rootOut, out)
    optDict = loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch is None:
        epoch = opt.nEpoch

    #############################################
    # load data
    dataset = classDB.DatasetLSTM(
        rootDB=opt.rootDB, subsetName=test,
        yrLst=np.arange(syr, eyr+1),
        var=(opt.var, opt.varC))
    x = dataset.readInput()
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    if opt.gpu > 0:
        xTest = xTest.cuda()
    nt, ngrid, nx = xTest.shape
    if opt.loss == 'sigma':
        ny = 2
    else:
        ny = 1

    #############################################
    # Load Model
    modelFile = os.path.join(outFolder, 'ep'+str(epoch)+'.pt')
    model = torch.load(modelFile)

    #############################################
    # save prediction
    model.train(mode=False)
    if testBatch > 0:
        yP = torch.zeros([nt, ngrid, ny])
        iS = np.arange(0, ngrid, testBatch)
        iE = np.append(iS[1:], ngrid)
        for i in range(0, len(iS)):
            xTemp = xTest[:, iS[i]:iE[i], :]
            yP[:, iS[i]:iE[i], :] = model(xTemp)
            model.zero_grad()
    else:
        yP = model(xTest)

    if opt.loss == 'sigma':
        yOut = yP[:, :, 0].detach().cpu().numpy().squeeze()
        sOut = yP[:, :, 1].detach().cpu().numpy().squeeze()
    else:
        yOut = yP.detach().cpu().numpy().squeeze()

    predName = 'test_{}_{}_{}_ep{}.csv'.format(
        test, str(syr), str(eyr), str(epoch))
    predFile = os.path.join(outFolder, predName)
    print('saving '+predFile)
    pd.DataFrame(yOut).to_csv(predFile, header=False, index=False)

    if opt.loss == 'sigma':
        sigmaName = 'testSigma_{}_{}_{}_ep{}.csv'.format(
            test, str(syr), str(eyr), str(epoch))
        sigmaFile = os.path.join(outFolder, sigmaName)
        print('saving '+sigmaFile)
        pd.DataFrame(sOut).to_csv(sigmaFile, header=False, index=False)

    #############################################
    # MC dropout
    if drMC > 0:
        # model.train()
        mcName = 'test_{}_{}_{}_ep{}_drM{}'.format(
            test, str(syr), str(eyr), str(epoch), str(drMC))
        mcFolder = os.path.join(outFolder, mcName)

        # remove existing files
        mcFile = os.path.join(outFolder, mcName+'.npy')
        mcSigmaName = 'testSigma_{}_{}_{}_ep{}_drM{}'.format(
            test, str(syr), str(eyr), str(epoch), str(drMC))
        mcSigmaFile = os.path.join(outFolder, mcSigmaName+'.npy')
        if os.path.isdir(mcFolder):
            shutil.rmtree(mcFolder)
        if os.path.isfile(mcFile):
            os.remove(mcFile)
        if os.path.isfile(mcSigmaFile):
            os.remove(mcSigmaFile)

        if not os.path.isdir(mcFolder):
            os.mkdir(mcFolder)
        for kk in range(0, drMC):
            if testBatch > 0:
                yP = torch.zeros([nt, ngrid, ny])
                iS = np.arange(0, ngrid, testBatch)
                iE = np.append(iS[1:], ngrid)
                for i in range(0, len(iS)):
                    xTemp = xTest[:, iS[i]:iE[i], :]
                    yP[:, iS[i]:iE[i], :] = model(xTemp, doDropMC=True)
                    model.zero_grad()
            else:
                yP = model(xTest, doDropMC=True)

            if opt.loss == 'sigma':
                yOut = yP[:, :, 0].detach().cpu().numpy().squeeze()
                sOut = yP[:, :, 1].detach().cpu().numpy().squeeze()

                sigmaName = 'drSigma_{}.csv'.format(str(kk))
                sigmaFile = os.path.join(mcFolder, sigmaName)
                print('saving '+sigmaFile)
                pd.DataFrame(sOut).to_csv(sigmaFile, header=False, index=False)
            else:
                yOut = yP.detach().cpu().numpy().squeeze()

            predName = 'drMC_{}.csv'.format(str(kk))
            predFile = os.path.join(mcFolder, predName)
            print('saving '+predFile)
            pd.DataFrame(yOut).to_csv(predFile, header=False, index=False)


def readPred(*, rootOut, out, test, syr, eyr, epoch=None, drMC=0,
             reReadMC=False):
    outFolder = os.path.join(rootOut, out)
    optDict = loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch is None:
        epoch = opt.nEpoch
    stat = funDB.readStat(
        rootDB=opt.rootDB, fieldName=opt.target, isConst=False)

    if opt.loss == 'sigma':
        sigmaName = 'testSigma_{}_{}_{}_ep{}.csv'.format(
            test, str(syr), str(eyr), str(epoch))
        sigmaFile = os.path.join(outFolder, sigmaName)
        print('reading '+sigmaFile)
        temp = pd.read_csv(sigmaFile, dtype=np.float,
                           header=None).values.swapaxes(1, 0)
        dataSigma = np.sqrt(np.exp(temp))*stat[3]
    else:
        dataSigma = None

    predName = 'test_{}_{}_{}_ep{}.csv'.format(
        test, str(syr), str(eyr), str(epoch))
    predFile = os.path.join(outFolder, predName)
    print('reading '+predFile)
    temp = pd.read_csv(predFile, dtype=np.float,
                       header=None).values.swapaxes(1, 0)
    dataPred = temp*stat[3]+stat[2]

    ngrid, nt = dataPred.shape
    if drMC > 0:
        dataSigmaBatch = np.empty([ngrid, nt, drMC])
        mcName = 'test_{}_{}_{}_ep{}_drM{}'.format(
            test, str(syr), str(eyr), str(epoch), str(drMC))
        mcSigmaName = 'testSigma_{}_{}_{}_ep{}_drM{}'.format(
            test, str(syr), str(eyr), str(epoch), str(drMC))
        mcFolder = os.path.join(outFolder, mcName)

        mcFile = os.path.join(outFolder, mcName+'.npy')
        mcSigmaFile = os.path.join(outFolder, mcSigmaName+'.npy')

        if not os.path.isfile(mcFile) or reReadMC:
            dataPredBatch = np.empty([ngrid, nt, drMC])
            for kk in range(0, drMC):
                predName = 'drMC_{}.csv'.format(str(kk))
                predFile = os.path.join(mcFolder, predName)
                print('reading '+predFile)
                temp = pd.read_csv(predFile, dtype=np.float,
                                   header=None).values.swapaxes(1, 0)
                temp = temp*stat[3]+stat[2]
                dataPredBatch[:, :, kk] = temp
            np.save(mcFile, dataPredBatch)
        else:
            dataPredBatch = np.load(mcFile)

        if opt.loss == 'sigma':
            if not os.path.isfile(mcSigmaFile) or reReadMC:
                for kk in range(0, drMC):
                    sigmaName = 'drSigma_{}.csv'.format(str(kk))
                    sigmaFile = os.path.join(mcFolder, sigmaName)
                    print('reading '+sigmaFile)
                    temp = pd.read_csv(sigmaFile, dtype=np.float,
                                       header=None).values.swapaxes(1, 0)
                    temp = np.sqrt(np.exp(temp))*stat[3]
                    dataSigmaBatch[:, :, kk] = temp
                np.save(mcSigmaFile, dataSigmaBatch)
            else:
                dataSigmaBatch = np.load(mcSigmaFile)
    else:
        dataSigmaBatch = None
        dataPredBatch = None

    return (dataPred, dataSigma, dataPredBatch, dataSigmaBatch)


def checkPred(*, rootOut, out, test, syr, eyr, epoch=None, drMC=0):
    outFolder = os.path.join(rootOut, out)
    optDict = loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch is None:
        epoch = opt.nEpoch

    if opt.loss == 'sigma':
        sigmaName = 'testSigma_{}_{}_{}_ep{}.csv'.format(
            test, str(syr), str(eyr), str(epoch))
        sigmaFile = os.path.join(outFolder, sigmaName)
        if not os.path.isfile(sigmaFile):
            return False

    predName = 'test_{}_{}_{}_ep{}.csv'.format(
        test, str(syr), str(eyr), str(epoch))
    predFile = os.path.join(outFolder, predName)
    if not os.path.isfile(predFile):
        return False

    if drMC > 0:
        mcName = 'test_{}_{}_{}_ep{}_drM{}'.format(
            test, str(syr), str(eyr), str(epoch), str(drMC))
        mcFolder = os.path.join(outFolder, mcName)
        if opt.loss == 'sigma':
            sigmaName = 'drSigma_{}.csv'.format(str(drMC-1))
            sigmaFile = os.path.join(mcFolder, sigmaName)
            if not os.path.isfile(sigmaFile):
                return False

        predName = 'drMC_{}.csv'.format(str(drMC-1))
        predFile = os.path.join(mcFolder, predName)
        if not os.path.isfile(predFile):
            return False

    return True
