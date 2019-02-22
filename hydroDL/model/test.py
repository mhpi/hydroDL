import os
import torch
import shutil
import numpy as np
import pandas as pd
from argparse import Namespace
import opt
import dbSMAP

def testLSTM(*, rootOut, out, test, syr, eyr, epoch=None, drMC=0, testBatch=0):
    outFolder = os.path.join(rootOut, out)
    optDict = opt.loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch is None:
        epoch = opt.nEpoch

    #############################################
    # load data
    dataset = db.dbSMAP_LSTM(
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
    optDict = opt.loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch is None:
        epoch = opt.nEpoch
    stat = dbSMAP.readStat(
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
