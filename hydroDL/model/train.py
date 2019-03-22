import numpy as np
import torch
import time
import os
import csv


def trainModel(model,
               x,
               y,
               lossFun,
               *,
               nEpoch=500,
               miniBatch=[100, 30],
               saveEpoch=100,
               saveFolder=None):
    batchSize, rho = miniBatch
    ngrid, nt, nx = x.shape
    nIterEp = int(
        np.ceil(np.log(0.01) / np.log(1 - batchSize * rho / ngrid / nt)))

    xTrain, yTrain = randomSubset(x, y, [batchSize, rho])
    if torch.cuda.is_available():
        xTrain = xTrain.cuda()
        yTrain = yTrain.cuda()
        lossFun = lossFun.cuda()
        model = model.cuda()

    optim = torch.optim.Adadelta(model.parameters())
    model.zero_grad()
    runFile = os.path.join(saveFolder, 'run.csv')
    rf = open(runFile, 'a+')
    for iEpoch in range(0, nEpoch):
        lossEp = 0
        t0 = time.time()
        for iIter in range(0, nIterEp):
            # training iterations
            yP = model(xTrain)
            loss = lossFun(yP, yTrain)
            loss.backward()
            optim.step()
            model.zero_grad()
            lossEp = lossEp + loss.item()
        # print loss
        lossEp = lossEp / nIterEp
        logStr = 'Epoch {} Loss {:.3f} time {:.2f}'.format(
            iEpoch, lossEp,
            time.time() - t0)
        print(logStr)
        rf.write(logStr + '\n')
        if iEpoch % saveEpoch == 0 and saveFolder is not None:
            # save model
            modelFile = os.path.join(saveFolder,
                                     'model_Ep' + str(iEpoch) + '.pt')
            torch.save(model, modelFile)
    rf.close()
    return model


def testModel(model, x, *, batchSize=None):
    ngrid, nt, nx = x.shape
    ny = model.ny
    if batchSize is None:
        batchSize = ngrid
    xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
    if torch.cuda.is_available():
        xTest = xTest.cuda()
        model = model.cuda()

    model.train(mode=False)
    yP = torch.zeros([nt, ngrid, ny])
    iS = np.arange(0, ngrid, batchSize)
    iE = np.append(iS[1:], ngrid)
    for i in range(0, len(iS)):
        xTemp = xTest[:, iS[i]:iE[i], :]
        yP[:, iS[i]:iE[i], :] = model(xTemp)
    yOut = yP.detach().cpu().numpy().swapaxes(0, 1)
    return yOut


def randomSubset(x, y, dimSubset):
    ngrid, nt, nx = x.shape
    batchSize, rho = dimSubset
    xTensor = torch.zeros([rho, batchSize, x.shape[-1]], requires_grad=False)
    yTensor = torch.zeros([rho, batchSize, y.shape[-1]], requires_grad=False)

    iGrid = np.random.randint(0, ngrid, [batchSize])
    iT = np.random.randint(0, nt - rho, [batchSize])
    for k in range(batchSize):
        temp = x[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        xTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k] + 1, np.arange(iT[k], iT[k] + rho), :]
        yTensor[:, k:k + 1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
    return xTensor, yTensor
