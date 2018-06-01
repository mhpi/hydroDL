import os
import rnnSMAP
import numpy as np
import pandas as pd
import torch
import time

import imp
from argparse import Namespace
from rnnSMAP import *

imp.reload(rnnSMAP)
rnnSMAP.reload()

# train model
optDict = rnnSMAP.classLSTM.optLSTM(
    out='CONUSv4f1_pytorch',
    rootDB=rnnSMAP.kPath['DBSMAP_L3_NA'],
    rootOut=rnnSMAP.kPath['OutSMAP_L3_NA'],
    syr=2015, eyr=2015,
    var='varLst_soilM', varC='varConstLst_Noah',
    train='CONUSv4f1',
    nEpoch=100
)

opt = Namespace(**optDict)

outFolder = os.path.join(opt.rootOut, opt.out)
if not os.path.isdir(outFolder):
    os.mkdir(outFolder)
funLSTM.saveOptLSTM(outFolder, optDict)
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
pBatch = nbatch*rho/ngrid/nt
if pBatch < 1:
    nIterEpoch = np.ceil(np.log(0.01)/np.log(1-nbatch*rho/ngrid/nt))
    nIterEpoch = nIterEpoch.astype(int)
else:
    nIterEpoch = 1

saveEpoch = opt.saveEpoch

iEpoch = 1
lossEpoch = 0
# timeIter = time.time()
timeEpoch = time.time()
rf = open(runFile, 'w')

rf.truncate()
rf.close
for iIter in range(1, nEpoch*nIterEpoch+1):
    # random a piece of time series
    iGrid = np.random.randint(0, ngrid, [nbatch])
    iT = np.random.randint(0, nt-rho+1, [nbatch])

    for k in range(nbatch):
        temp = x[iGrid[k]:iGrid[k]+1, np.arange(iT[k], iT[k]+rho), :]
        xTrain[:, k:k+1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))
        temp = y[iGrid[k]:iGrid[k]+1, np.arange(iT[k], iT[k]+rho)]
        yTrain[:, k:k+1, :] = torch.from_numpy(np.swapaxes(temp, 1, 0))

    # xTrain = torch.from_numpy(x).float()
    # yTrain = torch.from_numpy(y).float()

    if torch.cuda.is_available():
        xTrain = xTrain.cuda(opt.gpu)
        yTrain = yTrain.cuda(opt.gpu)

    yP = model(xTrain)
    loc0 = yTrain != yTrain
    loc1 = yTrain == yTrain    
    yT = torch.empty(yP.shape[0], yP.shape[1], 1).cuda()
    yT[loc0] = yP[loc0]
    yT[loc1] = yTrain[loc1]
    yT = yT.detach()

    optim.zero_grad()
    loss = crit(yP, yT)
    loss.backward()
    optim.step()
    # print('Epoch {} Iter {} Loss {:.3f} time {:.2f}'.format(iEpoch, iIter, loss.data[0]),time.time()-timeIter)
    # timeIter = time.time()
    lossEpoch = lossEpoch+loss.item()

    if iIter % nIterEpoch == 0:
        if iEpoch % saveEpoch == 0:
            modelFile = os.path.join(outFolder, 'ep'+str(iEpoch)+'.pt')
            torch.save(model, modelFile)
        # test on training
        xTest = torch.from_numpy(x).float()
        yTarget = torch.from_numpy(y).float()
        xTest = xTest.cuda(0)
        yTarget = yTarget.cuda(0)
        yTest = model(xTest)
        loc0 = yTarget != yTarget
        yTarget[loc0] = yTest[loc0]
        yTarget = yTarget.detach()
        lossTrain = crit(yTest, yTarget).item()
        with open(runFile, 'a+') as rf:
            _ = rf.write(str(lossEpoch/nIterEpoch)+', '+str(lossTrain)+'\n')
        print('Epoch {} Loss {:.3f} Loss Train {:.3f} time {:.2f}'.format(
            iEpoch, lossEpoch/nIterEpoch, lossTrain, time.time()-timeEpoch))
        lossEpoch = 0
        timeEpoch = time.time()
        iEpoch = iEpoch+1


rf.close()
