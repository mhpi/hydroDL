import os
import pandas as pd
import rnnSMAP
import numpy as np
import torch
from argparse import Namespace
from . import classDB
from . import funLSTM
import torch.nn.functional as F
from tqdm import tqdm


def readWeightDector(*, rootOut, out, test, syr, eyr,
                     wOpt='wc', epoch=None, redo=False, nPerm=100):
    outFolder = os.path.join(rootOut, out)
    optDict = funLSTM.loadOptLSTM(outFolder)
    opt = Namespace(**optDict)
    if epoch is None:
        epoch = opt.nEpoch

    fileName = wOpt+'X_{}_{}_{}_ep{}.npy'.format(
        test, str(syr), str(eyr), str(epoch))
    fileX = os.path.join(outFolder, fileName)
    fileName = wOpt+'H_{}_{}_{}_ep{}.npy'.format(
        test, str(syr), str(eyr), str(epoch))
    fileH = os.path.join(outFolder, fileName)

    if not os.path.isfile(fileX) or not os.path.isfile(fileH) or redo is True:
        #############################################
        # load data
        #############################################
        dataset = classDB.DatasetLSTM(
            rootDB=opt.rootDB, subsetName=test,
            yrLst=np.arange(syr, eyr+1),
            var=(opt.var, opt.varC))
        x = dataset.readInput()
        xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
        if opt.gpu > 0:
            xTest = xTest.cuda()
        nt, ngrid, nx = xTest.shape
        nh = opt.hiddenSize

        #############################################
        # Load Model
        #############################################
        modelFile = os.path.join(outFolder, 'ep'+str(epoch)+'.pt')
        model = torch.load(modelFile)
        model.train(mode=False)
        if opt.model == 'cudnn':
            w_ih = model.lstm.w_ih
            w_hh = model.lstm.w_hh
            b_ih = model.lstm.b_ih
            b_hh = model.lstm.b_hh
            xTemp = model.linearIn(xTest)
            xIn = model.relu(xTemp)
        cX = []
        cH = []

        #############################################
        # Weight Dector
        #############################################
        h0 = torch.zeros(ngrid, opt.hiddenSize).cuda()
        c0 = torch.zeros(ngrid, opt.hiddenSize).cuda()
        pbar = tqdm(total=nt, mininterval=0.01,
                    desc='dectecting weight cancellation')
        for k in range(0, nt):
            xx = xIn[k, :, :]
            gates = F.linear(xx, w_ih, b_ih) + F.linear(h0, w_hh, b_hh)
            gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)
            gate_i = F.sigmoid(gate_i)
            gate_f = F.sigmoid(gate_f)
            gate_c = F.tanh(gate_c)
            gate_o = F.sigmoid(gate_o)
            c1 = (gate_f * c0) + (gate_i * gate_c)
            h1 = gate_o * F.tanh(c1)
            c0 = c1
            h0 = h1

            if wOpt == 'wc':
                cXtemp = detectWeight(xx, w_ih, ngrid=ngrid, nh=nh)
                cX.append(cXtemp)
                cHtemp = detectWeight(h1, w_hh, ngrid=ngrid, nh=nh)
                cH.append(cHtemp)
            if wOpt == 'wp':
                cXtemp = permuteWeight(xx, w_ih, nh=nh, nPerm=nPerm)
                cX.append(cXtemp)
                cHtemp = permuteWeight(h1, w_hh, nh=nh, nPerm=nPerm)
                cH.append(cHtemp)
            if wOpt == 'wp2':
                cXtemp = permuteWeight2(xx, w_ih, ngrid=ngrid, nh=nh)
                cX.append(cXtemp)
                cHtemp = permuteWeight2(h1, w_hh, ngrid=ngrid, nh=nh)
                cH.append(cHtemp)

            pbar.update(1)
        cXout = torch.cat(cX, 0).view(nt, ngrid, nh*4).detach().cpu().numpy()
        cHout = torch.cat(cH, 0).view(nt, ngrid, nh*4).detach().cpu().numpy()

        #############################################
        # Save Data
        #############################################
        print('saving '+fileX)
        np.save(fileX, cXout)
        print('saving '+fileH)
        np.save(fileH, cHout)
    else:
        cXout = np.load(fileX)
        cHout = np.load(fileH)
    return(cXout, cHout)


def detectWeight(x, w, *, ngrid=None, nh=None):
    # detect large weight
    if ngrid is None:
        ngrid = x.shape[0]
    if nh is None:
        nh = w.shape[1]
    p1 = x.matmul(w.transpose(1, 0))
    pmul = p1.mul(p1)

    a = x.view(ngrid, nh, 1).repeat(1, 1, nh*4)
    b = w.transpose(1, 0).view(1, nh, nh*4).repeat(ngrid, 1, 1)
    c = a.mul(b)
    psum = c.mul(c).sum(dim=1)
    out = psum > pmul
    return out


def permuteWeight(x, w, *, nh=None, nPerm=100):
    if nh is None:
        nh = w.shape[1]

    p0X = x.matmul(w.transpose(1, 0))
    pX = p0X.clone().fill_(0)
    for k in range(0, nPerm):
        p1X = x[:, torch.randperm(nh)].matmul(w.transpose(1, 0))
        pX = pX+(p0X.abs() < p1X.abs()).to(dtype=torch.float32)
    out = pX/nPerm
    return out


def permuteWeight2(x, w, *, ngrid=None, nh=None):
    if ngrid is None:
        ngrid = x.shape[0]
    if nh is None:
        nh = w.shape[1]

    p0 = x.matmul(w.transpose(1, 0)).pow(2)
    xNorm = x.norm(p=2, dim=1).view(ngrid, 1).repeat(1, nh*4)
    wNorm = w.norm(p=2, dim=1).repeat(ngrid, 1)
    xSum = x.sum(dim=1).view(ngrid, 1).repeat(1, nh*4)
    wSum = w.sum(dim=1).repeat(ngrid, 1)

    p1 = xNorm.mul(wNorm)/nh + \
        (wSum.pow(2) - wNorm).mul(xSum.pow(2)-xNorm)/(nh*(nh-1))

    out = p0 < p1
    return out
