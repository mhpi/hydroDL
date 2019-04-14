import os
import pandas as pd
import rnnSMAP
from rnnSMAP import runTrainLSTM
import matplotlib.pyplot as plt
import numpy as np
import torch
from argparse import Namespace
import torch.nn.functional as F
import time

testName = 'CONUSv4f1'
out = 'CONUSv4f1_y15_Forcing'
rootOut = rnnSMAP.kPath['Out_L3_NA']
rootDB = rnnSMAP.kPath['DB_L3_NA']
epoch = 500
yr = 2015

outFolder = os.path.join(rootOut, out)
optDict = rnnSMAP.funLSTM.loadOptLSTM(outFolder)
opt = Namespace(**optDict)
dataset = rnnSMAP.classDB.DatasetLSTM(
    rootDB=rootDB, subsetName=testName,
    yrLst=[yr],
    var=(opt.var, opt.varC))
x = dataset.readInput()
xTest = torch.from_numpy(np.swapaxes(x, 1, 0)).float()
xTest = xTest.cuda()

modelFile = os.path.join(rootOut, out, 'ep'+str(epoch)+'.pt')
model = torch.load(modelFile)
model.train(mode=False)
yP = model(xTest)
nt, ngrid, nx = xTest.shape
nh = opt.hiddenSize

# test model
xm0 = xTest
xm1 = model.linearIn(xm0)
xm2 = model.relu(xm1)
ym0, (hm0, cm0) = model.lstm(xm2)
ym1 = model.linearOut(ym0)

# step by step model
w_ih = model.lstm.w_ih
w_hh = model.lstm.w_hh
b_ih = model.lstm.b_ih
b_hh = model.lstm.b_hh

h0 = torch.zeros(ngrid, opt.hiddenSize).cuda()
c0 = torch.zeros(ngrid, opt.hiddenSize).cuda()
hout = []
cX = []
cH = []
for kk in range(0, nt):
    tt = time.time()
    x0 = xm2[kk, :, :]
    gates = F.linear(x0, w_ih, b_ih) + F.linear(h0, w_hh, b_hh)
    gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)
    gate_i = F.sigmoid(gate_i)
    gate_f = F.sigmoid(gate_f)
    gate_c = F.tanh(gate_c)
    gate_o = F.sigmoid(gate_o)
    c1 = (gate_f * c0) + (gate_i * gate_c)
    h1 = gate_o * F.tanh(c1)
    c0 = c1
    h0 = h1
    hout.append(h0)

    # detect large weight
    p1 = x0.matmul(w_ih.transpose(1, 0))
    pmul = p1.mul(p1)
    psum = pmul.clone().fill_(0)
    # for k in range(0, ngrid):
    #     a = x0[k, :].repeat(1024, 1)
    #     p2 = a.mul(w_ih)
    #     psumTemp = p2.mul(p2).sum(dim=1)
    #     psum[k, :] = psumTemp
    a = x0.view(ngrid, nh, 1).repeat(1, 1, nh*4)
    b = w_ih.transpose(1, 0).view(1, nh, nh*4).repeat(ngrid, 1, 1)
    c = a.mul(b)
    psum = c.mul(c).sum(dim=1)
    cXtemp = psum > pmul
    cX.append(cXtemp)

    p1 = h0.matmul(w_hh.transpose(1, 0))
    pmul = p1.mul(p1)
    psum = pmul.clone().fill_(0)
    # for k in range(0, ngrid):
    #     a = h0[k, :].repeat(1024, 1)
    #     p2 = a.mul(w_hh)
    #     psumTemp = p2.mul(p2).sum(dim=1)
    #     psum[k, :] = psumTemp
    a = h0.view(ngrid, nh, 1).repeat(1, 1, nh*4)
    b = w_hh.transpose(1, 0).view(1, nh, nh*4).repeat(ngrid, 1, 1)
    c = a.mul(b)
    psum = c.mul(c).sum(dim=1)
    cHtemp = psum > pmul
    cH.append(cHtemp)
    print('time step {} time {:.5f}'.format(kk, time.time()-tt))

houtView = torch.cat(hout, 0).view(nt, *hout[0].size())
y1 = model.linearOut(houtView)

cXout = torch.cat(cX, 0).view(nt, ngrid,nh*4).detach().cpu().numpy().squeeze()/1024
cHout = torch.cat(cH, 0).view(nt, ngrid).detach().cpu().numpy().squeeze()/1024

saveFolder = '/mnt/sdc/rnnSMAP/Result_SMAPgrid/weightDetector/'
cXoutFile = os.path.join(saveFolder, testName+'_yr'+str(yr)+'_cX')
pd.DataFrame(cXout).to_csv(cXoutFile, header=False, index=False)
cHoutFile = os.path.join(saveFolder, testName+'_yr'+str(yr)+'_cH')
pd.DataFrame(cHout).to_csv(cHoutFile, header=False, index=False)
