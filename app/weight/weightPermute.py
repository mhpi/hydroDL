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

kk = 1
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

p0X = x0.matmul(w_ih.transpose(1, 0))
pX = p0X.clone().fill_(0)
for k in range(0, 100):
    p1X = x0[:, torch.randperm(256)].matmul(w_ih.transpose(1, 0))
    pX = pX+(p0X.abs() < p1X.abs()).to(dtype=torch.float32)
rX = pX.mean(dim=1)/100


p0 = x0.matmul(w_ih.transpose(1, 0)).pow(2)
p0 = p0.mul(p0)

xNorm = x0.norm(p=2, dim=1).view(ngrid, 1).repeat(1, nh*4)
wNorm = w_ih.norm(p=2, dim=1).repeat(ngrid, 1)
xSum = x0.sum(dim=1).view(ngrid, 1).repeat(1, nh*4)
wSum = w_ih.sum(dim=1).repeat(ngrid, 1)

p1 = xNorm.mul(wNorm)/nh+(wSum.pow(2)-wNorm).mul(xSum.pow(2)-xNorm)/(nh*(nh-1))
