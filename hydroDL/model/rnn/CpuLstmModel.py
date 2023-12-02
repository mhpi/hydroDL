import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import cnn, rnn
import csv
import numpy as np


class CpuLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5):
        super(CpuLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        self.lstm = rnn.LSTMcell_tied(
            inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr, drMethod="drW", gpu=-1
        )
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = -1
        self.name = "CpuLstmModel"
        self.is_legacy = True

    def forward(self, inputs, doDropMC=False):
        # x0 = F.relu(self.linearIn(x))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # return out
        results = {}
        x = inputs
        nt, ngrid, nx = x.shape
        yt = torch.zeros(ngrid, 1)
        results = torch.zeros(nt, ngrid, self.ny)
        ht = None
        ct = None
        resetMask = True
        for t in range(nt):
            xt = x[t, :, :]
            x0 = F.relu(self.linearIn(xt))
            ht, ct = self.lstm(x0, hidden=(ht, ct), resetMask=resetMask)
            yt = self.linearOut(ht)
            resetMask = False
            results[t, :, :] = yt
        return results
