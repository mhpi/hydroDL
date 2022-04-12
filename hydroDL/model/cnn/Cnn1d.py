import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import rnn, cnn
import csv
import numpy as np


class Cnn1d(nn.Module):
    def __init__(self, *, nx, nt, cnnSize=32, cp1=(64, 3, 2), cp2=(128, 5, 2)):
        super(Cnn1d, self).__init__()
        self.nx = nx
        self.nt = nt
        cOut, f, p = cp1
        self.conv1 = nn.Conv1d(nx, cOut, f)
        self.pool1 = nn.MaxPool1d(p)
        lTmp = int(cnn.calConvSize(nt, f, 0, 1, 1) / p)

        cIn = cOut
        cOut, f, p = cp2
        self.conv2 = nn.Conv1d(cIn, cOut, f)
        self.pool2 = nn.MaxPool1d(p)
        lTmp = int(cnn.calConvSize(lTmp, f, 0, 1, 1) / p)

        self.flatLength = int(cOut * lTmp)
        self.fc1 = nn.Linear(self.flatLength, cnnSize)
        self.fc2 = nn.Linear(cnnSize, cnnSize)
        self.name = "Cnn1d"
        self.is_legacy = True

    def forward(self, x):
        # x- [nt,ngrid,nx]
        x1 = x
        x1 = x1.permute(1, 2, 0)
        x1 = self.pool1(F.relu(self.conv1(x1)))
        x1 = self.pool2(F.relu(self.conv2(x1)))
        x1 = x1.view(-1, self.flatLength)
        x1 = F.relu(self.fc1(x1))
        x1 = self.fc2(x1)

        return x1
