import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import rnn, cnn
import csv
import numpy as np


class LstmCnn1d(torch.nn.Module):
    # Dense layer > reduce dim > dense
    def __init__(
        self,
        *,
        nx,
        ny,
        rho,
        nkernel=(10, 5),
        kernelSize=(3, 3),
        stride=(2, 1),
        padding=(1, 1),
        dr=0.5,
        poolOpt=None
    ):
        # two convolutional layer
        super(LstmCnn1d, self).__init__()
        self.nx = nx
        self.ny = ny
        self.rho = rho
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = nx
        Lout = rho
        for ii in range(nlayer):
            # First layer: no dimension reduction
            ConvLayer = cnn.CNN1dkernel(
                ninchannel=ninchan,
                nkernel=nkernel[ii],
                kernelSize=kernelSize[ii],
                stride=stride[ii],
                padding=padding[ii],
            )
            self.features.add_module("CnnLayer%d" % (ii + 1), ConvLayer)
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            if poolOpt is not None:
                self.features.add_module(
                    "Pooling%d" % (ii + 1), nn.MaxPool1d(poolOpt[ii])
                )
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(
            Lout * nkernel[-1]
        )  # total CNN feature number after convolution
        self.name = "LstmCnn1d"
        self.is_legacy = True

    def forward(self, x, doDropMC=False):
        out = self.features(x)
        # # z0 = (ntime*ngrid) * nkernel * sizeafterconv
        # z0 = z0.view(nt, ngrid, self.Ncnnout)
        # x0 = torch.cat((x, z0), dim=2)
        # x0 = F.relu(self.linearIn(x0))
        # outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        # out = self.linearOut(outLSTM)
        # # out = rho/time * batchsize * Ntargetvar
        return out
