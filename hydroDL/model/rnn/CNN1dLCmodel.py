import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import cnn
from hydroDL.model import rnn
import csv
import numpy as np


class CNN1dLCmodel(torch.nn.Module):
    # add the CNN extracted features into original LSTM input, then pass through linear layer
    def __init__(
        self,
        *,
        nx,
        ny,
        nobs,
        hiddenSize,
        nkernel=(10, 5),
        kernelSize=(3, 3),
        stride=(2, 1),
        dr=0.5,
        poolOpt=None,
        cnndr=0.0
    ):
        # two convolutional layer
        super(CNN1dLCmodel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.obs = nobs
        self.hiddenSize = hiddenSize
        nlayer = len(nkernel)
        self.features = nn.Sequential()
        ninchan = 1  # need to modify the hardcode: 4 for smap and 1 for FDC
        Lout = nobs
        for ii in range(nlayer):
            ConvLayer = cnn.CNN1dkernel(
                ninchannel=ninchan,
                nkernel=nkernel[ii],
                kernelSize=kernelSize[ii],
                stride=stride[ii],
            )
            self.features.add_module("CnnLayer%d" % (ii + 1), ConvLayer)
            if cnndr != 0.0:
                self.features.add_module("dropout%d" % (ii + 1), nn.Dropout(p=cnndr))
            ninchan = nkernel[ii]
            Lout = cnn.calConvSize(lin=Lout, kernel=kernelSize[ii], stride=stride[ii])
            self.features.add_module("Relu%d" % (ii + 1), nn.ReLU())
            if poolOpt is not None:
                self.features.add_module(
                    "Pooling%d" % (ii + 1), nn.MaxPool1d(poolOpt[ii])
                )
                Lout = cnn.calPoolSize(lin=Lout, kernel=poolOpt[ii])
        self.Ncnnout = int(
            Lout * nkernel[-1]
        )  # total CNN feature number after convolution
        Nf = self.Ncnnout + nx
        self.linearIn = torch.nn.Linear(Nf, hiddenSize)
        self.lstm = rnn.CudnnLstm(inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr)
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.name = "CNN1dLCmodel"
        self.is_legacy = True

    def forward(self, x, z, doDropMC=False):
        # z = ngrid*nVar add a channel dimension
        ngrid = z.shape[0]
        rho, BS, Nvar = x.shape
        if len(z.shape) == 2:  # for FDC, else 3 dimension for smap
            z = torch.unsqueeze(z, dim=1)
        z0 = self.features(z)
        # z0 = (ngrid) * nkernel * sizeafterconv
        z0 = z0.view(ngrid, self.Ncnnout).repeat(rho, 1, 1)
        x = torch.cat((x, z0), dim=2)
        x0 = F.relu(self.linearIn(x))
        outLSTM, (hn, cn) = self.lstm(x0, doDropMC=doDropMC)
        out = self.linearOut(outLSTM)
        # out = rho/time * batchsize * Ntargetvar
        return out
