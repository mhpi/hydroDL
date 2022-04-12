import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import rnn
import csv
import numpy as np


class CudnnLstmModel_R2P(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, filename):
        super(CudnnLstmModel_R2P, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        # self.linearR2P = torch.nn.Linear(nx[1], nx[2])
        self.linearR2Pa = torch.nn.Linear(nx[1], hiddenSize)
        self.linearR2Pa1 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.linearR2Pa2 = torch.nn.Linear(hiddenSize, hiddenSize)
        self.linearR2Pb = torch.nn.Linear(
            hiddenSize, nx[2] + 2
        )  # add two for final shift layer
        # self.linearR2Pb = torch.nn.Linear(hiddenSize, nx[2]) #add two for final shift layer
        self.linearDrop = nn.Dropout(dr)
        # self.bn1 = torch.nn.BatchNorm1d(num_features=hiddenSize)

        # self.lstm = rnn.CudnnLstmModel(
        #    nx=nx, ny=ny, hiddenSize=hiddenSize, dr=dr)
        self.lstm = torch.load(filename)

        # self.lstm.eval()

        for param in self.lstm.parameters():
            param.requires_grad = False

        self.linearRV2S = torch.nn.Linear(nx[1] + ny, ny)
        self.linearR2S = torch.nn.Linear(nx[1], ny)
        self.linearV2S = torch.nn.Linear(ny, ny)  # mapping to SMAP
        # self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.name = "CudnnLstmModel_R2P"
        self.is_legacy = True

    def forward(self, x, doDropMC=False, outModel=None):
        if type(x) is tuple or type(x) is list:
            Forcing, Raw = x

        ##Param = F.relu(self.linearR2P(Raw))
        # Param_a = self.linearDrop(torch.relu(self.linearR2Pa(Raw))) # dropout setting
        # Param_a = torch.relu(self.linearR2Pa(Raw))
        Param_a = self.linearDrop(torch.relu(self.linearR2Pa(Raw)))
        ##Param_bn = self.bn1(Param_a)

        # Param_a1 = torch.relu(self.linearR2Pa1(Param_a))
        Param_a1 = self.linearDrop(torch.relu(self.linearR2Pa1(Param_a)))
        # Param_a2 = self.linearDrop(torch.relu(self.linearR2Pa2(Param_a1)))

        # Param = torch.atan(self.linearR2Pb(Param_a1))
        ##Param = torch.relu(self.linearR2Pb(Param_a1))

        Param_two = torch.atan(self.linearR2Pb(Param_a1))
        dim = Param_two.shape
        Param = Param_two[:, :, 0 : dim[2] - 2]
        a = Param_two[:, :, dim[2] - 2 : dim[2] - 1]
        b = Param_two[
            :, :, dim[2] - 1 : dim[2]
        ]  ##Param = torch.rrelu(self.linearR2Pb(Param_bn))

        if outModel is None:
            x1 = torch.cat(
                (Forcing, Param), dim=len(Param.shape) - 1
            )  # by default cat along dim=0
            # self.lstm.eval()
            outLSTM_surrogate = self.lstm(x1, doDropMC=doDropMC, dropoutFalse=True)
            # outLSTM_SMAP = torch.atan(self.linearV2S(outLSTM_surrogate))  # mapping to SMAP
            # x2 = torch.cat((outLSTM_surrogate, Raw), dim=len(Raw.shape)-1)
            outLSTM_SMAP = a * outLSTM_surrogate + b  # mapping to SMAP
            # outLSTM_SMAP = outLSTM_surrogate  # mapping to SMAP
            # return outLSTM_surrogate, Param
            return (outLSTM_SMAP, Param)
        else:
            # outQ_hymod, outE_hymod = self.lstm.advance(Forcing[:,:,0], Forcing[:,:,1], \
            #     Param[:,:,0], Param[:,:,1], Param[:,:,2], Param[:,:,3], Param[:,:,4], Param[:,:,5],\
            #         Param[:,:,6], Param[:,:,7])
            # outQ_hymod = self.hymode.advance(Forcing[:,:,0], Forcing[:,:,1], Param[:,:,:])
            # outQ_hymod = outQ_hymod.unsqueeze(2)
            # return outQ_hymod, Param

            # ====== reasonable hyod parameters ======
            out = "/mnt/sdc/SUR_VIC/multiOutput_CONUSv16f1_VIC/hymod/"
            with open(out + "parameters_hymod") as f:
                reader = csv.reader(f, delimiter=",")
                parameters = list(reader)
                parameters = np.array(parameters).astype(float)

            parameters = torch.from_numpy(parameters)
            return parameters

        # self.lstm.train()
        # out = self.linearOut(outLSTM)
        # return outLSTM
