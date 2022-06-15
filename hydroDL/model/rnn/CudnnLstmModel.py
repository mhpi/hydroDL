"""A class for an LSTM model that uses Cuda"""
from hydroDL.model.rnn.CudnnLstm import CudnnLstm
import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
from hydroDL.model import rnn
import csv
import numpy as n


class CudnnLstmModel(torch.nn.Module):
    def __init__(self, *, nx, ny, hiddenSize, dr=0.5, warmUpDay=None):
        super(CudnnLstmModel, self).__init__()
        self.nx = nx
        self.ny = ny
        self.hiddenSize = hiddenSize
        self.ct = 0
        self.nLayer = 1
        self.linearIn = torch.nn.Linear(nx, hiddenSize)
        if torch.__version__ > "1.9":
            # 2021-10-24. SCP: incorporate newer version of torch LSTM to avoid "weights not contiguous on memory" issue
            self.lstm = torch.nn.LSTM(hiddenSize, hiddenSize, 2, dropout=dr)
        else:
            self.lstm = rnn.CudnnLstm(
                inputSize=hiddenSize, hiddenSize=hiddenSize, dr=dr
            )
        self.linearOut = torch.nn.Linear(hiddenSize, ny)
        self.gpu = 1
        self.name = "CudnnLstmModel"
        self.is_legacy = True
        # self.drtest = torch.nn.Dropout(p=0.4)
        self.warmUpDay = warmUpDay

    def forward(self, x, doDropMC=False, dropoutFalse=False):
        """

        :param inputs: a dictionary of input data (x and potentially z data)
        :param doDropMC:
        :param dropoutFalse:
        :return:
        """
        if not self.warmUpDay is None:
            x, warmUpDay = self.extend_day(x, warmUpDay=self.warmUpDay)

        x0 = F.relu(self.linearIn(x))
        if torch.__version__ > "1.9":
            outLSTM, (hn, cn) = self.lstm(x0)
        else:
            outLSTM, (hn, cn) = self.lstm(
                x0, doDropMC=doDropMC, dropoutFalse=dropoutFalse
            )
        # outLSTMdr = self.drtest(outLSTM)
        out = self.linearOut(outLSTM)

        if not self.warmUpDay is None:
            out = self.reduce_day(out, warmUpDay=warmUpDay)

        return out

    def extend_day(self, x, warm_up_day):
        x_num_day = x.shape[0]
        warm_up_day = min(x_num_day, warm_up_day)
        x_select = x[:warm_up_day, :, :]
        x = torch.cat([x_select, x], dim=0)
        return x, warm_up_day

    def reduce_day(self, x, warm_up_day):
        x = x[warm_up_day:,:,:]
        return x