import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
import csv
import numpy as np


class LSTMcell_tied(torch.nn.Module):
    def __init__(
        self,
        *,
        inputSize,
        hiddenSize,
        mode="train",
        dr=0.5,
        drMethod="drX+drW+drC",
        gpu=1,
        seed=42
    ):
        super(LSTMcell_tied, self).__init__()

        self.inputSize = inputSize
        self.hiddenSize = hiddenSize
        self.dr = dr
        self.name = "LSTMcell_tied"
        self.is_legacy = True
        self.seed=seed
        self.w_ih = Parameter(torch.Tensor(hiddenSize * 4, inputSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize * 4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize * 4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize * 4))

        self.drMethod = drMethod.split("+")
        self.gpu = gpu
        self.mode = mode
        if mode == "train":
            self.train(mode=True)
        elif mode == "test":
            self.train(mode=False)
        elif mode == "drMC":
            self.train(mode=False)

        if gpu >= 0:
            self = self.cuda()
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def reset_mask(self, x, h, c):
        self.maskX = createMask(x, self.dr, self.seed)
        self.maskH = createMask(h, self.dr, self.seed)
        self.maskC = createMask(c, self.dr, self.seed)
        self.maskW_ih = createMask(self.w_ih, self.dr, self.seed)
        self.maskW_hh = createMask(self.w_hh, self.dr, self.seed)

    def forward(self, x, hidden, *, resetMask=True, doDropMC=False):
        if self.dr > 0 and (doDropMC is True or self.training is True):
            doDrop = True
        else:
            doDrop = False

        batchSize = x.size(0)
        h0, c0 = hidden
        if h0 is None:
            h0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)
        if c0 is None:
            c0 = x.new_zeros(batchSize, self.hiddenSize, requires_grad=False)

        if self.dr > 0 and self.training is True and resetMask is True:
            self.reset_mask(x, h0, c0)

        if doDrop is True and "drH" in self.drMethod:
            h0 = DropMask.apply(h0, self.maskH, True)

        if doDrop is True and "drX" in self.drMethod:
            x = DropMask.apply(x, self.maskX, True)

        if doDrop is True and "drW" in self.drMethod:
            w_ih = DropMask.apply(self.w_ih, self.maskW_ih, True)
            w_hh = DropMask.apply(self.w_hh, self.maskW_hh, True)
        else:
            # self.w are parameters, while w are not
            w_ih = self.w_ih
            w_hh = self.w_hh

        gates = F.linear(x, w_ih, self.b_ih) + F.linear(h0, w_hh, self.b_hh)
        gate_i, gate_f, gate_c, gate_o = gates.chunk(4, 1)

        gate_i = torch.sigmoid(gate_i)
        gate_f = torch.sigmoid(gate_f)
        gate_c = torch.tanh(gate_c)
        gate_o = torch.sigmoid(gate_o)

        if self.training is True and "drC" in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * torch.tanh(c1)

        return h1, c1
