#!/usr/bin/env python

import math
import torch
import torch.nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

"""
Implementation of LSTM variants.

For now, they only support a sequence size of 1, and are ideal for RL use-cases.
Besides that, they are a stripped-down version of PyTorch's RNN layers.
(no bidirectional, no num_layers, no batch_first)
"""


class untiedLSTMcell(torch.nn.Module):

    """
    """

    def __init__(self, *, xSize, hiddenSize, train=True,
                 dr=0.5, drMethod='gal+sem', gpu=0):
        super(untiedLSTMcell, self).__init__()
        self.xSize = xSize
        self.hiddenSize = xSize
        self.dr = dr

        self.w_xi = Parameter(torch.Tensor(hiddenSize, xSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, xSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, xSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, xSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hiddenSize)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def init_mask(self, x, h, c):
        self.maskXI = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskXF = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskXC = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskXO = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskHI = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskHF = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskHC = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskHO = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskC = c.bernoulli(1-self.dr).div_(1-self.dr)

        self.maskXI = self.maskXI.detach()
        self.maskXF = self.maskXF.detach()
        self.maskXC = self.maskXC.detach()
        self.maskXO = self.maskXO.detach()
        self.maskHI = self.maskHI.detach()
        self.maskHF = self.maskHF.detach()
        self.maskHC = self.maskHC.detach()
        self.maskHO = self.maskHO.detach()
        self.maskC = self.maskC.detach()

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'gal' in self.drMethod:
            h0I = h0.mul(self.maskHI)
            h0F = h0.mul(self.maskHF)
            h0C = h0.mul(self.maskHC)
            h0O = h0.mul(self.maskHO)
        else:
            h0I = h0
            h0F = h0
            h0C = h0
            h0O = h0

        gateI = F.linear(x, self.w_xi)+F.linear(h0I, self.w_hi) + self.b_i
        gateF = F.linear(x, self.w_xf)+F.linear(h0F, self.w_hf) + self.b_f
        gateC = F.linear(x, self.w_xc)+F.linear(h0C, self.w_hc) + self.b_c
        gateO = F.linear(x, self.w_xo)+F.linear(h0O, self.w_ho) + self.b_o

        gateI = F.sigmoid(gateI)
        gateF = F.sigmoid(gateF)
        gateC = F.tanh(gateC)
        gateO = F.sigmoid(gateO)

        if doDrop and 'sem' in self.drMethod:
            gateC = gateC.mul(self.maskC)

        c1 = (gateF * c0) + (gateI * gateC)
        h1 = gateO * F.tanh(c1)

        return h1, c1


class tiedLSTMcell(torch.nn.Module):
    """
    """

    def __init__(self, *, xSize, hiddenSize, train=True,
                 dr=0.5, drMethod='gal+sem', gpu=0):
        super(tiedLSTMcell, self).__init__()

        self.xSize = xSize
        self.hiddenSize = hiddenSize
        self.dr = dr

        self.w_ih = Parameter(torch.Tensor(hiddenSize*4, xSize))
        self.w_hh = Parameter(torch.Tensor(hiddenSize*4, hiddenSize))
        self.b_ih = Parameter(torch.Tensor(hiddenSize*4))
        self.b_hh = Parameter(torch.Tensor(hiddenSize*4))

        self.drMethod = drMethod.split('+')
        self.gpu = gpu
        self.train = train
        if gpu >= 0:
            self = self.cuda(gpu)
            self.is_cuda = True
        else:
            self.is_cuda = False
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hiddenSize)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def init_mask(self, x, h, c):
        self.maskX = x.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskH = h.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskC = c.bernoulli(1-self.dr).div_(1-self.dr)
        self.maskX = self.maskX.detach()
        self.maskH = self.maskH.detach()
        self.maskC = self.maskC.detach()

    def forward(self, x, hidden):
        nbatch = x.size(0)
        doDrop = self.training and self.dr > 0.0

        h0, c0 = hidden

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and 'gal' in self.drMethod:
            h0 = h0.mul(self.maskH)
            # x = x.mul(self.maskX)

        gates = F.linear(x, self.w_ih, self.b_ih) + \
            F.linear(h0, self.w_hh, self.b_hh)
        ingagateIte, gateF, gateC, gateO = gates.chunk(4, 1)

        gateI = F.sigmoid(gateI)
        gateF = F.sigmoid(gateF)
        gateC = F.tanh(gateC)
        gateO = F.sigmoid(gateO)

        if doDrop and 'sem' in self.drMethod:
            gateC = gateC.mul(self.maskC)

        c1 = (gateF * c0) + (gateI * gateC)
        h1 = gateO * F.tanh(c1)

        return h1, c1
