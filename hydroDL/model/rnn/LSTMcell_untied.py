import math
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from hydroDL.model.dropout import DropMask, createMask
import csv
import numpy as np


class LSTMcell_untied(torch.nn.Module):
    def __init__(
        self, *, inputSize, hiddenSize, train=True, dr=0.5, drMethod="gal+sem", gpu=0, seed=42
    ):
        super(LSTMcell_untied, self).__init__()
        self.inputSize = inputSize
        self.hiddenSize = inputSize
        self.dr = dr
        self.seed = seed
        self.name = "LSTMcell_untied"
        self.is_legacy = True

        self.w_xi = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xf = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xo = Parameter(torch.Tensor(hiddenSize, inputSize))
        self.w_xc = Parameter(torch.Tensor(hiddenSize, inputSize))

        self.w_hi = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hf = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_ho = Parameter(torch.Tensor(hiddenSize, hiddenSize))
        self.w_hc = Parameter(torch.Tensor(hiddenSize, hiddenSize))

        self.b_i = Parameter(torch.Tensor(hiddenSize))
        self.b_f = Parameter(torch.Tensor(hiddenSize))
        self.b_o = Parameter(torch.Tensor(hiddenSize))
        self.b_c = Parameter(torch.Tensor(hiddenSize))

        self.drMethod = drMethod.split("+")
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
        self.maskX_i = createMask(x, self.dr, self.seed)
        self.maskX_f = createMask(x, self.dr, self.seed)
        self.maskX_c = createMask(x, self.dr, self.seed)
        self.maskX_o = createMask(x, self.dr, self.seed)

        self.maskH_i = createMask(h, self.dr, self.seed)
        self.maskH_f = createMask(h, self.dr, self.seed)
        self.maskH_c = createMask(h, self.dr, self.seed)
        self.maskH_o = createMask(h, self.dr, self.seed)

        self.maskC = createMask(c, self.dr, self.seed)

        self.maskW_xi = createMask(self.w_xi, self.dr, self.seed)
        self.maskW_xf = createMask(self.w_xf, self.dr, self.seed)
        self.maskW_xc = createMask(self.w_xc, self.dr, self.seed)
        self.maskW_xo = createMask(self.w_xo, self.dr, self.seed)
        self.maskW_hi = createMask(self.w_hi, self.dr, self.seed)
        self.maskW_hf = createMask(self.w_hf, self.dr, self.seed)
        self.maskW_hc = createMask(self.w_hc, self.dr, self.seed)
        self.maskW_ho = createMask(self.w_ho, self.dr, self.seed)

    def forward(self, x, hidden):
        h0, c0 = hidden
        doDrop = self.training and self.dr > 0.0

        if doDrop:
            self.init_mask(x, h0, c0)

        if doDrop and "drH" in self.drMethod:
            h0_i = h0.mul(self.maskH_i)
            h0_f = h0.mul(self.maskH_f)
            h0_c = h0.mul(self.maskH_c)
            h0_o = h0.mul(self.maskH_o)
        else:
            h0_i = h0
            h0_f = h0
            h0_c = h0
            h0_o = h0

        if doDrop and "drX" in self.drMethod:
            x_i = x.mul(self.maskX_i)
            x_f = x.mul(self.maskX_f)
            x_c = x.mul(self.maskX_c)
            x_o = x.mul(self.maskX_o)
        else:
            x_i = x
            x_f = x
            x_c = x
            x_o = x

        if doDrop and "drW" in self.drMethod:
            w_xi = self.w_xi.mul(self.maskW_xi)
            w_xf = self.w_xf.mul(self.maskW_xf)
            w_xc = self.w_xc.mul(self.maskW_xc)
            w_xo = self.w_xo.mul(self.maskW_xo)
            w_hi = self.w_hi.mul(self.maskW_hi)
            w_hf = self.w_hf.mul(self.maskW_hf)
            w_hc = self.w_hc.mul(self.maskW_hc)
            w_ho = self.w_ho.mul(self.maskW_ho)
        else:
            w_xi = self.w_xi
            w_xf = self.w_xf
            w_xc = self.w_xc
            w_xo = self.w_xo
            w_hi = self.w_hi
            w_hf = self.w_hf
            w_hc = self.w_hc
            w_ho = self.w_ho

        gate_i = F.linear(x_i, w_xi) + F.linear(h0_i, w_hi) + self.b_i
        gate_f = F.linear(x_f, w_xf) + F.linear(h0_f, w_hf) + self.b_f
        gate_c = F.linear(x_c, w_xc) + F.linear(h0_c, w_hc) + self.b_c
        gate_o = F.linear(x_o, w_xo) + F.linear(h0_o, w_ho) + self.b_o

        gate_i = F.sigmoid(gate_i)
        gate_f = F.sigmoid(gate_f)
        gate_c = F.tanh(gate_c)
        gate_o = F.sigmoid(gate_o)

        if doDrop and "drC" in self.drMethod:
            gate_c = gate_c.mul(self.maskC)

        c1 = (gate_f * c0) + (gate_i * gate_c)
        h1 = gate_o * F.tanh(c1)

        return h1, c1
